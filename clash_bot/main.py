#!/usr/bin/env python3
"""
Clash Royale - bot autónomo de bucle cerrado (captura -> percepción -> decisión -> acción).

Arquitectura
------------
    ScreenCapture   : localiza la ventana del emulador y la captura con `mss` (~30 FPS).
    Perception      : recorta ROIs fijas (elixir, 4 cartas, arena) y extrae el estado:
                        - elixir por umbral HSV del morado/rosa de la barra
                        - identidad de cada carta por dHash perceptual (64 bits)
                        - amenazas enemigas por máscara de rojo (barras de vida / etiquetas rojas)
    CardMemory      : base de datos auto-aprendida (cards_db.json). Cada hash nuevo se registra
                      solo; al jugarlo se mide la caída de elixir para inferir su coste y se le
                      asigna el nombre/rol del mazo configurado en DECK con ese coste.
    TacticalEngine  : reglas priorizadas y configurables (defensa, castigo de exceso de elixir,
                      push con tanque, exploración de cartas desconocidas).
    HumanInput      : convierte (slot, baldosa) en clics de `pyautogui` con jitter y retrasos.

Todas las coordenadas son FRACCIONES (0..1) del área de juego, por lo que funcionan con
cualquier resolución mientras la relación de aspecto sea la del juego (vertical ~9:16).

Uso
---
    python main.py               # juega
    python main.py --debug       # juega y muestra ventana con las ROIs y el estado
    python main.py --calibrate   # guarda calibration.png con las ROIs dibujadas y sale

Parada de emergencia: mover el ratón a una esquina de la pantalla (FAILSAFE de pyautogui)
o Ctrl+C en la consola.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import mss
import numpy as np
import pyautogui

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CONFIG = {
    # --- Localización del juego ------------------------------------------------
    # Parte del título de la ventana del emulador (BlueStacks, LDPlayer, MuMu...).
    "window_title": "BlueStacks",
    # Si no se encuentra la ventana (o no usas Windows) se usa esta región absoluta.
    "fallback_region": {"left": 0, "top": 0, "width": 540, "height": 960},
    # Márgenes de la ventana que NO son juego (barra de título, barra lateral del emulador),
    # en píxeles. Ajusta con --calibrate.
    "window_margins": {"top": 40, "bottom": 0, "left": 0, "right": 40},

    "target_fps": 30,

    # --- ROIs (fracciones del área de juego) --------------------------------------
    # Barra de elixir: franja horizontal fina bajo las cartas.
    "elixir_roi": (0.285, 0.958, 0.975, 0.972),   # x1, y1, x2, y2
    # Las 4 cartas de la mano.
    "card_rois": [
        (0.235, 0.835, 0.410, 0.945),
        (0.420, 0.835, 0.595, 0.945),
        (0.605, 0.835, 0.780, 0.945),
        (0.790, 0.835, 0.965, 0.945),
    ],
    # Arena jugable completa (18 x 32 baldosas). La fila 0 es la del fondo enemigo.
    "arena_roi": (0.025, 0.075, 0.975, 0.795),
    "arena_tiles": (18, 32),
    "river_rows": (15, 16),

    # --- Umbrales de color (OpenCV HSV: H 0-180, S 0-255, V 0-255) ----------------
    "elixir_hsv_low": (135, 90, 110),
    "elixir_hsv_high": (170, 255, 255),
    # Rojo enemigo (dos rangos porque el rojo envuelve el círculo de H).
    "enemy_hsv_ranges": [((0, 160, 150), (8, 255, 255)), ((170, 160, 150), (180, 255, 255))],
    "card_gray_saturation": 45,   # saturación media por debajo => carta gris (sin elixir)

    # --- Reconocimiento de cartas -------------------------------------------------
    "hash_match_distance": 10,    # distancia de Hamming máx. para considerar misma carta
    "cards_db": "cards_db.json",

    # --- Detección de amenazas ----------------------------------------------------
    "threat_min_area": 12,        # px de rojo mínimos para considerar un blob como unidad
    "threat_heavy_area": 140,     # px de rojo a partir del cual se considera tanque (P.E.K.K.A...)
    "threat_swarm_count": 4,      # nº de blobs en un mismo carril => enjambre

    # --- Ritmo / humanización -----------------------------------------------------
    "min_seconds_between_plays": 1.2,
    "click_jitter_px": 4,
    "click_delay_range": (0.05, 0.16),
    "move_duration_range": (0.04, 0.12),
}

# Mazo configurado: nombre -> (coste, rol). El bot asocia cada carta vista con una de estas
# entradas midiendo su coste real al jugarla. Roles disponibles:
#   tank, tank_killer, splash, spell_small, spell_big, support, cycle, building
DECK = {
    "Giant":        (5, "tank"),
    "Mini P.E.K.K.A": (4, "tank_killer"),
    "Musketeer":    (4, "support"),
    "Valkyrie":     (4, "splash"),
    "Fireball":     (4, "spell_big"),
    "Zap":          (2, "spell_small"),
    "Skeletons":    (1, "cycle"),
    "Cannon":       (3, "building"),
}

# Respuestas defensivas preferidas según el tipo de amenaza (en orden de preferencia).
COUNTERS = {
    "heavy": ["tank_killer", "building", "cycle", "support"],
    "swarm": ["spell_small", "splash", "spell_big", "support"],
    "light": ["splash", "support", "cycle", "tank_killer", "building"],
}

log = logging.getLogger("crbot")


# =============================================================================
# CAPTURA DE PANTALLA
# =============================================================================

class ScreenCapture:
    """Captura el área de juego del emulador con mss (rápido, sin foco necesario)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sct = mss.mss()
        self.region = self._locate()
        log.info("Área de juego: %s", self.region)

    def _locate(self) -> dict:
        m = self.cfg["window_margins"]
        try:
            import pygetwindow as gw  # sólo Windows/macOS
            wins = [w for w in gw.getWindowsWithTitle(self.cfg["window_title"]) if w.width > 100]
            if wins:
                w = wins[0]
                try:
                    if w.isMinimized:
                        w.restore()
                except Exception:
                    pass
                return {
                    "left": w.left + m["left"],
                    "top": w.top + m["top"],
                    "width": w.width - m["left"] - m["right"],
                    "height": w.height - m["top"] - m["bottom"],
                }
            log.warning("Ventana '%s' no encontrada; usando fallback_region.", self.cfg["window_title"])
        except Exception as exc:  # pygetwindow no disponible
            log.warning("pygetwindow no disponible (%s); usando fallback_region.", exc)
        return dict(self.cfg["fallback_region"])

    def grab(self) -> np.ndarray:
        """Devuelve el frame en BGR (formato OpenCV)."""
        raw = np.asarray(self.sct.grab(self.region))      # BGRA
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    # Conversión de fracciones del área de juego a píxeles absolutos de pantalla.
    def to_screen(self, fx: float, fy: float) -> tuple[int, int]:
        r = self.region
        return int(r["left"] + fx * r["width"]), int(r["top"] + fy * r["height"])


# =============================================================================
# PERCEPCIÓN
# =============================================================================

def crop(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def dhash(img: np.ndarray, size: int = 8) -> int:
    """Hash perceptual por diferencias: robusto a escala y pequeños cambios de brillo."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class Threat:
    tile: tuple[int, int]      # (columna, fila) de la arena
    area: int
    lane: str                  # "left" | "right"


@dataclass
class HandCard:
    slot: int
    hash: int
    available: bool            # False si la carta aparece en gris
    name: Optional[str] = None
    cost: Optional[int] = None
    role: Optional[str] = None


@dataclass
class GameState:
    timestamp: float
    in_battle: bool
    elixir: float
    hand: list[HandCard] = field(default_factory=list)
    threats_own_half: list[Threat] = field(default_factory=list)
    threats_enemy_half: list[Threat] = field(default_factory=list)


class Perception:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.e_low = np.array(cfg["elixir_hsv_low"], np.uint8)
        self.e_high = np.array(cfg["elixir_hsv_high"], np.uint8)
        self.cols, self.rows = cfg["arena_tiles"]

    # ---- Elixir ---------------------------------------------------------------
    def read_elixir(self, frame: np.ndarray) -> tuple[float, bool]:
        """Elixir 0..10 midiendo qué fracción horizontal de la barra es morada.

        Devuelve (elixir, barra_visible). Una columna cuenta como llena si >40 % de sus
        píxeles caen dentro del rango HSV del elixir. El llenado es contiguo desde la
        izquierda, así que se toma la última columna llena del primer tramo continuo.
        """
        bar = crop(frame, self.cfg["elixir_roi"])
        if bar.size == 0:
            return 0.0, False
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.e_low, self.e_high)
        col_fill = (mask > 0).mean(axis=0) > 0.4
        visible = col_fill.mean() > 0.02 or self._bar_background_visible(hsv)
        if not col_fill.any():
            return 0.0, visible
        # Primer tramo continuo desde la izquierda (tolerando huecos de 2px por los separadores).
        filled = 0
        gap = 0
        for v in col_fill:
            if v:
                filled += 1 + gap
                gap = 0
            else:
                gap += 1
                if gap > max(3, len(col_fill) // 60):
                    break
        return round(10.0 * filled / len(col_fill), 1), True

    @staticmethod
    def _bar_background_visible(hsv: np.ndarray) -> bool:
        # Con 0 de elixir la barra es oscura y poco saturada; sirve para saber que seguimos en batalla.
        return bool((hsv[..., 2] < 70).mean() > 0.6)

    # ---- Cartas ---------------------------------------------------------------
    def read_hand(self, frame: np.ndarray) -> list[HandCard]:
        hand = []
        for i, roi in enumerate(self.cfg["card_rois"]):
            img = crop(frame, roi)
            h, w = img.shape[:2]
            core = img[int(h * 0.12):int(h * 0.80), int(w * 0.12):int(w * 0.88)]  # sin borde ni coste
            if core.size == 0:
                continue
            sat = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)[..., 1].mean()
            # El hash se calcula sobre la versión en gris para que la misma carta tenga el
            # mismo hash esté o no disponible.
            hand.append(HandCard(slot=i, hash=dhash(core),
                                 available=sat > self.cfg["card_gray_saturation"]))
        return hand

    # ---- Amenazas -------------------------------------------------------------
    def read_threats(self, frame: np.ndarray) -> tuple[list[Threat], list[Threat]]:
        arena = crop(frame, self.cfg["arena_roi"])
        ah, aw = arena.shape[:2]
        hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
        mask = np.zeros((ah, aw), np.uint8)
        for low, high in self.cfg["enemy_hsv_ranges"]:
            mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))

        # Ignorar las torres enemigas (filas superiores) para no confundirlas con tropas.
        mask[: int(ah * 7 / self.rows), :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        own, enemy = [], []
        river_top = self.cfg["river_rows"][0]
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < self.cfg["threat_min_area"]:
                continue
            cx, cy = centroids[k]
            col = int(np.clip(cx / aw * self.cols, 0, self.cols - 1))
            row = int(np.clip(cy / ah * self.rows, 0, self.rows - 1))
            t = Threat(tile=(col, row), area=area, lane="left" if col < self.cols / 2 else "right")
            (own if row >= river_top else enemy).append(t)
        return own, enemy

    # ---- Estado completo ------------------------------------------------------
    def analyse(self, frame: np.ndarray) -> GameState:
        elixir, bar_visible = self.read_elixir(frame)
        own, enemy = self.read_threats(frame) if bar_visible else ([], [])
        return GameState(
            timestamp=time.time(),
            in_battle=bar_visible,
            elixir=elixir,
            hand=self.read_hand(frame) if bar_visible else [],
            threats_own_half=own,
            threats_enemy_half=enemy,
        )


# =============================================================================
# MEMORIA DE CARTAS (auto-aprendizaje, sin recortes manuales)
# =============================================================================

class CardMemory:
    """Asocia hashes visuales a cartas del mazo. Persiste en JSON entre partidas."""

    def __init__(self, path: Path, deck: dict, max_dist: int):
        self.path = path
        self.deck = deck
        self.max_dist = max_dist
        self.entries: list[dict] = []    # {"hash": int, "name": str|None, "cost_samples": [..]}
        if path.exists():
            try:
                self.entries = json.loads(path.read_text())
                log.info("Cargadas %d cartas aprendidas de %s", len(self.entries), path)
            except Exception:
                log.warning("cards_db corrupto; empezando de cero.")

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False))

    def _find(self, h: int) -> Optional[dict]:
        best, best_d = None, self.max_dist + 1
        for e in self.entries:
            d = hamming(h, e["hash"])
            if d < best_d:
                best, best_d = e, d
        return best

    def enrich(self, card: HandCard) -> None:
        e = self._find(card.hash)
        if e is None:
            e = {"hash": card.hash, "name": None, "cost_samples": []}
            self.entries.append(e)
            log.info("Nueva carta vista en slot %d (hash %016x)", card.slot + 1, card.hash)
            self.save()
        if e["name"]:
            card.name = e["name"]
            card.cost, card.role = self.deck[e["name"]]

    def learn_cost(self, h: int, measured_drop: float) -> None:
        """Registra la caída de elixir observada al jugar la carta y la etiqueta."""
        e = self._find(h)
        if e is None or e["name"]:
            return
        e["cost_samples"].append(measured_drop)
        cost = int(round(float(np.median(e["cost_samples"]))))
        taken = {x["name"] for x in self.entries if x["name"]}
        candidates = [n for n, (c, _) in self.deck.items() if c == cost and n not in taken]
        if not candidates:  # coste medido ligeramente desviado: probar ±1
            candidates = [n for n, (c, _) in self.deck.items()
                          if abs(c - cost) == 1 and n not in taken]
        if candidates and (len(e["cost_samples"]) >= 2 or len(candidates) == 1):
            e["name"] = candidates[0]
            log.info("Carta aprendida: %016x -> %s (coste %d)", h, e["name"], self.deck[e["name"]][0])
        self.save()


# =============================================================================
# MOTOR TÁCTICO
# =============================================================================

@dataclass
class Action:
    slot: int
    tile: tuple[int, int]      # (columna, fila)
    reason: str
    card_hash: int = 0


class TacticalEngine:
    """Reglas evaluadas por prioridad; la primera que devuelve una acción gana.

    Para añadir una regla: escribe un método `rule_xxx(self, state) -> Optional[Action]`
    y añádelo a `self.rules` en la posición de prioridad deseada.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cols, self.rows = cfg["arena_tiles"]
        self.rules: list[Callable[[GameState], Optional[Action]]] = [
            self.rule_defend,
            self.rule_counter_push,
            self.rule_tank_push,
            self.rule_prevent_leak,
            self.rule_explore_unknown,
        ]

    # ---- utilidades -----------------------------------------------------------
    @staticmethod
    def _playable(state: GameState) -> list[HandCard]:
        return [c for c in state.hand if c.available and
                (c.cost is None or c.cost <= state.elixir)]

    def _by_role(self, state: GameState, roles: list[str]) -> Optional[HandCard]:
        cards = self._playable(state)
        for role in roles:
            for c in sorted(cards, key=lambda c: c.cost or 99):
                if c.role == role:
                    return c
        return None

    def _clamp_own(self, col: int, row: int) -> tuple[int, int]:
        return int(np.clip(col, 0, self.cols - 1)), int(np.clip(row, self.cfg["river_rows"][1] + 2, self.rows - 1))

    def _lane_col(self, lane: str, back: bool = False) -> int:
        # Columnas de los puentes / detrás de las torres de princesa.
        return (3 if lane == "left" else self.cols - 4) if not back else (2 if lane == "left" else self.cols - 3)

    @staticmethod
    def _classify(threats: list[Threat], heavy_area: int, swarm_n: int) -> str:
        if any(t.area >= heavy_area for t in threats):
            return "heavy"
        if len(threats) >= swarm_n:
            return "swarm"
        return "light"

    # ---- reglas ---------------------------------------------------------------
    def rule_defend(self, s: GameState) -> Optional[Action]:
        """Hay tropas enemigas en nuestra mitad: responder con el counter adecuado."""
        if not s.threats_own_half:
            return None
        lanes: dict[str, list[Threat]] = {}
        for t in s.threats_own_half:
            lanes.setdefault(t.lane, []).append(t)
        lane, threats = max(lanes.items(), key=lambda kv: sum(t.area for t in kv[1]))
        kind = self._classify(threats, self.cfg["threat_heavy_area"], self.cfg["threat_swarm_count"])
        card = self._by_role(s, COUNTERS[kind])
        if card is None:
            return None
        lead = max(threats, key=lambda t: t.tile[1])          # el más adelantado (más cerca de nuestra torre)
        cx = int(np.mean([t.tile[0] for t in threats]))
        cy = int(np.mean([t.tile[1] for t in threats]))
        if card.role and card.role.startswith("spell"):
            tile = (cx, cy)                                    # hechizo encima del grupo
        elif card.role == "building":
            tile = self._clamp_own(self.cols // 2 + (-1 if lane == "left" else 1), 23)  # centro, tira la tropa
        else:
            tile = self._clamp_own(lead.tile[0], lead.tile[1] + 3)  # justo delante de la amenaza
        return Action(card.slot, tile, f"defensa {kind} en carril {lane} con {card.name}", card.hash)

    def rule_counter_push(self, s: GameState) -> Optional[Action]:
        """Tras defender, si el enemigo tiene poca presencia y hay elixir, apoyar el carril."""
        if s.threats_own_half or s.elixir < 6:
            return None
        card = self._by_role(s, ["support", "splash"])
        if card is None or not s.threats_enemy_half:
            return None
        # Apoya el carril con menos enemigos visibles.
        left = sum(t.area for t in s.threats_enemy_half if t.lane == "left")
        right = sum(t.area for t in s.threats_enemy_half if t.lane == "right")
        lane = "left" if left < right else "right"
        return Action(card.slot, self._clamp_own(self._lane_col(lane), 19),
                      f"apoyo en puente {lane} con {card.name}", card.hash)

    def rule_tank_push(self, s: GameState) -> Optional[Action]:
        """Elixir >= 7 y el rival no ataca: tanque desde atrás."""
        if s.elixir < 7 or s.threats_own_half:
            return None
        card = self._by_role(s, ["tank"])
        if card is None:
            return None
        lane = random.choice(["left", "right"])
        return Action(card.slot, (self.cols // 2 + (-1 if lane == "left" else 0), self.rows - 1),
                      f"tanque atrás ({lane}) con {card.name}", card.hash)

    def rule_prevent_leak(self, s: GameState) -> Optional[Action]:
        """Con el elixir casi lleno, no desperdiciar: jugar algo barato en la retaguardia."""
        if s.elixir < 9.5:
            return None
        cards = [c for c in self._playable(s) if not (c.role or "").startswith("spell")]
        if not cards:
            return None
        card = min(cards, key=lambda c: c.cost or 5)
        lane = random.choice(["left", "right"])
        return Action(card.slot, self._clamp_own(self._lane_col(lane, back=True), self.rows - 2),
                      f"evitar fuga de elixir con {card.name or 'carta desconocida'}", card.hash)

    def rule_explore_unknown(self, s: GameState) -> Optional[Action]:
        """Jugar cartas aún no identificadas en zona segura para medir su coste."""
        if s.elixir < 8 or s.threats_own_half:
            return None
        unknown = [c for c in s.hand if c.available and c.name is None]
        if not unknown:
            return None
        card = random.choice(unknown)
        return Action(card.slot, (self.cols // 2, self.rows - 2),
                      f"explorar carta desconocida (slot {card.slot + 1})", card.hash)

    def decide(self, state: GameState) -> Optional[Action]:
        for rule in self.rules:
            action = rule(state)
            if action:
                return action
        return None


# =============================================================================
# EJECUCIÓN DE CLICS HUMANIZADA
# =============================================================================

class HumanInput:
    def __init__(self, cfg: dict, capture: ScreenCapture):
        self.cfg = cfg
        self.cap = capture
        pyautogui.FAILSAFE = True   # ratón a una esquina => excepción y parada
        pyautogui.PAUSE = 0

    def _tap(self, x: int, y: int) -> None:
        j = self.cfg["click_jitter_px"]
        x += random.randint(-j, j)
        y += random.randint(-j, j)
        pyautogui.moveTo(x, y, duration=random.uniform(*self.cfg["move_duration_range"]),
                         tween=pyautogui.easeOutQuad)
        time.sleep(random.uniform(0.01, 0.04))
        pyautogui.click()
        time.sleep(random.uniform(*self.cfg["click_delay_range"]))

    def slot_center(self, slot: int) -> tuple[int, int]:
        x1, y1, x2, y2 = self.cfg["card_rois"][slot]
        return self.cap.to_screen((x1 + x2) / 2, (y1 + y2) / 2)

    def tile_center(self, col: int, row: int) -> tuple[int, int]:
        x1, y1, x2, y2 = self.cfg["arena_roi"]
        cols, rows = self.cfg["arena_tiles"]
        fx = x1 + (col + 0.5) / cols * (x2 - x1)
        fy = y1 + (row + 0.5) / rows * (y2 - y1)
        return self.cap.to_screen(fx, fy)

    def play(self, action: Action) -> None:
        self._tap(*self.slot_center(action.slot))
        self._tap(*self.tile_center(*action.tile))


# =============================================================================
# DEPURACIÓN VISUAL
# =============================================================================

def draw_overlay(frame: np.ndarray, cfg: dict, state: Optional[GameState]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    def rect(roi, color):
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, 1)

    rect(cfg["elixir_roi"], (255, 0, 255))
    for roi in cfg["card_rois"]:
        rect(roi, (0, 255, 255))
    rect(cfg["arena_roi"], (0, 255, 0))
    ax1, ay1, ax2, ay2 = cfg["arena_roi"]
    cols, rows = cfg["arena_tiles"]
    for r in range(rows + 1):
        y = int((ay1 + r / rows * (ay2 - ay1)) * h)
        cv2.line(out, (int(ax1 * w), y), (int(ax2 * w), y), (0, 90, 0), 1)
    for c in range(cols + 1):
        x = int((ax1 + c / cols * (ax2 - ax1)) * w)
        cv2.line(out, (x, int(ay1 * h)), (x, int(ay2 * h)), (0, 90, 0), 1)

    if state:
        cv2.putText(out, f"Elixir {state.elixir:.1f}  batalla={state.in_battle}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for c in state.hand:
            x1, y1, _, _ = cfg["card_rois"][c.slot]
            label = f"{c.name or '?'}{'' if c.available else ' (gris)'}"
            cv2.putText(out, label, (int(x1 * w), int(y1 * h) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        for t in state.threats_own_half + state.threats_enemy_half:
            fx = ax1 + (t.tile[0] + 0.5) / cols * (ax2 - ax1)
            fy = ay1 + (t.tile[1] + 0.5) / rows * (ay2 - ay1)
            cv2.circle(out, (int(fx * w), int(fy * h)), 6, (0, 0, 255), 2)
    return out


# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================

class Bot:
    def __init__(self, cfg: dict, debug: bool):
        self.cfg = cfg
        self.debug = debug
        self.capture = ScreenCapture(cfg)
        self.perception = Perception(cfg)
        self.memory = CardMemory(Path(__file__).with_name(cfg["cards_db"]), DECK, cfg["hash_match_distance"])
        self.engine = TacticalEngine(cfg)
        self.input = HumanInput(cfg, self.capture)
        self.last_play = 0.0
        self.pending_cost: Optional[tuple[int, float, float]] = None   # (hash, elixir_antes, t)

    def _check_cost_measurement(self, state: GameState) -> None:
        """~0.5 s después de jugar, la caída de elixir indica el coste de la carta."""
        if not self.pending_cost:
            return
        h, before, t0 = self.pending_cost
        dt = time.time() - t0
        if dt < 0.5:
            return
        regen = dt / 2.8                      # regeneración normal ≈ 1 elixir / 2.8 s
        drop = before - state.elixir + regen
        if 0.5 <= drop <= 10.5:
            self.memory.learn_cost(h, drop)
        self.pending_cost = None

    def step(self) -> Optional[GameState]:
        frame = self.capture.grab()
        state = self.perception.analyse(frame)
        if state.in_battle:
            for card in state.hand:
                self.memory.enrich(card)
            self._check_cost_measurement(state)

            if time.time() - self.last_play >= self.cfg["min_seconds_between_plays"] and not self.pending_cost:
                action = self.engine.decide(state)
                if action:
                    log.info("[%.1f elixir] %s -> slot %d, baldosa %s",
                             state.elixir, action.reason, action.slot + 1, action.tile)
                    self.input.play(action)
                    self.last_play = time.time()
                    self.pending_cost = (action.card_hash, state.elixir, self.last_play)

        if self.debug:
            cv2.imshow("crbot debug", draw_overlay(frame, self.cfg, state))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
        return state

    def run(self) -> None:
        period = 1.0 / self.cfg["target_fps"]
        log.info("Bot en marcha. Ctrl+C o ratón a una esquina para detener.")
        frames, t_fps = 0, time.time()
        while True:
            t0 = time.perf_counter()
            self.step()
            frames += 1
            if time.time() - t_fps >= 10:
                log.debug("FPS medio: %.1f", frames / (time.time() - t_fps))
                frames, t_fps = 0, time.time()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def calibrate(cfg: dict) -> None:
    cap = ScreenCapture(cfg)
    frame = cap.grab()
    state = Perception(cfg).analyse(frame)
    out = Path(__file__).with_name("calibration.png")
    cv2.imwrite(str(out), draw_overlay(frame, cfg, state))
    print(f"Guardado {out}. Elixir leído: {state.elixir}. Ajusta las ROIs en CONFIG si no encajan.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bot autónomo de Clash Royale para emulador.")
    parser.add_argument("--debug", action="store_true", help="muestra ventana con ROIs y estado")
    parser.add_argument("--calibrate", action="store_true", help="guarda calibration.png y sale")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.calibrate:
        calibrate(CONFIG)
        return
    bot = Bot(CONFIG, debug=args.debug)
    time.sleep(2)  # tiempo para poner el emulador en primer plano
    try:
        bot.run()
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        log.info("Detenido por el usuario.")
    finally:
        bot.memory.save()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())

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
    YoloDetector    : (opcional) detecta tropas concretas (P.E.K.K.A, Montapuercos...) con un
                      modelo YOLO de ultralytics; el bando se deduce del color del indicador.
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

Detección de tropas con YOLO (opcional; sin modelo se usa la detección por color)
    python main.py --collect [--no-play]   # 1) guarda frames de la arena en dataset/images
    #                                        2) etiqueta las tropas (CVAT / Label Studio / Roboflow)
    #                                           y exporta en formato YOLO con su data.yaml
    python main.py --train dataset/data.yaml   # 3) entrena y deja cr_yolo.pt junto a main.py
    python main.py                             # 4) el bot usa YOLO automáticamente

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

    # --- Detección de tropas con YOLO (ultralytics) -------------------------------
    # Si el modelo no existe o ultralytics no está instalado, se usa automáticamente la
    # detección por manchas rojas de arriba.
    "yolo": {
        "enabled": True,
        "model": "cr_yolo.pt",        # ruta relativa a main.py (sale de --train)
        "base_weights": "yolo11n.pt", # pesos preentrenados para --train (se descargan solos)
        "conf": 0.40,
        "iou": 0.50,
        "imgsz": 640,
        "device": None,               # None = auto (GPU si hay), "cpu", "0"...
        "every_n_frames": 2,          # inferencia 1 de cada N frames (reutiliza el resultado)
        # Colores del indicador de nivel / barra de vida que llevan las tropas encima.
        "team_hsv_blue": ((95, 140, 140), (125, 255, 255)),
    },
    # Recolección automática de frames para crear el dataset (--collect).
    "dataset_dir": "dataset",
    "collect_interval_s": 1.5,

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
    "air": ["support", "spell_big", "building", "spell_small"],
    "building": ["spell_big", "tank", "tank_killer"],
}

# Perfil de cada clase que puede devolver el modelo YOLO -> tipo de amenaza.
# Los nombres se normalizan (minúsculas, espacios/guiones -> "_", sin prefijo enemy_/ally_).
# Las clases que no estén aquí se tratan como "light". "ignore" = no es una amenaza.
TROOP_PROFILES = {
    "pekka": "heavy", "mega_knight": "heavy", "giant": "heavy", "golem": "heavy",
    "royal_giant": "heavy", "hog_rider": "heavy", "ram_rider": "heavy", "elixir_golem": "heavy",
    "electro_giant": "heavy", "goblin_giant": "heavy", "giant_skeleton": "heavy",
    "balloon": "air", "lava_hound": "air", "baby_dragon": "air", "minions": "air",
    "minion_horde": "air", "bats": "air", "inferno_dragon": "air", "mega_minion": "air",
    "skeleton_army": "swarm", "goblin_gang": "swarm", "barbarians": "swarm", "skeletons": "swarm",
    "goblins": "swarm", "spear_goblins": "swarm", "guards": "swarm", "royal_recruits": "swarm",
    "x_bow": "building", "mortar": "building", "goblin_hut": "building", "furnace": "building",
    "tombstone": "building", "barbarian_hut": "building",
    "king_tower": "ignore", "princess_tower": "ignore", "arrows": "ignore", "fireball": "ignore",
    "zap": "ignore", "log": "ignore", "clock": "ignore",
}

# Counters específicos por tropa (tienen prioridad sobre COUNTERS genéricos).
SPECIFIC_COUNTERS = {
    "pekka": ["tank_killer", "building", "cycle"],
    "mega_knight": ["tank_killer", "building", "support"],
    "hog_rider": ["building", "tank_killer", "cycle"],
    "balloon": ["support", "building"],
    "skeleton_army": ["spell_small", "splash"],
    "goblin_gang": ["spell_small", "splash"],
    "minion_horde": ["spell_big", "support"],
    "x_bow": ["spell_big", "tank"],
}

KIND_PRIORITY = ["heavy", "building", "air", "swarm", "light"]

TEAM_PREFIXES = {"enemy_": "enemy", "red_": "enemy", "opp_": "enemy",
                 "ally_": "ally", "own_": "ally", "blue_": "ally"}

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
    label: Optional[str] = None        # clase YOLO normalizada (p.ej. "pekka"); None = blob rojo
    kind: Optional[str] = None         # heavy | air | swarm | light | building
    conf: float = 1.0
    bbox: Optional[tuple[int, int, int, int]] = None   # px dentro del recorte de la arena


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


def normalize_label(name: str) -> tuple[str, Optional[str]]:
    """'Enemy PEKKA' -> ('pekka', 'enemy'); 'mini-pekka' -> ('mini_pekka', None)."""
    n = name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")
    for prefix, team in TEAM_PREFIXES.items():
        if n.startswith(prefix):
            return n[len(prefix):], team
    return n, None


class YoloDetector:
    """Detector de tropas con un modelo YOLO entrenado sobre la arena de Clash Royale.

    El bando de cada detección sale del prefijo de la clase (enemy_/ally_) si el modelo
    lo distingue, o si no del color del indicador que la tropa lleva encima
    (rojo = enemigo, azul = aliado).
    """

    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg
        self.ycfg = cfg["yolo"]
        self.ok = False
        self.model = None
        if not self.ycfg["enabled"]:
            return
        path = base_dir / self.ycfg["model"]
        if not path.exists():
            log.warning("Modelo YOLO '%s' no encontrado: uso detección por color. "
                        "Crea uno con --collect y --train.", path)
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            log.warning("ultralytics no instalado: uso detección por color.")
            return
        self.model = YOLO(str(path))
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.red_ranges = [(np.array(lo, np.uint8), np.array(hi, np.uint8))
                           for lo, hi in cfg["enemy_hsv_ranges"]]
        lo, hi = self.ycfg["team_hsv_blue"]
        self.blue = (np.array(lo, np.uint8), np.array(hi, np.uint8))
        self.ok = True
        log.info("YOLO cargado (%d clases) desde %s", len(self.names), path)

    def _team_by_color(self, hsv: np.ndarray, box: tuple[int, int, int, int]) -> str:
        x1, y1, x2, y2 = box
        h = max(1, y2 - y1)
        # El indicador de nivel / barra de vida está en la parte alta de la caja o justo encima.
        top = max(0, y1 - h // 3)
        roi = hsv[top:y1 + h // 3, max(0, x1):max(x1 + 1, x2)]
        if roi.size == 0:
            return "enemy"
        red = sum(int(cv2.inRange(roi, lo, hi).sum()) for lo, hi in self.red_ranges)
        blue = int(cv2.inRange(roi, *self.blue).sum())
        return "enemy" if red >= blue else "ally"

    def detect(self, arena: np.ndarray) -> list[tuple[str, str, float, tuple[int, int, int, int]]]:
        """Devuelve [(label, team, conf, (x1, y1, x2, y2)), ...] en px del recorte."""
        res = self.model.predict(arena, conf=self.ycfg["conf"], iou=self.ycfg["iou"],
                                 imgsz=self.ycfg["imgsz"], device=self.ycfg["device"],
                                 verbose=False)[0]
        hsv = None
        out = []
        for xyxy, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                   res.boxes.conf.tolist()):
            label, team = normalize_label(self.names[int(cls)])
            box = tuple(int(v) for v in xyxy)
            if team is None:
                if hsv is None:
                    hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
                team = self._team_by_color(hsv, box)
            out.append((label, team, float(conf), box))
        return out


class Perception:
    def __init__(self, cfg: dict, detector: Optional[YoloDetector] = None):
        self.cfg = cfg
        self.detector = detector if detector and detector.ok else None
        self._frame_idx = 0
        self._cached_threats: tuple[list[Threat], list[Threat]] = ([], [])
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
        if self.detector is None:
            return self.read_threats_color(frame)
        self._frame_idx += 1
        if self._frame_idx % self.cfg["yolo"]["every_n_frames"] == 0 or self._frame_idx == 1:
            self._cached_threats = self.read_threats_yolo(frame)
        return self._cached_threats

    def read_threats_yolo(self, frame: np.ndarray) -> tuple[list[Threat], list[Threat]]:
        arena = crop(frame, self.cfg["arena_roi"])
        ah, aw = arena.shape[:2]
        own, enemy = [], []
        river_top = self.cfg["river_rows"][0]
        for label, team, conf, (x1, y1, x2, y2) in self.detector.detect(arena):
            kind = TROOP_PROFILES.get(label, "light")
            if team != "enemy" or kind == "ignore":
                continue
            # Posición = pies de la tropa (base de la caja), que es lo que importa en la arena.
            cx, cy = (x1 + x2) / 2, y2 - (y2 - y1) * 0.15
            col = int(np.clip(cx / aw * self.cols, 0, self.cols - 1))
            row = int(np.clip(cy / ah * self.rows, 0, self.rows - 1))
            t = Threat(tile=(col, row), area=(x2 - x1) * (y2 - y1),
                       lane="left" if col < self.cols / 2 else "right",
                       label=label, kind=kind, conf=conf, bbox=(x1, y1, x2, y2))
            (own if row >= river_top else enemy).append(t)
        return own, enemy

    def read_threats_color(self, frame: np.ndarray) -> tuple[list[Threat], list[Threat]]:
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
    def _classify(threats: list[Threat], heavy_area: int, swarm_n: int) -> tuple[str, Optional[str]]:
        """Devuelve (tipo de amenaza, tropa concreta más peligrosa si YOLO la identificó)."""
        labelled = [t for t in threats if t.kind]
        if labelled:
            for kind in KIND_PRIORITY:
                group = [t for t in labelled if t.kind == kind]
                if group:
                    if kind == "light" and len(labelled) >= swarm_n:
                        return "swarm", None
                    top = max(group, key=lambda t: t.conf)
                    return kind, top.label
        # Sin etiquetas (detección por color): heurística por tamaño / número de manchas.
        if any(t.area >= heavy_area for t in threats):
            return "heavy", None
        if len(threats) >= swarm_n:
            return "swarm", None
        return "light", None

    # ---- reglas ---------------------------------------------------------------
    def rule_defend(self, s: GameState) -> Optional[Action]:
        """Hay tropas enemigas en nuestra mitad: responder con el counter adecuado."""
        if not s.threats_own_half:
            return None
        lanes: dict[str, list[Threat]] = {}
        for t in s.threats_own_half:
            lanes.setdefault(t.lane, []).append(t)
        lane, threats = max(lanes.items(), key=lambda kv: sum(t.area for t in kv[1]))
        kind, label = self._classify(threats, self.cfg["threat_heavy_area"], self.cfg["threat_swarm_count"])
        card = self._by_role(s, SPECIFIC_COUNTERS.get(label, COUNTERS[kind])) if label else None
        card = card or self._by_role(s, COUNTERS[kind])
        if card is None:
            return None
        targets = [t for t in threats if t.label == label] if label else threats
        lead = max(targets, key=lambda t: t.tile[1])          # el más adelantado (más cerca de nuestra torre)
        cx = int(np.mean([t.tile[0] for t in threats]))
        cy = int(np.mean([t.tile[1] for t in threats]))
        if card.role and card.role.startswith("spell"):
            tile = (cx, cy)                                    # hechizo encima del grupo
        elif card.role == "building":
            tile = self._clamp_own(self.cols // 2 + (-1 if lane == "left" else 1), 23)  # centro, tira la tropa
        else:
            tile = self._clamp_own(lead.tile[0], lead.tile[1] + 3)  # justo delante de la amenaza
        return Action(card.slot, tile, f"defensa {label or kind} en carril {lane} con {card.name}", card.hash)

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
            if t.bbox:   # detección YOLO: caja + etiqueta
                bx1, by1, bx2, by2 = t.bbox
                ox, oy = int(ax1 * w), int(ay1 * h)
                cv2.rectangle(out, (ox + bx1, oy + by1), (ox + bx2, oy + by2), (0, 0, 255), 1)
                cv2.putText(out, f"{t.label} {t.conf:.2f}", (ox + bx1, oy + by1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            else:        # detección por color: círculo en la baldosa
                fx = ax1 + (t.tile[0] + 0.5) / cols * (ax2 - ax1)
                fy = ay1 + (t.tile[1] + 0.5) / rows * (ay2 - ay1)
                cv2.circle(out, (int(fx * w), int(fy * h)), 6, (0, 0, 255), 2)
    return out


# =============================================================================
# DATASET Y ENTRENAMIENTO YOLO
# =============================================================================

class DatasetCollector:
    """Guarda recortes de la arena durante las partidas para entrenar YOLO.

    Junto a cada imagen escribe una pre-etiqueta en formato YOLO (clase 0 = "enemy_unit")
    generada con la detección por color, para que en la herramienta de etiquetado
    (CVAT, Label Studio, Roboflow...) sólo haya que corregir cajas y poner el nombre de la tropa.
    """

    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg
        self.img_dir = base_dir / cfg["dataset_dir"] / "images"
        self.lbl_dir = base_dir / cfg["dataset_dir"] / "prelabels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.color = Perception(cfg)       # sin detector => manchas rojas
        self.last = 0.0
        self.count = 0

    def maybe_save(self, frame: np.ndarray, state: GameState) -> None:
        if not state.in_battle or time.time() - self.last < self.cfg["collect_interval_s"]:
            return
        self.last = time.time()
        arena = crop(frame, self.cfg["arena_roi"])
        name = f"cr_{int(self.last * 1000)}"
        cv2.imwrite(str(self.img_dir / f"{name}.jpg"), arena, [cv2.IMWRITE_JPEG_QUALITY, 92])
        own, enemy = self.color.read_threats_color(frame)
        cols, rows = self.cfg["arena_tiles"]
        lines = []
        for t in own + enemy:
            # Caja aproximada de ~1.5 x 2 baldosas alrededor de la mancha (la tropa queda debajo).
            cx, cy = (t.tile[0] + 0.5) / cols, (t.tile[1] + 1.0) / rows
            lines.append(f"0 {cx:.5f} {cy:.5f} {1.5 / cols:.5f} {2.0 / rows:.5f}")
        (self.lbl_dir / f"{name}.txt").write_text("\n".join(lines))
        self.count += 1
        if self.count % 50 == 0:
            log.info("Dataset: %d imágenes guardadas en %s", self.count, self.img_dir)


def train_yolo(cfg: dict, data_yaml: str, epochs: int) -> None:
    """Entrena (fine-tuning) YOLO sobre el dataset etiquetado y deja el modelo en cfg['yolo']['model']."""
    import shutil
    from ultralytics import YOLO

    y = cfg["yolo"]
    base_dir = Path(__file__).parent
    model = YOLO(y["base_weights"])
    results = model.train(data=data_yaml, epochs=epochs, imgsz=y["imgsz"], device=y["device"],
                          project=str(base_dir / "runs"), name="cr_yolo", exist_ok=True,
                          # El juego no se ve nunca volteado en vertical ni con tonos cambiados:
                          # el color rojo/azul es información (bando), así que se limita hsv_h.
                          flipud=0.0, fliplr=0.5, hsv_h=0.0, mosaic=1.0)
    best = Path(results.save_dir) / "weights" / "best.pt"
    dst = base_dir / y["model"]
    shutil.copy(best, dst)
    log.info("Modelo entrenado copiado a %s", dst)


# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================

class Bot:
    def __init__(self, cfg: dict, debug: bool, collect: bool = False, play: bool = True):
        base_dir = Path(__file__).parent
        self.cfg = cfg
        self.debug = debug
        self.play = play
        self.capture = ScreenCapture(cfg)
        self.perception = Perception(cfg, YoloDetector(cfg, base_dir))
        self.collector = DatasetCollector(cfg, base_dir) if collect else None
        self.memory = CardMemory(base_dir / cfg["cards_db"], DECK, cfg["hash_match_distance"])
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
            if self.collector:
                self.collector.maybe_save(frame, state)

            if self.play and time.time() - self.last_play >= self.cfg["min_seconds_between_plays"] and not self.pending_cost:
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
    state = Perception(cfg, YoloDetector(cfg, Path(__file__).parent)).analyse(frame)
    out = Path(__file__).with_name("calibration.png")
    cv2.imwrite(str(out), draw_overlay(frame, cfg, state))
    print(f"Guardado {out}. Elixir leído: {state.elixir}. Ajusta las ROIs en CONFIG si no encajan.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bot autónomo de Clash Royale para emulador.")
    parser.add_argument("--debug", action="store_true", help="muestra ventana con ROIs y estado")
    parser.add_argument("--calibrate", action="store_true", help="guarda calibration.png y sale")
    parser.add_argument("--collect", action="store_true",
                        help="guarda frames de la arena en dataset/ para entrenar YOLO")
    parser.add_argument("--no-play", action="store_true",
                        help="sólo observa (útil con --collect mientras juegas tú)")
    parser.add_argument("--train", metavar="DATA_YAML",
                        help="entrena YOLO con un dataset etiquetado (formato ultralytics) y sale")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.train:
        train_yolo(CONFIG, args.train, args.epochs)
        return
    if args.calibrate:
        calibrate(CONFIG)
        return
    bot = Bot(CONFIG, debug=args.debug, collect=args.collect, play=not args.no_play)
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

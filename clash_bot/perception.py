"""Visión: captura de pantalla, lectura de elixir/cartas y detección de tropas."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import TEAM_PREFIXES, TROOP_PROFILES

log = logging.getLogger("crbot")


# =============================================================================
# CAPTURA
# =============================================================================

def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    best, start = (0, 0), None
    for i, v in enumerate(list(flags) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def find_game_rect(img: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Localiza el juego (franja vertical 9:16) dentro de una ventana con bandas negras.

    Devuelve (x1, y1, x2, y2) en píxeles de la imagen, o None si no se ve claro.
    """
    bright = img[..., :3].max(axis=2) > 24
    x1, x2 = _longest_run(bright.mean(axis=0) > 0.3)
    if x2 - x1 < 100:
        return None
    y1, y2 = _longest_run(bright[:, x1:x2].mean(axis=1) > 0.3)
    expected_h = (x2 - x1) * 16 / 9
    if y2 - y1 > expected_h * 1.03:      # sobra por arriba: barra de título del emulador
        y1 = y2 - int(expected_h)
    if y2 - y1 < 150 or not 0.42 <= (x2 - x1) / (y2 - y1) <= 0.66:
        return None
    return x1, y1, x2, y2


class ScreenCapture:
    """Captura el área de juego del emulador con mss."""

    def __init__(self, cfg: dict):
        import mss
        self.cfg = cfg
        self.sct = mss.mss()
        self.found = False
        self.window = None
        self.region = self._locate()
        log.info("Área de juego: %s", self.region)

    def _locate(self) -> dict:
        m = self.cfg["window_margins"]
        try:
            import pygetwindow as gw
            wins = [w for w in gw.getWindowsWithTitle(self.cfg["window_title"]) if w.width > 100]
            if wins:
                w = wins[0]
                try:
                    if w.isMinimized:
                        w.restore()
                except Exception:
                    pass
                self.found = True
                self.window = w
                return {"left": w.left + m["left"], "top": w.top + m["top"],
                        "width": w.width - m["left"] - m["right"],
                        "height": w.height - m["top"] - m["bottom"]}
            log.warning("Ventana '%s' no encontrada; usando fallback_region.", self.cfg["window_title"])
        except Exception as exc:
            log.warning("pygetwindow no disponible (%s); usando fallback_region.", exc)
        return dict(self.cfg["fallback_region"])

    def focus(self) -> None:
        """Pone la ventana del emulador delante (la captura toma lo que se ve en pantalla)."""
        if self.window is None:
            return
        try:
            if self.window.isMinimized:
                self.window.restore()
            self.window.activate()
        except Exception:
            try:   # truco habitual cuando Windows no deja activar la ventana directamente
                self.window.minimize()
                self.window.restore()
            except Exception as exc:
                log.warning("No he podido poner el emulador delante: %s", exc)
        time.sleep(0.8)
        self.region = self._locate()

    def auto_fit(self) -> Optional[dict]:
        """Ajusta los márgenes solo, buscando el juego dentro de la ventana del emulador."""
        if self.window is None:
            return None
        w = self.window
        full = {"left": w.left, "top": w.top, "width": w.width, "height": w.height}
        rect = find_game_rect(np.asarray(self.sct.grab(full)))
        if rect is None:
            return None
        x1, y1, x2, y2 = rect
        margins = {"top": int(y1), "bottom": int(w.height - y2), "left": int(x1), "right": int(w.width - x2)}
        self.cfg["window_margins"] = margins
        self.region = self._locate()
        log.info("Juego localizado dentro de la ventana; márgenes: %s", margins)
        return margins

    def grab(self) -> np.ndarray:
        raw = np.asarray(self.sct.grab(self.region))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def to_screen(self, fx: float, fy: float) -> tuple[int, int]:
        r = self.region
        return int(r["left"] + fx * r["width"]), int(r["top"] + fy * r["height"])


# =============================================================================
# UTILIDADES
# =============================================================================

def crop(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def dhash(img: np.ndarray, size: int = 8) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def normalize_label(name: str) -> tuple[str, Optional[str]]:
    """'Enemy PEKKA' -> ('pekka', 'enemy'); 'mini-pekka' -> ('mini_pekka', None)."""
    n = name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")
    for prefix, team in TEAM_PREFIXES.items():
        if n.startswith(prefix):
            return n[len(prefix):], team
    return n, None


def hsv_ranges(ranges) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in ranges]


def mask_ranges(hsv: np.ndarray, ranges) -> np.ndarray:
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, lo, hi)
    return m


def tile_rect_px(rect, aw: int, ah: int, cols: int, rows: int, pad: int = 0):
    c1, r1, c2, r2 = rect
    return (max(0, int((c1 - pad) / cols * aw)), max(0, int((r1 - pad) / rows * ah)),
            min(aw, int((c2 + pad) / cols * aw)), min(ah, int((r2 + pad) / rows * ah)))


# =============================================================================
# ESTADO
# =============================================================================

@dataclass
class Unit:
    tile: tuple[int, int]               # (columna, fila)
    area: int
    lane: str                           # "left" | "right"
    label: Optional[str] = None         # clase YOLO normalizada; None = detección por color
    kind: Optional[str] = None          # heavy | air | swarm | building | light | spell
    conf: float = 1.0
    bbox: Optional[tuple[int, int, int, int]] = None


Threat = Unit  # alias por compatibilidad


@dataclass
class HandCard:
    slot: int
    hash: int
    available: bool
    name: Optional[str] = None
    cost: Optional[int] = None
    role: Optional[str] = None


@dataclass
class GameState:
    timestamp: float
    in_battle: bool
    elixir: float
    hand: list[HandCard] = field(default_factory=list)
    threats_own_half: list[Unit] = field(default_factory=list)
    threats_enemy_half: list[Unit] = field(default_factory=list)
    enemy_spells: list[Unit] = field(default_factory=list)
    allies: list[Unit] = field(default_factory=list)

    @property
    def enemies(self) -> list[Unit]:
        return self.threats_own_half + self.threats_enemy_half


# =============================================================================
# YOLO
# =============================================================================

class YoloDetector:
    """Detector de tropas YOLO; el bando sale del prefijo de clase o del color del indicador."""

    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg
        self.ycfg = cfg["yolo"]
        self.ok = False
        self.red = hsv_ranges(cfg["enemy_hsv_ranges"])
        self.blue = hsv_ranges([cfg["ally_hsv"]])
        if not self.ycfg["enabled"]:
            return
        path = base_dir / self.ycfg["model"]
        if not path.exists():
            log.warning("Modelo YOLO '%s' no encontrado: uso detección por color.", path)
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            log.warning("ultralytics no instalado: uso detección por color.")
            return
        self.model = YOLO(str(path))
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.ok = True
        log.info("YOLO cargado (%d clases) desde %s", len(self.names), path)

    def _team_by_color(self, hsv: np.ndarray, box) -> str:
        x1, y1, x2, y2 = box
        h = max(1, y2 - y1)
        roi = hsv[max(0, y1 - h // 3):y1 + h // 3, max(0, x1):max(x1 + 1, x2)]
        if roi.size == 0:
            return "enemy"
        red = int(mask_ranges(roi, self.red).sum())
        blue = int(mask_ranges(roi, self.blue).sum())
        return "enemy" if red >= blue else "ally"

    def detect(self, arena: np.ndarray):
        """[(label, team, conf, (x1, y1, x2, y2)), ...] en px del recorte de la arena."""
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


# =============================================================================
# PERCEPCIÓN
# =============================================================================

class Perception:
    def __init__(self, cfg: dict, detector: Optional[YoloDetector] = None):
        self.cfg = cfg
        self.detector = detector if detector and detector.ok else None
        self.e_low = np.array(cfg["elixir_hsv_low"], np.uint8)
        self.e_high = np.array(cfg["elixir_hsv_high"], np.uint8)
        self.red = hsv_ranges(cfg["enemy_hsv_ranges"])
        self.blue = hsv_ranges([cfg["ally_hsv"]])
        self.cols, self.rows = cfg["arena_tiles"]
        self._frame_idx = 0
        self._cache = ([], [], [])
        self._static_mask: Optional[np.ndarray] = None
        self._bg: Optional[np.ndarray] = None       # fondo de la arena (float32)
        self._bg_samples: list[np.ndarray] = []

    def reset_background(self) -> None:
        """Llamar al empezar cada partida: el fondo se aprende con la arena vacía."""
        self._bg = None
        self._bg_samples = []

    def _foreground(self, arena: np.ndarray) -> Optional[np.ndarray]:
        """Máscara de lo que NO es escenario (tropas, proyectiles). None mientras aprende el fondo.

        Muchas arenas tienen caminos y adornos rojos/azules; sólo cuenta como tropa el color
        de equipo que además se diferencia del fondo aprendido.
        """
        if self._bg is None or self._bg.shape != arena.shape:
            if self._bg_samples and self._bg_samples[0].shape != arena.shape:
                self._bg_samples = []
            self._bg_samples.append(arena.copy())
            if len(self._bg_samples) < 8:
                return None
            self._bg = np.median(np.stack(self._bg_samples), axis=0).astype(np.float32)
            self._bg_samples = []
        diff = cv2.absdiff(arena, cv2.convertScaleAbs(self._bg)).max(axis=2) > 32
        fg = cv2.dilate(diff.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        # El fondo se adapta rápido donde no hay nada y muy despacio bajo las tropas.
        alpha = np.where(fg > 0, 0.003, 0.05).astype(np.float32)[..., None]
        self._bg = self._bg * (1 - alpha) + arena.astype(np.float32) * alpha
        return fg

    # ---- Elixir ---------------------------------------------------------------
    def read_elixir(self, frame: np.ndarray) -> tuple[float, bool]:
        """(elixir 0..10, en_batalla). En batalla = se ve la gota morada junto a la barra."""
        icon = crop(frame, self.cfg["elixir_icon_roi"])
        bar = crop(frame, self.cfg["elixir_roi"])
        if bar.size == 0 or icon.size == 0:
            return 0.0, False
        icon_mask = cv2.inRange(cv2.cvtColor(icon, cv2.COLOR_BGR2HSV), self.e_low, self.e_high)
        if (icon_mask > 0).mean() < 0.3:
            return 0.0, False
        mask = cv2.inRange(cv2.cvtColor(bar, cv2.COLOR_BGR2HSV), self.e_low, self.e_high)
        col_fill = ((mask > 0).mean(axis=0) > 0.4).astype(np.uint8)
        # Quitar ruido suelto; el número de elixir tapa el inicio de la barra, así que se
        # mide hasta dónde llega el relleno (última columna morada), no un tramo continuo.
        k = max(3, len(col_fill) // 50)
        col_fill = cv2.morphologyEx(col_fill.reshape(1, -1), cv2.MORPH_OPEN, np.ones((1, k), np.uint8)).ravel()
        idx = np.flatnonzero(col_fill)
        if idx.size == 0:
            return 0.0, True
        return round(min(10.0, 10.0 * (idx[-1] + 1) / len(col_fill)), 1), True

    # ---- Cartas ---------------------------------------------------------------
    def read_hand(self, frame: np.ndarray) -> list[HandCard]:
        hand = []
        for i, roi in enumerate(self.cfg["card_rois"]):
            img = crop(frame, roi)
            h, w = img.shape[:2]
            core = img[int(h * 0.12):int(h * 0.80), int(w * 0.12):int(w * 0.88)]
            if core.size == 0:
                continue
            sat = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)[..., 1].mean()
            hand.append(HandCard(slot=i, hash=dhash(core),
                                 available=bool(sat > self.cfg["card_gray_saturation"])))
        return hand

    # ---- Tropas ---------------------------------------------------------------
    def _to_tile(self, cx: float, cy: float, aw: int, ah: int) -> tuple[int, int]:
        return (int(np.clip(cx / aw * self.cols, 0, self.cols - 1)),
                int(np.clip(cy / ah * self.rows, 0, self.rows - 1)))

    def _lane(self, col: int) -> str:
        return "left" if col < self.cols / 2 else "right"

    def _mask_static(self, ah: int, aw: int) -> np.ndarray:
        """Máscara (255 = analizar) que excluye torres y río (colores fijos del escenario)."""
        if self._static_mask is None or self._static_mask.shape != (ah, aw):
            m = np.full((ah, aw), 255, np.uint8)
            for rect in self.cfg["towers"].values():
                x1, y1, x2, y2 = tile_rect_px(rect, aw, ah, self.cols, self.rows, pad=1)
                m[y1:y2, x1:x2] = 0
            r1, r2 = self.cfg["river_rows"]
            m[int(r1 / self.rows * ah):int((r2 + 1) / self.rows * ah), :] = 0
            self._static_mask = m
        return self._static_mask

    def read_units(self, frame: np.ndarray) -> tuple[list[Unit], list[Unit], list[Unit]]:
        """(enemigos, aliados, hechizos_enemigos), cacheado cada N frames si hay YOLO."""
        if self.detector is None:
            return self._units_color(frame)
        self._frame_idx += 1
        if self._frame_idx % self.cfg["yolo"]["every_n_frames"] == 0 or self._frame_idx == 1:
            self._cache = self._units_yolo(frame)
        return self._cache

    def _units_yolo(self, frame: np.ndarray):
        arena = crop(frame, self.cfg["arena_roi"])
        ah, aw = arena.shape[:2]
        enemies, allies, spells = [], [], []
        for label, team, conf, (x1, y1, x2, y2) in self.detector.detect(arena):
            kind = TROOP_PROFILES.get(label, "light")
            if kind == "ignore":
                continue
            cx, cy = (x1 + x2) / 2, y2 - (y2 - y1) * 0.15
            tile = self._to_tile(cx, cy, aw, ah)
            u = Unit(tile, (x2 - x1) * (y2 - y1), self._lane(tile[0]), label, kind, conf,
                     (x1, y1, x2, y2))
            if team == "ally":
                allies.append(u)
            elif kind == "spell":
                spells.append(u)
            else:
                enemies.append(u)
        return enemies, allies, spells

    def _blobs(self, mask: np.ndarray, aw: int, ah: int) -> list[Unit]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < self.cfg["threat_min_area"]:
                continue
            tile = self._to_tile(*centroids[k], aw, ah)
            out.append(Unit(tile, area, self._lane(tile[0])))
        return out

    def _units_color(self, frame: np.ndarray):
        arena = crop(frame, self.cfg["arena_roi"])
        ah, aw = arena.shape[:2]
        fg = self._foreground(arena)
        if fg is None:
            return [], [], []
        hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
        static = self._mask_static(ah, aw) & fg
        enemies = self._blobs(mask_ranges(hsv, self.red) & static, aw, ah)
        allies = self._blobs(mask_ranges(hsv, self.blue) & static, aw, ah)
        return enemies, allies, []

    # ---- Estado completo ------------------------------------------------------
    def analyse(self, frame: np.ndarray) -> GameState:
        elixir, visible = self.read_elixir(frame)
        st = GameState(timestamp=time.time(), in_battle=visible, elixir=elixir)
        if not visible:
            return st
        st.hand = self.read_hand(frame)
        enemies, st.allies, st.enemy_spells = self.read_units(frame)
        river_top = self.cfg["river_rows"][0]
        for u in enemies:
            (st.threats_own_half if u.tile[1] >= river_top else st.threats_enemy_half).append(u)
        return st


# =============================================================================
# MEMORIA DE CARTAS
# =============================================================================

class CardMemory:
    """Asocia hashes visuales a cartas del mazo midiendo su coste al jugarlas."""

    def __init__(self, path: Path, deck: dict, max_dist: int):
        self.path = path
        self.deck = deck
        self.max_dist = max_dist
        self.entries: list[dict] = []
        if path.exists():
            try:
                self.entries = json.loads(path.read_text())
                log.info("Cargadas %d cartas aprendidas de %s", len(self.entries), path)
            except Exception:
                log.warning("cards_db corrupto; empezando de cero.")

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False))

    def find(self, h: int) -> Optional[dict]:
        best, best_d = None, self.max_dist + 1
        for e in self.entries:
            d = hamming(h, e["hash"])
            if d < best_d:
                best, best_d = e, d
        return best

    def enrich(self, card: HandCard) -> None:
        e = self.find(card.hash)
        if e is None:
            e = {"hash": card.hash, "name": None, "cost_samples": []}
            self.entries.append(e)
            log.info("Nueva carta vista en slot %d (hash %016x)", card.slot + 1, card.hash)
            self.save()
        if e["name"] and e["name"] in self.deck:
            card.name = e["name"]
            card.cost, card.role = self.deck[e["name"]]

    def learn_cost(self, h: int, measured_drop: float) -> None:
        e = self.find(h)
        if e is None or e["name"]:
            return
        e["cost_samples"].append(round(measured_drop, 2))
        cost = int(round(float(np.median(e["cost_samples"]))))
        taken = {x["name"] for x in self.entries if x["name"]}
        candidates = [n for n, (c, _) in self.deck.items() if c == cost and n not in taken]
        if not candidates:
            candidates = [n for n, (c, _) in self.deck.items() if abs(c - cost) == 1 and n not in taken]
        if candidates and (len(e["cost_samples"]) >= 2 or len(candidates) == 1):
            e["name"] = candidates[0]
            log.info("Carta aprendida: %016x -> %s", h, e["name"])
        self.save()

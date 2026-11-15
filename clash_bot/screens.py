"""Fase 1 - Autonomía: reconocer pantallas, navegar menús, seguir torres/coronas y estadísticas."""

from __future__ import annotations

import json
import logging
import random
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from opponent import elixir_multiplier
from perception import GameState, crop, hsv_ranges, mask_ranges, tile_rect_px

log = logging.getLogger("crbot")


class Screen(Enum):
    BATTLE = "battle"
    END = "end"
    MENU = "menu"
    UNKNOWN = "unknown"


class ScreenDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.s = cfg["screens"]

    def button_fill(self, frame: np.ndarray, roi, hsv_range) -> float:
        img = crop(frame, roi)
        if img.size == 0:
            return 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lo, hi = hsv_range
        return float((cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0).mean())

    def classify(self, frame: np.ndarray, state: GameState) -> Screen:
        if state.in_battle:
            return Screen.BATTLE
        if self.button_fill(frame, self.s["ok_button_roi"], self.s["ok_button_hsv"]) >= self.s["min_button_fill"]:
            return Screen.END
        if self.button_fill(frame, self.s["battle_button_roi"], self.s["battle_button_hsv"]) >= self.s["min_button_fill"]:
            return Screen.MENU
        return Screen.UNKNOWN

    def debug_info(self, frame: np.ndarray) -> dict:
        return {"ok_fill": round(self.button_fill(frame, self.s["ok_button_roi"], self.s["ok_button_hsv"]), 3),
                "battle_fill": round(self.button_fill(frame, self.s["battle_button_roi"],
                                                      self.s["battle_button_hsv"]), 3)}


class MatchTracker:
    """Cronómetro, multiplicador de elixir y estado de las 6 torres (por su color de equipo)."""

    BASELINE_S = 3.0

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cols, self.rows = cfg["arena_tiles"]
        self.red = hsv_ranges(cfg["enemy_hsv_ranges"])
        self.blue = hsv_ranges([cfg["ally_hsv"]])
        self.active = False

    def start(self, now: float) -> None:
        self.t0 = now
        self.active = True
        self.samples: dict[str, list[float]] = {k: [] for k in self.cfg["towers"]}
        self.baseline: dict[str, float] = {}
        self.low_since: dict[str, Optional[float]] = {k: None for k in self.cfg["towers"]}
        self.destroyed: set[str] = set()

    def elapsed(self, now: float) -> float:
        return now - self.t0 if self.active else 0.0

    def multiplier(self, now: float) -> float:
        return elixir_multiplier(self.cfg["elixir_schedule"], self.elapsed(now))

    def _fills(self, frame: np.ndarray) -> dict[str, float]:
        arena = crop(frame, self.cfg["arena_roi"])
        ah, aw = arena.shape[:2]
        hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
        out = {}
        for name, rect in self.cfg["towers"].items():
            x1, y1, x2, y2 = tile_rect_px(rect, aw, ah, self.cols, self.rows)
            roi = hsv[y1:y2, x1:x2]
            ranges = self.red if name.startswith("enemy") else self.blue
            out[name] = float((mask_ranges(roi, ranges) > 0).mean()) if roi.size else 0.0
        return out

    def update(self, frame: np.ndarray, now: float) -> list[str]:
        """Devuelve eventos: 'tower_taken:<torre>' / 'tower_lost:<torre>'."""
        fills = self._fills(frame)
        if self.elapsed(now) < self.BASELINE_S:
            for k, v in fills.items():
                self.samples[k].append(v)
            return []
        if not self.baseline:
            self.baseline = {k: float(np.median(v)) if v else 0.0 for k, v in self.samples.items()}
            weak = [k for k, v in self.baseline.items() if v < 0.01]
            if weak:
                log.warning("Torres sin color de referencia (ajusta 'towers'/colores): %s", weak)
        events = []
        for k, v in fills.items():
            base = self.baseline.get(k, 0.0)
            if k in self.destroyed or base < 0.01:
                continue
            if v < base * self.cfg["tower_destroyed_ratio"]:
                if self.low_since[k] is None:
                    self.low_since[k] = now
                elif now - self.low_since[k] >= self.cfg["tower_confirm_s"]:
                    self.destroyed.add(k)
                    ev = ("tower_taken:" if k.startswith("enemy") else "tower_lost:") + k
                    events.append(ev)
                    log.info("Torre destruida: %s", k)
            else:
                self.low_since[k] = None
        return events

    @staticmethod
    def _lanes(destroyed: set[str], side: str) -> set[str]:
        return {k.split("_")[1] for k in destroyed if k.startswith(side)}

    def enemy_down(self) -> set[str]:
        return self._lanes(self.destroyed, "enemy")

    def own_down(self) -> set[str]:
        return self._lanes(self.destroyed, "own")

    def crowns(self) -> tuple[int, int]:
        ours = 3 if "enemy_king" in self.destroyed else len(self.enemy_down() - {"king"})
        theirs = 3 if "own_king" in self.destroyed else len(self.own_down() - {"king"})
        return ours, theirs

    def result(self) -> str:
        if not self.baseline or all(v < 0.01 for v in self.baseline.values()):
            return "unknown"
        ours, theirs = self.crowns()
        return "win" if ours > theirs else ("loss" if theirs > ours else "draw")


class Navigator:
    """Fuera de batalla: pulsa OK en la pantalla final y Batalla en el menú."""

    def __init__(self, cfg: dict, controls, max_games: int = 0):
        self.cfg = cfg
        self.s = cfg["screens"]
        self.controls = controls
        self.max_games = max_games or self.s["max_games"]
        self.games_started = 0
        self.games_completed = 0
        self.last_tap = 0.0
        self.last_known = time.time()
        self.next_battle_at = 0.0

    def finished(self) -> bool:
        return bool(self.max_games) and self.games_completed >= self.max_games

    def on_game_over(self, now: float) -> None:
        self.games_completed += 1
        self.next_battle_at = now + random.uniform(*self.s["pause_between_games_s"])

    def step(self, screen: Screen, now: float) -> None:
        if screen != Screen.UNKNOWN:
            self.last_known = now
        if now - self.last_tap < 2.0:
            return
        if screen == Screen.END:
            log.info("Pantalla final: pulsando OK")
            self.controls.tap_fraction(*self.s["ok_button"])
            self.last_tap = now
        elif screen == Screen.MENU:
            if (self.max_games and self.games_started >= self.max_games) or now < self.next_battle_at:
                return
            log.info("Menú principal: buscando partida (%d)", self.games_started + 1)
            self.controls.tap_fraction(*self.s["battle_button"])
            self.games_started += 1
            self.last_tap = now
        elif screen == Screen.UNKNOWN and now - self.last_known > self.s["unknown_action_after_s"]:
            log.info("Pantalla desconocida durante %ds: pulsando 'atrás'", self.s["unknown_action_after_s"])
            self.controls.press(self.s["back_key"])
            self.last_tap = now
            self.last_known = now


class Stats:
    def __init__(self, path: Path):
        self.path = path

    def record(self, result: str, crowns: tuple[int, int], duration: float, plays: int) -> None:
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "result": result, "crowns": list(crowns),
               "duration_s": round(duration, 1), "plays": plays}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        log.info("Partida: %s %d-%d (%.0fs). %s", result, crowns[0], crowns[1], duration, self.summary())

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def summary(self, last: int = 0) -> str:
        rows = self.rows()[-last:] if last else self.rows()
        if not rows:
            return "Sin partidas registradas."
        w = sum(r["result"] == "win" for r in rows)
        l = sum(r["result"] == "loss" for r in rows)
        d = sum(r["result"] == "draw" for r in rows)
        u = len(rows) - w - l - d
        decided = w + l
        wr = f"{100 * w / decided:.1f}%" if decided else "n/d"
        return f"{len(rows)} partidas: {w}V {l}D {d}E {u}? | % victorias {wr}"

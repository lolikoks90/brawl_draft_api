"""Clics humanizados con pyautogui."""

from __future__ import annotations

import random
import time

import pyautogui

from tactics import Action


class HumanInput:
    def __init__(self, cfg: dict, capture):
        self.cfg = cfg
        self.cap = capture
        pyautogui.FAILSAFE = True   # ratón a una esquina => parada
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

    def tap_fraction(self, fx: float, fy: float) -> None:
        self._tap(*self.cap.to_screen(fx, fy))

    def press(self, key: str) -> None:
        pyautogui.press(key)

    def slot_center(self, slot: int) -> tuple[int, int]:
        x1, y1, x2, y2 = self.cfg["card_rois"][slot]
        return self.cap.to_screen((x1 + x2) / 2, (y1 + y2) / 2)

    def tile_center(self, col: int, row: int) -> tuple[int, int]:
        x1, y1, x2, y2 = self.cfg["arena_roi"]
        cols, rows = self.cfg["arena_tiles"]
        return self.cap.to_screen(x1 + (col + 0.5) / cols * (x2 - x1), y1 + (row + 0.5) / rows * (y2 - y1))

    def play(self, action: Action) -> None:
        self._tap(*self.slot_center(action.slot))
        self._tap(*self.tile_center(*action.tile))

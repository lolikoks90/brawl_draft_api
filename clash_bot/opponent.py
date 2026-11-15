"""Modelo del rival: estima su elixir y su ciclo de cartas a partir de lo que despliega."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

from config import DEFAULT_ENEMY_COST, ENEMY_CARD_COSTS, SPAWNS
from perception import GameState, Unit

log = logging.getLogger("crbot")


def elixir_multiplier(schedule, elapsed: float) -> float:
    mult = 1.0
    for start, m in schedule:
        if elapsed >= start:
            mult = m
    return mult


@dataclass
class Track:
    label: Optional[str]
    tile: tuple[float, float]
    first_seen: float
    last_seen: float
    hits: int = 1
    counted: bool = False


@dataclass
class Deploy:
    t: float
    label: Optional[str]
    cost: float
    lane: str
    tile: tuple[int, int]


class OpponentModel:
    """Sigue las unidades enemigas entre frames y descuenta elixir por cada despliegue nuevo.

    - Con YOLO: coste exacto por etiqueta (ENEMY_CARD_COSTS) y ciclo de cartas.
    - Sin YOLO: se cuenta cada mancha roja nueva en su mitad como una carta de coste medio.
    """

    MATCH_DIST = 3.0     # baldosas
    GROUP_DIST = 4.0     # unidades del mismo enjambre aparecen juntas
    GROUP_S = 1.2
    CONFIRM_HITS = 3
    EXPIRE_S = 1.5

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reset(5.0, 0.0)

    def reset(self, start_elixir: float, t0: float) -> None:
        self.elixir = float(start_elixir)
        self.t0 = t0
        self.last_t = t0
        self.tracks: list[Track] = []
        self.deploys: list[Deploy] = []
        self.cycle: deque[str] = deque(maxlen=8)
        self.deck: list[str] = []
        self.known = True

    # ---- seguimiento ----------------------------------------------------------
    @staticmethod
    def _dist(a, b) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _is_spawn(self, label: Optional[str], tile) -> bool:
        if label is None:
            return False
        for tr in self.tracks:
            if tr.label and label in SPAWNS.get(tr.label, []) and self._dist(tr.tile, tile) < 5:
                return True
        return False

    def _grouped(self, label: Optional[str], tile, now: float) -> bool:
        return any(d.label == label and now - d.t < self.GROUP_S and
                   self._dist(d.tile, tile) < self.GROUP_DIST for d in self.deploys[-6:])

    def _register_deploy(self, tr: Track, now: float) -> None:
        tr.counted = True
        if self._is_spawn(tr.label, tr.tile) or self._grouped(tr.label, tr.tile, now):
            return
        cost = ENEMY_CARD_COSTS.get(tr.label, DEFAULT_ENEMY_COST) if tr.label else DEFAULT_ENEMY_COST
        tile = (int(tr.tile[0]), int(tr.tile[1]))
        lane = "left" if tile[0] < self.cfg["arena_tiles"][0] / 2 else "right"
        self.deploys.append(Deploy(now, tr.label, cost, lane, tile))
        self.elixir = max(0.0, self.elixir - cost)
        if tr.label:
            self.cycle.append(tr.label)
            if tr.label not in self.deck and len(self.deck) < 8:
                self.deck.append(tr.label)
        log.debug("Rival despliega %s (-%.1f) -> elixir rival ~%.1f", tr.label or "?", cost, self.elixir)

    def update(self, state: GameState, now: float) -> None:
        dt = max(0.0, now - self.last_t)
        self.last_t = now
        mult = elixir_multiplier(self.cfg["elixir_schedule"], now - self.t0)
        self.elixir = min(10.0, self.elixir + dt / self.cfg["elixir_regen_s"] * mult)

        units: list[Unit] = state.enemies + state.enemy_spells
        matched = set()
        for u in units:
            best, best_d = None, self.MATCH_DIST
            for i, tr in enumerate(self.tracks):
                if i in matched or tr.label != u.label:
                    continue
                d = self._dist(tr.tile, u.tile)
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                self.tracks.append(Track(u.label, (float(u.tile[0]), float(u.tile[1])), now, now))
                matched.add(len(self.tracks) - 1)
            else:
                tr = self.tracks[best]
                tr.tile = (0.6 * tr.tile[0] + 0.4 * u.tile[0], 0.6 * tr.tile[1] + 0.4 * u.tile[1])
                tr.last_seen = now
                tr.hits += 1
                matched.add(best)

        river_top = self.cfg["river_rows"][0]
        for tr in self.tracks:
            # Sin YOLO sólo se cuentan apariciones en su mitad (allí es donde se despliega).
            if not tr.counted and tr.hits >= self.CONFIRM_HITS and (tr.label or tr.tile[1] < river_top):
                self._register_deploy(tr, now)
            elif not tr.counted and tr.hits >= self.CONFIRM_HITS:
                tr.counted = True
        self.tracks = [tr for tr in self.tracks if now - tr.last_seen < self.EXPIRE_S]

    # ---- consultas ------------------------------------------------------------
    def recent(self, within_s: float, now: float) -> list[Deploy]:
        return [d for d in self.deploys if now - d.t <= within_s]

    def likely_in_hand(self) -> list[str]:
        """Si conocemos sus 8 cartas, las 4 en mano son las que no jugó en sus 4 últimas."""
        if len(self.deck) < 8:
            return []
        last4 = list(self.cycle)[-4:]
        return [c for c in self.deck if c not in last4]

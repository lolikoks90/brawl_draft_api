"""Motor táctico: genera jugadas candidatas con una prioridad a priori.

Cada regla devuelve 0..N `Action`. El aprendizaje (fase 2) elige entre todas ellas; si
está desactivado se juega la de mayor prioridad.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import COUNTERS, HIGH_VALUE_TARGETS, KIND_PRIORITY, SPECIFIC_COUNTERS
from perception import GameState, HandCard, Unit


@dataclass
class Action:
    slot: int
    tile: tuple[int, int]
    reason: str
    card_hash: int = 0
    rule: str = ""
    variant: str = ""
    role: Optional[str] = None
    card_name: Optional[str] = None
    prior: float = 0.5


@dataclass
class BattleContext:
    now: float
    elapsed: float = 0.0
    multiplier: float = 1.0
    opp_elixir: Optional[float] = None
    opp_recent: list = field(default_factory=list)       # [Deploy]
    opp_likely_hand: list = field(default_factory=list)
    enemy_towers_down: set = field(default_factory=set)  # {"left", "right", "king"}
    own_towers_down: set = field(default_factory=set)


@dataclass
class Situation:
    kind: str                      # heavy | air | swarm | building | light | idle
    label: Optional[str] = None
    lane: Optional[str] = None
    threats: list = field(default_factory=list)


class TacticalEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.t = cfg["tactics"]
        self.cols, self.rows = cfg["arena_tiles"]
        self.history: list[tuple[float, Action]] = []   # nuestras jugadas recientes
        self.rules = [
            self.rule_defend,
            self.rule_spell_value,
            self.rule_punish,
            self.rule_siege,
            self.rule_counter_push,
            self.rule_follow_tank,
            self.rule_tank_push,
            self.rule_prevent_leak,
            self.rule_explore_unknown,
        ]

    # ---- utilidades -----------------------------------------------------------
    def notify_played(self, action: Action, now: float) -> None:
        self.history.append((now, action))
        self.history = self.history[-20:]

    @staticmethod
    def playable(state: GameState) -> list[HandCard]:
        return [c for c in state.hand if c.available and
                (c.cost <= state.elixir if c.cost is not None else state.elixir >= 2.0)]

    def cards_for(self, state: GameState, roles: list[str], limit: int = 3,
                  allow_unknown: bool = False) -> list[tuple[int, HandCard]]:
        """[(rango_del_rol, carta)] en orden de preferencia, sin repetir carta."""
        out, used = [], set()
        cards = self.playable(state)
        for rank, role in enumerate(roles):
            for c in sorted(cards, key=lambda c: c.cost or 99):
                if c.role == role and c.slot not in used:
                    out.append((rank, c))
                    used.add(c.slot)
        if not out and allow_unknown:
            # Arranque en frío: cartas aún sin identificar sirven antes que no hacer nada.
            out = [(len(roles), c) for c in cards if c.role is None]
        return out[:limit]

    def lane_col(self, lane: str, back: bool = False) -> int:
        if back:
            return 2 if lane == "left" else self.cols - 3
        return 3 if lane == "left" else self.cols - 4

    def legal_tile(self, tile: tuple[int, int], role: Optional[str], ctx: BattleContext) -> tuple[int, int]:
        col = int(np.clip(tile[0], 0, self.cols - 1))
        row = int(np.clip(tile[1], 0, self.rows - 1))
        if role and role.startswith("spell"):
            return col, row
        lane = "left" if col < self.cols / 2 else "right"
        min_row = self.cfg["river_rows"][1] + 1
        if lane in ctx.enemy_towers_down:   # con su torre caída se puede desplegar en su lado
            min_row = 10
        return col, max(row, min_row)

    def situation(self, s: GameState) -> Situation:
        if not s.threats_own_half:
            return Situation("idle")
        lanes: dict[str, list[Unit]] = {}
        for u in s.threats_own_half:
            lanes.setdefault(u.lane, []).append(u)
        lane, threats = max(lanes.items(), key=lambda kv: sum(u.area for u in kv[1]))
        labelled = [u for u in threats if u.kind]
        swarm_n = self.cfg["threat_swarm_count"]
        if labelled:
            for kind in KIND_PRIORITY:
                group = [u for u in labelled if u.kind == kind]
                if group:
                    if kind == "light" and len(labelled) >= swarm_n:
                        return Situation("swarm", None, lane, threats)
                    return Situation(kind, max(group, key=lambda u: u.conf).label, lane, threats)
        if any(u.area >= self.cfg["threat_heavy_area"] for u in threats):
            return Situation("heavy", None, lane, threats)
        if len(threats) >= swarm_n:
            return Situation("swarm", None, lane, threats)
        return Situation("light", None, lane, threats)

    def _mk(self, card: HandCard, tile, ctx, rule, variant, prior, reason) -> Action:
        return Action(card.slot, self.legal_tile(tile, card.role, ctx), reason, card.hash,
                      rule, variant, card.role, card.name, round(float(prior), 3))

    # ---- reglas ---------------------------------------------------------------
    def rule_defend(self, s: GameState, ctx: BattleContext, sit: Situation) -> list[Action]:
        if sit.kind == "idle":
            return []
        roles = list(dict.fromkeys(SPECIFIC_COUNTERS.get(sit.label, []) + COUNTERS[sit.kind]))
        if sit.kind == "heavy":            # hechizos pequeños no paran tanques
            roles = [r for r in roles if r != "spell_small"]
        targets = [u for u in sit.threats if u.label == sit.label] if sit.label else sit.threats
        lead = max(targets, key=lambda u: u.tile[1])
        cx = int(round(np.mean([u.tile[0] for u in targets])))
        cy = int(round(np.mean([u.tile[1] for u in targets])))
        tower_col = self.lane_col(sit.lane)
        center = self.cols // 2 + (-1 if sit.lane == "left" else 0)
        what = sit.label or sit.kind
        out = []
        for rank, c in self.cards_for(s, roles, allow_unknown=True):
            base = max(0.4, 0.95 - 0.12 * rank)
            if (c.role or "").startswith("spell"):
                out.append(self._mk(c, (cx, cy), ctx, "defend", "on_top", base,
                                    f"hechizo sobre {what}"))
            elif c.role == "building":
                out.append(self._mk(c, (center, 22), ctx, "defend", "center_pull", base,
                                    f"edificio central contra {what}"))
            else:
                out.append(self._mk(c, (lead.tile[0], lead.tile[1] + 3), ctx, "defend", "front", base,
                                    f"{c.name} delante de {what}"))
                out.append(self._mk(c, (center, lead.tile[1] + 4), ctx, "defend", "pull_center",
                                    base - 0.08, f"{c.name} tirando {what} al centro"))
                if c.role in ("support", "splash"):
                    out.append(self._mk(c, (tower_col, 26), ctx, "defend", "behind_tower",
                                        base - 0.1, f"{c.name} tras la torre contra {what}"))
        return out

    def rule_spell_value(self, s: GameState, ctx: BattleContext, sit: Situation) -> list[Action]:
        enemies = s.enemies
        if len(enemies) < 2:
            return []
        r = self.t["spell_cluster_radius"]
        best, best_score, best_pos = None, 0.0, None
        for u in enemies:
            group = [v for v in enemies if (v.tile[0] - u.tile[0]) ** 2 + (v.tile[1] - u.tile[1]) ** 2 <= r * r]
            score = len(group) + 2 * sum(1 for v in group if v.label in HIGH_VALUE_TARGETS)
            if score > best_score:
                best_score, best = score, group
                best_pos = (int(round(np.mean([v.tile[0] for v in group]))),
                            int(round(np.mean([v.tile[1] for v in group]))))
        if best_score < self.t["spell_cluster_min"]:
            return []
        out = []
        for _, c in self.cards_for(s, ["spell_big", "spell_small"], limit=2):
            if c.role == "spell_small" and not any(v.kind in ("swarm", "air", None) for v in best):
                continue
            out.append(self._mk(c, best_pos, ctx, "spell_value", "cluster", 0.6 + 0.05 * min(best_score, 6),
                                f"{c.name} sobre grupo de {len(best)}"))
        return out

    def rule_punish(self, s: GameState, ctx: BattleContext, sit: Situation) -> list[Action]:
        if sit.kind != "idle" or ctx.opp_elixir is None:
            return []
        if ctx.opp_elixir > self.t["punish_opp_elixir_max"] or s.elixir < self.t["punish_min_own_elixir"]:
            return []
        big = [d for d in ctx.opp_recent if d.cost >= 5]
        if big:
            lane = "right" if big[-1].lane == "left" else "left"   # castigar por el carril contrario
        elif ctx.enemy_towers_down & {"left", "right"}:
            lane = sorted(ctx.enemy_towers_down & {"left", "right"})[0]
        else:
            lane = random.choice(["left", "right"])
        col = self.lane_col(lane)
        out = []
        for rank, c in self.cards_for(s, ["siege", "tank", "tank_killer", "support", "splash"]):
            out.append(self._mk(c, (col, 17), ctx, "punish", "bridge", 0.8 - 0.08 * rank,
                                f"castigo con {c.name} en puente {lane} (rival ~{ctx.opp_elixir:.1f})"))
        return out

    def rule_counter_push(self, s: GameState, ctx: BattleContext, sit: Situation) -> list[Action]:
        if sit.kind != "idle" or s.elixir < 4:
            return []
        movers = [a for a in s.allies if a.tile[1] <= 24]
        if not movers:
            return []
        lead = min(movers, key=lambda a: a.tile[1])
        out = []
        for rank, c in self.cards_for(s, ["support", "splash", "tank_killer"], limit=2):
            out.append(self._mk(c, (lead.tile[0], lead.tile[1] + 2), ctx, "counter_push", "behind_ally",
                                0.65 - 0.08 * rank, f"contraataque: {c.name} tras nuestras tropas"))
        return out

    def rule_follow_tank(self, s: GameState, ctx: BattleContext, sit: Situation) -> list[Action]:
        if sit.kind != "idle" or s.elixir < 3:
            return []
        lo, hi = self.t["follow_tank_window_s"]
        tanks = [a for t, a in self.history if a.role in ("tank", "siege") and lo <= ctx.now - t <= hi]
        if not tanks:
            return []
        last = tanks[-1]
        lane_col = last.tile[0]
        row = min(self.rows - 1, last.tile[1] + 3) if last.role == "siege" else 25
        what = "proteger el asedio" if last.role == "siege" else "apoyo al tanque"
        out = []
        for rank, c in self.cards_for(s, ["support", "splash", "tank_killer"], limit=2):
            out.append(self._mk(c, (lane_col, row), ctx, "follow_tank", "behind_tank", 0.7 - 0.08 * rank,
                                f"{what} con {c.name}"))
        return out

    def rule_siege(self, s: GameState, ctx: BattleContext, sit: Situation) -> list[Action]:
        """Mortero / Ballesta en el puente para atacar la torre, si no hay uno reciente."""
        if sit.kind != "idle":
            return []
        if any(a.role == "siege" and ctx.now - t < 10 for t, a in self.history):
            return []
        cards = self.cards_for(s, ["siege"], limit=1)
        if not cards:
            return []
        c = cards[0][1]
        spare = 0 if ctx.multiplier >= 2 else 1   # un elixir de margen para defenderlo
        if s.elixir < (c.cost or 4) + spare:
            return []
        down = ctx.enemy_towers_down & {"left", "right"}
        big = [d for d in ctx.opp_recent if d.cost >= 5]
        if down:
            lane = sorted(down)[0]
        elif big:
            lane = "right" if big[-1].lane == "left" else "left"
        else:
            lane = random.choice(["left", "right"])
        col = self.lane_col(lane)
        return [self._mk(c, (col, 17), ctx, "siege", "bridge", 0.7, f"{c.name} en el puente ({lane})"),
                self._mk(c, (col, 19), ctx, "siege", "safe", 0.6, f"{c.name} algo retrasado ({lane})")]

    def rule_tank_push(self, s: GameState, ctx: BattleContext, sit: Situation) -> list[Action]:
        need = self.t["tank_push_elixir_double"] if ctx.multiplier >= 2 else self.t["tank_push_elixir"]
        if sit.kind != "idle" or s.elixir < need:
            return []
        down = ctx.enemy_towers_down & {"left", "right"}
        lane = sorted(down)[0] if down else random.choice(["left", "right"])
        out = []
        for _, c in self.cards_for(s, ["tank"], limit=1):
            out.append(self._mk(c, (self.cols // 2 + (-1 if lane == "left" else 0), self.rows - 1), ctx,
                                "tank_push", "back", 0.55, f"tanque atrás ({lane}) con {c.name}"))
            if ctx.multiplier >= 2:
                out.append(self._mk(c, (self.lane_col(lane), 17), ctx, "tank_push", "bridge", 0.5,
                                    f"tanque en puente ({lane}) con {c.name}"))
        return out

    def rule_prevent_leak(self, s: GameState, ctx: BattleContext, sit: Situation) -> list[Action]:
        if s.elixir < self.t["leak_elixir"] or sit.kind != "idle":
            return []
        cards = [c for c in self.playable(s) if not (c.role or "").startswith("spell") and c.role != "siege"]
        if not cards:
            return []
        c = min(cards, key=lambda c: c.cost or 5)
        lane = random.choice(["left", "right"])
        return [self._mk(c, (self.lane_col(lane, back=True), self.rows - 2), ctx, "prevent_leak", "back_corner",
                         0.45, f"evitar fuga de elixir con {c.name or 'carta desconocida'}")]

    def rule_explore_unknown(self, s: GameState, ctx: BattleContext, sit: Situation) -> list[Action]:
        if s.elixir < 8 or sit.kind != "idle":
            return []
        unknown = [c for c in s.hand if c.available and c.name is None]
        if not unknown:
            return []
        c = random.choice(unknown)
        return [Action(c.slot, (self.cols // 2, self.rows - 2), f"explorar carta desconocida (slot {c.slot + 1})",
                       c.hash, "explore", "back", None, None, 0.35)]

    # ---- API ------------------------------------------------------------------
    def candidates(self, state: GameState, ctx: BattleContext) -> tuple[Situation, list[Action]]:
        sit = self.situation(state)
        out: list[Action] = []
        for rule in self.rules:
            out.extend(rule(state, ctx, sit))
        out.sort(key=lambda a: -a.prior)
        return sit, out

"""Fase 2 - Aprendizaje por refuerzo (bandido contextual con UCB).

Cada vez que hay que decidir, el contexto se resume en una clave discreta
(situación, elixir propio, fase de elixir, elixir del rival) y cada jugada candidata en un
"brazo" (regla:rol:variante). Tras la partida, cada decisión recibe una recompensa:

    recompensa = final_weight * resultado (+1 / 0 / -1)
               + (1 - final_weight) * recompensa local
    local: +1 si cae una torre rival poco después, -1 si cae una nuestra,
           +0.3 si una defensa limpia nuestra mitad.

Las estadísticas se guardan en learner_db.json, así que mejora partida tras partida.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from tactics import Action, BattleContext, Situation

log = logging.getLogger("crbot")

WAIT = "wait"


class Learner:
    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg["learner"]
        self.enabled = self.cfg["enabled"]
        self.path = base_dir / self.cfg["db"]
        self.stats: dict[str, dict[str, list[float]]] = {}
        if self.path.exists():
            try:
                self.stats = json.loads(self.path.read_text())
                log.info("Aprendizaje: %d contextos cargados", len(self.stats))
            except Exception:
                log.warning("learner_db corrupto; empezando de cero.")
        self.decisions: list[dict] = []
        self._last_wait = -1e9

    # ---- claves ---------------------------------------------------------------
    @staticmethod
    def context_key(sit: Situation, elixir: float, ctx: BattleContext) -> str:
        e = "e_low" if elixir < 4 else ("e_mid" if elixir < 7 else "e_high")
        phase = "x1" if ctx.multiplier < 2 else ("x2" if ctx.multiplier < 3 else "x3")
        if ctx.opp_elixir is None:
            o = "o_unk"
        else:
            o = "o_low" if ctx.opp_elixir < 4 else ("o_mid" if ctx.opp_elixir < 7 else "o_high")
        return f"{sit.kind}|{e}|{phase}|{o}"

    @staticmethod
    def arm_key(a: Action) -> str:
        return f"{a.rule}:{a.role or '?'}:{a.variant}"

    # ---- elección -------------------------------------------------------------
    def _score(self, ctx_key: str, arm: str, prior: float, total: float) -> float:
        n, s = self.stats.get(ctx_key, {}).get(arm, [0.0, 0.0])
        pw = self.cfg["prior_weight"]
        mean = (s + (2 * prior - 1) * pw) / (n + pw)          # prior en [0,1] -> recompensa [-1,1]
        return mean + self.cfg["ucb_c"] * math.sqrt(math.log(total + 2) / (n + 1))

    def choose(self, sit: Situation, elixir: float, ctx: BattleContext,
               candidates: list[Action]) -> Optional[Action]:
        if not candidates:
            return None
        if not self.enabled:
            return candidates[0]
        can_wait = sit.kind == "idle"
        if can_wait and ctx.now - self._last_wait < self.cfg["wait_decision_interval_s"]:
            return None                       # seguimos esperando la decisión anterior
        key = self.context_key(sit, elixir, ctx)
        arms: dict[str, Action] = {}
        for a in candidates:                  # un brazo por clave, el de mayor prioridad
            k = self.arm_key(a)
            if k not in arms or a.prior > arms[k].prior:
                arms[k] = a
        total = sum(v[0] for v in self.stats.get(key, {}).values())
        scored = [(self._score(key, k, a.prior, total), k, a) for k, a in arms.items()]
        if can_wait:
            scored.append((self._score(key, WAIT, self.cfg["wait_prior"], total), WAIT, None))
        _, arm, action = max(scored, key=lambda x: x[0])
        self.decisions.append({"ctx": key, "arm": arm, "t": ctx.now, "local": 0.0,
                               "defend": sit.kind != "idle"})
        if action is None:
            self._last_wait = ctx.now
        return action

    # ---- recompensas ----------------------------------------------------------
    def start_game(self) -> None:
        self.decisions = []
        self._last_wait = -1e9

    def on_event(self, kind: str, now: float) -> None:
        w = self.cfg["local_window_s"]
        for d in self.decisions:
            age = now - d["t"]
            if age < 0 or age > w:
                continue
            if kind == "tower_taken" and d["arm"] != WAIT:
                d["local"] += 1.0
            elif kind == "tower_lost":
                d["local"] -= 1.0
            elif kind == "threat_cleared" and d["defend"] and age <= 8:
                d["local"] += 0.3

    def end_game(self, result: str) -> None:
        if not self.enabled or not self.decisions:
            return
        final = {"win": 1.0, "loss": -1.0}.get(result, 0.0)
        fw = self.cfg["final_weight"]
        for d in self.decisions:
            r = fw * final + (1 - fw) * max(-1.0, min(1.0, d["local"]))
            n_s = self.stats.setdefault(d["ctx"], {}).setdefault(d["arm"], [0.0, 0.0])
            n_s[0] += 1
            n_s[1] += r
        self.path.write_text(json.dumps(self.stats, indent=1))
        log.info("Aprendizaje actualizado con %d decisiones (resultado: %s)", len(self.decisions), result)
        self.decisions = []

    def summary(self, top: int = 15) -> list[tuple[str, str, int, float]]:
        rows = []
        for ctx, arms in self.stats.items():
            for arm, (n, s) in arms.items():
                rows.append((ctx, arm, int(n), s / n if n else 0.0))
        rows.sort(key=lambda r: -r[2])
        return rows[:top]

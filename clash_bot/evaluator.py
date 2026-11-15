"""Evaluación de posiciones al estilo Stockfish.

Stockfish tiene una función que puntúa cualquier posición y busca la jugada que lleva a la
mejor. Aquí:

  1. Una red de valor V(estado) estima la probabilidad de ganar desde el estado actual.
     Se entrena sola con las partidas del bot: cada estado grabado se etiqueta con el
     resultado final (victoria = 1, empate = 0.5, derrota = 0).
  2. Antes de jugar, para cada jugada candidata se simula cómo quedaría el tablero
     (elixir gastado, carta fuera de la mano, tropa nueva o enemigos eliminados por el
     hechizo) y se calcula V(después) - V(ahora).
  3. Esa ventaja se suma a la prioridad de la jugada, y el bandido de la fase 2 decide.

Es una búsqueda de 1 jugada (no un árbol profundo): Clash Royale es en tiempo real y no
hay simulador exacto, así que mirar más allá sería sobre todo ruido.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np

from imitation import encode_state, feature_dim
from perception import GameState, Unit
from tactics import Action, BattleContext

log = logging.getLogger("crbot")

RESULT_VALUE = {"win": 1.0, "draw": 0.5, "loss": 0.0}
ROLE_KIND = {"tank": "heavy", "tank_killer": "light", "splash": "light", "support": "light",
             "cycle": "swarm", "building": "building", "siege": "building"}


def simulate(state: GameState, action: Action, cfg: dict) -> GameState:
    """Estado aproximado justo después de jugar `action`."""
    hand = [dataclasses.replace(c) for c in state.hand]
    card = next((c for c in hand if c.slot == action.slot), None)
    cost = card.cost if card and card.cost is not None else 3.5
    if card:
        card.available = False
    own, enemy = list(state.threats_own_half), list(state.threats_enemy_half)
    allies = list(state.allies)
    role = action.role or ""
    if role.startswith("spell"):
        r = 2.5 if role == "spell_big" else 2.0
        heavy = cfg["threat_heavy_area"]

        def survives(u: Unit) -> bool:
            hit = (u.tile[0] - action.tile[0]) ** 2 + (u.tile[1] - action.tile[1]) ** 2 <= r * r
            if not hit:
                return True
            if role == "spell_small":   # un hechizo pequeño sólo elimina unidades débiles
                return u.kind in ("heavy", "building") or (u.kind is None and u.area >= heavy)
            return u.kind == "heavy" or (u.kind is None and u.area >= heavy)

        own, enemy = [u for u in own if survives(u)], [u for u in enemy if survives(u)]
    else:
        lane = "left" if action.tile[0] < cfg["arena_tiles"][0] / 2 else "right"
        n = 3 if role == "cycle" else 1
        allies += [Unit(action.tile, 100, lane, None, ROLE_KIND.get(role)) for _ in range(n)]
    return dataclasses.replace(state, elixir=max(0.0, state.elixir - cost), hand=hand,
                               threats_own_half=own, threats_enemy_half=enemy, allies=allies)


def _build_net(torch, in_dim: int):
    nn = torch.nn
    return nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))


def train_value(cfg: dict, base_dir: Path) -> bool:
    try:
        import torch
    except ImportError:
        log.warning("PyTorch no instalado: no se puede entrenar la red de valor.")
        return False
    vcfg = cfg["value"]
    Xs, Ys, games = [], [], 0
    for f in sorted((base_dir / cfg["imitation"]["data_dir"]).glob("own_*.npz")):
        d = np.load(f)
        y = RESULT_VALUE.get(str(d["source"]))
        if y is None or d["X"].shape[1] != feature_dim(cfg):
            continue
        Xs.append(d["X"])
        Ys.append(np.full(len(d["X"]), y, np.float32))
        games += 1
    if games < vcfg["min_games"]:
        log.info("Red de valor: %d partidas con resultado (< %d), se entrenará más adelante.",
                 games, vcfg["min_games"])
        return False
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    torch.manual_seed(0)
    perm = np.random.default_rng(0).permutation(len(X))
    n_val = max(1, len(X) // 10)
    val, tr = perm[:n_val], perm[n_val:]
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    net = _build_net(torch, X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(vcfg["epochs"]):
        net.train()
        np.random.shuffle(tr)
        for b in range(0, len(tr), 256):
            idx = torch.as_tensor(tr[b:b + 256])
            loss = bce(net(Xt[idx]).squeeze(1), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Xt[val]).squeeze(1))
        decided = Yt[val] != 0.5
        acc = ((p[decided] > 0.5).float() == Yt[val][decided]).float().mean().item() if decided.any() else float("nan")
    torch.save({"state": net.state_dict(), "in_dim": X.shape[1]}, base_dir / vcfg["model"])
    log.info("Red de valor entrenada con %d estados de %d partidas. Acierto ganador (validación): %.1f%%",
             len(X), games, 100 * acc)
    return True


class ValueEvaluator:
    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg
        self.path = base_dir / cfg["value"]["model"]
        self.ok = False
        self.reload()

    def reload(self) -> None:
        self.ok = False
        if not self.cfg["value"]["enabled"] or not self.path.exists():
            return
        try:
            import torch
        except ImportError:
            return
        ck = torch.load(self.path, map_location="cpu")
        if ck["in_dim"] != feature_dim(self.cfg):
            log.warning("value.pt no coincide con la configuración actual; se reentrenará.")
            return
        self.torch = torch
        self.net = _build_net(torch, ck["in_dim"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.ok = True
        log.info("Red de valor cargada (%s)", self.path)

    def win_prob(self, state: GameState, ctx: BattleContext) -> float:
        x = encode_state(state, self.cfg, ctx.multiplier, ctx.elapsed, ctx.opp_elixir)
        with self.torch.no_grad():
            return float(self.torch.sigmoid(self.net(self.torch.as_tensor(x)[None]))[0, 0])

    def rerank(self, state: GameState, ctx: BattleContext, candidates: list[Action]) -> float:
        """Ajusta la prioridad de cada candidata con V(después) - V(ahora). Devuelve V(ahora)."""
        if not self.ok or not candidates:
            return float("nan")
        feats = [encode_state(state, self.cfg, ctx.multiplier, ctx.elapsed, ctx.opp_elixir)]
        feats += [encode_state(simulate(state, a, self.cfg), self.cfg, ctx.multiplier, ctx.elapsed,
                               ctx.opp_elixir) for a in candidates]
        with self.torch.no_grad():
            v = self.torch.sigmoid(self.net(self.torch.as_tensor(np.stack(feats)))).squeeze(1).numpy()
        w = self.cfg["value"]["weight"]
        for a, v_after in zip(candidates, v[1:]):
            adv = float(v_after - v[0])
            a.prior = float(np.clip(a.prior + w * adv, 0.01, 0.99))
            a.reason += f" | eval {adv:+.2f}"
        candidates.sort(key=lambda a: -a.prior)
        return float(v[0])

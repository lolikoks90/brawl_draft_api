"""Fase 3 - Aprendizaje por imitación.

Fuentes de datos (todas se procesan sin etiquetar nada a mano):
  1. Vídeos de partidas de profesionales grabados en vertical con la mano visible
     (--ingest-videos CARPETA). De cada vídeo se detecta automáticamente:
        - cuándo se juega una carta: cambia el hash de un slot y baja el elixir;
        - su coste: la caída de elixir;
        - dónde se colocó: la zona de la arena que cambia justo después y que antes
          estaba quieta (diferencia de frames).
  2. Las partidas del propio bot: cada decisión se guarda con su estado y, al acabar,
     las victorias se usan como ejemplos ("auto-imitación").

Con eso se entrena una red pequeña (PyTorch) que, dado el estado, predice qué slot jugar
(o esperar) y en qué zona. En juego, sus propuestas compiten con las reglas y el bandido
(fase 2) aprende en qué situaciones conviene hacerle caso.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import KINDS, ROLES
from opponent import OpponentModel, elixir_multiplier
from perception import GameState, Perception, crop, hamming
from tactics import Action, BattleContext

log = logging.getLogger("crbot")


# =============================================================================
# CODIFICACIÓN DEL ESTADO
# =============================================================================

def feature_dim(cfg: dict) -> int:
    gx, gy = cfg["imitation"]["grid"]
    return (len(KINDS) + 1) * gx * gy + 4 * (2 + len(ROLES)) + 5


def encode_state(state: GameState, cfg: dict, multiplier: float, elapsed: float,
                 opp_elixir: Optional[float]) -> np.ndarray:
    gx, gy = cfg["imitation"]["grid"]
    cols, rows = cfg["arena_tiles"]
    grid = np.zeros((len(KINDS) + 1, gy, gx), np.float32)
    heavy_area = cfg["threat_heavy_area"]

    def cell(u):
        return min(gy - 1, u.tile[1] * gy // rows), min(gx - 1, u.tile[0] * gx // cols)

    for u in state.enemies:
        kind = u.kind if u.kind in KINDS else ("heavy" if u.area >= heavy_area else "light")
        cy, cx = cell(u)
        grid[KINDS.index(kind), cy, cx] += 1
    for u in state.allies:
        cy, cx = cell(u)
        grid[-1, cy, cx] += 1
    grid = np.minimum(grid, 5) / 5

    hand = np.zeros((4, 2 + len(ROLES)), np.float32)
    for c in state.hand[:4]:
        cost = c.cost if c.cost is not None else 3.5
        hand[c.slot, 0] = cost / 10
        hand[c.slot, 1] = float(c.available and cost <= state.elixir)
        if c.role in ROLES:
            hand[c.slot, 2 + ROLES.index(c.role)] = 1
    misc = np.array([state.elixir / 10, multiplier / 3, min(elapsed / 300, 1),
                     (opp_elixir / 10) if opp_elixir is not None else 0.5,
                     float(opp_elixir is not None)], np.float32)
    return np.concatenate([grid.ravel(), hand.ravel(), misc])


def tile_to_cell(tile, cfg) -> int:
    tx, ty = cfg["imitation"]["tile_grid"]
    cols, rows = cfg["arena_tiles"]
    return min(ty - 1, tile[1] * ty // rows) * tx + min(tx - 1, tile[0] * tx // cols)


def cell_to_tile(cell: int, cfg) -> tuple[int, int]:
    tx, ty = cfg["imitation"]["tile_grid"]
    cols, rows = cfg["arena_tiles"]
    cy, cx = divmod(int(cell), tx)
    return int((cx + 0.5) * cols / tx), int((cy + 0.5) * rows / ty)


# =============================================================================
# GRABACIÓN DE LAS PARTIDAS DEL BOT
# =============================================================================

class GameRecorder:
    WAIT_EVERY_S = 1.5

    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg
        self.dir = base_dir / cfg["imitation"]["data_dir"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.start_game()

    def start_game(self) -> None:
        self.X, self.ys, self.yc = [], [], []
        self.last_wait = -1e9

    def record(self, state: GameState, ctx: BattleContext, action: Optional[Action]) -> None:
        if action is None:
            if ctx.now - self.last_wait < self.WAIT_EVERY_S:
                return
            self.last_wait = ctx.now
        self.X.append(encode_state(state, self.cfg, ctx.multiplier, ctx.elapsed, ctx.opp_elixir))
        self.ys.append(0 if action is None else action.slot + 1)
        self.yc.append(-1 if action is None else tile_to_cell(action.tile, self.cfg))

    def end_game(self, result: str) -> None:
        if len(self.X) >= 10:
            path = self.dir / f"own_{int(time.time())}_{result}.npz"
            np.savez_compressed(path, X=np.stack(self.X), y_slot=np.array(self.ys, np.int64),
                                y_cell=np.array(self.yc, np.int64), source=np.array(result))
        self.start_game()


# =============================================================================
# EXTRACCIÓN DE JUGADAS DESDE VÍDEOS
# =============================================================================

@dataclasses.dataclass
class _Rec:
    t: float
    state: GameState
    small: np.ndarray       # recorte de la arena reducido (BGR)


def _role_from(cost: float, rows: list[int], river_top: int) -> str:
    if rows and np.mean([r < river_top for r in rows]) > 0.5:
        return "spell_big" if cost >= 4 else "spell_small"
    if cost >= 5:
        return "tank"
    if cost >= 4:
        return "support"
    if cost >= 3:
        return "splash"
    return "cycle"


class VideoIngestor:
    DIFF_THR = 40

    def __init__(self, cfg: dict, base_dir: Path, detector=None):
        self.cfg = cfg
        self.icfg = cfg["imitation"]
        self.detector = detector
        self.out = base_dir / self.icfg["data_dir"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.maxd = cfg["hash_match_distance"]
        self.cols, self.rows = cfg["arena_tiles"]

    # ---- lectura --------------------------------------------------------------
    def _read(self, path: Path) -> list[_Rec]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            log.warning("No se pudo abrir %s", path)
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / self.icfg["video_fps"])))
        perception = Perception(self.cfg, self.detector)
        recs, idx = [], 0
        while cap.grab():
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (540, int(540 * h / w)))
                    st = perception.analyse(frame)
                    arena = crop(frame, self.cfg["arena_roi"])
                    ah, aw = arena.shape[:2]
                    small = cv2.resize(arena, (90, max(1, int(90 * ah / aw))), interpolation=cv2.INTER_AREA)
                    recs.append(_Rec(idx / fps, st, small))
            idx += 1
        cap.release()
        return recs

    def _segments(self, recs: list[_Rec]) -> list[list[_Rec]]:
        segs, cur, last_t = [], [], None
        for r in recs:
            if r.state.in_battle and len(r.state.hand) == 4:
                if cur and r.t - last_t > 1.0:
                    segs.append(cur)
                    cur = []
                cur.append(r)
                last_t = r.t
        if cur:
            segs.append(cur)
        return [s for s in segs if s[-1].t - s[0].t >= self.icfg["min_battle_s"]]

    # ---- detección de jugadas -------------------------------------------------
    def _placement(self, seg: list[_Rec], i: int) -> Optional[tuple[int, int]]:
        n = len(seg)
        before, after, prev = seg[i - 2].small, seg[min(i + 5, n - 1)].small, seg[max(i - 7, 0)].small
        # Diferencia de color (máximo por canal): una tropa nueva cambia el tono aunque no el brillo.
        d_after = (cv2.absdiff(after, before).max(axis=2) > self.DIFF_THR).astype(np.uint8)
        moving = cv2.dilate((cv2.absdiff(before, prev).max(axis=2) > self.DIFF_THR).astype(np.uint8),
                            np.ones((5, 5), np.uint8))
        cand = d_after & (1 - moving)
        num, _, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
        if num <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < 4:
            return None
        gh, gw = cand.shape
        cx, cy = cents[k]
        return (int(np.clip(cx / gw * self.cols, 0, self.cols - 1)),
                int(np.clip(cy / gh * self.rows, 0, self.rows - 1)))

    def _events(self, seg: list[_Rec]) -> list[tuple[int, int, int, float, tuple[int, int]]]:
        dt = (seg[-1].t - seg[0].t) / max(1, len(seg) - 1)
        events, last_by_slot = [], {}
        for i in range(2, len(seg) - 6):
            prev, cur = seg[i - 1].state, seg[i].state
            for k in range(4):
                hp, hn = prev.hand[k].hash, cur.hand[k].hash
                if hamming(hp, hn) <= self.maxd or i - last_by_slot.get(k, -99) < 5:
                    continue
                if any(hamming(seg[i + j].state.hand[k].hash, hn) > self.maxd for j in (1, 2)):
                    continue
                mult = elixir_multiplier(self.cfg["elixir_schedule"], seg[i].t - seg[0].t)
                drop = seg[i - 2].state.elixir - seg[i + 3].state.elixir + 5 * dt / self.cfg["elixir_regen_s"] * mult
                if drop < 0.7:
                    continue
                tile = self._placement(seg, i)
                if tile is None:
                    continue
                events.append((i, k, hp, drop, tile))
                last_by_slot[k] = i
        return events

    # ---- construcción de ejemplos ---------------------------------------------
    def _process_segment(self, seg: list[_Rec], name: str) -> int:
        events = self._events(seg)
        if len(events) < 5:
            return 0
        clusters: list[dict] = []

        def cluster(h: int) -> dict:
            for c in clusters:
                if hamming(c["hash"], h) <= self.maxd:
                    return c
            c = {"hash": h, "costs": [], "rows": []}
            clusters.append(c)
            return c

        for _, _, h, drop, tile in events:
            c = cluster(h)
            c["costs"].append(drop)
            c["rows"].append(tile[1])
        river_top = self.cfg["river_rows"][0]
        for c in clusters:
            c["cost"] = int(round(float(np.median(c["costs"])))) if c["costs"] else None
            c["role"] = _role_from(c["cost"], c["rows"], river_top) if c["cost"] else None

        def enriched(st: GameState) -> GameState:
            hand = []
            for hc in st.hand:
                c = cluster(hc.hash)
                hand.append(dataclasses.replace(hc, cost=c.get("cost"), role=c.get("role")))
            return dataclasses.replace(st, hand=hand)

        opp = OpponentModel(self.cfg)
        opp.reset(seg[0].state.elixir, seg[0].t)
        opp_e = []
        for r in seg:
            opp.update(r.state, r.t)
            opp_e.append(opp.elixir)

        def feats(i: int) -> np.ndarray:
            el = seg[i].t - seg[0].t
            return encode_state(enriched(seg[i].state), self.cfg,
                                elixir_multiplier(self.cfg["elixir_schedule"], el), el, opp_e[i])

        X, ys, yc = [], [], []
        ev_idx = {e[0] for e in events}
        for i, k, _, _, tile in events:
            X.append(feats(i - 2))
            ys.append(k + 1)
            yc.append(tile_to_cell(tile, self.cfg))
        for i in range(2, len(seg) - 6, 15):
            if all(abs(i - j) > 15 for j in ev_idx):
                X.append(feats(i))
                ys.append(0)
                yc.append(-1)
        np.savez_compressed(self.out / f"video_{name}.npz", X=np.stack(X), y_slot=np.array(ys, np.int64),
                            y_cell=np.array(yc, np.int64), source=np.array("pro"))
        return len(events)

    def ingest(self, folder: Path) -> int:
        exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        videos = [p for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in exts]
        if not videos:
            log.warning("No hay vídeos en %s", folder)
            return 0
        total = 0
        for v in videos:
            recs = self._read(v)
            segs = self._segments(recs)
            n = sum(self._process_segment(s, f"{v.stem}_{j}") for j, s in enumerate(segs))
            log.info("%s: %d partidas, %d jugadas extraídas", v.name, len(segs), n)
            total += n
        log.info("Total: %d jugadas de %d vídeos", total, len(videos))
        return total


# =============================================================================
# ENTRENAMIENTO Y POLÍTICA
# =============================================================================

def _build_net(torch, in_dim: int, n_cells: int):
    nn = torch.nn

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(256, 256), nn.ReLU())
            self.slot = nn.Linear(256, 5)       # 0 = esperar, 1..4 = slot
            self.cell = nn.Linear(256, n_cells)

        def forward(self, x):
            h = self.body(x)
            return self.slot(h), self.cell(h)

    return PolicyNet()


def train_policy(cfg: dict, base_dir: Path) -> bool:
    try:
        import torch
    except ImportError:
        log.warning("PyTorch no instalado: no se puede entrenar la política.")
        return False
    icfg = cfg["imitation"]
    weights = icfg["source_weights"]
    Xs, S, C, W = [], [], [], []
    for f in sorted((base_dir / icfg["data_dir"]).glob("*.npz")):
        d = np.load(f)
        w = weights.get(str(d["source"]), 0.0)
        if w <= 0 or d["X"].shape[1] != feature_dim(cfg):
            continue
        Xs.append(d["X"]); S.append(d["y_slot"]); C.append(d["y_cell"])
        W.append(np.full(len(d["X"]), w, np.float32))
    if not Xs:
        log.info("Política: aún no hay datos de entrenamiento.")
        return False
    X, ys, yc, w = map(np.concatenate, (Xs, S, C, W))
    w = np.where(ys == 0, w * 0.5, w)      # hay muchas más esperas que jugadas
    if len(X) < icfg["min_samples"]:
        log.info("Política: %d ejemplos (< %d), se entrenará más adelante.", len(X), icfg["min_samples"])
        return False

    tx, ty = icfg["tile_grid"]
    torch.manual_seed(0)
    perm = np.random.default_rng(0).permutation(len(X))
    n_val = max(1, len(X) // 10)
    val, tr = perm[:n_val], perm[n_val:]
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    Xt, St, Ct, Wt = T(X), T(ys, torch.long), T(yc, torch.long), T(w)
    net = _build_net(torch, X.shape[1], tx * ty)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    for epoch in range(icfg["epochs"]):
        net.train()
        np.random.shuffle(tr)
        for b in range(0, len(tr), 256):
            idx = torch.as_tensor(tr[b:b + 256])
            ls, lc = net(Xt[idx])
            wb = Wt[idx]
            loss = (wb * ce(ls, St[idx])).mean()
            m = Ct[idx] >= 0
            if m.any():
                loss = loss + (wb[m] * ce(lc[m], Ct[idx][m])).sum() / wb[m].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        ls, lc = net(Xt[val])
        acc_s = (ls.argmax(1) == St[val]).float().mean().item()
        m = Ct[val] >= 0
        acc_c = (lc[m].argmax(1) == Ct[val][m]).float().mean().item() if m.any() else float("nan")
    torch.save({"state": net.state_dict(), "in_dim": X.shape[1], "n_cells": tx * ty},
               base_dir / icfg["model"])
    log.info("Política entrenada con %d ejemplos. Validación: acierto slot %.1f%%, zona %.1f%%",
             len(X), 100 * acc_s, 100 * acc_c)
    return True


class ImitationPolicy:
    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg
        self.path = base_dir / cfg["imitation"]["model"]
        self.ok = False
        self.reload()

    def reload(self) -> None:
        self.ok = False
        if not self.cfg["imitation"]["enabled"] or not self.path.exists():
            return
        try:
            import torch
        except ImportError:
            log.warning("PyTorch no instalado: política de imitación desactivada.")
            return
        ck = torch.load(self.path, map_location="cpu")
        if ck["in_dim"] != feature_dim(self.cfg):
            log.warning("policy.pt no coincide con la configuración actual; reentrena con --train-policy.")
            return
        self.torch = torch
        self.net = _build_net(torch, ck["in_dim"], ck["n_cells"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.ok = True
        log.info("Política de imitación cargada (%s)", self.path)

    def propose(self, state: GameState, ctx: BattleContext, engine) -> list[Action]:
        if not self.ok or len(state.hand) < 4:
            return []
        x = encode_state(state, self.cfg, ctx.multiplier, ctx.elapsed, ctx.opp_elixir)
        with self.torch.no_grad():
            ls, lc = self.net(self.torch.as_tensor(x)[None])
            ps = self.torch.softmax(ls, 1)[0].numpy()
            cell = int(lc.argmax(1)[0])
        out = []
        for slot in np.argsort(-ps[1:]) + 1:
            p = float(ps[slot])
            if p < self.cfg["imitation"]["min_prob"]:
                break
            card = state.hand[slot - 1]
            if not card.available or (card.cost is not None and card.cost > state.elixir):
                continue
            tile = engine.legal_tile(cell_to_tile(cell, self.cfg), card.role, ctx)
            out.append(Action(card.slot, tile, f"imitación {card.name or '?'} ({p:.0%})", card.hash,
                              "imitation", "policy", card.role, card.name, 0.5 + 0.5 * p))
        return out

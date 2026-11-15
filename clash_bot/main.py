#!/usr/bin/env python3
"""
Clash Royale - bot autónomo (captura -> percepción -> decisión -> acción) en 3 fases.

  Fase 1  Autonomía total: busca partida, juega, pulsa OK al terminar y vuelve a empezar.
          Sigue torres/coronas, estima el elixir y el ciclo del rival y usa tácticas
          (defensa, castigo, contraataque, apoyo al tanque, hechizos de valor...).
  Fase 2  Aprendizaje por refuerzo: elige entre jugadas candidatas con un bandido
          contextual (UCB) que aprende de cada victoria/derrota (learner_db.json).
  Fase 3  Imitación: una red entrenada con vídeos de profesionales y con las victorias
          del propio bot propone jugadas; se reentrena sola cada N partidas.

Módulos: config.py (todo lo ajustable), perception.py, opponent.py, tactics.py,
learner.py, screens.py, imitation.py, controls.py, tools.py.

Uso
---
    python main.py                       # juega partidas sin parar
    python main.py --max-games 20        # para tras 20 partidas
    python main.py --debug               # + ventana con lo que detecta (q para salir)
    python main.py --calibrate           # guarda calibration.png y muestra diagnósticos
    python main.py --stats               # resumen de partidas y de lo aprendido

  YOLO (detección de tropas concretas, opcional)
    python main.py --collect [--no-play] # guarda imágenes de la arena en dataset/
    python main.py --train dataset/data.yaml

  Imitación (fase 3)
    python main.py --ingest-videos videos/   # extrae jugadas de vídeos de profesionales
    python main.py --train-policy            # entrena policy.pt (también se hace solo)

Parada: ratón a una esquina de la pantalla, Ctrl+C, o 'q' en la ventana de depuración.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from config import CONFIG, DECK
from imitation import GameRecorder, ImitationPolicy, VideoIngestor, train_policy
from learner import Learner
from opponent import OpponentModel
from perception import CardMemory, GameState, Perception, ScreenCapture, YoloDetector
from screens import MatchTracker, Navigator, Screen, ScreenDetector, Stats
from tactics import BattleContext, TacticalEngine
from tools import DatasetCollector, draw_overlay, train_yolo

log = logging.getLogger("crbot")
BASE_DIR = Path(__file__).parent


class Bot:
    def __init__(self, cfg: dict, debug: bool = False, collect: bool = False,
                 play: bool = True, max_games: int = 0, capture=None, controls=None):
        self.cfg = cfg
        self.debug = debug
        self.play = play
        self.capture = capture or ScreenCapture(cfg)
        if controls is None:
            from controls import HumanInput   # importa pyautogui sólo cuando se va a jugar
            controls = HumanInput(cfg, self.capture)
        self.controls = controls
        self.detector = YoloDetector(cfg, BASE_DIR)
        self.perception = Perception(cfg, self.detector)
        self.memory = CardMemory(BASE_DIR / cfg["cards_db"], DECK, cfg["hash_match_distance"])
        self.engine = TacticalEngine(cfg)
        self.screens = ScreenDetector(cfg)
        self.match = MatchTracker(cfg)
        self.opponent = OpponentModel(cfg)
        self.learner = Learner(cfg, BASE_DIR)
        self.recorder = GameRecorder(cfg, BASE_DIR)
        self.policy = ImitationPolicy(cfg, BASE_DIR)
        self.navigator = Navigator(cfg, self.controls, max_games)
        self.stats = Stats(BASE_DIR / cfg["stats_file"])
        self.collector = DatasetCollector(cfg, BASE_DIR) if collect else None

        self.in_match = False
        self.no_bar_since: Optional[float] = None
        self.last_play = 0.0
        self.pending_cost: Optional[tuple[int, float, float]] = None
        self.plays = 0
        self.threat_seen = False
        self.clear_since: Optional[float] = None
        self.games_since_retrain = 0
        self.ctx = BattleContext(now=time.time())

    # ---- ciclo de partida -----------------------------------------------------
    def _start_match(self, state: GameState, now: float) -> None:
        log.info("=== Partida iniciada ===")
        self.in_match = True
        self.no_bar_since = None
        self.plays = 0
        self.pending_cost = None
        self.threat_seen, self.clear_since = False, None
        self.match.start(now)
        self.opponent.reset(state.elixir, now)   # ambos empiezan con el mismo elixir
        self.engine.history.clear()
        self.learner.start_game()
        self.recorder.start_game()

    def _end_match(self, now: float) -> None:
        self.in_match = False
        result = self.match.result()
        self.stats.record(result, self.match.crowns(), self.match.elapsed(now), self.plays)
        self.learner.end_game(result)
        self.recorder.end_game(result)
        self.memory.save()
        self.navigator.on_game_over(now)
        self.games_since_retrain += 1
        every = self.cfg["imitation"]["auto_retrain_every_games"]
        if self.cfg["imitation"]["enabled"] and every and self.games_since_retrain >= every:
            self.games_since_retrain = 0
            log.info("Reentrenando la política de imitación con las últimas partidas...")
            if train_policy(self.cfg, BASE_DIR):
                self.policy.reload()

    def _check_cost_measurement(self, state: GameState, now: float) -> None:
        if not self.pending_cost:
            return
        h, before, t0 = self.pending_cost
        dt = now - t0
        if dt < 0.5:
            return
        drop = before - state.elixir + dt / self.cfg["elixir_regen_s"] * self.match.multiplier(now)
        if 0.5 <= drop <= 10.5:
            self.memory.learn_cost(h, drop)
        self.pending_cost = None

    def _track_threats(self, state: GameState, now: float) -> None:
        if state.threats_own_half:
            self.threat_seen, self.clear_since = True, None
        elif self.threat_seen:
            self.clear_since = self.clear_since or now
            if now - self.clear_since >= 2.0:
                self.learner.on_event("threat_cleared", now)
                self.threat_seen, self.clear_since = False, None

    def _battle_step(self, frame, state: GameState, now: float) -> None:
        for card in state.hand:
            self.memory.enrich(card)
        self._check_cost_measurement(state, now)
        for ev in self.match.update(frame, now):
            self.learner.on_event(ev.split(":")[0], now)
        self.opponent.update(state, now)
        self._track_threats(state, now)
        if self.collector:
            self.collector.maybe_save(frame, state)

        self.ctx = BattleContext(
            now=now, elapsed=self.match.elapsed(now), multiplier=self.match.multiplier(now),
            opp_elixir=self.opponent.elixir, opp_recent=self.opponent.recent(6.0, now),
            opp_likely_hand=self.opponent.likely_in_hand(),
            enemy_towers_down=self.match.enemy_down(), own_towers_down=self.match.own_down())

        if not self.play or self.pending_cost:
            return
        if now - self.last_play < self.cfg["min_seconds_between_plays"]:
            return
        sit, cands = self.engine.candidates(state, self.ctx)
        cands += self.policy.propose(state, self.ctx, self.engine)
        cands.sort(key=lambda a: -a.prior)
        action = self.learner.choose(sit, state.elixir, self.ctx, cands)
        self.recorder.record(state, self.ctx, action)
        if action is None:
            return
        log.info("[%.1f elixir | rival ~%.1f | x%.0f] %s -> slot %d, baldosa %s (%s)",
                 state.elixir, self.ctx.opp_elixir, self.ctx.multiplier, action.reason,
                 action.slot + 1, action.tile, self.learner.arm_key(action))
        self.controls.play(action)
        self.last_play = time.time()
        self.plays += 1
        self.engine.notify_played(action, self.last_play)
        self.pending_cost = (action.card_hash, state.elixir, self.last_play)

    # ---- bucle ----------------------------------------------------------------
    def step(self) -> None:
        now = time.time()
        frame = self.capture.grab()
        state = self.perception.analyse(frame)
        screen = self.screens.classify(frame, state)

        if screen == Screen.BATTLE:
            self.no_bar_since = None
            if not self.in_match:
                self._start_match(state, now)
            self._battle_step(frame, state, now)
        else:
            if self.in_match:
                self.no_bar_since = self.no_bar_since or now
                if now - self.no_bar_since >= self.cfg["screens"]["end_confirm_s"]:
                    self._end_match(now)
            elif self.play:
                self.navigator.step(screen, now)

        if self.debug:
            extra = f"[{screen.value}]"
            if self.in_match:
                extra += f" t={self.ctx.elapsed:.0f}s x{self.ctx.multiplier:.0f} rival~{self.ctx.opp_elixir:.1f}"
            cv2.imshow("crbot debug", draw_overlay(frame, self.cfg, state, extra))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    def run(self) -> None:
        period = 1.0 / self.cfg["target_fps"]
        log.info("Bot en marcha. %s", self.stats.summary())
        while True:
            t0 = time.perf_counter()
            self.step()
            if not self.in_match and self.navigator.finished():
                log.info("Límite de partidas alcanzado.")
                return
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))


# =============================================================================
# COMANDOS
# =============================================================================

def calibrate(cfg: dict) -> None:
    frame = ScreenCapture(cfg).grab()
    state = Perception(cfg, YoloDetector(cfg, BASE_DIR)).analyse(frame)
    det = ScreenDetector(cfg)
    screen = det.classify(frame, state)
    out = BASE_DIR / "calibration.png"
    cv2.imwrite(str(out), draw_overlay(frame, cfg, state, f"[{screen.value}]"))
    print(f"Guardado {out}")
    print(f"Pantalla detectada: {screen.value} | botones: {det.debug_info(frame)} "
          f"(umbral {cfg['screens']['min_button_fill']})")
    print(f"Elixir leído: {state.elixir} | cartas: {['ok' if c.available else 'gris' for c in state.hand]}")
    fills = {k: round(v, 3) for k, v in MatchTracker(cfg)._fills(frame).items()}
    print(f"Color de torres (en batalla debe ser > 0.01): {fills}")
    print(f"Enemigos: {len(state.enemies)} | aliados: {len(state.allies)}")


def show_stats(cfg: dict) -> None:
    stats = Stats(BASE_DIR / cfg["stats_file"])
    print("Global:     ", stats.summary())
    print("Últimas 20: ", stats.summary(last=20))
    rows = Learner(cfg, BASE_DIR).summary()
    if rows:
        print("\nJugadas más usadas (contexto | jugada | veces | recompensa media):")
        for ctx, arm, n, mean in rows:
            print(f"  {ctx:32s} {arm:36s} {n:5d}  {mean:+.2f}")


def main() -> int:
    p = argparse.ArgumentParser(description="Bot autónomo de Clash Royale para emulador.")
    p.add_argument("--debug", action="store_true", help="ventana con lo que detecta")
    p.add_argument("--calibrate", action="store_true", help="guarda calibration.png y diagnósticos")
    p.add_argument("--max-games", type=int, default=0, help="parar tras N partidas (0 = sin límite)")
    p.add_argument("--stats", action="store_true", help="resumen de partidas y aprendizaje")
    p.add_argument("--collect", action="store_true", help="guarda imágenes para entrenar YOLO")
    p.add_argument("--no-play", action="store_true", help="sólo observa, no hace clics")
    p.add_argument("--train", metavar="DATA_YAML", help="entrena el detector YOLO")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--ingest-videos", metavar="CARPETA", help="extrae jugadas de vídeos (fase 3)")
    p.add_argument("--train-policy", action="store_true", help="entrena la política de imitación")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.train:
        train_yolo(CONFIG, args.train, args.epochs)
        return 0
    if args.stats:
        show_stats(CONFIG)
        return 0
    if args.ingest_videos:
        VideoIngestor(CONFIG, BASE_DIR, YoloDetector(CONFIG, BASE_DIR)).ingest(Path(args.ingest_videos))
        train_policy(CONFIG, BASE_DIR)
        return 0
    if args.train_policy:
        return 0 if train_policy(CONFIG, BASE_DIR) else 1
    if args.calibrate:
        calibrate(CONFIG)
        return 0

    import pyautogui
    bot = Bot(CONFIG, debug=args.debug, collect=args.collect, play=not args.no_play,
              max_games=args.max_games)
    time.sleep(2)
    try:
        bot.run()
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        log.info("Detenido por el usuario.")
    finally:
        bot.memory.save()
        cv2.destroyAllWindows()
        log.info(bot.stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())

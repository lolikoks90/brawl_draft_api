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
  Eval.   Estilo Stockfish: una red de valor estima la probabilidad de ganar y cada jugada
          candidata se puntúa por cómo cambia esa probabilidad (evaluator.py).

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
from evaluator import ValueEvaluator, train_value
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
        self.value = ValueEvaluator(cfg, BASE_DIR)
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
        self.last_screen = Screen.UNKNOWN
        self.match_ended_at = 0.0
        self._warned_cards = False

    # ---- ciclo de partida -----------------------------------------------------
    def _start_match(self, state: GameState, now: float) -> None:
        log.info("=== Partida iniciada ===")
        self.in_match = True
        self.no_bar_since = None
        self.plays = 0
        self.pending_cost = None
        self.threat_seen, self.clear_since = False, None
        self.match.start(now)
        self.perception.reset_background()
        self.opponent.reset(state.elixir, now)   # ambos empiezan con el mismo elixir
        self.engine.history.clear()
        self.learner.start_game()
        self.recorder.start_game()

    def _end_match(self, now: float) -> None:
        self.in_match = False
        self.match_ended_at = now
        if self.memory.unknown_count():
            log.warning("Hay %d cartas sin identificar: abre la pestaña 'Cartas' del lanzador y di cuál es "
                        "cada una. El bot NO juega las cartas que no conoce.", self.memory.unknown_count())
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
            log.info("Reentrenando la política de imitación y la red de valor...")
            if train_policy(self.cfg, BASE_DIR):
                self.policy.reload()
            if self.cfg["value"]["enabled"] and train_value(self.cfg, BASE_DIR):
                self.value.reload()

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
        self.value.rerank(state, self.ctx, cands)      # "Stockfish": evalúa el tablero tras cada jugada
        action = self.learner.choose(sit, state.elixir, self.ctx, cands)
        self.recorder.record(state, self.ctx, action)
        if action is None:
            return
        log.info("[%.1f elixir | rival ~%.1f | x%.0f] %s -> slot %d, baldosa %s (%s)",
                 state.elixir, self.ctx.opp_elixir, self.ctx.multiplier, action.reason,
                 action.slot + 1, action.tile, self.learner.arm_key(action))
        self._save_debug(frame, state, action)
        self.controls.play(action)
        self.last_play = time.time()
        self.plays += 1
        self.engine.notify_played(action, self.last_play)
        self.pending_cost = (action.card_hash, state.elixir, self.last_play)

    def _save_debug(self, frame, state: GameState, action) -> None:
        """Guarda lo que veía el bot al decidir cada jugada (carpeta capturas/, máx. 40)."""
        try:
            d = BASE_DIR / "capturas"
            d.mkdir(exist_ok=True)
            if not getattr(self, "_debug_cleared", False):
                for f in d.glob("*.jpg"):
                    f.unlink()
                self._debug_cleared, self._debug_n = True, 0
            if self._debug_n >= 40:
                return
            self._debug_n += 1
            img = draw_overlay(frame, self.cfg, state, action.reason[:40])
            r = self.capture.region
            x1, y1, x2, y2 = self.cfg["arena_roi"]
            cols, rows = self.cfg["arena_tiles"]
            px = int((x1 + (action.tile[0] + 0.5) / cols * (x2 - x1)) * r["width"])
            py = int((y1 + (action.tile[1] + 0.5) / rows * (y2 - y1)) * r["height"])
            cv2.drawMarker(img, (px, py), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 24, 3)
            cv2.imwrite(str(d / f"jugada_{self._debug_n:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as exc:
            log.debug("No pude guardar la captura de depuración: %s", exc)

    # ---- bucle ----------------------------------------------------------------
    def step(self) -> None:
        now = time.time()
        frame = self.capture.grab()
        state = self.perception.analyse(frame)
        screen = self.screens.classify(frame, state)
        self.last_screen = screen

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
            if not getattr(self, "_win_placed", False):
                # Ventana de depuración a la derecha del emulador, para que no lo tape.
                r = self.capture.region
                cv2.namedWindow("crbot debug", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("crbot debug", max(200, r["width"] // 2), max(300, r["height"] // 2))
                cv2.moveWindow("crbot debug", r["left"] + r["width"] + 10, r["top"])
                self._win_placed = True
            extra = f"[{screen.value}]"
            if self.in_match:
                extra += f" t={self.ctx.elapsed:.0f}s x{self.ctx.multiplier:.0f} rival~{self.ctx.opp_elixir:.1f}"
            cv2.imshow("crbot debug", draw_overlay(frame, self.cfg, state, extra))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    def run(self) -> None:
        period = 1.0 / self.cfg["target_fps"]
        log.info("Bot en marcha. %s", self.stats.summary())
        stop_file = BASE_DIR / "STOP"
        while True:
            if stop_file.exists():          # el lanzador pide parar de forma ordenada
                stop_file.unlink(missing_ok=True)
                log.info("Parada solicitada desde el lanzador.")
                return
            t0 = time.perf_counter()
            self.step()
            if not self.in_match and self.navigator.finished():
                # Antes de salir hay que pulsar OK y volver al menú (máx. 40 s).
                back = self.last_screen == Screen.MENU or time.time() - self.match_ended_at > 40
                if not self.play or back:
                    log.info("Límite de partidas alcanzado.")
                    return
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))


# =============================================================================
# COMANDOS
# =============================================================================

def _find_button(frame, hsv_range, y_from: float, y_to: float):
    """Mayor mancha del color del botón en la franja vertical indicada -> (roi, centro) en fracciones."""
    import numpy as np
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi = hsv_range
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask[: int(y_from * h)] = 0
    mask[int(y_to * h):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = max(5, int(0.04 * w)) | 1          # une el botón por encima del texto blanco
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[k]
    if area < 0.004 * h * w:
        return None
    # Recorte interior (15 %) para quedarnos con el relleno del botón, no con el borde.
    roi = [round((x + 0.15 * bw) / w, 3), round((y + 0.15 * bh) / h, 3),
           round((x + 0.85 * bw) / w, 3), round((y + 0.85 * bh) / h, 3)]
    return roi, [round((x + bw / 2) / w, 3), round((y + bh / 2) / h, 3)]


def calibrate(cfg: dict, expect: Optional[str] = None, delay: float = 0) -> bool:
    from config import save_settings

    cap = ScreenCapture(cfg)
    if delay:
        print(f"Poniendo el emulador delante y capturando en {delay:.0f} segundos...", flush=True)
        cap.focus()
        time.sleep(delay)
    margins = cap.auto_fit()
    if margins:
        save_settings({"window_margins": margins})
    frame = cap.grab()
    ok = True

    def good(msg):
        print("  OK  ", msg)

    def bad(msg):
        nonlocal ok
        ok = False
        print("  FALLO", msg)

    print("\n=== CALIBRACIÓN", f"({expect})" if expect else "", "===")
    if cap.found:
        good(f"Ventana del emulador encontrada ('{cfg['window_title']}').")
        if margins:
            good(f"Juego localizado dentro de la ventana (márgenes guardados: {margins}).")
        else:
            print("  AVISO No he podido localizar el juego dentro de la ventana; uso los márgenes guardados.")
    else:
        bad(f"No encuentro una ventana llamada '{cfg['window_title']}'. En el lanzador pulsa "
            "'Detectar ventanas' y elige la del emulador.")

    sc = cfg["screens"]
    if expect == "menu":
        found = _find_button(frame, sc["battle_button_hsv"], 0.71, 0.90)
        if found:
            sc["battle_button_roi"], sc["battle_button"] = found
            save_settings({"screens": {"battle_button_roi": found[0], "battle_button": found[1]}})
            good("He encontrado el botón amarillo 'Batalla' y he guardado su posición.")
        else:
            bad("No veo el botón amarillo 'Batalla'. ¿Está el juego en el menú principal?")
    elif expect == "end":
        found = _find_button(frame, sc["ok_button_hsv"], 0.7, 1.0)
        if found:
            sc["ok_button_roi"], sc["ok_button"] = found
            save_settings({"screens": {"ok_button_roi": found[0], "ok_button": found[1]}})
            good("He encontrado el botón azul 'OK' y he guardado su posición.")
        else:
            bad("No veo el botón azul 'OK'. ¿Está el juego en la pantalla de final de partida?")

    state = Perception(cfg, YoloDetector(cfg, BASE_DIR)).analyse(frame)
    det = ScreenDetector(cfg)
    screen = det.classify(frame, state)
    names = {"menu": "menú principal", "battle": "batalla", "end": "final de partida", "unknown": "desconocida"}
    if expect:
        if screen.value == expect:
            good(f"El bot reconoce esta pantalla como: {names[screen.value]}.")
        else:
            bad(f"Esperaba '{names[expect]}' pero el bot ve '{names[screen.value]}'.")
    else:
        print(f"  Pantalla reconocida: {names[screen.value]}")

    if expect == "battle" or (expect is None and state.in_battle):
        if state.in_battle and 0 < state.elixir <= 10:
            good(f"Barra de elixir leída: {state.elixir} (comprueba que coincide con el juego).")
        else:
            bad("No consigo leer la barra de elixir. Revisa los márgenes de la ventana.")
        n_cards = sum(1 for c in state.hand)
        (good if n_cards == 4 else bad)(f"Cartas en mano detectadas: {n_cards} de 4.")
        fills = MatchTracker(cfg)._fills(frame)
        towers_ok = sum(v > 0.01 for v in fills.values())
        (good if towers_ok == 6 else bad)(f"Torres reconocidas por su color: {towers_ok} de 6.")

    out = BASE_DIR / "calibration.png"
    cv2.imwrite(str(out), draw_overlay(frame, cfg, state, f"[{screen.value}]"))
    print(f"\nImagen guardada en {out}")
    print("Todo correcto." if ok else
          "Hay fallos: mira la imagen. Los recuadros de colores deben caer encima de la barra de "
          "elixir, las 4 cartas y la arena. Si no cuadran, ajusta los márgenes de la ventana "
          "(o envía la imagen a quien te ayude).")
    return ok


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
    p.add_argument("--train", metavar="ZIP_O_YAML", help="entrena YOLO con el ZIP de Roboflow o su data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--ingest-videos", metavar="CARPETA", help="extrae jugadas de vídeos (fase 3)")
    p.add_argument("--train-policy", action="store_true", help="entrena la política de imitación y la red de valor")
    p.add_argument("--delay", type=float, default=0, help="segundos de espera antes de capturar")
    p.add_argument("--expect", choices=["menu", "battle", "end"], help="pantalla que se calibra")
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
        ok_p = train_policy(CONFIG, BASE_DIR)
        ok_v = train_value(CONFIG, BASE_DIR)
        return 0 if (ok_p or ok_v) else 1
    if args.calibrate:
        return 0 if calibrate(CONFIG, args.expect, args.delay) else 1

    import pyautogui
    bot = Bot(CONFIG, debug=args.debug, collect=args.collect, play=not args.no_play,
              max_games=args.max_games)
    bot.capture.focus()
    bot.capture.auto_fit()
    time.sleep(1)
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

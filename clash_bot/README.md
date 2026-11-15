# Bot de Clash Royale (emulador de PC)

## Instalación
```
pip install mss opencv-python numpy pyautogui pygetwindow
pip install ultralytics   # opcional: YOLO (fase 1+) y PyTorch para la imitación (fase 3)
```
Windows con la escala de pantalla al 100 %, emulador en vertical y visible.

## Puesta en marcha
1. En `config.py`: `window_title` (título del emulador) y `DECK` (tus 8 cartas con coste y rol).
2. `python main.py --calibrate` en **tres pantallas**: menú principal, batalla y pantalla final.
   Comprueba en `calibration.png` y en la consola que:
   - la pantalla detectada es `menu` / `battle` / `end` respectivamente
     (si no, ajusta `screens` en `config.py`: posiciones y colores de los botones);
   - en batalla el elixir leído coincide y las 6 torres tienen color > 0.01 (ajusta `towers`).
3. `python main.py --debug --max-games 1` para ver una partida con lo que detecta.
4. `python main.py` y a jugar. `python main.py --stats` para ver cómo va.

Parar: ratón a una esquina, Ctrl+C o `q` en la ventana de depuración.

## Las 3 fases
| Fase | Qué hace | Archivos |
|---|---|---|
| 1. Autonomía | Busca partida, juega, pulsa OK y repite. Torres/coronas, elixir y ciclo del rival, tácticas (defensa, castigo, contraataque, apoyo al tanque, hechizos de valor, fuga de elixir). | `screens.py`, `opponent.py`, `tactics.py` |
| 2. Refuerzo | Bandido contextual (UCB): aprende qué jugada funciona en cada situación a partir de victorias/derrotas y torres ganadas/perdidas. | `learner.py` → `learner_db.json` |
| 3. Imitación | Red que imita a profesionales (vídeos) y las victorias del propio bot. Se reentrena sola cada 10 partidas. | `imitation.py` → `policy.pt` |

### Fase 3 con vídeos de profesionales
Pon en `videos/` grabaciones **verticales, a pantalla completa y con la mano visible**
(idealmente grabadas en tu mismo emulador, con la misma interfaz). Luego:
```
python main.py --ingest-videos videos/
```
Detecta automáticamente cada carta jugada, su coste y dónde se colocó, y entrena `policy.pt`.

### YOLO (opcional, mejora todo lo anterior)
Sin modelo se detectan tropas por color (rojo = rival, azul = tuyo). Con YOLO se reconoce
cada tropa (P.E.K.K.A, Montapuercos...) y el elixir del rival se estima con coste exacto.
```
python main.py --collect --no-play     # guarda imágenes mientras juegas
# etiqueta en CVAT / Label Studio / Roboflow y exporta en formato YOLO
python main.py --train dataset/data.yaml
```

## Archivos que genera
`cards_db.json` (cartas aprendidas), `learner_db.json` (fase 2), `policy_data/` y `policy.pt`
(fase 3), `match_history.jsonl` (resultados). Borra uno para reiniciar ese aprendizaje.

"""Lanzador con botones: usa el bot sin escribir comandos.

Doble clic en INICIAR.bat (o: python lanzador.py).
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

BASE = Path(__file__).resolve().parent
MAIN = BASE / "main.py"
sys.path.insert(0, str(BASE))

import config  # noqa: E402

ROLE_LABELS = {
    "tank": "Tanque (Gigante, Montapuercos, Golem...)",
    "tank_killer": "Mata-tanques (Mini P.E.K.K.A, Infernal...)",
    "splash": "Daño en área (Valquiria, Mago, Bebé dragón...)",
    "support": "Apoyo a distancia (Mosquetera, Arqueras...)",
    "cycle": "Barata / ciclo (Esqueletos, Espíritu de hielo...)",
    "building": "Edificio (Cañón, Tesla, Torre bombardera...)",
    "spell_small": "Hechizo pequeño (Descarga, Tronco, Flechas...)",
    "spell_big": "Hechizo grande (Bola de fuego, Veneno, Cohete...)",
}
LABEL_TO_ROLE = {v: k for k, v in ROLE_LABELS.items()}
COMMON_EMULATORS = ["BlueStacks App Player", "BlueStacks", "LDPlayer", "MuMu Player", "MEmu", "Nox"]


class Runner:
    """Ejecuta main.py en segundo plano y pasa su salida a la ventana."""

    def __init__(self, app: "App"):
        self.app = app
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, args: list[str], title: str, on_done=None) -> None:
        if self.busy():
            messagebox.showinfo("Espera", "Ya hay una tarea en marcha. Pulsa 'Detener' primero.")
            return
        (BASE / "STOP").unlink(missing_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.app.log(f"\n▶ {title}\n")
        self.proc = subprocess.Popen([sys.executable, "-u", str(MAIN), *args], cwd=str(BASE),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", env=env, creationflags=flags)
        self.app.set_status(f"En marcha: {title}")

        def reader(proc=self.proc):
            for line in proc.stdout:
                self.q.put(line)
            proc.wait()
            self.q.put(("DONE", proc.returncode, on_done))

        threading.Thread(target=reader, daemon=True).start()

    def stop(self) -> None:
        if not self.busy():
            return
        (BASE / "STOP").write_text("stop")
        self.app.log("\n⏹ Deteniendo...\n")

        def killer(proc=self.proc):
            time.sleep(6)
            if proc.poll() is None:
                proc.terminate()

        threading.Thread(target=killer, daemon=True).start()

    def poll(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item[0] == "DONE":
                    _, code, on_done = item
                    self.app.set_status("Listo." if code == 0 else f"Terminado (código {code}).")
                    if on_done:
                        on_done(code)
                else:
                    self.app.log(item)
        except queue.Empty:
            pass
        self.app.root.after(100, self.poll)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Bot de Clash Royale")
        root.geometry("900x720")
        self.runner = Runner(self)

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=False, padx=8, pady=6)
        self._tab_config(nb)
        self._tab_calibrate(nb)
        self._tab_play(nb)
        self._tab_train(nb)

        bottom = ttk.Frame(root)
        bottom.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        bar = ttk.Frame(bottom)
        bar.pack(fill="x")
        self.status = ttk.Label(bar, text="Listo.")
        self.status.pack(side="left")
        ttk.Button(bar, text="⏹ Detener", command=self.runner.stop).pack(side="right")
        ttk.Button(bar, text="Copiar registro", command=self.copy_log).pack(side="right", padx=4)
        self.text = tk.Text(bottom, height=14, wrap="word", font=("Consolas", 9))
        sb = ttk.Scrollbar(bottom, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)
        self.runner.poll()
        self.log("Bienvenido. Sigue las pestañas en orden: 1 → 2 → 3. La 4 es opcional (mejorar el bot).\n")

    # ---- utilidades -----------------------------------------------------------
    def log(self, msg: str) -> None:
        self.text.insert("end", msg)
        self.text.see("end")

    def set_status(self, msg: str) -> None:
        self.status.configure(text=msg)

    def copy_log(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.text.get("1.0", "end"))
        self.set_status("Registro copiado: pégalo donde te estén ayudando.")

    @staticmethod
    def _help(parent, text: str) -> None:
        ttk.Label(parent, text=text, wraplength=840, justify="left").pack(anchor="w", pady=(4, 8))

    # ---- 1. Configurar --------------------------------------------------------
    def _tab_config(self, nb) -> None:
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="1. Configurar")
        self._help(f, "Abre tu emulador con Clash Royale. Pulsa 'Detectar ventanas' y elige la del emulador. "
                      "Después escribe las 8 cartas de tu mazo, su coste de elixir y qué tipo de carta es. "
                      "Pulsa 'Guardar' al terminar.")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Ventana del emulador:").pack(side="left")
        self.win = ttk.Combobox(row, width=45, values=COMMON_EMULATORS)
        self.win.set(config.CONFIG["window_title"])
        self.win.pack(side="left", padx=6)
        ttk.Button(row, text="Detectar ventanas", command=self.detect_windows).pack(side="left")

        mrow = ttk.Frame(f)
        mrow.pack(fill="x", pady=6)
        ttk.Label(mrow, text="Márgenes (píxeles que no son juego) — arriba/abajo/izq./der.:").pack(side="left")
        self.margins = {}
        for k in ("top", "bottom", "left", "right"):
            e = ttk.Spinbox(mrow, from_=0, to=300, width=5)
            e.set(config.CONFIG["window_margins"][k])
            e.pack(side="left", padx=2)
            self.margins[k] = e

        grid = ttk.LabelFrame(f, text="Tu mazo", padding=6)
        grid.pack(fill="x", pady=6)
        ttk.Label(grid, text="Nombre de la carta").grid(row=0, column=0, sticky="w")
        ttk.Label(grid, text="Elixir").grid(row=0, column=1)
        ttk.Label(grid, text="Tipo").grid(row=0, column=2, sticky="w")
        self.deck_rows = []
        items = list(config.DECK.items()) + [("", (3, "support"))] * (8 - len(config.DECK))
        for i, (name, (cost, role)) in enumerate(items[:8], start=1):
            n = ttk.Entry(grid, width=24)
            n.insert(0, name)
            c = ttk.Spinbox(grid, from_=1, to=10, width=4)
            c.set(cost)
            r = ttk.Combobox(grid, width=48, state="readonly", values=list(ROLE_LABELS.values()))
            r.set(ROLE_LABELS.get(role, ROLE_LABELS["support"]))
            n.grid(row=i, column=0, padx=2, pady=1)
            c.grid(row=i, column=1, padx=2)
            r.grid(row=i, column=2, padx=2, sticky="w")
            self.deck_rows.append((n, c, r))
        ttk.Button(f, text="💾 Guardar", command=self.save_config).pack(anchor="e")

    def detect_windows(self) -> None:
        try:
            import pygetwindow as gw
            titles = sorted({t for t in gw.getAllTitles() if t.strip()})
        except Exception as exc:
            messagebox.showerror("Error", f"No puedo listar ventanas: {exc}")
            return
        self.win.configure(values=titles)
        guess = [t for t in titles if any(e.lower() in t.lower() for e in COMMON_EMULATORS)]
        if guess:
            self.win.set(guess[0])
        self.log("Ventanas abiertas: " + " | ".join(titles) + "\n")

    def save_config(self) -> None:
        deck = {}
        for n, c, r in self.deck_rows:
            name = n.get().strip()
            if name:
                deck[name] = [int(c.get()), LABEL_TO_ROLE[r.get()]]
        if len(deck) != 8:
            if not messagebox.askyesno("Mazo incompleto", f"Has escrito {len(deck)} cartas de 8. ¿Guardar igualmente?"):
                return
        try:
            margins = {k: int(e.get()) for k, e in self.margins.items()}
        except ValueError:
            messagebox.showerror("Error", "Los márgenes deben ser números.")
            return
        config.save_settings({"window_title": self.win.get().strip(), "window_margins": margins, "deck": deck})
        # Si el mazo cambia, las cartas aprendidas ya no sirven.
        db = BASE / config.CONFIG["cards_db"]
        if db.exists() and messagebox.askyesno("Mazo", "¿Has cambiado de mazo? Si es así, olvido las cartas aprendidas."):
            db.unlink()
        self.set_status("Ajustes guardados.")
        self.log("✔ Ajustes guardados en ajustes.json\n")

    # ---- 2. Calibrar ----------------------------------------------------------
    def _tab_calibrate(self, nb) -> None:
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="2. Calibrar")
        self._help(f, "Pon el juego en cada pantalla y pulsa el botón correspondiente. El bot comprobará que la "
                      "reconoce y buscará solo los botones. Si sale FALLO, mira la imagen y ajusta los márgenes "
                      "en la pestaña 1. Haz las tres.")
        for expect, text in [("menu", "A) Estoy en el MENÚ PRINCIPAL (se ve el botón amarillo 'Batalla')"),
                             ("battle", "B) Estoy en una BATALLA (amistosa o entrenamiento, con cartas en mano)"),
                             ("end", "C) Estoy en la pantalla FINAL de una partida (botón azul 'OK')")]:
            ttk.Button(f, text=text, command=lambda e=expect: self.calibrate(e)).pack(fill="x", pady=3)
        ttk.Button(f, text="Ver última imagen de calibración", command=self.show_calibration).pack(anchor="w", pady=6)

    def calibrate(self, expect: str) -> None:
        self.runner.start(["--calibrate", "--expect", expect], f"Calibrar ({expect})",
                          on_done=lambda code: self.show_calibration())

    def show_calibration(self) -> None:
        img_path = BASE / "calibration.png"
        if not img_path.exists():
            messagebox.showinfo("Calibración", "Aún no hay imagen. Pulsa uno de los botones de calibrar.")
            return
        top = tk.Toplevel(self.root)
        top.title("Lo que ve el bot")
        img = tk.PhotoImage(file=str(img_path))
        factor = max(1, int(img.height() / 850) + (1 if img.height() > 850 else 0))
        if factor > 1:
            img = img.subsample(factor)
        lbl = ttk.Label(top, image=img)
        lbl.image = img
        lbl.pack()
        ttk.Label(top, text="Morado = barra de elixir · Amarillo = cartas · Verde = arena · Rojo/azul = torres · "
                            "Naranja = botones Batalla/OK", wraplength=500).pack(pady=4)

    # ---- 3. Jugar -------------------------------------------------------------
    def _tab_play(self, nb) -> None:
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="3. Jugar")
        self._help(f, "Deja el emulador visible y no toques el ratón mientras juega. Para pararlo: botón 'Detener' "
                      "o lleva el ratón a una esquina de la pantalla. Recomendado: usa una cuenta secundaria.")
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Número de partidas (0 = sin límite):").pack(side="left")
        self.games = ttk.Spinbox(row, from_=0, to=1000, width=6)
        self.games.set(1)
        self.games.pack(side="left", padx=6)
        self.show_view = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Mostrar lo que ve el bot", variable=self.show_view).pack(side="left", padx=10)
        ttk.Button(f, text="▶ JUGAR", command=self.play).pack(fill="x", pady=8)
        ttk.Button(f, text="📊 Ver estadísticas", command=lambda: self.runner.start(["--stats"], "Estadísticas")).pack(anchor="w")

    def play(self) -> None:
        args = ["--max-games", str(int(self.games.get() or 0))]
        if self.show_view.get():
            args.append("--debug")
        self.log("El bot empieza en 2 segundos: pon el emulador delante.\n")
        self.runner.start(args, "Jugar")

    # ---- 4. Mejorar -----------------------------------------------------------
    def _tab_train(self, nb) -> None:
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="4. Mejorar el bot (opcional)")
        a = ttk.LabelFrame(f, text="Reconocer tropas (Roboflow)", padding=6)
        a.pack(fill="x", pady=4)
        steps = [
            ("1) Empezar a guardar imágenes (juega tú mientras tanto)",
             lambda: self.runner.start(["--collect", "--no-play"], "Guardar imágenes")),
            ("2) Crear ZIP de imágenes para subir a Roboflow", self.make_zip),
            ("3) Abrir Roboflow (crea proyecto 'Object Detection', sube el ZIP y etiqueta)",
             lambda: webbrowser.open("https://app.roboflow.com")),
            ("4) Entrenar con el ZIP exportado de Roboflow (formato YOLOv11)", self.train_zip),
        ]
        for text, cmd in steps:
            ttk.Button(a, text=text, command=cmd).pack(fill="x", pady=2)
        b = ttk.LabelFrame(f, text="Aprender de profesionales", padding=6)
        b.pack(fill="x", pady=4)
        ttk.Button(b, text="Elegir carpeta con vídeos de partidas (verticales, con la mano visible)",
                   command=self.ingest_videos).pack(fill="x", pady=2)
        ttk.Button(b, text="Reentrenar ahora con todo lo aprendido",
                   command=lambda: self.runner.start(["--train-policy"], "Reentrenar")).pack(fill="x", pady=2)

    def make_zip(self) -> None:
        imgs = BASE / config.CONFIG["dataset_dir"] / "images"
        n = len(list(imgs.glob("*.jpg"))) if imgs.exists() else 0
        if n == 0:
            messagebox.showinfo("Sin imágenes", "Primero pulsa el paso 1 y juega unas partidas.")
            return
        desktop = next((d for d in [Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop",
                                    Path.home() / "Escritorio"] if d.is_dir()), BASE)
        out = shutil.make_archive(str(desktop / "clash_imagenes"), "zip", imgs)
        self.log(f"✔ ZIP con {n} imágenes creado en: {out}\n")
        messagebox.showinfo("ZIP creado", f"{n} imágenes en:\n{out}\n\nSúbelo a Roboflow en el paso 3.")
        if sys.platform == "win32":
            os.startfile(desktop)

    def train_zip(self) -> None:
        path = filedialog.askopenfilename(title="Elige el ZIP exportado de Roboflow",
                                          filetypes=[("ZIP", "*.zip"), ("data.yaml", "*.yaml")])
        if path:
            self.log("El entrenamiento puede tardar desde minutos (con tarjeta gráfica) hasta horas.\n")
            self.runner.start(["--train", path], "Entrenar detector de tropas")

    def ingest_videos(self) -> None:
        folder = filedialog.askdirectory(title="Carpeta con vídeos")
        if folder:
            self.runner.start(["--ingest-videos", folder], "Aprender de vídeos")


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if sys.platform == "win32" else "clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

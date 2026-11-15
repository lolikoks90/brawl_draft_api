"""Depuración visual, recolección de dataset y entrenamiento de YOLO."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from perception import GameState, Perception, crop, tile_rect_px

log = logging.getLogger("crbot")


def draw_overlay(frame: np.ndarray, cfg: dict, state: Optional[GameState], extra: str = "") -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    def rect(roi, color):
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, 1)

    rect(cfg["elixir_roi"], (255, 0, 255))
    for roi in cfg["card_rois"]:
        rect(roi, (0, 255, 255))
    rect(cfg["screens"]["battle_button_roi"], (0, 200, 255))
    rect(cfg["screens"]["ok_button_roi"], (255, 150, 0))
    ax1, ay1, ax2, ay2 = cfg["arena_roi"]
    rect(cfg["arena_roi"], (0, 255, 0))
    cols, rows = cfg["arena_tiles"]
    ox, oy, aw, ah = int(ax1 * w), int(ay1 * h), int((ax2 - ax1) * w), int((ay2 - ay1) * h)
    for r in range(rows + 1):
        y = oy + int(r / rows * ah)
        cv2.line(out, (ox, y), (ox + aw, y), (0, 90, 0), 1)
    for c in range(cols + 1):
        x = ox + int(c / cols * aw)
        cv2.line(out, (x, oy), (x, oy + ah), (0, 90, 0), 1)
    for name, tr in cfg["towers"].items():
        x1, y1, x2, y2 = tile_rect_px(tr, aw, ah, cols, rows)
        color = (0, 0, 255) if name.startswith("enemy") else (255, 0, 0)
        cv2.rectangle(out, (ox + x1, oy + y1), (ox + x2, oy + y2), color, 2)

    def put(text, x, y, color, s=0.4):
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, 1)

    if state:
        put(f"Elixir {state.elixir:.1f}  batalla={state.in_battle} {extra}", 8, 20, (255, 255, 255), 0.45)
        for c in state.hand:
            x1, y1, _, _ = cfg["card_rois"][c.slot]
            put(f"{c.name or '?'}{'' if c.available else ' (gris)'}", int(x1 * w), int(y1 * h) - 4, (0, 255, 255), 0.35)
        for u, color in [(u, (0, 0, 255)) for u in state.enemies + state.enemy_spells] + \
                        [(u, (255, 128, 0)) for u in state.allies]:
            if u.bbox:
                bx1, by1, bx2, by2 = u.bbox
                cv2.rectangle(out, (ox + bx1, oy + by1), (ox + bx2, oy + by2), color, 1)
                put(f"{u.label} {u.conf:.2f}", ox + bx1, oy + by1 - 3, color, 0.35)
            else:
                cx = ox + int((u.tile[0] + 0.5) / cols * aw)
                cy = oy + int((u.tile[1] + 0.5) / rows * ah)
                cv2.circle(out, (cx, cy), 6, color, 2)
    return out


class DatasetCollector:
    """Guarda recortes de la arena + pre-etiquetas YOLO (clase 0) para entrenar el detector."""

    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg
        self.img_dir = base_dir / cfg["dataset_dir"] / "images"
        self.lbl_dir = base_dir / cfg["dataset_dir"] / "prelabels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.color = Perception(cfg)
        self.last = 0.0
        self.count = 0

    def maybe_save(self, frame: np.ndarray, state: GameState) -> None:
        if not state.in_battle or time.time() - self.last < self.cfg["collect_interval_s"]:
            return
        self.last = time.time()
        name = f"cr_{int(self.last * 1000)}"
        cv2.imwrite(str(self.img_dir / f"{name}.jpg"), crop(frame, self.cfg["arena_roi"]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        enemies, allies, _ = self.color._units_color(frame)
        cols, rows = self.cfg["arena_tiles"]
        lines = [f"0 {(u.tile[0] + 0.5) / cols:.5f} {(u.tile[1] + 1.0) / rows:.5f} {1.5 / cols:.5f} {2.0 / rows:.5f}"
                 for u in enemies + allies]
        (self.lbl_dir / f"{name}.txt").write_text("\n".join(lines))
        self.count += 1
        if self.count % 50 == 0:
            log.info("Dataset: %d imágenes en %s", self.count, self.img_dir)


def prepare_dataset(path: str, base_dir: Path) -> Path:
    """Acepta el ZIP exportado de Roboflow (o su data.yaml) y deja las rutas listas para entrenar."""
    import yaml
    import zipfile

    p = Path(path)
    if p.suffix.lower() == ".zip":
        dest = base_dir / "dataset_rf" / p.stem
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as z:
            z.extractall(dest)
        found = sorted(dest.rglob("data.yaml"))
        if not found:
            raise FileNotFoundError(f"El ZIP {p.name} no contiene data.yaml (¿exportaste en formato YOLO?)")
        p = found[0]
    root = p.parent
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    data["path"] = str(root.resolve())
    for key, names in (("train", ["train"]), ("val", ["valid", "val"]), ("test", ["test"])):
        for n in names:
            if (root / n / "images").is_dir():
                data[key] = f"{n}/images"
                break
        else:
            if key == "test":
                data.pop("test", None)
            elif not (root / str(data.get(key, ""))).exists():
                raise FileNotFoundError(f"No encuentro la carpeta de imágenes '{key}' en {root}")
    if "val" not in data:
        data["val"] = data["train"]
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    log.info("Dataset listo: %s (%s clases)", p, len(data.get("names", [])))
    return p


def train_yolo(cfg: dict, data_yaml: str, epochs: int) -> None:
    from ultralytics import YOLO

    y = cfg["yolo"]
    base_dir = Path(__file__).parent
    data_yaml = str(prepare_dataset(data_yaml, base_dir))
    model = YOLO(y["base_weights"])
    results = model.train(data=data_yaml, epochs=epochs, imgsz=y["imgsz"], device=y["device"],
                          project=str(base_dir / "runs"), name="cr_yolo", exist_ok=True,
                          flipud=0.0, fliplr=0.5, hsv_h=0.0, mosaic=1.0)
    dst = base_dir / y["model"]
    shutil.copy(Path(results.save_dir) / "weights" / "best.pt", dst)
    log.info("Modelo YOLO copiado a %s", dst)

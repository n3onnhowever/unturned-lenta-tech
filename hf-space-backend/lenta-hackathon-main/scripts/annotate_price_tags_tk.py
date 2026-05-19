"""
Простая разметка bbox ценников под YOLO (CSV для build_yolo_dataset_from_image_bbox_csv.py).

Запуск (из корня репозитория, нужен дисплей — не SSH без X):
  pip install Pillow opencv-python-headless
  python scripts/annotate_price_tags_tk.py --images-dir frames/materials_data_43_15_43_15 --csv annotations/my_price_tags.csv

Управление:
  ЛКМ + тянуть — новый прямоугольник (в пикселях исходного кадра).
  «Сохранить кадр» — дописать в CSV все прямоугольники этого кадра и перейти к следующему.
  «Пропуск» — следующий кадр без записи.
  «Отменить последний» — убрать последний нарисованный прямоугольник на текущем кадре.

CSV: image_path,x_min,y_min,x_max,y_max (пути относительно текущей рабочей папки, если получается).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:
    print("Нужен tkinter (обычно есть в установке Python на Windows).", file=sys.stderr)
    raise


def rel_path_or_abs(p: Path, cwd: Path) -> str:
    try:
        return str(p.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(p.resolve())


class AnnotatorApp:
    def __init__(
        self,
        images: list[Path],
        csv_out: Path,
        cwd: Path,
        max_display: int,
    ) -> None:
        self.images = images
        self.idx = 0
        self.csv_out = csv_out
        self.cwd = cwd
        self.max_display = max_display

        self.orig_bgr = None
        self.oh = self.ow = 0
        self.disp_w = self.disp_h = 0
        self.scale = 1.0
        self.photo = None
        self.boxes: list[tuple[int, int, int, int]] = []
        self.drag_start: tuple[int, int] | None = None

        self.root = tk.Tk()
        self.root.title("Разметка ценников (bbox)")
        self.root.minsize(640, 480)

        self.var_state = tk.StringVar()
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.var_state).pack(side=tk.LEFT)

        btn = ttk.Frame(self.root, padding=6)
        btn.pack(fill=tk.X)
        ttk.Button(btn, text="Сохранить кадр и далее", command=self.save_and_next).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn, text="Пропуск", command=self.skip).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn, text="Отменить последний bbox", command=self.undo_last).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn, text="Выход", command=self.root.quit).pack(side=tk.RIGHT, padx=4)

        self.canvas = tk.Canvas(self.root, bg="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        self.csv_out.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

        self.load_current()

    def _update_state(self) -> None:
        if not self.images:
            self.var_state.set("Нет изображений")
            return
        p = self.images[self.idx]
        self.var_state.set(f"{self.idx + 1}/{len(self.images)}  {p.name}  bbox на кадре: {len(self.boxes)}")

    def _ensure_csv_header(self) -> None:
        if self.csv_out.exists() and self.csv_out.stat().st_size > 0:
            return
        with self.csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "x_min", "y_min", "x_max", "y_max"])

    def _to_orig(self, xd: int, yd: int) -> tuple[int, int]:
        x = int(round(xd / self.scale))
        y = int(round(yd / self.scale))
        return max(0, min(self.ow - 1, x)), max(0, min(self.oh - 1, y))

    def load_current(self) -> None:
        self.canvas.delete("all")
        self.boxes.clear()
        self.drag_start = None

        if self.idx >= len(self.images):
            messagebox.showinfo("Готово", "Все кадры просмотрены.")
            self.root.quit()
            return

        path = self.images[self.idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            messagebox.showwarning("Ошибка", f"Не удалось открыть:\n{path}")
            self.idx += 1
            self.load_current()
            return

        self.orig_bgr = bgr
        self.oh, self.ow = bgr.shape[:2]
        m = max(self.ow, self.oh)
        if m > self.max_display:
            self.scale = self.max_display / m
        else:
            self.scale = 1.0
        self.disp_w = max(1, int(round(self.ow * self.scale)))
        self.disp_h = max(1, int(round(self.oh * self.scale)))
        small = cv2.resize(bgr, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.config(width=self.disp_w, height=self.disp_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self._redraw_boxes()
        self._update_state()

    def _redraw_boxes(self) -> None:
        self.canvas.delete("box")
        for x0, y0, x1, y1 in self.boxes:
            xd0, yd0 = int(x0 * self.scale), int(y0 * self.scale)
            xd1, yd1 = int(x1 * self.scale), int(y1 * self.scale)
            self.canvas.create_rectangle(
                xd0, yd0, xd1, yd1, outline="#00ff00", width=2, tags="box"
            )

    def on_down(self, e: tk.Event) -> None:
        self.drag_start = (e.x, e.y)

    def on_move(self, e: tk.Event) -> None:
        if self.drag_start is None:
            return
        self.canvas.delete("temp")
        x0, y0 = self.drag_start
        self.canvas.create_rectangle(
            x0, y0, e.x, e.y, outline="#ffff00", width=2, tags="temp"
        )

    def on_up(self, e: tk.Event) -> None:
        if self.drag_start is None:
            return
        x0d, y0d = self.drag_start
        x1d, y1d = e.x, e.y
        self.canvas.delete("temp")
        self.drag_start = None
        if abs(x1d - x0d) < 4 or abs(y1d - y0d) < 4:
            return
        ox0, oy0 = self._to_orig(min(x0d, x1d), min(y0d, y1d))
        ox1, oy1 = self._to_orig(max(x0d, x1d), max(y0d, y1d))
        if ox1 <= ox0 + 2 or oy1 <= oy0 + 2:
            return
        self.boxes.append((ox0, oy0, ox1, oy1))
        self._redraw_boxes()
        self._update_state()

    def undo_last(self) -> None:
        if self.boxes:
            self.boxes.pop()
            self._redraw_boxes()
            self._update_state()

    def save_and_next(self) -> None:
        path = self.images[self.idx]
        rel = rel_path_or_abs(path, self.cwd)
        with self.csv_out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for x0, y0, x1, y1 in self.boxes:
                w.writerow([rel, x0, y0, x1, y1])
        self.idx += 1
        self.load_current()

    def skip(self) -> None:
        self.idx += 1
        self.load_current()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Разметка bbox ценников (tkinter).")
    ap.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Папка с .jpg/.png (например frames/materials_data_43_15_43_15)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("annotations/price_tags_manual.csv"),
        help="Куда дописывать CSV",
    )
    ap.add_argument(
        "--max-display",
        type=int,
        default=1400,
        help="Макс. сторона превью (картинка масштабируется)",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Ограничить число кадров (с начала списка)",
    )
    args = ap.parse_args()

    cwd = Path.cwd()
    d = args.images_dir
    if not d.is_dir():
        print(f"Нет папки: {d}", file=sys.stderr)
        return 1
    imgs = sorted(
        list(d.glob("*.jpg"))
        + list(d.glob("*.jpeg"))
        + list(d.glob("*.png"))
        + list(d.glob("*.JPG"))
    )
    if args.max_images is not None:
        imgs = imgs[: max(0, args.max_images)]
    if not imgs:
        print("В папке нет изображений.", file=sys.stderr)
        return 1

    app = AnnotatorApp(imgs, args.csv, cwd, args.max_display)
    app.run()
    print(f"CSV: {args.csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Разметка полей ценника на кропах (из manual_label_sample.csv).

  python scripts/label_unreadable_crops_tk.py \\
    --csv output/unreadable_splits/manual_label_sample.csv

Управление:
  Заполните название и цены как на ценнике.
  «Сохранить» — записать строку и перейти дальше.
  «Пропуск» — следующий кадр без изменений.
  Стрелки Left/Right или кнопки — навигация.
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
    print("Нужен tkinter.", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_ocr_fields import parse_ocr_text, price_fields_to_display_line


def load_rows(csv_path: Path) -> tuple[list[dict], list[str]]:
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in (
        "label_product_name",
        "label_price_default",
        "label_price_card",
        "label_price_line",
        "label_notes",
    ):
        if col not in fieldnames:
            fieldnames.append(col)
    return rows, fieldnames


def save_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class LabelApp:
    def __init__(self, rows: list[dict], fieldnames: list[str], csv_path: Path) -> None:
        self.rows = rows
        self.fieldnames = fieldnames
        self.csv_path = csv_path
        self.idx = 0
        self.photo = None

        self.root = tk.Tk()
        self.root.title("Разметка ценников (поля)")
        self.root.minsize(900, 700)

        self.var_state = tk.StringVar()
        ttk.Label(self.root, textvariable=self.var_state, padding=6).pack(fill=tk.X)

        body = ttk.Frame(self.root, padding=6)
        body.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(body, bg="#222", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        form = ttk.Frame(body, padding=8, width=320)
        form.pack(side=tk.RIGHT, fill=tk.Y)
        form.pack_propagate(False)

        ttk.Label(form, text="OCR (подсказка):").pack(anchor=tk.W)
        self.txt_raw = tk.Text(form, height=4, width=36, wrap=tk.WORD)
        self.txt_raw.pack(fill=tk.X, pady=4)

        ttk.Label(form, text="Название (как на ценнике):").pack(anchor=tk.W, pady=(8, 0))
        self.ent_name = ttk.Entry(form, width=40)
        self.ent_name.pack(fill=tk.X)

        ttk.Label(form, text="Цена обычная (например 3.68):").pack(anchor=tk.W, pady=(8, 0))
        self.ent_pd = ttk.Entry(form, width=40)
        self.ent_pd.pack(fill=tk.X)

        ttk.Label(form, text="Цена по карте (например 3.05):").pack(anchor=tk.W, pady=(8, 0))
        self.ent_pc = ttk.Entry(form, width=40)
        self.ent_pc.pack(fill=tk.X)

        ttk.Label(form, text="Строка цен на ценнике (368 99 …):").pack(anchor=tk.W, pady=(8, 0))
        self.ent_pline = ttk.Entry(form, width=40)
        self.ent_pline.pack(fill=tk.X)

        ttk.Label(form, text="Заметки:").pack(anchor=tk.W, pady=(8, 0))
        self.ent_notes = ttk.Entry(form, width=40)
        self.ent_notes.pack(fill=tk.X)

        ttk.Button(form, text="Подставить из OCR", command=self.fill_from_ocr).pack(
            fill=tk.X, pady=8
        )

        nav = ttk.Frame(self.root, padding=6)
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="← Назад", command=self.prev_item).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="Сохранить →", command=self.save_and_next).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="Пропуск", command=self.skip).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="Выход", command=self.on_exit).pack(side=tk.RIGHT, padx=4)

        self.root.bind("<Left>", lambda e: self.prev_item())
        self.root.bind("<Right>", lambda e: self.save_and_next())
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.show_item()

    def row(self) -> dict:
        return self.rows[self.idx]

    def show_item(self) -> None:
        r = self.row()
        labeled = sum(1 for x in self.rows if (x.get("label_product_name") or "").strip())
        self.var_state.set(
            f"{self.idx + 1}/{len(self.rows)} | размечено: {labeled} | {r.get('crop_filename', '')}"
        )
        self.txt_raw.delete("1.0", tk.END)
        self.txt_raw.insert("1.0", r.get("raw_text") or "(пусто)")

        self.ent_name.delete(0, tk.END)
        self.ent_name.insert(0, r.get("label_product_name") or "")

        self.ent_pd.delete(0, tk.END)
        self.ent_pd.insert(0, r.get("label_price_default") or "")

        self.ent_pc.delete(0, tk.END)
        self.ent_pc.insert(0, r.get("label_price_card") or "")

        self.ent_pline.delete(0, tk.END)
        self.ent_pline.insert(0, r.get("label_price_line") or "")

        self.ent_notes.delete(0, tk.END)
        self.ent_notes.insert(0, r.get("label_notes") or "")

        path = Path(r.get("crop_path") or "")
        if not path.is_file():
            path = Path(r.get("crop_filename") or "")
        bgr = cv2.imread(str(path)) if path.is_file() else None
        self.canvas.delete("all")
        if bgr is None:
            self.canvas.create_text(200, 200, text="Нет изображения", fill="white")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_w, max_h = 620, 820
        scale = min(max_w / w, max_h / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        img = Image.fromarray(rgb).resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def fill_from_ocr(self) -> None:
        raw = self.row().get("raw_text") or ""
        p = parse_ocr_text(raw)
        if p.get("product_name") and not self.ent_name.get().strip():
            self.ent_name.insert(0, p["product_name"])
        if p.get("price_default") and not self.ent_pd.get().strip():
            self.ent_pd.insert(0, p["price_default"])
        if p.get("price_card") and not self.ent_pc.get().strip():
            self.ent_pc.insert(0, p["price_card"])
        if not self.ent_pline.get().strip():
            line = price_fields_to_display_line(p.get("price_default"), p.get("price_card"), raw)
            if line:
                self.ent_pline.insert(0, line)

    def write_fields_to_row(self) -> None:
        r = self.row()
        r["label_product_name"] = self.ent_name.get().strip()
        r["label_price_default"] = self.ent_pd.get().strip()
        r["label_price_card"] = self.ent_pc.get().strip()
        pline = self.ent_pline.get().strip()
        if not pline:
            pline = price_fields_to_display_line(r["label_price_default"], r["label_price_card"], r.get("raw_text", ""))
        r["label_price_line"] = pline
        r["label_notes"] = self.ent_notes.get().strip()

    def save_and_next(self) -> None:
        self.write_fields_to_row()
        save_rows(self.csv_path, self.rows, self.fieldnames)
        if self.idx < len(self.rows) - 1:
            self.idx += 1
            self.show_item()
        else:
            messagebox.showinfo("Готово", f"Сохранено в {self.csv_path}")
            self.root.destroy()

    def skip(self) -> None:
        if self.idx < len(self.rows) - 1:
            self.idx += 1
            self.show_item()

    def prev_item(self) -> None:
        if self.idx > 0:
            self.write_fields_to_row()
            save_rows(self.csv_path, self.rows, self.fieldnames)
            self.idx -= 1
            self.show_item()

    def on_exit(self) -> None:
        self.write_fields_to_row()
        save_rows(self.csv_path, self.rows, self.fieldnames)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("output/unreadable_splits/manual_label_sample.csv"),
    )
    args = ap.parse_args()
    if not args.csv.is_file():
        print(f"Нет файла: {args.csv}")
        return 1
    rows, fieldnames = load_rows(args.csv)
    if not rows:
        print("CSV пуст")
        return 1
    LabelApp(rows, fieldnames, args.csv).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

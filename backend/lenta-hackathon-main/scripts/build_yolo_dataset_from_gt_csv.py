"""
Сбор датасета YOLO (детекция ценника) из эталонных CSV в materials/data.

В CSV заданы filename, frame_timestamp (мс), x_min..y_max — это разметка «где ценник»
на кадре. Кадры вырезаются из соответствующих .mp4. Класс один: price_tag.

Разрешение видео в папках может отличаться от имени в CSV (например 2.mp4 vs 25_12-20.mp4):
используется поиск .mp4 в подпапке.

Пример:
  python scripts/build_yolo_dataset_from_gt_csv.py --materials materials/data --out dataset/price_tags

Честная проверка по новому магазину/видео:
  python scripts/build_yolo_dataset_from_gt_csv.py --materials materials/data --out dataset/price_tags_holdout_43_15 --holdout-sources 43_15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2


def parse_float(s: str) -> float | None:
    s = (s or "").strip().strip('"').replace(" ", "")
    if not s or s.lower() in ("нет", "nan", "-"):
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def resolve_video(materials: Path, filename_cell: str) -> Path | None:
    raw = (filename_cell or "").strip().replace("\\", "/")
    if not raw:
        return None
    parts = [p for p in raw.split("/") if p]
    base = parts[-1]
    sub = parts[-2] if len(parts) >= 2 else None

    candidates: list[Path] = []
    if sub:
        candidates.append(materials / sub / base)
    candidates.append(materials / base)
    if sub:
        d = materials / sub
        if d.is_dir():
            for p in sorted(d.glob("*.mp4")):
                candidates.append(p)
            for p in sorted(d.glob("*.MP4")):
                candidates.append(p)
    else:
        stem_dir = materials / Path(base).stem
        if stem_dir.is_dir():
            for p in sorted(stem_dir.glob("*.mp4")):
                candidates.append(p)

    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.is_file():
            return c
    return None


def frame_index_from_ms(fps: float, ts_ms: float, nfc: int) -> int:
    if fps <= 1e-3:
        fps = 25.0
    n = int(round(ts_ms / 1000.0 * fps))
    if nfc > 0:
        n = max(0, min(nfc - 1, n))
    else:
        n = max(0, n)
    return n


def get_video_meta(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 25.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    nfc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 1e-3:
        fps = 25.0
    return fps, nfc


def read_frame(path: Path, frame_idx: int) -> object | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok and frame is not None else None


def read_video_frames(path: Path, frame_indices: set[int]) -> dict[int, object]:
    if not frame_indices:
        return {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    out: dict[int, object] = {}
    for frame_idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            out[frame_idx] = frame
    cap.release()
    return out


def frame_dir_for_video(frames_root: Path, video: Path) -> Path:
    return frames_root / f"materials_data_{video.parent.name}_{video.stem}"


def list_extracted_frame_indices(frames_root: Path | None, video: Path) -> list[int]:
    if frames_root is None:
        return []
    frame_dir = frame_dir_for_video(frames_root, video)
    if not frame_dir.is_dir():
        return []
    out: list[int] = []
    for p in frame_dir.glob("*.jpg"):
        try:
            out.append(int(p.stem))
        except ValueError:
            continue
    return sorted(out)


def read_extracted_frames(frames_root: Path | None, video: Path, frame_indices: set[int]) -> dict[int, object]:
    if frames_root is None or not frame_indices:
        return {}
    frame_dir = frame_dir_for_video(frames_root, video)
    if not frame_dir.is_dir():
        return {}
    out: dict[int, object] = {}
    for frame_idx in sorted(frame_indices):
        img_path = frame_dir / f"{frame_idx:06d}.jpg"
        frame = cv2.imread(str(img_path))
        if frame is not None:
            out[frame_idx] = frame
    return out


def read_frames(frames_root: Path | None, video: Path, frame_indices: set[int]) -> dict[int, object]:
    extracted = read_extracted_frames(frames_root, video, frame_indices)
    missing = set(frame_indices) - set(extracted)
    if missing:
        extracted.update(read_video_frames(video, missing))
    return extracted


def mean_absdiff(a: object, b: object) -> float:
    return float(cv2.absdiff(a, b).mean())


def xyxy_to_yolo_line(x1: float, y1: float, x2: float, y2: float, iw: int, ih: int) -> str | None:
    w = x2 - x1
    h = y2 - y1
    if w < 2 or h < 2:
        return None
    xc = (x1 + x2) / 2.0 / iw
    yc = (y1 + y2) / 2.0 / ih
    nw = w / iw
    nh = h / ih
    if nw <= 0 or nh <= 0 or xc <= 0 or yc <= 0 or xc >= 1 or yc >= 1:
        return None
    xc = min(1.0, max(0.0, xc))
    yc = min(1.0, max(0.0, yc))
    nw = min(1.0, max(1e-6, nw))
    nh = min(1.0, max(1e-6, nh))
    return f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"


def read_gt_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def source_from_video(path: Path) -> str:
    return path.parent.name


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO-датасет из GT CSV (bbox + timestamp).")
    ap.add_argument("--materials", type=Path, default=Path("materials/data"))
    ap.add_argument(
        "--frames-root",
        type=Path,
        default=Path("frames"),
        help="Папка с заранее извлечёнными кадрами; значительно быстрее mp4 seek.",
    )
    ap.add_argument("--out", type=Path, default=Path("dataset/price_tags"))
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Доля кадров в val (по ключу видео+кадр)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--holdout-sources",
        nargs="*",
        default=[],
        help="Источники, которые нужно полностью убрать из train и положить в val, например: 43_15 49_5",
    )
    ap.add_argument(
        "--train-sources",
        nargs="*",
        default=[],
        help="Если задано, брать в train только эти источники; остальные источники попадут в val.",
    )
    ap.add_argument(
        "--adjacent-frames",
        type=int,
        default=0,
        help="Добавить соседние кадры вокруг GT timestamp только для train-eligible источников.",
    )
    ap.add_argument(
        "--adjacent-max-mean-diff",
        type=float,
        default=8.0,
        help="Максимальный средний absdiff с GT-кадром для соседнего кадра; <=0 отключает guard.",
    )
    ap.add_argument(
        "--negative-frames-per-source",
        type=int,
        default=0,
        help="Добавить до N background кадров на train-source с пустыми label-файлами.",
    )
    ap.add_argument(
        "--negative-frame-stride",
        type=int,
        default=25,
        help="Шаг отбора кандидатов для negative frames из заранее извлечённых кадров.",
    )
    ap.add_argument(
        "--negative-exclude-radius",
        type=int,
        default=10,
        help="Не брать negative frame, если рядом есть GT frame в пределах этого радиуса.",
    )
    args = ap.parse_args()

    if not args.materials.is_dir():
        print(f"Нет каталога: {args.materials}", file=sys.stderr)
        return 1

    csv_files = sorted(args.materials.rglob("*.csv"))
    if not csv_files:
        print("CSV не найдены.", file=sys.stderr)
        return 1

    holdout_sources = set(args.holdout_sources)
    train_sources = set(args.train_sources)
    if holdout_sources & train_sources:
        print(
            f"Источник не может быть одновременно train и holdout: {sorted(holdout_sources & train_sources)}",
            file=sys.stderr,
        )
        return 1
    if args.adjacent_frames < 0:
        print("--adjacent-frames должен быть >= 0", file=sys.stderr)
        return 1
    if args.negative_frames_per_source < 0:
        print("--negative-frames-per-source должен быть >= 0", file=sys.stderr)
        return 1
    if args.negative_frame_stride <= 0:
        print("--negative-frame-stride должен быть > 0", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)

    # (resolved_video, frame_idx) -> list of xyxy pixel boxes
    base_groups: dict[tuple[str, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    videos_by_source: dict[str, set[str]] = defaultdict(set)
    base_group_count = 0
    adjacent_added = 0
    adjacent_skipped_motion = 0
    adjacent_skipped_read = 0
    negative_keys: set[tuple[str, int]] = set()
    meta_cache: dict[str, tuple[float, int]] = {}

    def meta(v: Path) -> tuple[float, int]:
        k = str(v.resolve())
        if k not in meta_cache:
            meta_cache[k] = get_video_meta(v)
        return meta_cache[k]

    for csv_path in csv_files:
        for row in read_gt_rows(csv_path):
            fn = row.get("filename") or ""
            vid = resolve_video(args.materials, fn)
            if vid is None:
                continue
            ts = parse_float(row.get("frame_timestamp", "") or "")
            if ts is None:
                continue
            x1 = parse_float(row.get("x_min", ""))
            y1 = parse_float(row.get("y_min", ""))
            x2 = parse_float(row.get("x_max", ""))
            y2 = parse_float(row.get("y_max", ""))
            if None in (x1, y1, x2, y2):
                continue
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue

            fps, nfc = meta(vid)
            fi = frame_index_from_ms(fps, float(ts), nfc)
            vid_resolved = str(vid.resolve())
            videos_by_source[source_from_video(vid)].add(vid_resolved)
            key = (vid_resolved, fi)
            base_groups[key].append((x1, y1, x2, y2))
            base_group_count += 1

    if not base_groups:
        print("Нет валидных групп (видео+кадр+bbox). Проверьте пути к .mp4 и CSV.", file=sys.stderr)
        return 1

    groups: dict[tuple[str, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    for key, boxes in base_groups.items():
        groups[key].extend(boxes)

    if args.adjacent_frames > 0:
        wanted_by_video: dict[str, set[int]] = defaultdict(set)
        for vid_path, fi in base_groups:
            vid = Path(vid_path)
            src = source_from_video(vid)
            train_eligible = src not in holdout_sources and (not train_sources or src in train_sources)
            if not train_eligible:
                continue
            _fps, nfc = meta(vid)
            wanted_by_video[vid_path].add(fi)
            for off in range(-args.adjacent_frames, args.adjacent_frames + 1):
                if off == 0:
                    continue
                adj_fi = fi + off
                if adj_fi < 0 or (nfc > 0 and adj_fi >= nfc):
                    adjacent_skipped_read += 1
                    continue
                wanted_by_video[vid_path].add(adj_fi)

        frames_by_video = {
            vid_path: read_frames(args.frames_root, Path(vid_path), indices)
            for vid_path, indices in wanted_by_video.items()
        }
        for (vid_path, fi), boxes in base_groups.items():
            if vid_path not in frames_by_video:
                continue
            frames = frames_by_video[vid_path]
            base_frame = frames.get(fi)
            if base_frame is None:
                adjacent_skipped_read += args.adjacent_frames * 2
                continue
            vid = Path(vid_path)
            _fps, nfc = meta(vid)
            for off in range(-args.adjacent_frames, args.adjacent_frames + 1):
                if off == 0:
                    continue
                adj_fi = fi + off
                if adj_fi < 0 or (nfc > 0 and adj_fi >= nfc):
                    continue
                adj_frame = frames.get(adj_fi)
                if adj_frame is None:
                    adjacent_skipped_read += 1
                    continue
                if args.adjacent_max_mean_diff > 0 and mean_absdiff(base_frame, adj_frame) > args.adjacent_max_mean_diff:
                    adjacent_skipped_motion += 1
                    continue
                groups[(vid_path, adj_fi)].extend(boxes)
                adjacent_added += len(boxes)

    if args.negative_frames_per_source > 0:
        positive_by_video: dict[str, set[int]] = defaultdict(set)
        for vid_path, fi in groups:
            positive_by_video[vid_path].add(fi)

        for src, video_paths in sorted(videos_by_source.items()):
            train_eligible = src not in holdout_sources and (not train_sources or src in train_sources)
            if not train_eligible:
                continue

            candidates: list[tuple[str, int]] = []
            for vid_path in sorted(video_paths):
                vid = Path(vid_path)
                extracted_indices = list_extracted_frame_indices(args.frames_root, vid)
                if not extracted_indices:
                    continue
                positive = positive_by_video.get(vid_path, set())
                forbidden: set[int] = set()
                for fi in positive:
                    forbidden.update(
                        range(
                            max(0, fi - args.negative_exclude_radius),
                            fi + args.negative_exclude_radius + 1,
                        )
                    )
                for fi in extracted_indices[:: args.negative_frame_stride]:
                    if fi in forbidden:
                        continue
                    candidates.append((vid_path, fi))

            rng.shuffle(candidates)
            for key in candidates[: args.negative_frames_per_source]:
                if key in groups:
                    continue
                groups[key] = []
                negative_keys.add(key)

    keys = list(groups.keys())
    rng.shuffle(keys)

    random_val_count = max(1, int(len(keys) * args.val_ratio)) if len(keys) > 1 else 0
    if args.val_ratio <= 0:
        random_val_count = 0
    random_val_set = set(keys[:random_val_count]) if random_val_count else set()

    def is_val_key(key: tuple[str, int]) -> bool:
        src = source_from_video(Path(key[0]))
        if src in holdout_sources:
            return True
        if train_sources and src not in train_sources:
            return True
        return key in random_val_set

    img_train = args.out / "images" / "train"
    lbl_train = args.out / "labels" / "train"
    img_val = args.out / "images" / "val"
    lbl_val = args.out / "labels" / "val"
    for d in (img_train, lbl_train, img_val, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    written_train = written_val = 0
    skipped = 0
    needed_by_video: dict[str, set[int]] = defaultdict(set)
    for vid_path, fi in keys:
        needed_by_video[vid_path].add(fi)
    output_frames_by_video = {
        vid_path: read_frames(args.frames_root, Path(vid_path), indices)
        for vid_path, indices in needed_by_video.items()
    }

    for key in keys:
        vid_path, fi = key
        boxes = groups[key]
        vid = Path(vid_path)
        frame = output_frames_by_video.get(vid_path, {}).get(fi)
        if frame is None:
            skipped += 1
            continue

        ih, iw = frame.shape[:2]
        lines: list[str] = []
        for x1, y1, x2, y2 in boxes:
            x1c = max(0.0, min(float(iw - 1), x1))
            y1c = max(0.0, min(float(ih - 1), y1))
            x2c = max(0.0, min(float(iw), x2))
            y2c = max(0.0, min(float(ih), y2))
            ln = xyxy_to_yolo_line(x1c, y1c, x2c, y2c, iw, ih)
            if ln:
                lines.append(ln)
        is_negative = key in negative_keys
        if not lines and not is_negative:
            skipped += 1
            continue

        hshort = hashlib.sha1(f"{vid_path}:{fi}".encode("utf-8")).hexdigest()[:10]
        stem = f"{vid.parent.name}_{fi:06d}_{hshort}"
        is_val = is_val_key(key)
        img_dir = img_val if is_val else img_train
        lbl_dir = lbl_val if is_val else lbl_train

        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"
        cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        with lbl_path.open("w", encoding="utf-8") as tf:
            tf.writelines(lines)

        if is_val:
            written_val += 1
        else:
            written_train += 1

    yaml_path = args.out / "data.yaml"
    root_abs = args.out.resolve()
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root_abs.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: price_tag",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        f"Кадров train={written_train}, val={written_val}, пропусков={skipped}, "
        f"уникальных (видео,кадр)={len(keys)}, base_rows={base_group_count}, "
        f"adjacent_added={adjacent_added}, adjacent_skip_motion={adjacent_skipped_motion}, "
        f"adjacent_skip_read={adjacent_skipped_read}, negatives={len(negative_keys)}, yaml={yaml_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Сравнение двух детекторов ценников на одинаковых кадрах из ваших .mp4:

1) merged YOLOv8n — полный кадр (как predict_materials_videos.py).
2) YOLO11n коллеги — тайлинг 768 / stride 384, затем NMS (как predict_frame в
   NIK/lenta_tech_ml/qr_research/price_tag_tiled_detector_eval.py).

По умолчанию без цветового warm-filter и без QR (как их eval без --require-qr).

Пример:
  python scripts/compare_video_detectors_materials.py --vid-stride 4 --conf-merged 0.5 --conf-nik 0.15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


def make_tiles(width: int, height: int, tile_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    tile_width = min(tile_size, width)
    tile_height = min(tile_size, height)
    xs = list(range(0, max(1, width - tile_width + 1), stride))
    ys = list(range(0, max(1, height - tile_height + 1), stride))
    if not xs or xs[-1] != width - tile_width:
        xs.append(width - tile_width)
    if not ys or ys[-1] != height - tile_height:
        ys.append(height - tile_height)
    return [(x, y, x + tile_width, y + tile_height) for y in ys for x in xs]


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def nms_predictions(preds: list[dict[str, Any]], iou_thresh: float) -> list[dict[str, Any]]:
    ordered = sorted(preds, key=lambda item: float(item["conf"]), reverse=True)
    kept: list[dict[str, Any]] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [
            pred for pred in ordered if iou_xyxy(tuple(current["bbox"]), tuple(pred["bbox"])) < iou_thresh
        ]
    return kept


def predict_nik_tiled(
    model: YOLO,
    frame,
    *,
    tile_size: int,
    stride: int,
    imgsz: int,
    conf: float,
    model_iou: float,
    nms_iou: float,
    device: str | None,
) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    preds: list[dict[str, Any]] = []
    dev_kw = {"device": device} if device else {}
    for tile_idx, (left, top, right, bottom) in enumerate(make_tiles(width, height, tile_size, stride)):
        tile = frame[top:bottom, left:right]
        result = model.predict(
            tile,
            imgsz=imgsz,
            conf=conf,
            iou=model_iou,
            verbose=False,
            **dev_kw,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for box, score in zip(xyxy, scores):
            x1, y1, x2, y2 = [float(v) for v in box]
            preds.append(
                {
                    "bbox": (x1 + left, y1 + top, x2 + left, y2 + top),
                    "conf": float(score),
                    "tile_idx": tile_idx,
                }
            )
    return nms_predictions(preds, iou_thresh=nms_iou)


def conf_stats(confs: list[float]) -> tuple[float, float, float]:
    if not confs:
        return 0.0, 0.0, 0.0
    return float(sum(confs) / len(confs)), float(max(confs)), float(min(confs))


def main() -> int:
    ap = argparse.ArgumentParser(description="Сравнение merged full-frame vs NIK tiled на тех же кадрах.")
    ap.add_argument("--materials", type=Path, default=Path("materials/data"), help="Корень с видео")
    ap.add_argument(
        "--weights-merged",
        type=Path,
        default=Path("runs/detect/runs/detect/price_tag_merged/weights/best.pt"),
    )
    ap.add_argument(
        "--weights-nik",
        type=Path,
        default=Path(
            "NIK/lenta_tech_ml/experiments/price_tag_detector_runs/"
            "detect_yolo11n_tiles768_fulltags_oldtrain_newval_e12_i640/weights/best.pt"
        ),
    )
    ap.add_argument("--vid-stride", type=int, default=4, help="Как в predict_materials_videos: каждый N-й кадр")
    ap.add_argument("--imgsz-merged", type=int, default=1280)
    ap.add_argument("--conf-merged", type=float, default=0.5)
    ap.add_argument("--imgsz-nik", type=int, default=640)
    ap.add_argument("--conf-nik", type=float, default=0.15)
    ap.add_argument("--tile-size", type=int, default=768)
    ap.add_argument("--tile-stride", type=int, default=384)
    ap.add_argument("--nik-model-iou", type=float, default=0.6)
    ap.add_argument("--nik-nms-iou", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="", help="Пусто = авто (CUDA если есть)")
    ap.add_argument("--max-frames-per-video", type=int, default=0, help="0 = без лимита")
    ap.add_argument("--max-videos", type=int, default=0, help="0 = все найденные .mp4")
    ap.add_argument("--progress-every", type=int, default=25, help="Печать прогресса каждые N обработанных кадров (0 = только по видео)")
    ap.add_argument("--out-csv", type=Path, default=Path("runs/compare_detectors_materials_frames.csv"))
    ap.add_argument("--out-summary", type=Path, default=Path("runs/compare_detectors_materials_summary.json"))
    args = ap.parse_args()

    device = args.device if args.device else None

    if not args.weights_merged.is_file():
        print(f"Нет весов merged: {args.weights_merged}", file=sys.stderr)
        return 1
    if not args.weights_nik.is_file():
        print(f"Нет весов NIK: {args.weights_nik}", file=sys.stderr)
        return 1
    if not args.materials.is_dir():
        print(f"Нет папки: {args.materials}", file=sys.stderr)
        return 1

    vids = sorted(args.materials.rglob("*.mp4")) + sorted(args.materials.rglob("*.MP4"))
    vids = list(dict.fromkeys(vids))
    if args.max_videos > 0:
        vids = vids[: args.max_videos]
    if not vids:
        print("Видео .mp4 не найдены.", file=sys.stderr)
        return 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    mat_root = args.materials.resolve()
    print(
        f"Видео: {len(vids)}, vid_stride={args.vid_stride}, merged imgsz/conf={args.imgsz_merged}/{args.conf_merged}, "
        f"NIK tiled {args.tile_size}/{args.tile_stride} imgsz/conf={args.imgsz_nik}/{args.conf_nik}",
        flush=True,
    )

    model_m = YOLO(str(args.weights_merged))
    model_n = YOLO(str(args.weights_nik))
    dev_kw = {"device": device} if device else {}

    summary_videos: list[dict[str, Any]] = []
    row_count = 0

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "video_rel",
                "frame_idx",
                "merged_n",
                "merged_conf_mean",
                "merged_conf_max",
                "merged_conf_min",
                "nik_n",
                "nik_conf_mean",
                "nik_conf_max",
                "nik_conf_min",
            ]
        )

        def avg(xs: list[int]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        for vid in vids:
            try:
                vid_rel = str(vid.resolve().relative_to(mat_root))
            except ValueError:
                vid_rel = str(vid)
            print(f"[видео] {vid_rel}", flush=True)

            cap = cv2.VideoCapture(str(vid))
            if not cap.isOpened():
                print(f"Не открыть видео: {vid}", file=sys.stderr)
                continue

            merged_ns: list[int] = []
            nik_ns: list[int] = []
            merged_confs_all: list[float] = []
            nik_confs_all: list[float] = []
            frames_used = 0

            fi = -1
            while True:
                ok, frame = cap.read()
                fi += 1
                if not ok or frame is None:
                    break
                if args.vid_stride > 1 and (fi % args.vid_stride) != 0:
                    continue

                rm = model_m.predict(frame, imgsz=args.imgsz_merged, conf=args.conf_merged, verbose=False, **dev_kw)[0]
                merged_boxes = rm.boxes
                if merged_boxes is None or len(merged_boxes) == 0:
                    mc: list[float] = []
                else:
                    mc = [float(c) for c in merged_boxes.conf.cpu().numpy().tolist()]

                preds_n = predict_nik_tiled(
                    model_n,
                    frame,
                    tile_size=args.tile_size,
                    stride=args.tile_stride,
                    imgsz=args.imgsz_nik,
                    conf=args.conf_nik,
                    model_iou=args.nik_model_iou,
                    nms_iou=args.nik_nms_iou,
                    device=device,
                )
                nc = [p["conf"] for p in preds_n]

                mm, mx, mn = conf_stats(mc)
                nm, nx, nn = conf_stats(nc)
                w.writerow(
                    [
                        vid_rel,
                        fi,
                        len(mc),
                        f"{mm:.6f}",
                        f"{mx:.6f}",
                        f"{mn:.6f}",
                        len(nc),
                        f"{nm:.6f}",
                        f"{nx:.6f}",
                        f"{nn:.6f}",
                    ]
                )
                row_count += 1
                merged_ns.append(len(mc))
                nik_ns.append(len(nc))
                merged_confs_all.extend(mc)
                nik_confs_all.extend(nc)
                frames_used += 1

                if args.progress_every > 0 and frames_used % args.progress_every == 0:
                    print(
                        f"  ... кадр fi={fi}, merged={len(mc)}, nik={len(nc)}, "
                        f"всего строк CSV={row_count}",
                        flush=True,
                    )

                if args.max_frames_per_video > 0 and frames_used >= args.max_frames_per_video:
                    break

            cap.release()

            print(
                f"  готово: кадров={frames_used}, merged боксов={sum(merged_ns)}, nik={sum(nik_ns)}",
                flush=True,
            )

            summary_videos.append(
                {
                    "video_rel": vid_rel,
                    "frames_sampled": frames_used,
                    "vid_stride": args.vid_stride,
                    "merged_mean_boxes_per_frame": avg(merged_ns),
                    "nik_mean_boxes_per_frame": avg(nik_ns),
                    "merged_total_boxes": sum(merged_ns),
                    "nik_total_boxes": sum(nik_ns),
                    "merged_conf_mean_global": float(sum(merged_confs_all) / len(merged_confs_all))
                    if merged_confs_all
                    else 0.0,
                    "nik_conf_mean_global": float(sum(nik_confs_all) / len(nik_confs_all)) if nik_confs_all else 0.0,
                }
            )

    total_frames = sum(s["frames_sampled"] for s in summary_videos)
    total_m_boxes = sum(s["merged_total_boxes"] for s in summary_videos)
    total_n_boxes = sum(s["nik_total_boxes"] for s in summary_videos)

    out_summary_obj = {
        "settings": {
            "weights_merged": str(args.weights_merged),
            "weights_nik": str(args.weights_nik),
            "vid_stride": args.vid_stride,
            "imgsz_merged": args.imgsz_merged,
            "conf_merged": args.conf_merged,
            "imgsz_nik": args.imgsz_nik,
            "conf_nik": args.conf_nik,
            "tile_size": args.tile_size,
            "tile_stride": args.tile_stride,
            "nik_model_iou": args.nik_model_iou,
            "nik_nms_iou": args.nik_nms_iou,
            "device": args.device or "auto",
            "max_frames_per_video": args.max_frames_per_video,
            "max_videos": args.max_videos,
            "progress_every": args.progress_every,
        },
        "aggregate": {
            "videos": len(summary_videos),
            "frames_sampled_total": total_frames,
            "merged_boxes_total": total_m_boxes,
            "nik_boxes_total": total_n_boxes,
            "merged_mean_boxes_per_frame": total_m_boxes / total_frames if total_frames else 0.0,
            "nik_mean_boxes_per_frame": total_n_boxes / total_frames if total_frames else 0.0,
        },
        "per_video": summary_videos,
    }
    args.out_summary.write_text(json.dumps(out_summary_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Строк в CSV: {row_count} -> {args.out_csv.resolve()}")
    print(f"Сводка JSON: {args.out_summary.resolve()}")
    agg = out_summary_obj["aggregate"]
    print(
        f"Итого кадров: {agg['frames_sampled_total']}, merged боксов: {agg['merged_boxes_total']} "
        f"(ср./кадр {agg['merged_mean_boxes_per_frame']:.3f}), "
        f"NIK tiled боксов: {agg['nik_boxes_total']} (ср./кадр {agg['nik_mean_boxes_per_frame']:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

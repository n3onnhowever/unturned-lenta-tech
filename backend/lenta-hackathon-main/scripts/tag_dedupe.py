"""Deduplicate hackathon tag rows: one physical tag per video."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def _bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    def as_float(value: Any) -> float:
        try:
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return 0.0

    return (
        as_float(row.get("x_min")),
        as_float(row.get("y_min")),
        as_float(row.get("x_max")),
        as_float(row.get("y_max")),
    )


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def row_quality_score(row: dict[str, Any], meta: dict[str, Any] | None = None) -> float:
    meta = meta or {}
    score = float(meta.get("confidence", 0) or 0)
    if (row.get("barcode") or "").strip():
        score += 8.0
    if (row.get("price_card") or "").strip():
        score += 4.0
    if (row.get("price_default") or "").strip():
        score += 4.0
    if (row.get("product_name") or "").strip():
        score += 2.0
    disc = (row.get("discount_amount") or "").strip()
    if disc and disc not in ("нет", ""):
        score += 1.5
    if (row.get("color") or "").strip():
        score += 0.5
    return score


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


def _price_pair(row: dict[str, Any]) -> tuple[str, str]:
    return (_clean(row.get("price_default")), _clean(row.get("price_card")))


def _name_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = _clean(left.get("product_name"))
    b = _clean(right.get("product_name"))
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _center_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    ax, ay = _centroid(left)
    bx, by = _centroid(right)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _same_identity(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    iou_threshold: float,
    spatial_px: float,
) -> bool:
    left_bc = _clean(left.get("barcode"))
    right_bc = _clean(right.get("barcode"))
    if len(left_bc) >= 8 and left_bc == right_bc:
        return True

    overlap = iou(_bbox(left), _bbox(right))
    if overlap >= iou_threshold:
        return True

    if spatial_px > 0 and _center_distance(left, right) <= spatial_px:
        lp = _price_pair(left)
        rp = _price_pair(right)
        price_agrees = bool(lp[0] and lp[0] == rp[0]) or bool(lp[1] and lp[1] == rp[1])
        if price_agrees and (_name_similarity(left, right) >= 0.35 or not left_bc and not right_bc):
            return True
    return False


def _dedupe_index_pool(
    rows: list[dict[str, Any]],
    pool: list[int],
    meta_by_index: list[dict[str, Any]],
    iou_threshold: float,
) -> list[int]:
    """IoU clustering inside one frame."""
    pool = list(pool)
    chosen: list[int] = []
    while pool:
        seed = pool.pop(0)
        cluster = [seed]
        rest = []
        box_s = _bbox(rows[seed])
        for j in pool:
            if iou(box_s, _bbox(rows[j])) >= iou_threshold or _center_distance(rows[seed], rows[j]) <= 8.0:
                cluster.append(j)
            else:
                rest.append(j)
        pool = rest
        best = max(cluster, key=lambda i: row_quality_score(rows[i], meta_by_index[i]))
        chosen.append(best)
    return chosen


def dedupe_tag_rows(
    rows: list[dict[str, Any]],
    *,
    meta_by_index: list[dict[str, Any]] | None = None,
    iou_threshold: float = 0.45,
    dedupe_spatial_px: float = 0.0,
) -> list[dict[str, Any]]:
    """
    1) Within each frame: merge duplicate detections by IoU / near-identical centers.
    2) Across video: merge strong identities by barcode, or optional spatial+price/name agreement.
    """
    if not rows:
        return []

    meta_by_index = meta_by_index or [{} for _ in rows]

    by_frame: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        ts = (row.get("frame_timestamp") or "").strip() or "0"
        by_frame.setdefault(ts, []).append(i)

    frame_winners: list[int] = []
    for idxs in by_frame.values():
        frame_winners.extend(_dedupe_index_pool(rows, idxs, meta_by_index, iou_threshold))

    final = _cluster_identities(
        frame_winners,
        rows,
        meta_by_index,
        iou_threshold=iou_threshold,
        spatial_px=dedupe_spatial_px,
    )
    final.sort()
    return [rows[i] for i in final]


def _centroid(row: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = _bbox(row)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dedupe_spatial(
    indices: list[int],
    rows: list[dict[str, Any]],
    meta_by_index: list[dict[str, Any]],
    dist_px: float,
) -> list[int]:
    """Merge revisits of the same shelf slot on different frames."""
    pool = list(indices)
    chosen: list[int] = []
    while pool:
        seed = pool.pop(0)
        cluster = [seed]
        rest = []
        sx, sy = _centroid(rows[seed])
        for j in pool:
            cx, cy = _centroid(rows[j])
            if ((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5 <= dist_px:
                cluster.append(j)
            else:
                rest.append(j)
        pool = rest
        best = max(cluster, key=lambda i: row_quality_score(rows[i], meta_by_index[i]))
        chosen.append(best)
    return chosen


def _cluster_identities(
    indices: list[int],
    rows: list[dict[str, Any]],
    meta_by_index: list[dict[str, Any]],
    *,
    iou_threshold: float,
    spatial_px: float,
) -> list[int]:
    pool = list(indices)
    chosen: list[int] = []
    while pool:
        seed = pool.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            rest = []
            for idx in pool:
                if any(
                    _same_identity(
                        rows[idx],
                        rows[member],
                        iou_threshold=iou_threshold,
                        spatial_px=spatial_px,
                    )
                    for member in cluster
                ):
                    cluster.append(idx)
                    changed = True
                else:
                    rest.append(idx)
            pool = rest
        chosen.append(max(cluster, key=lambda i: row_quality_score(rows[i], meta_by_index[i])))
    return chosen

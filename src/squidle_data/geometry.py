"""Squidle annotation geometry helpers for exemplar prompts and COCO export."""

from __future__ import annotations

import ast
import json
from typing import List, Sequence, Tuple

import pandas as pd
from sam3_exemplar.geometry import bbox_of_vertices, is_axis_aligned_bbox, polygon_area_pixels

COL_LIKELIHOOD = "likelihood"


def parse_polygon_rel(poly_val) -> List[Tuple[float, float]]:
    """Parse a polygon cell into relative (dx, dy) vertex offsets."""
    if pd.isna(poly_val):
        return []
    if isinstance(poly_val, (list, tuple)):
        seq = poly_val
    else:
        s = str(poly_val).strip()
        if not s:
            return []
        try:
            seq = ast.literal_eval(s)
        except Exception:
            seq = json.loads(s)

    out: List[Tuple[float, float]] = []
    for item in seq:
        if isinstance(item, dict) and "xy" in item:
            x, y = item["xy"]
        else:
            x, y = item
        out.append((float(x), float(y)))
    return out


def clip_norm_xy(x: float, y: float) -> Tuple[float, float]:
    """Clamp relative image coordinates to [0, 1]."""
    return max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y)))


def squidle_offsets_to_norm_vertices(
    x: float,
    y: float,
    poly_val,
    *,
    clip: bool = True,
) -> List[Tuple[float, float]]:
    """Convert Squidle ``point.polygon`` offsets and click ``(x, y)`` to norm vertices."""
    vertices: List[Tuple[float, float]] = []
    for dx, dy in parse_polygon_rel(poly_val):
        nx, ny = float(x) + dx, float(y) + dy
        if clip:
            nx, ny = clip_norm_xy(nx, ny)
        vertices.append((nx, ny))
    return vertices


def append_exemplar_geometry(
    bboxes: list,
    points: list,
    polygons: list,
    x: float,
    y: float,
    polygon_vertices: Sequence[Sequence[float]],
    *,
    bbox_format: str = "xywh",
) -> None:
    """Append one exemplar; use polygon sampling when polygon is not a bbox."""
    points.append((float(x), float(y)))
    if is_axis_aligned_bbox(polygon_vertices):
        bboxes.append(bbox_of_vertices(polygon_vertices, fmt=bbox_format))
        polygons.append(None)
    else:
        bboxes.append(None)
        polygons.append(list(polygon_vertices))


def point_to_bbox_pixels(
    x_rel: float,
    y_rel: float,
    width: int,
    height: int,
    *,
    frac: float = 0.01,
) -> List[float]:
    """Tiny COCO-style bbox around a relative point (for exports without polygons)."""
    px = x_rel * width
    py = y_rel * height
    half = max(2.0, frac * min(width, height))
    x0 = max(0.0, px - half)
    y0 = max(0.0, py - half)
    x1 = min(float(width), px + half)
    y1 = min(float(height), py + half)
    return [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]


def annotation_bbox_pixels(
    row,
    width: int,
    height: int,
    *,
    poly_col: str | None,
    x_col: str,
    y_col: str,
) -> List[float]:
    x = float(row[x_col])
    y = float(row[y_col])
    if poly_col and poly_col in row.index and pd.notna(row.get(poly_col)):
        poly_rel = parse_polygon_rel(row[poly_col])
        if poly_rel:
            verts_rel = [(x + dx, y + dy) for dx, dy in poly_rel]
            x0, y0, bw, bh = bbox_of_vertices(verts_rel, fmt="xywh")
            return [x0 * width, y0 * height, bw * width, bh * height]
    return point_to_bbox_pixels(x, y, width, height)


def annotation_segmentation_pixels(
    row,
    width: int,
    height: int,
    *,
    poly_col: str | None,
    x_col: str,
    y_col: str,
) -> List[List[float]]:
    """COCO segmentation from Squidle point.polygon (relative to point.x/y)."""
    if not poly_col or poly_col not in row.index or pd.isna(row.get(poly_col)):
        return []
    x = float(row[x_col])
    y = float(row[y_col])
    poly_rel = parse_polygon_rel(row[poly_col])
    if len(poly_rel) < 3:
        return []
    ring: List[float] = []
    verts_px: List[Tuple[float, float]] = []
    for dx, dy in poly_rel:
        px = (x + dx) * width
        py = (y + dy) * height
        ring.extend([px, py])
        verts_px.append((px, py))
    if polygon_area_pixels(verts_px) <= 0:
        return []
    return [ring]

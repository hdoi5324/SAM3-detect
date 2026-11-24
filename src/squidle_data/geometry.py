"""Squidle annotation geometry helpers for exemplar prompts and COCO export."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd

# Generic helpers moved to the Squidle-free core (sam3_exemplar.geometry,
# SAM3-FSOD sync manifest); re-exported here for backwards compatibility.
from sam3_exemplar.geometry import (  # noqa: F401
    append_exemplar_geometry,
    bbox_of_vertices,
    clip_norm_xy,
    is_axis_aligned_bbox,
    offsets_to_norm_vertices,
    parse_polygon_rel,
    polygon_area_pixels,
    squidle_offsets_to_norm_vertices,
)

COL_LIKELIHOOD = "likelihood"


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

"""Shared 2D geometry helpers for exemplar prompts and COCO export.

This module is part of the Squidle-free public core (SAM3-FSOD): it must
never import ``squidle_data``, ``sqapi`` or ``sqbot_framework``.
"""

from __future__ import annotations

import ast
import json
import math
from typing import List, Sequence, Tuple

import numpy as np


def polygon_is_bbox(coordinates) -> bool:
    """True if vertices form an axis-aligned rectangle (Squidle bbox polygon)."""
    if not isinstance(coordinates, list):
        return False
    if len(coordinates) not in (4, 5) or not all(
        isinstance(i, float) for p in coordinates for i in p
    ):
        return False

    ind = 0 if coordinates[0][0] == coordinates[1][0] else 1
    matches = []
    for i in range(4):
        matches.append(coordinates[i][ind] == coordinates[(i + 1) % 4][ind])
        ind = 0 if ind == 1 else 1
    return all(matches)


# Alias used by SAM3 exemplar / polygon prompt code.
is_axis_aligned_bbox = polygon_is_bbox


def bbox_of_vertices(
    vertices: Sequence[Sequence[float]],
    fmt: str = "xywh",
) -> tuple[float, float, float, float]:
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x1, y1 = float(np.min(xs)), float(np.min(ys))
    x2, y2 = float(np.max(xs)), float(np.max(ys))
    if fmt.lower() == "xyxy":
        return x1, y1, x2, y2
    if fmt.lower() == "xywh":
        return x1, y1, (x2 - x1), (y2 - y1)
    raise ValueError("fmt must be 'xyxy' or 'xywh'")


def polygon_area_pixels(vertices: Sequence[Sequence[float]]) -> float:
    """Shoelace area for a polygon in pixel coordinates."""
    if len(vertices) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(i + 1) % len(vertices)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _is_missing(value) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def parse_polygon_rel(poly_val) -> List[Tuple[float, float]]:
    """Parse a polygon cell into relative (dx, dy) vertex offsets.

    Accepts a list/tuple of ``[dx, dy]`` (or ``{"xy": [dx, dy]}``) items, or a
    string encoding of one (Python-literal or JSON). Empty/missing -> ``[]``.
    """
    if _is_missing(poly_val):
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


def offsets_to_norm_vertices(
    x: float,
    y: float,
    poly_val,
    *,
    clip: bool = True,
) -> List[Tuple[float, float]]:
    """Convert relative polygon offsets around a click ``(x, y)`` to norm vertices."""
    vertices: List[Tuple[float, float]] = []
    for dx, dy in parse_polygon_rel(poly_val):
        nx, ny = float(x) + dx, float(y) + dy
        if clip:
            nx, ny = clip_norm_xy(nx, ny)
        vertices.append((nx, ny))
    return vertices


# Backwards-compatible name (originally in squidle_data.geometry; the polygon
# format originated in Squidle exports but the helper is format-generic).
squidle_offsets_to_norm_vertices = offsets_to_norm_vertices


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


def should_process_point(point: dict) -> bool:
    """Return False if the Squidle point already has a non-bbox polygon."""
    plgn = point.get("data", {}).get("polygon")
    is_bbox_plgn = polygon_is_bbox(plgn)
    if not is_bbox_plgn and isinstance(plgn, list) and len(plgn) >= 3:
        return False
    x, y = point.get("x"), point.get("y")
    return isinstance(x, float) and isinstance(y, float)

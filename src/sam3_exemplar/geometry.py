"""Shared 2D geometry helpers for exemplar prompts and COCO export."""

from __future__ import annotations

from typing import Sequence

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


def should_process_point(point: dict) -> bool:
    """Return False if the Squidle point already has a non-bbox polygon."""
    plgn = point.get("data", {}).get("polygon")
    is_bbox_plgn = polygon_is_bbox(plgn)
    if not is_bbox_plgn and isinstance(plgn, list) and len(plgn) >= 3:
        return False
    x, y = point.get("x"), point.get("y")
    return isinstance(x, float) and isinstance(y, float)

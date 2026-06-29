"""Helpers for SAM3 exemplar prompts from polygon annotations."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon


def is_axis_aligned_bbox(coordinates) -> bool:
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


def sample_points_in_polygon(
    polygon_xy,
    *,
    n_points: int = 8,
    include_click: tuple[float, float] | list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[list[float]]:
    """
    Sample normalized [x, y] points inside a polygon for SAM3 point prompts.

    ``polygon_xy`` and ``include_click`` use Squidle normalized coordinates in [0, 1].
    """
    if n_points <= 0:
        return []

    poly = ShapelyPolygon(polygon_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        if include_click is not None:
            return [[float(include_click[0]), float(include_click[1])]]
        return []

    coords: list[tuple[float, float]] = []
    if include_click is not None:
        coords.append((float(include_click[0]), float(include_click[1])))

    minx, miny, maxx, maxy = poly.bounds
    rng = rng or np.random.default_rng(42)
    max_attempts = max(n_points * 50, 100)
    attempts = 0
    while len(coords) < n_points and attempts < max_attempts:
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if poly.contains(Point(x, y)):
            coords.append((x, y))
        attempts += 1

    if len(coords) < n_points:
        rep = poly.representative_point()
        coords.append((float(rep.x), float(rep.y)))

    return [[x, y] for x, y in coords[:n_points]]


def absolute_polygon_from_point(x: float, y: float, relative_polygon) -> list[list[float]]:
    """Convert Squidle point-relative polygon vertices to absolute normalized coords."""
    return [[float(vx + x), float(vy + y)] for vx, vy in relative_polygon]


def bbox_xywh_from_vertices(vertices) -> tuple[float, float, float, float]:
    """Bounding box as normalized top-left xywh from absolute polygon vertices."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x1, y1 = float(min(xs)), float(min(ys))
    x2, y2 = float(max(xs)), float(max(ys))
    return x1, y1, (x2 - x1), (y2 - y1)

"""Mask and polygon helpers for exemplar detection and segmentation."""

from __future__ import annotations

import cv2
import numpy as np
from shapely import Polygon
from shapely.geometry import Point as ShapelyPoint


def is_area_greater_than(coords, min_area: float) -> bool:
    return Polygon(coords).area > float(min_area)


def clean_polygon(list_of_xy, width: int, height: int) -> list[tuple[float, float]]:
    checked_points = []
    for x, y in list_of_xy:
        if x <= 0:
            x = 0
        if x >= width:
            x = width - 1
        if y <= 0:
            y = 0
        if y >= height:
            y = height - 1
        checked_points.append((x, y))
    return checked_points


def check_polygon(list_of_xy, width: int, height: int, min_percent: float = 0.001) -> bool:
    if not (isinstance(list_of_xy, list) and len(list_of_xy) > 2):
        return False
    min_area = width * height * min_percent
    return is_area_greater_than(list_of_xy, min_area)


def _mask_uint8(mask) -> np.ndarray | None:
    mask_arr = np.asarray(mask).squeeze()
    if mask_arr.size == 0:
        return None
    if mask_arr.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, got shape {mask_arr.shape}")
    if np.issubdtype(mask_arr.dtype, np.floating):
        mask_uint8 = (mask_arr >= 0.5).astype(np.uint8)
    else:
        mask_uint8 = mask_arr.astype(np.uint8)
    if not mask_uint8.any():
        return None
    return mask_uint8


def mask_to_polygon(mask, tolerance: float = 2.0) -> list[list[list[float]]]:
    """Extract simplified polygon(s) from a mask using OpenCV."""
    mask_uint8 = _mask_uint8(mask)
    if mask_uint8 is None:
        return []
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        poly = cv2.approxPolyDP(contour, epsilon=tolerance, closed=True)
        points = poly.reshape(-1, 2).tolist()
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def primary_mask_polygon(mask, tolerance: float = 2.0) -> list[list[float]] | None:
    """Return the largest contour as a single polygon, or ``None`` if empty."""
    mask_uint8 = _mask_uint8(mask)
    if mask_uint8 is None:
        return None
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(contour, epsilon=tolerance, closed=True)
    points = poly.reshape(-1, 2).tolist()
    return points if len(points) >= 3 else None


def normalized_point_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Squidle normalized ``(x, y)`` → pixel ``(row, col)`` for mask indexing."""
    row = int(np.clip(y * height, 0, height - 1))
    col = int(np.clip(x * width, 0, width - 1))
    return row, col


def _pixel_in_mask_array(mask, row: int, col: int, threshold: float = 0.5) -> bool:
    m = np.asarray(mask).squeeze()
    if m.size == 0 or m.ndim != 2:
        return False
    mh, mw = m.shape
    row = int(np.clip(row, 0, mh - 1))
    col = int(np.clip(col, 0, mw - 1))
    val = m[row, col]
    if m.dtype == np.bool_ or val in (0, 1):
        return bool(val)
    return float(val) >= threshold


def normalized_point_in_mask(
    mask,
    x: float,
    y: float,
    width: int,
    height: int,
    threshold: float = 0.5,
) -> bool:
    """True if the normalized click lies inside the mask."""
    m = np.asarray(mask).squeeze()
    if m.size == 0 or m.ndim != 2:
        return False
    mh, mw = m.shape
    row, col = normalized_point_to_pixel(x, y, width, height)
    if mw != width or mh != height:
        col = int(round(col * (mw - 1) / max(width - 1, 1)))
        row = int(round(row * (mh - 1) / max(height - 1, 1)))
    return _pixel_in_mask_array(m, row, col, threshold=threshold)


def normalized_point_in_polygon(
    polygon,
    x: float,
    y: float,
    width: int,
    height: int,
) -> bool:
    """True if the normalized click lies inside (or on) the polygon in pixel space."""
    if not polygon or len(polygon) < 3:
        return False
    px = x * width
    py = y * height
    shape = Polygon(polygon)
    if not shape.is_valid:
        shape = shape.buffer(0)
    if shape.is_empty or not shape.is_valid:
        return False
    pt = ShapelyPoint(px, py)
    return shape.contains(pt) or shape.boundary.distance(pt) <= 1.0


def scale_polygon(polygon, from_w: int, from_h: int, to_w: int, to_h: int):
    if from_w == to_w and from_h == to_h:
        return polygon
    sx = to_w / from_w
    sy = to_h / from_h
    return [[p[0] * sx, p[1] * sy] for p in polygon]


def mask_contains_point(mask, row: int, col: int, threshold: float = 0.5) -> bool:
    """Legacy row/col check; prefer :func:`normalized_point_in_mask` when x,y are known."""
    return _pixel_in_mask_array(mask, row, col, threshold=threshold)


def select_best_mask_index(
    masks,
    scores,
    x: float,
    y: float,
    width: int,
    height: int,
    min_score: float | None = None,
) -> int | None:
    """Index of the highest-scoring mask that contains the normalized query point, or ``None``."""
    masks_arr = np.asarray(masks)
    scores_arr = np.asarray(scores).reshape(-1)
    n = min(len(scores_arr), masks_arr.shape[0])
    best_i = None
    best_score = -np.inf
    for i in range(n):
        score = float(scores_arr[i])
        if min_score is not None and score <= min_score:
            continue
        if normalized_point_in_mask(masks_arr[i], x, y, width, height) and score > best_score:
            best_score = score
            best_i = i
    return best_i


def polygon_from_mask_covering_point(
    mask,
    x: float,
    y: float,
    mask_width: int,
    mask_height: int,
    tolerance: float = 3.0,
) -> list[list[float]] | None:
    """
    Extract a polygon from ``mask`` only if the normalized point lies inside it.

    Returns ``None`` when the point is not covered or contour extraction fails.
    """
    if not normalized_point_in_mask(mask, x, y, mask_width, mask_height):
        return None
    for polygon in mask_to_polygon(mask, tolerance=tolerance):
        if normalized_point_in_polygon(polygon, x, y, mask_width, mask_height):
            return polygon
    return None

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


@dataclass
class PolygonCandidate:
    score: float
    label_id: int
    poly: List[List[float]]
    hit_point_indices: List[int]
    is_new: bool
    fallback_hit_point_indices: List[int] = field(default_factory=list)
    assigned_point_index: int | None = None


def polygon_iou(poly_a: ShapelyPolygon, poly_b: ShapelyPolygon) -> float:
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return inter / union if union > 0 else 0.0


def build_candidate(
    *,
    score: float,
    label_id: int,
    poly: List[List[float]],
    hit_point_indices: Iterable[int],
    fallback_hit_point_indices: Iterable[int] = (),
) -> PolygonCandidate | None:
    """
    Create a candidate from polygon + metadata.

    ``hit_point_indices``: points with matching label_id inside the mask.
    ``fallback_hit_point_indices``: unlabeled points (label_id None) inside the mask,
    used when no matching-label point is available for assignment.

    Returns None if polygon is invalid after a best-effort cleanup.
    """
    shape = ShapelyPolygon(poly)
    if not shape.is_valid:
        shape = shape.buffer(0)
    if shape.is_empty or not shape.is_valid:
        return None

    hits = list(hit_point_indices)
    fallback_hits = list(fallback_hit_point_indices)
    return PolygonCandidate(
        score=float(score),
        label_id=int(label_id),
        poly=poly,
        hit_point_indices=hits,
        fallback_hit_point_indices=fallback_hits,
        is_new=(len(hits) == 0 and len(fallback_hits) == 0),
    )


def suppress_overlapping_candidates(
    candidates: Sequence[PolygonCandidate],
    iou_threshold: float,
) -> List[PolygonCandidate]:
    """
    Global polygon NMS across all labels, highest score first.
    """
    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    accepted: List[PolygonCandidate] = []
    accepted_shapes: List[ShapelyPolygon] = []

    for cand in sorted_cands:
        cand_shape = ShapelyPolygon(cand.poly)
        if not cand_shape.is_valid:
            cand_shape = cand_shape.buffer(0)
        if cand_shape.is_empty or not cand_shape.is_valid:
            continue

        redundant = False
        for shape in accepted_shapes:
            if polygon_iou(cand_shape, shape) > iou_threshold:
                redundant = True
                break
        if redundant:
            continue

        accepted.append(cand)
        accepted_shapes.append(cand_shape)

    return accepted


def assign_candidates_to_points(
    candidates: Sequence[PolygonCandidate],
    *,
    add_new_annotations: bool,
    add_new_threshold: float,
) -> List[PolygonCandidate]:
    """
    Enforce one-to-one point assignment after NMS.

    Prefer points with matching label_id; if none are free, use an unlabeled
    point (label_id None) from ``fallback_hit_point_indices``.
    """
    accepted: List[PolygonCandidate] = []
    used_points: set[int] = set()

    for cand in candidates:
        if cand.is_new:
            if add_new_annotations and cand.score > add_new_threshold:
                accepted.append(cand)
            continue

        available_idx = next(
            (idx for idx in cand.hit_point_indices if idx not in used_points),
            None,
        )
        if available_idx is None:
            available_idx = next(
                (idx for idx in cand.fallback_hit_point_indices if idx not in used_points),
                None,
            )
        if available_idx is None:
            continue
        cand.assigned_point_index = int(available_idx)
        used_points.add(int(available_idx))
        accepted.append(cand)

    return accepted


def points_inside_mask(mask_np: np.ndarray, pts_coords: np.ndarray) -> np.ndarray:
    """
    Return point indices where mask is true.
    """
    if pts_coords.size == 0:
        return np.array([], dtype=int)
    inside_mask = mask_np[pts_coords[:, 0], pts_coords[:, 1]]
    return np.where(inside_mask)[0]


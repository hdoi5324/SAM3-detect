"""Shared exemplar media dict layout for SAM3 online bot and offline CLI."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from squidle_data.geometry import append_exemplar_geometry, squidle_offsets_to_norm_vertices

# Canonical layout: one media key → six parallel lists consumed by encode_exemplar_prompts.
IDX_BBOXES = 0
IDX_POINTS = 1
IDX_NEG_BBOXES = 2
IDX_NEG_POINTS = 3
IDX_POLYGONS = 4
IDX_NEG_POLYGONS = 5

__all__ = [
    "IDX_BBOXES",
    "IDX_NEG_BBOXES",
    "IDX_NEG_POINTS",
    "IDX_NEG_POLYGONS",
    "IDX_POINTS",
    "IDX_POLYGONS",
    "append_negative_geometry_lists",
    "append_positive_geometry_to_entry",
    "build_exemplar_data_dict",
    "empty_exemplar_media_lists",
    "get_or_create_media_entry",
    "parse_exemplar_media_fields",
    "squidle_row_norm_geometry",
]


def empty_exemplar_media_lists() -> list:
    return [[], [], [], [], [], []]


def get_or_create_media_entry(exemplars_for_label: dict, media_key: str) -> list:
    if media_key not in exemplars_for_label:
        exemplars_for_label[media_key] = empty_exemplar_media_lists()
    return exemplars_for_label[media_key]


def append_positive_geometry_to_entry(
    entry: list,
    x: float,
    y: float,
    polygon_vertices: Sequence[Sequence[float]] | None,
    *,
    bbox_format: str = "xywh",
    clip: bool = True,
) -> None:
    """Append one positive exemplar geometry row to a six-list media entry."""
    if polygon_vertices is None:
        entry[IDX_POINTS].append((float(x), float(y)))
        return
    append_exemplar_geometry(
        entry[IDX_BBOXES],
        entry[IDX_POINTS],
        entry[IDX_POLYGONS],
        x,
        y,
        polygon_vertices,
        bbox_format=bbox_format,
    )
    entry[IDX_POINTS][-1] = [float(x), float(y)]


def append_negative_geometry_lists(
    entry: list,
    neg_bboxes: Sequence,
    neg_points: Sequence,
    neg_polygons: Sequence,
) -> None:
    entry[IDX_NEG_BBOXES].extend(neg_bboxes)
    entry[IDX_NEG_POINTS].extend(neg_points)
    entry[IDX_NEG_POLYGONS].extend(neg_polygons)


def squidle_row_norm_geometry(
    x,
    y,
    polygon_cell,
    *,
    clip: bool = False,
) -> tuple[float, float, list[tuple[float, float]] | None]:
    """Squidle CSV row → click + normalized polygon vertices (image media)."""
    xf, yf = float(x), float(y)
    if polygon_cell is None or (isinstance(polygon_cell, float) and str(polygon_cell) == "nan"):
        return xf, yf, None
    vertices = squidle_offsets_to_norm_vertices(xf, yf, polygon_cell, clip=clip)
    if not vertices:
        return xf, yf, None
    return xf, yf, vertices


def parse_exemplar_media_fields(exemplar_fields) -> Tuple[list, list, list, list, list, list]:
    """Return pos/neg boxes, points, polygons from a 6-field exemplar dict value."""
    boxes = list(exemplar_fields[IDX_BBOXES]) if len(exemplar_fields) > IDX_BBOXES else []
    points = list(exemplar_fields[IDX_POINTS]) if len(exemplar_fields) > IDX_POINTS else []
    neg_boxes = list(exemplar_fields[IDX_NEG_BBOXES]) if len(exemplar_fields) > IDX_NEG_BBOXES else []
    neg_points = list(exemplar_fields[IDX_NEG_POINTS]) if len(exemplar_fields) > IDX_NEG_POINTS else []
    polygons = (
        list(exemplar_fields[IDX_POLYGONS])
        if len(exemplar_fields) > IDX_POLYGONS
        else [None] * len(points)
    )
    neg_polygons = (
        list(exemplar_fields[IDX_NEG_POLYGONS])
        if len(exemplar_fields) > IDX_NEG_POLYGONS
        else [None] * len(neg_points)
    )
    if len(polygons) < len(points):
        polygons.extend([None] * (len(points) - len(polygons)))
    if len(neg_polygons) < len(neg_points):
        neg_polygons.extend([None] * (len(neg_points) - len(neg_polygons)))
    return boxes, points, neg_boxes, neg_points, polygons, neg_polygons


def build_exemplar_data_dict(
    exemplar_images: List[str],
    exemplar_bboxes: List,
    exemplar_points: List,
    exemplar_polygons: List,
    *,
    neg_fields_fn: Callable[[str], tuple[list, list, list]] | None = None,
) -> Dict[str, list]:
    """Build ``{media_key: [bboxes, points, neg_*, polygons, neg_polygons]}`` for encoding."""
    out: Dict[str, list] = {}
    for idx, img in enumerate(exemplar_images):
        neg_bboxes: list = []
        neg_points: list = []
        neg_polygons: list = []
        if neg_fields_fn is not None:
            neg_bboxes, neg_points, neg_polygons = neg_fields_fn(img)
        out[img] = [
            exemplar_bboxes[idx],
            exemplar_points[idx],
            neg_bboxes,
            neg_points,
            exemplar_polygons[idx],
            neg_polygons,
        ]
    return out

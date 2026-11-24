"""Squidle CSV exemplar loading (PRIVATE adapter — never exported to SAM3-FSOD).

Moved out of ``file_exemplars`` so the COCO/file exemplar core stays
Squidle-free. ``file_exemplars.load_exemplar_annotations`` dispatches here
lazily for ``annotation_file_type="squidle"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

from sam3_exemplar.exemplar_data import squidle_row_norm_geometry
from sam3_exemplar.file_exemplars import DEFAULT_IMAGE_CACHE_DIR
from sam3_exemplar.geometry import append_exemplar_geometry

__all__ = [
    "load_squidle_annotation_df",
    "load_squidle_exemplar_annotations",
    "squidle_negative_exemplars_for_image",
    "squidle_positive_label_ids_on_image",
]


def load_squidle_annotation_df(annotation_filename: str | Path) -> pd.DataFrame:
    """Load a Squidle export CSV with standard column cleanup."""
    df = pd.read_csv(annotation_filename)
    drop_cols = [c for c in df.columns if c.startswith("point.pixels")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    required = ["point.media.path_best", "point.x", "point.y"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    return df[df["point.media.path_best"].notna()].copy()

def squidle_positive_label_ids_on_image(df: pd.DataFrame, media_path: str) -> set[Any]:
    if "label.id" not in df.columns:
        return set()
    return set(df.loc[df["point.media.path_best"] == media_path, "label.id"].dropna())

def squidle_negative_exemplars_for_image(
    df: pd.DataFrame,
    media_path: str,
    positive_label_ids: set[Any],
    n_negative: int,
    *,
    bbox_format: str = "xywh",
    seed: int = 42,
) -> Tuple[
    List[Optional[Tuple[float, float, float, float]]],
    List[Tuple[float, float]],
    List[Optional[List[Tuple[float, float]]]],
]:
    """Sample negative exemplar geometry from other labels on the same image."""
    if n_negative <= 0 or "label.id" not in df.columns or not positive_label_ids:
        return [], [], []

    neg_df = df.loc[
        (~df["label.id"].isin(positive_label_ids))
        & (df["point.media.path_best"] == media_path)
    ]
    if neg_df.empty:
        return [], [], []

    neg_df = neg_df.sample(n=min(n_negative, len(neg_df)), random_state=seed)
    neg_bboxes: List[Optional[Tuple[float, float, float, float]]] = []
    neg_points: List[Tuple[float, float]] = []
    neg_polygons: List[Optional[List[Tuple[float, float]]]] = []
    poly_col = "point.polygon" if "point.polygon" in neg_df.columns else None

    for _, row in neg_df.iterrows():
        try:
            x = float(row["point.x"])
            y = float(row["point.y"])
        except (TypeError, ValueError):
            continue
        if poly_col and pd.notna(row.get(poly_col)):
            x, y, vertices = squidle_row_norm_geometry(
                row["point.x"], row["point.y"], row[poly_col], clip=True
            )
            if vertices:
                append_exemplar_geometry(
                    neg_bboxes, neg_points, neg_polygons, x, y, vertices, bbox_format=bbox_format
                )
                continue
        neg_points.append((x, y))

    return neg_bboxes, neg_points, neg_polygons

def load_squidle_exemplar_annotations(
        annotation_filename: str | Path,
        sqapi,
        bbox_format="xywh",
        image_cache_dir: str | Path | None = DEFAULT_IMAGE_CACHE_DIR,
        **_,
) -> Tuple[
    List[str],
    List[List[Optional[Tuple[float, float, float, float]]]],
    List[List[Tuple[float, float]]],
    List[List[Optional[List[Tuple[float, float]]]]],
    List[Tuple[int, int]],
]:
    """
    Load through sqapi:
      - image_paths: sorted unique list of `point.media.path_best`
      - bboxes_by_image: per-image box prompts (None when polygon prompt is used)
      - points_by_image: per-image click points in relative coords
      - polygons_by_image: per-image polygon vertices (None when bbox prompt is used)
      - image_sizes: (width, height) per image

    Notes
    -----
    - Uses only relative coordinates (no pixel conversion).
    - For each row, polygon vertices are interpreted as relative offsets around (point.x, point.y).
      Each vertex: (point.x + dx, point.y + dy), then a per-row bbox is computed from these vertices.
    - Rows without a valid polygon are skipped for bbox generation (point still recorded).
    - Columns starting with 'point.pixels' are ignored entirely.
    - Only image media are used; each group is checked via ``/api/media/<id>`` before
      the image file is opened.

    Parameters
    ----------
    sqapi
        Squidle API client (required for image-media filtering).
    annotation_filename : str | Path
        Path to the CSV (export) file.
    bbox_format : {'xyxy','xywh'}, default 'xyxy'
        Bounding box output format in relative coordinates.

    Returns
    -------
    image_paths : List[str]
        Sorted unique `point.media.path_best` strings.
    bboxes_by_image : List[List[Tuple[float,float,float,float]]]
        Per image, list of bboxes in relative coords (format per `bbox_format`).
    points_by_image : List[List[Tuple[float,float]]]
        Per image, list of relative (point.x, point.y).

    Notes on image sizing
    ---------------------
    Image dimensions are resolved by opening each grouped image path via PIL. Remote URLs
    are cached under ``image_cache_dir`` (default ``.cache/exemplar_seg_images``) so repeated
    runs avoid re-downloading. If PIL/open fails, falls back to ``SQMediaObject`` dimensions.
    """
    from sqapi.media import SQMediaObject

    from squidle_data.cache import open_pil_from_uri_or_path
    from squidle_data.media import (
        filter_annotations_to_image_media as filter_exemplar_df_to_image_media,
    )
    from squidle_data.media import is_squidle_image_media

    df = load_squidle_annotation_df(annotation_filename)
    df = filter_exemplar_df_to_image_media(df, sqapi)

    # Keep optional polygon column if present
    poly_col = "point.polygon" if "point.polygon" in df.columns else None

    # Sort by path to provide deterministic output order
    df = df.sort_values(by="point.media.path_best").reset_index(drop=True)

    # Group by image path
    grouped = df.groupby("point.media.path_best", sort=True)

    image_paths: List[str] = []
    image_sizes: List[Tuple[int, int]] = []
    bboxes_by_image: List[List[Optional[Tuple[float, float, float, float]]]] = []
    points_by_image: List[List[Tuple[float, float]]] = []
    polygons_by_image: List[List[Optional[List[Tuple[float, float]]]]] = []

    cache_dir: Path | None = None
    if image_cache_dir is not None:
        cache_dir = Path(image_cache_dir).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = (Path.cwd() / cache_dir).resolve()
        else:
            cache_dir = cache_dir.resolve()

    for img_path, g in grouped:
        media_id = pd.to_numeric(g["point.media.id"].iloc[0], errors="coerce")
        if pd.isna(media_id) or not is_squidle_image_media(sqapi, int(media_id)):
            continue

        # Resolve image size through the shared PIL/cache path first.
        try:
            im = open_pil_from_uri_or_path(str(img_path), cache_dir)
            width, height = im.size
            im.close()
        except Exception:
            # Fallback for odd URLs/media that PIL cannot open.
            sq_media = SQMediaObject(img_path)
            sq_media.data()
            width, height = sq_media.width, sq_media.height
        image_sizes.append((width, height))

        image_paths.append(str(img_path))
        img_bboxes: List[Optional[Tuple[float, float, float, float]]] = []
        img_points: List[Tuple[float, float]] = []
        img_polygons: List[Optional[List[Tuple[float, float]]]] = []

        for _, row in g.iterrows():
            try:
                x = float(row["point.x"])
                y = float(row["point.y"])
            except (TypeError, ValueError):
                continue

            if poly_col and pd.notna(row.get(poly_col)):
                x, y, vertices = squidle_row_norm_geometry(
                    row["point.x"], row["point.y"], row[poly_col], clip=True
                )
                if vertices:
                    append_exemplar_geometry(
                        img_bboxes, img_points, img_polygons, x, y, vertices, bbox_format=bbox_format
                    )
                    continue
            img_points.append((x, y))

        bboxes_by_image.append(img_bboxes)
        points_by_image.append(img_points)
        polygons_by_image.append(img_polygons)

    return image_paths, bboxes_by_image, points_by_image, polygons_by_image, image_sizes

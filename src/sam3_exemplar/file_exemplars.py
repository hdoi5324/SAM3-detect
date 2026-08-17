"""Load exemplar annotations from COCO JSON or Squidle CSV (offline CLI)."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sqapi.media import SQMediaObject

from sam3_exemplar.geometry import bbox_of_vertices, is_axis_aligned_bbox
from squidle_data.geometry import append_exemplar_geometry, squidle_offsets_to_norm_vertices
from squidle_data.cache import (
    open_pil_from_uri_or_path,
)
from sam3_exemplar.exemplar_data import squidle_row_norm_geometry
from squidle_data.media import (
    filter_annotations_to_image_media as filter_exemplar_df_to_image_media,
    is_squidle_image_media,
)

DEFAULT_IMAGE_CACHE_DIR = Path(".cache/exemplar_seg_images")


def squidle_offsets_to_pixel_ring(
    x: float,
    y: float,
    poly_val,
    width: int,
    height: int,
    *,
    clip: bool = True,
) -> List[List[float]]:
    """Return one COCO segmentation ring ``[[x_px, y_px], ...]`` in pixel coordinates."""
    w, h = float(width), float(height)
    return [
        [nx * w, ny * h]
        for nx, ny in squidle_offsets_to_norm_vertices(x, y, poly_val, clip=clip)
    ]


def _ring_to_pairs(ring: Sequence) -> List[Tuple[float, float]]:
    """Convert one COCO polygon ring to (x, y) pixel pairs."""
    if not ring:
        return []
    if isinstance(ring[0], (int, float)):
        if len(ring) < 6:
            return []
        return [(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring) - 1, 2)]
    pairs: List[Tuple[float, float]] = []
    for pt in ring:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            pairs.append((float(pt[0]), float(pt[1])))
    return pairs


def _parse_coco_segmentation(segmentation) -> List[List[Tuple[float, float]]]:
    """
    Parse COCO ``segmentation`` into polygon rings as pixel (x, y) pairs.

    Supports flat rings ``[x1, y1, x2, y2, ...]`` and nested ``[[x, y], ...]``.
    RLE dict segmentations are skipped (caller should fall back to bbox).
    """
    if not segmentation or isinstance(segmentation, dict):
        return []
    if not isinstance(segmentation, list):
        return []

    polys: List[List[Tuple[float, float]]] = []
    for ring in segmentation:
        pairs = _ring_to_pairs(ring)
        if len(pairs) >= 3:
            polys.append(pairs)
    return polys


def _append_coco_exemplar_geometry(
    bboxes: List[Optional[Tuple[float, float, float, float]]],
    polygons: List[Optional[List[Tuple[float, float]]]],
    vertices_rel: List[Tuple[float, float]],
    bbox_xywh_rel: Optional[Tuple[float, float, float, float]],
    *,
    bbox_format: str = "xywh",
) -> None:
    """Append one COCO annotation using segmentation when available, else bbox."""
    if is_axis_aligned_bbox(vertices_rel):
        if bbox_xywh_rel is not None:
            bboxes.append(bbox_xywh_rel)
        else:
            bboxes.append(bbox_of_vertices(vertices_rel, fmt=bbox_format))
        polygons.append(None)
    else:
        bboxes.append(None)
        polygons.append(vertices_rel)


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


def annotated_image_indices(
    bboxes_by_image: Sequence[Sequence],
    points_by_image: Sequence[Sequence],
    polygons_by_image: Sequence[Sequence],
) -> List[int]:
    """Return indices of images that have at least one bbox, polygon, or point prompt."""
    indices: List[int] = []
    for i, (bboxes, points, polygons) in enumerate(
        zip(bboxes_by_image, points_by_image, polygons_by_image)
    ):
        has_geometry = (
            any(b is not None for b in bboxes)
            or any(p for p in polygons if p)
            or bool(points)
        )
        if has_geometry:
            indices.append(i)
    return indices


def annotation_indices(
    bboxes_by_image: Sequence[Sequence],
    points_by_image: Sequence[Sequence],
    polygons_by_image: Sequence[Sequence],
) -> List[Tuple[int, int]]:
    """Return ``(image_idx, anno_idx)`` for every annotation with usable geometry."""
    out: List[Tuple[int, int]] = []
    for img_idx, (bboxes, points, polygons) in enumerate(
        zip(bboxes_by_image, points_by_image, polygons_by_image)
    ):
        n = max(len(bboxes), len(points), len(polygons))
        for anno_idx in range(n):
            box = bboxes[anno_idx] if anno_idx < len(bboxes) else None
            point = points[anno_idx] if anno_idx < len(points) else None
            poly = polygons[anno_idx] if anno_idx < len(polygons) else None
            if box is not None or (poly is not None and len(poly) >= 3) or point is not None:
                out.append((img_idx, anno_idx))
    return out


def select_k_shot_annotations(
    exemplar_images: Sequence[str],
    exemplar_bboxes: Sequence[Sequence],
    exemplar_points: Sequence[Sequence],
    exemplar_polygons: Sequence[Sequence],
    *,
    k_shot: int,
    seed: int = 42,
) -> Tuple[
    List[str],
    List[List],
    List[List],
    List[List],
    List[Tuple[int, int]],
]:
    """
    Sample up to ``k_shot`` annotations (not images) and regroup by media path.

    Returns selected image lists plus the chosen ``(image_idx, anno_idx)`` pairs.
    """
    candidates = annotation_indices(exemplar_bboxes, exemplar_points, exemplar_polygons)
    if not candidates:
        raise ValueError("No exemplar annotations with bbox, polygon, or point prompts found.")
    if k_shot <= 0:
        raise ValueError(f"k_shot must be >= 1; got {k_shot}")

    k = min(len(candidates), int(k_shot))
    rng = random.Random(seed)
    selected = rng.sample(candidates, k=k)

    by_image: Dict[int, List[int]] = {}
    for img_idx, anno_idx in selected:
        by_image.setdefault(img_idx, []).append(anno_idx)

    sel_images: List[str] = []
    sel_bboxes: List[List] = []
    sel_points: List[List] = []
    sel_polygons: List[List] = []
    for img_idx in sorted(by_image):
        anno_idxs = sorted(by_image[img_idx])
        boxes = exemplar_bboxes[img_idx]
        points = exemplar_points[img_idx]
        polygons = exemplar_polygons[img_idx]
        sel_images.append(exemplar_images[img_idx])
        sel_bboxes.append([boxes[j] if j < len(boxes) else None for j in anno_idxs])
        # COCO leaves points empty (no click prompts); keep that rather than padding None.
        if len(points) == 0:
            sel_points.append([])
        else:
            sel_points.append([points[j] if j < len(points) else None for j in anno_idxs])
        sel_polygons.append([polygons[j] if j < len(polygons) else None for j in anno_idxs])

    return sel_images, sel_bboxes, sel_points, sel_polygons, selected


def load_exemplar_annotations(
        annotation_filename: str | Path,
        annotation_file_type: str,
        **kwargs,
) -> Tuple[
    List[str],
    List[List[Optional[Tuple[float, float, float, float]]]],
    List[List[Tuple[float, float]]],
    List[List[Optional[List[Tuple[float, float]]]]],
    List[Tuple[int, int]],
]:
    if annotation_file_type.lower() == "squidle":
        return load_squidle_exemplar_annotations(annotation_filename, **kwargs)

    if annotation_file_type.lower() == "coco":
        return load_coco_exemplar_annotations(annotation_filename, **kwargs)


def load_coco_exemplar_annotations(
        annotation_filename: str,
        exemplar_dir: str,
        bbox_format: str = "xywh",
        scale=1.0,
        **_,
) -> Tuple[
    List[str],  # image_paths
    List[List[Optional[Tuple[float, float, float, float]]]],  # bboxes_by_image
    List[List[Tuple[float, float]]],  # points_by_image (empty lists)
    List[List[Optional[List[Tuple[float, float]]]]],  # polygons_by_image
    List[Tuple[int, int]]  # image_sizes (width, height)
]:
    """
    Load COCO annotations and return image info aligned across lists.

    Returns:
        image_paths: list of image file names (as stored in the COCO "images" array).
        bboxes_by_image: per-image relative xywh bboxes (None when polygon prompt is used).
        points_by_image: for each image, an empty list (COCO has no click points).
        polygons_by_image: relative polygon vertices when segmentation is non-bbox.
        image_sizes: for each image, (width, height) ints from the COCO "images" entries.
    """
    # Read the COCO JSON file
    with open(annotation_filename, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # COCO structure:
    # coco["images"] -> list of {id, file_name, width, height, ...}
    # coco["annotations"] -> list of {image_id, bbox=[x,y,w,h], ...}
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    # Map image_id -> index in our output lists
    image_id_to_index: Dict[int, int] = {}
    image_paths: List[str] = []
    image_sizes: List[Tuple[int, int]] = []
    bboxes_by_image: List[List[Optional[Tuple[float, float, float, float]]]] = []
    points_by_image: List[List[Tuple[float, float]]] = []
    polygons_by_image: List[List[Optional[List[Tuple[float, float]]]]] = []

    # Initialize per-image containers
    for idx, img in enumerate(images):
        img_id = img["id"]
        image_id_to_index[img_id] = idx
        image_paths.append(img.get("file_name", ""))
        image_sizes.append((int(img.get("width", 0)), int(img.get("height", 0))))
        bboxes_by_image.append([])  # will be filled from annotations
        points_by_image.append([])  # intentionally left empty (no points for COCO bboxes)
        polygons_by_image.append([])

    # Collect geometry per image (COCO bbox is [x, y, w, h] in pixels).
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is None:
            continue

        idx = image_id_to_index.get(img_id)
        if idx is None:
            continue

        width, height = image_sizes[idx]
        if width <= 0 or height <= 0:
            continue

        bbox = ann.get("bbox")
        bbox_xywh_rel: Optional[Tuple[float, float, float, float]] = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = bbox
            bbox_xywh_rel = (
                float(x / width),
                float(y / height),
                float(w / width),
                float(h / height),
            )

        polygons_px = _parse_coco_segmentation(ann.get("segmentation"))
        if polygons_px:
            vertices_rel = [(px / width, py / height) for px, py in polygons_px[0]]
            _append_coco_exemplar_geometry(
                bboxes_by_image[idx],
                polygons_by_image[idx],
                vertices_rel,
                bbox_xywh_rel,
                bbox_format=bbox_format,
            )
        elif bbox_xywh_rel is not None:
            bboxes_by_image[idx].append(bbox_xywh_rel)
            polygons_by_image[idx].append(None)

    return image_paths, bboxes_by_image, points_by_image, polygons_by_image, image_sizes


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



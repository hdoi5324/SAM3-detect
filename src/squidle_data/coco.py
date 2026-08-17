"""Build COCO JSON from Squidle annotation and media collection DataFrames."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from squidle_data.columns import (
    COL_FILE_NAME,
    COL_IMAGE_ID,
    COL_LABEL_ID,
    COL_LABEL_NAME,
    COL_POINT_MEDIA_ID,
    COL_POINT_MEDIA_PATH,
    COL_POINT_POLYGON,
    COL_POINT_X,
    COL_POINT_Y,
    COL_URI,
)
from sam3_exemplar.geometry import polygon_area_pixels
from squidle_data.geometry import (
    COL_LIKELIHOOD,
    annotation_bbox_pixels,
    annotation_segmentation_pixels,
)
from squidle_data.cache import image_size_from_uri, resolve_image_cache_dir
from squidle_data.tables import clean_export_columns


def categories_from_annotation_df(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    if df.empty or COL_LABEL_ID not in df.columns:
        return [{"id": 1, "name": "object", "supercategory": "none"}], {}
    sub = df[[COL_LABEL_ID]].dropna().copy()
    if COL_LABEL_NAME in df.columns:
        sub[COL_LABEL_NAME] = df[COL_LABEL_NAME]
    else:
        sub[COL_LABEL_NAME] = ""
    pairs = sub.drop_duplicates(subset=[COL_LABEL_ID]).sort_values(COL_LABEL_ID)
    squidle_to_cat: Dict[int, int] = {}
    categories: List[Dict[str, Any]] = []
    for i, (_, r) in enumerate(pairs.iterrows(), start=1):
        lid = int(r[COL_LABEL_ID])
        squidle_to_cat[lid] = i
        categories.append(
            {"id": i, "name": str(r.get(COL_LABEL_NAME, "")), "supercategory": ""},
        )
    return categories, squidle_to_cat


def build_coco_from_squidle_annotations(
    images_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    sizes_from_collection: Optional[Dict[int, Tuple[int, int]]] = None,
    image_cache_dir=None,
    *,
    exclude_label_ids: Optional[Sequence[int]] = None,
    description: str | None = None,
    include_scores: bool = False,
    include_segmentation: bool = True,
    fixed_categories: Optional[Sequence[Dict[str, Any]]] = None,
    label_to_category_id: Optional[Dict[int, int]] = None,
    allowed_image_ids: Optional[Sequence[int]] = None,
    log_prefix: str = "[squidle_data]",
) -> Dict[str, Any]:
    """
    Build a COCO-style dict from Squidle annotation and collection DataFrames.

    When ``fixed_categories`` and ``label_to_category_id`` are set (prediction export),
    category ids follow the ground-truth map and rows with unmapped labels are skipped.
    """
    df = clean_export_columns(annotations_df)
    source_label = description or "Squidle annotations DataFrame"

    drop_cols = [c for c in df.columns if str(c).startswith("point.pixels")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    required = [COL_POINT_MEDIA_PATH, COL_POINT_X, COL_POINT_Y, COL_LABEL_ID, COL_POINT_MEDIA_ID]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column {col!r}")

    if exclude_label_ids:
        ignore = {int(x) for x in exclude_label_ids}
        df = df[~df[COL_LABEL_ID].isin(ignore)].copy()

    df = df[df[COL_POINT_MEDIA_PATH].notna()].copy()
    poly_col = COL_POINT_POLYGON if COL_POINT_POLYGON in df.columns else None

    for col in (COL_IMAGE_ID, COL_URI, COL_FILE_NAME):
        if col not in images_df.columns:
            raise KeyError(f"images_df missing required column: {col!r}")

    image_rows = images_df.drop_duplicates(subset=[COL_IMAGE_ID], keep="first").copy()
    gt_image_ids = set(int(v) for v in image_rows[COL_IMAGE_ID].tolist())
    if allowed_image_ids is not None:
        allowed = {int(x) for x in allowed_image_ids}
        gt_image_ids &= allowed
        image_rows = image_rows[image_rows[COL_IMAGE_ID].isin(allowed)].copy()

    ann_media_in_df = set(int(x) for x in df[COL_POINT_MEDIA_ID].dropna().astype(int).unique())
    missing_from_collection = sorted(ann_media_in_df - gt_image_ids)
    if missing_from_collection:
        n_orphan_rows = int((~df[COL_POINT_MEDIA_ID].isin(gt_image_ids)).sum())
        preview = missing_from_collection[:15]
        extra = (
            f" … (+{len(missing_from_collection) - len(preview)} more)"
            if len(missing_from_collection) > len(preview)
            else ""
        )
        print(
            f"{log_prefix} warn: annotation rows for media_id(s) not in images_df "
            f"({n_orphan_rows} row(s), {len(missing_from_collection)} distinct id(s)); "
            f"examples: {preview}{extra}"
        )

    df = df[df[COL_POINT_MEDIA_ID].isin(gt_image_ids)].copy()

    provided_sizes = dict(sizes_from_collection or {})
    resolved_size: Dict[int, Tuple[int, int]] = {}

    mids_arr = image_rows[COL_IMAGE_ID].astype(int).to_numpy()
    fns_arr = image_rows[COL_FILE_NAME].astype(str).to_numpy()

    for mid in np.unique(mids_arr):
        mid_i = int(mid)
        if mid_i in provided_sizes:
            resolved_size[mid_i] = provided_sizes[mid_i]

    sub = image_rows.drop_duplicates(subset=[COL_IMAGE_ID])
    cache_dir = resolve_image_cache_dir(image_cache_dir) if image_cache_dir is not None else None
    pending_fetch = [
        (int(m), str(u))
        for m, u in zip(sub[COL_IMAGE_ID].astype(int), sub[COL_URI].astype(str))
        if int(m) not in resolved_size
    ]

    if pending_fetch:
        if cache_dir is None:
            raise ValueError(
                "image_cache_dir is required to resolve image width/height "
                "(Squidle media export does not include pixel dimensions)."
            )

        def _fetch_wh(mid_uri: Tuple[int, str]) -> Tuple[int, Tuple[int, int]]:
            mid, uri = mid_uri
            return mid, image_size_from_uri(uri, cache_dir)

        n_workers = max(1, min(24, len(pending_fetch)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            iterator = ex.map(_fetch_wh, pending_fetch)
            for mid, wh in tqdm(iterator, total=len(pending_fetch), desc="Image sizes"):
                resolved_size[mid] = wh

    images_out = [
        {
            "id": int(m),
            "width": resolved_size[int(m)][0],
            "height": resolved_size[int(m)][1],
            "file_name": str(fn),
        }
        for m, fn in zip(mids_arr, fns_arr)
    ]

    categories: List[Dict[str, Any]]
    squidle_label_to_cat: Dict[int, int]

    if fixed_categories is not None and label_to_category_id is not None:
        categories = list(fixed_categories)
        squidle_label_to_cat = dict(label_to_category_id)
        df = df[df[COL_LABEL_ID].isin(squidle_label_to_cat.keys())].copy()
    else:
        categories, squidle_label_to_cat = categories_from_annotation_df(df)

    anns: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        mid = int(row[COL_POINT_MEDIA_ID])
        w, h = resolved_size[mid]
        lid = int(row[COL_LABEL_ID])
        cat_id = squidle_label_to_cat.get(lid)
        if cat_id is None:
            continue

        bbox = annotation_bbox_pixels(
            row, w, h, poly_col=poly_col, x_col=COL_POINT_X, y_col=COL_POINT_Y
        )
        segmentation = (
            annotation_segmentation_pixels(
                row, w, h, poly_col=poly_col, x_col=COL_POINT_X, y_col=COL_POINT_Y
            )
            if include_segmentation
            else []
        )
        if segmentation:
            verts = [
                (segmentation[0][j], segmentation[0][j + 1])
                for j in range(0, len(segmentation[0]), 2)
            ]
            area = polygon_area_pixels(verts)
        else:
            area = float(bbox[2] * bbox[3])

        rec: Dict[str, Any] = {
            "id": i,
            "image_id": mid,
            "category_id": cat_id,
            "bbox": [float(b) for b in bbox],
            "area": area,
            "iscrowd": 0,
            "segmentation": segmentation,
        }
        if include_scores:
            score = row.get(COL_LIKELIHOOD, 1.0)
            if COL_LIKELIHOOD in row.index and pd.notna(score):
                rec["score"] = float(score)
            elif include_scores:
                rec["score"] = 1.0
        anns.append(rec)

    return {
        "info": {"description": source_label},
        "licenses": [],
        "categories": categories,
        "images": images_out,
        "annotations": anns,
    }

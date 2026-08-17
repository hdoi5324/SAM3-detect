"""Build SAM3 exemplar dicts from Squidle annotation exports (image + WMS media)."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sqapi_contrib.media_image import fetch_media_image
from sqapi_contrib.media_metadata import parse_media_data
from sqapi_contrib.wms import get_wms_bbox_for_point, load_wms_tile_image
from sqbot_framework.handlers import WmsConfig
from sqbot_framework.handlers.coordinates import LonLatBbox, wms_storage_polygon_to_tile_norm
from sqbot_framework.handlers.tile_quality import assess_tile_rgb

from sam3_exemplar.exemplar_data import (
    append_negative_geometry_lists,
    append_positive_geometry_to_entry,
    get_or_create_media_entry,
    squidle_row_norm_geometry,
)
from squidle_data.geometry import append_exemplar_geometry

DEFAULT_WMS_EXEMPLAR_CACHE_DIR = Path(".cache/wms_exemplar_tiles")
DEFAULT_IMAGE_CACHE_DIR = Path(".cache/exemplar_seg_images")

from squidle_data.media import (
    filter_annotations_to_image_media as filter_exemplar_df_to_image_media,
    is_squidle_image_media,
    squidle_media_type_name,
)

__all__ = [
    "DEFAULT_WMS_EXEMPLAR_CACHE_DIR",
    "DEFAULT_IMAGE_CACHE_DIR",
    "build_exemplars_by_label_id_from_df",
    "fetch_and_cache_wms_exemplar_tile",
    "filter_exemplar_df_to_image_media",
    "is_squidle_image_media",
    "media_layer_params",
    "parse_wms_polygon_cell",
    "squidle_media_type_name",
    "wms_exemplar_cache_path",
    "wms_row_to_norm_geometry",
]


# --- WMS exemplar tile fetch ---


def wms_exemplar_cache_path(
    media_id: int,
    point_id: int,
    *,
    cache_dir: Path | None = None,
) -> Path:
    root = cache_dir or DEFAULT_WMS_EXEMPLAR_CACHE_DIR
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root / f"{int(media_id)}_{int(point_id)}.jpg"


def _fetch_wms_tile_pil(
    sqapi,
    media_id: int,
    bbox_lonlat: LonLatBbox,
    *,
    config: WmsConfig,
    media_url: str | None,
    layer_params: dict | None,
    size_px: int,
) -> Image.Image:
    if config.fetch_via_api:
        return fetch_media_image(sqapi, int(media_id), bbox_lonlat, size=size_px)
    if not media_url:
        raise ValueError("media_url required when fetch_via_api is False")
    return load_wms_tile_image(
        media_url,
        bbox_lonlat,
        size=size_px,
        scale_to_bbox=True,
        layer_params=layer_params or {},
    )


def fetch_and_cache_wms_exemplar_tile(
    sqapi,
    media_id: int,
    point_id: int,
    anchor_lon: float,
    anchor_lat: float,
    *,
    config: WmsConfig,
    media_url: str | None = None,
    layer_params: dict | None = None,
    cache_dir: Path | None = None,
) -> tuple[Path, int, int, LonLatBbox] | None:
    """Return ``(cache_path, width, height, bbox_lonlat)`` for a point-centred exemplar tile."""
    cache_path = wms_exemplar_cache_path(media_id, point_id, cache_dir=cache_dir)
    bbox = get_wms_bbox_for_point(lat=anchor_lat, lon=anchor_lon, size_m=config.point_size_m)

    if cache_path.is_file():
        with Image.open(cache_path) as im:
            w, h = im.size
        return cache_path, w, h, bbox

    pil = _fetch_wms_tile_pil(
        sqapi,
        media_id,
        bbox,
        config=config,
        media_url=media_url,
        layer_params=layer_params,
        size_px=config.tile_size_px,
    )
    rgb = pil.convert("RGB")
    if config.skip_blank_tiles and assess_tile_rgb(np.asarray(rgb)).is_blank:
        print(f"  skip blank WMS exemplar tile for point {point_id}")
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(cache_path, quality=92)
    return cache_path, rgb.width, rgb.height, bbox


def parse_wms_polygon_cell(poly_val) -> list[list[float]]:
    if poly_val is None or (isinstance(poly_val, float) and np.isnan(poly_val)):
        return []
    if isinstance(poly_val, str):
        poly_val = ast.literal_eval(poly_val)
    return [[float(p[0]), float(p[1])] for p in poly_val]


def wms_row_to_norm_geometry(
    anchor_lon: float,
    anchor_lat: float,
    polygon_storage: list[list[float]],
    bbox_lonlat: LonLatBbox,
    tile_width: int,
    tile_height: int,
) -> tuple[float, float, list[list[float]]]:
    """WMS Squidle storage → normalized tile coords for SAM3 exemplar prompts."""
    anchor_x, anchor_y, vertices = wms_storage_polygon_to_tile_norm(
        polygon_storage,
        anchor_lon,
        anchor_lat,
        bbox_lonlat,
        tile_width,
        tile_height,
    )
    return anchor_x, anchor_y, vertices


def media_layer_params(media_meta: dict | None) -> dict:
    if not media_meta:
        return {}
    return parse_media_data(media_meta).get("layer_params") or {}


# --- Squidle annotation → exemplar dict ---


def _append_exemplar_geometry(bboxes, points, polygons, x, y, polygon):
    append_exemplar_geometry(bboxes, points, polygons, x, y, polygon)
    points[-1] = [float(x), float(y)]


def _exemplar_key_is_image(key: str) -> bool:
    return key.startswith("http://") or key.startswith("https://")


def _row_point_id(row) -> int | None:
    point_id = pd.to_numeric(row.get("point.id"), errors="coerce")
    if pd.isna(point_id):
        point_id = pd.to_numeric(row.get("id"), errors="coerce")
    if pd.isna(point_id):
        return None
    return int(point_id)


def _potential_negative_count(df: pd.DataFrame, label_id, media_path: str) -> int:
    if "label.id" not in df.columns or "point.media.path_best" not in df.columns:
        return 0
    return int(
        ((df["label.id"] != label_id) & (df["point.media.path_best"] == media_path)).sum()
    )


def _annotation_rank_key(ann: dict, df: pd.DataFrame, label_id, n_negative: int) -> tuple:
    path = ann.get("path")
    if path is not None:
        return (_potential_negative_count(df, label_id, path) >= n_negative,)
    return (False,)


def _collect_exemplar_annotations(df_label: pd.DataFrame, sqapi) -> list[dict]:
    """One candidate per Squidle annotation (point/polygon), never per image."""
    annotations: list[dict] = []

    for _, row in df_label.iterrows():
        media_id = pd.to_numeric(row.get("point.media.id"), errors="coerce")
        if pd.isna(media_id):
            continue
        media_type = squidle_media_type_name(sqapi, int(media_id))

        if media_type == "image":
            x, y, polygon = squidle_row_norm_geometry(
                row["point.x"], row["point.y"], row["point.polygon"], clip=False
            )
            if polygon is None:
                continue
            annotations.append(
                {
                    "path": str(row["point.media.path_best"]),
                    "row": row,
                    "x": x,
                    "y": y,
                    "polygon": polygon,
                }
            )
            continue

        if media_type != "wms":
            continue
        if not parse_wms_polygon_cell(row.get("point.polygon")):
            continue
        if _row_point_id(row) is None:
            continue
        annotations.append({"path": None, "row": row})

    return annotations


def _select_exemplar_annotations(
    annotations: list[dict],
    *,
    k_shot: int | None,
    df: pd.DataFrame,
    label_id,
    n_negative: int,
) -> list[dict]:
    if not annotations:
        return []
    ranked = sorted(
        annotations,
        key=lambda ann: _annotation_rank_key(ann, df, label_id, n_negative),
        reverse=True,
    )
    if k_shot is None or k_shot <= 0:
        return ranked
    return ranked[: min(int(k_shot), len(ranked))]


def build_exemplars_by_label_id_from_df(
    sqapi,
    df: pd.DataFrame,
    *,
    wms_config: WmsConfig | None = None,
    label_ids_to_ignore: Sequence[int] = (),
    label_ids: Optional[Sequence[int]] = None,
    n_negative: int = 0,
    k_shot: int | None = None,
) -> Dict[int, Dict[str, list]]:
    """
    Build ``exemplars_by_label_id`` from up to ``k_shot`` annotations per label.

    Dict keys are loadable image paths: image URLs or cached WMS tile paths.
    Multiple selected annotations that share a path are grouped under one media entry
    for encoding only; selection itself is always annotation-level (k-shot).
    """
    ignore = set(label_ids_to_ignore or [])
    if df.empty:
        return {}

    if label_ids:
        lids = {int(x) for x in label_ids}
        df = df[df["label.id"].notna() & df["label.id"].isin(lids)]

    media_meta_cache: dict[int, dict] = {}
    media_url_cache: dict[int, str | None] = {}
    layer_params_cache: dict[int, dict] = {}

    def _media_meta(media_id: int) -> dict:
        if media_id not in media_meta_cache:
            media_meta_cache[media_id] = sqapi.get(f"/api/media/{int(media_id)}").execute().json()
        return media_meta_cache[media_id]

    exemplars_by_label_id: Dict[int, Dict[str, list]] = defaultdict(dict)

    for label_id, df_label in df.groupby("label.id", sort=False, dropna=False):
        if label_id in ignore:
            continue
        if label_ids and int(label_id) not in {int(x) for x in label_ids}:
            continue

        annotations = _collect_exemplar_annotations(df_label, sqapi)
        selected = _select_exemplar_annotations(
            annotations,
            k_shot=k_shot,
            df=df,
            label_id=label_id,
            n_negative=n_negative,
        )
        if k_shot and len(annotations) > len(selected):
            print(
                f"\nLabel: {label_id} | Using {len(selected)}/{len(annotations)} "
                f"annotation(s) (k_shot={k_shot})"
            )
        else:
            print(
                f"\nLabel: {label_id} | Subrows: {len(df_label)} | "
                f"Annotations: {len(selected)}"
            )

        for ann in selected:
            row = ann["row"]
            if ann.get("path") is not None:
                key = ann["path"]
                entry = get_or_create_media_entry(exemplars_by_label_id[label_id], key)
                append_positive_geometry_to_entry(
                    entry, ann["x"], ann["y"], ann["polygon"], clip=False
                )
                continue

            if wms_config is None:
                continue

            media_id = int(pd.to_numeric(row["point.media.id"]))
            point_id = _row_point_id(row)
            if point_id is None:
                continue

            anchor_lon, anchor_lat = float(row["point.x"]), float(row["point.y"])
            polygon_storage = parse_wms_polygon_cell(row.get("point.polygon"))
            if not polygon_storage:
                continue

            if media_id not in media_url_cache:
                meta = _media_meta(media_id)
                media_url_cache[media_id] = meta.get("path_best")
                layer_params_cache[media_id] = media_layer_params(meta)

            fetched = fetch_and_cache_wms_exemplar_tile(
                sqapi,
                media_id,
                point_id,
                anchor_lon,
                anchor_lat,
                config=wms_config,
                media_url=media_url_cache[media_id],
                layer_params=layer_params_cache[media_id],
            )
            if fetched is None:
                continue

            cache_path, tile_w, tile_h, bbox = fetched
            key = str(cache_path.resolve())
            x, y, polygon = wms_row_to_norm_geometry(
                anchor_lon,
                anchor_lat,
                polygon_storage,
                bbox,
                tile_w,
                tile_h,
            )
            entry = get_or_create_media_entry(exemplars_by_label_id[label_id], key)
            append_positive_geometry_to_entry(entry, x, y, polygon, clip=False)
            print(f"  WMS exemplar point {point_id} → {key}")

    if n_negative > 0:
        image_df = df[
            df["point.media.id"].apply(
                lambda mid: not pd.isna(mid) and is_squidle_image_media(sqapi, int(mid))
            )
        ]
        for label_id, label_data in exemplars_by_label_id.items():
            for key, media_data in label_data.items():
                if not _exemplar_key_is_image(key):
                    continue
                neg_data = image_df.loc[
                    (image_df["label.id"] != label_id)
                    & (image_df["point.media.path_best"] == key)
                ]
                if neg_data.empty:
                    continue
                neg_data = neg_data.sample(n=min(n_negative, len(neg_data)), random_state=42)
                neg_bboxes, neg_points, neg_polygons = [], [], []
                for _, a in neg_data.iterrows():
                    x, y, polygon = squidle_row_norm_geometry(
                        a["point.x"], a["point.y"], a["point.polygon"], clip=False
                    )
                    if polygon is None:
                        continue
                    _append_exemplar_geometry(neg_bboxes, neg_points, neg_polygons, x, y, polygon)
                append_negative_geometry_lists(media_data, neg_bboxes, neg_points, neg_polygons)

    for label_id, inner_dict in exemplars_by_label_id.items():
        sorted_items = sorted(
            inner_dict.items(),
            key=lambda x: (len(x[1][2]) >= n_negative, len(x[1][0])),
            reverse=True,
        )
        exemplars_by_label_id[label_id] = dict(sorted_items)

    return dict(exemplars_by_label_id)

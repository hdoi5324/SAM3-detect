"""Fetch Squidle annotation sets and media collections via sqapi."""

from __future__ import annotations

import io
import os
import pathlib

import pandas as pd
from dotenv import load_dotenv
from sqapi.api import SQAPI

# Columns for api/annotation_set/{id}/export.
ANNOTATION_SET_EXPORT_COLUMNS = [
    "label.id",
    "label.uuid",
    "label.name",
    "label.lineage_names",
    "comment",
    "needs_review",
    "tag_names",
    "updated_at",
    "point.id",
    "point.x",
    "point.y",
    "point.t",
    "point.is_targeted",
    "point.media.id",
    "point.media.key",
    "point.media.path_best",
    "point.polygon",
    "likelihood",
]

# Alias used by coco_export re-exports.
ANNOTATION_EXPORT_COLUMNS = ANNOTATION_SET_EXPORT_COLUMNS

# Legacy name — same columns as annotation-set export.
EXPORT_COLUMNS = ANNOTATION_SET_EXPORT_COLUMNS

MEDIA_EXPORT_COLUMNS = [
    "id",
    "key",
    "path_best",
    "pixel_width",
    "pixel_height",
]

_DEFAULT_FILEOPS = [
    dict(module="pandas", method="json_normalize"),
    dict(method="sort_index", kwargs=dict(axis=1)),
]


def connect_sqapi(
    *,
    api_key: str | None = None,
    host: str | None = None,
    sqapi=None,
    env_path: pathlib.Path | None = None,
):
    """Build or reuse an authenticated SQAPI client."""
    if sqapi is not None:
        return sqapi

    if env_path is not None:
        load_dotenv(env_path)
    elif api_key is None:
        load_dotenv()

    key = api_key or os.environ.get("SQUIDLE_API_TOKEN") or os.environ.get("SQUIDLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Pass api_key= or set SQUIDLE_API_TOKEN / SQUIDLE_API_KEY in the environment."
        )

    kwargs = dict(api_key=key)
    if host is not None:
        kwargs["host"] = host
    return SQAPI(**kwargs)


def export_endpoint_to_df(
    sqapi,
    endpoint: str,
    *,
    include_columns: list[str],
    filters: list[dict],
) -> pd.DataFrame:
    """Run a Squidle export endpoint and return a DataFrame."""
    request = sqapi.export(
        endpoint,
        include_columns=include_columns,
        filters=filters,
        fileops=_DEFAULT_FILEOPS,
        qsparams=dict(template="dataframe.csv", disposition="attachment"),
    )
    result = request.execute()
    text = result.content.decode("utf-8")
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


def get_media_collection_id(sqapi, annotation_set_id: int) -> int:
    payload = sqapi.get(f"/api/annotation_set/{int(annotation_set_id)}").execute().json()
    media_collection = payload.get("media_collection") or {}
    media_collection_id = media_collection.get("id")
    if media_collection_id is None:
        raise ValueError(
            f"annotation_set_id={annotation_set_id} has no media_collection.id in Squidle API response"
        )
    return int(media_collection_id)


def _annotation_set_export_filters(
    *,
    label_ids: list[int] | None = None,
    label_ids_to_ignore: list[int] | None = None,
    point_has_xy: bool = True,
    require_polygon: bool = False,
    label_id_is_not_null: bool = True,
) -> list[dict]:
    filters: list[dict] = []
    if point_has_xy:
        filters.append(dict(name="point", op="has", val=dict(name="has_xy", op="eq", val=True)))
    if require_polygon:
        filters.append(
            dict(name="point", op="has", val=dict(name="has_polygon", op="eq", val=True))
        )
    if label_id_is_not_null:
        filters.append(dict(name="label_id", op="is_not_null"))

    ignore = list(label_ids_to_ignore or [])
    if ignore:
        filters.append(dict(name="label_id", op="not_in", val=ignore))
    if label_ids:
        filters.append(dict(name="label_id", op="in", val=list(label_ids)))
    return filters


def load_squidle_annotations_to_df(
    sqapi,
    annotation_set_id: int,
    label_ids_to_ignore: list[int] | None = None,
    *,
    label_ids: list[int] | None = None,
    require_polygon: bool = False,
) -> pd.DataFrame:
    """Load annotations from a Squidle annotation set via set-scoped export."""
    return export_endpoint_to_df(
        sqapi,
        f"api/annotation_set/{int(annotation_set_id)}/export",
        include_columns=ANNOTATION_SET_EXPORT_COLUMNS,
        filters=_annotation_set_export_filters(
            label_ids=label_ids,
            label_ids_to_ignore=label_ids_to_ignore,
            require_polygon=require_polygon,
        ),
    )


def load_squidle_media_collection_to_df(sqapi, media_collection_id: int) -> pd.DataFrame:
    """Load the full media list for a Squidle media collection."""
    return export_endpoint_to_df(
        sqapi,
        f"api/media_collection/{int(media_collection_id)}/export",
        include_columns=MEDIA_EXPORT_COLUMNS,
        filters=[],
    )


def load_squidle_annotation_set_and_media(
    sqapi,
    annotation_set_id: int,
    *,
    label_ids: list[int] | None = None,
    label_ids_to_ignore: list[int] | None = None,
    require_polygon: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load annotation rows and the parent media collection for an annotation set."""
    media_collection_id = get_media_collection_id(sqapi, annotation_set_id)
    ann_df = load_squidle_annotations_to_df(
        sqapi,
        annotation_set_id,
        label_ids_to_ignore=label_ids_to_ignore,
        label_ids=label_ids,
        require_polygon=require_polygon,
    )
    media_df = load_squidle_media_collection_to_df(sqapi, media_collection_id)
    return ann_df, media_df

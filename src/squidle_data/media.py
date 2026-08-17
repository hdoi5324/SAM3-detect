"""Squidle media type helpers (image / WMS / video discrimination)."""

from __future__ import annotations

import pandas as pd

from squidle_data.columns import COL_POINT_MEDIA_ID

_MEDIA_TYPE_CACHE: dict[int, str | None] = {}


def squidle_media_type_name(sqapi, media_id: int) -> str | None:
    """Fetch ``media_type.name`` for a Squidle media id (cached)."""
    mid = int(media_id)
    if mid not in _MEDIA_TYPE_CACHE:
        meta = sqapi.get(f"/api/media/{mid}").execute().json()
        _MEDIA_TYPE_CACHE[mid] = (meta.get("media_type") or {}).get("name")
    return _MEDIA_TYPE_CACHE[mid]


def is_squidle_image_media(sqapi, media_id: int) -> bool:
    return squidle_media_type_name(sqapi, media_id) == "image"


def filter_annotations_to_image_media(df: pd.DataFrame, sqapi) -> pd.DataFrame:
    """Keep annotation rows whose point media is ``image`` (via ``/api/media/<id>``)."""
    if df.empty:
        return df
    if COL_POINT_MEDIA_ID not in df.columns:
        raise ValueError(f"{COL_POINT_MEDIA_ID!r} is required to filter by image media")
    row_ids = pd.to_numeric(df[COL_POINT_MEDIA_ID], errors="coerce")
    unique_ids = row_ids.dropna().astype(int).unique()
    image_ids = {int(mid) for mid in unique_ids if is_squidle_image_media(sqapi, int(mid))}
    out = df[row_ids.isin(image_ids)].copy()
    dropped = len(df) - len(out)
    if dropped:
        print(
            f"[squidle_data] image-media filter: kept {len(out)}/{len(df)} annotation(s) "
            f"({dropped} on non-image media excluded)"
        )
    return out


__all__ = [
    "filter_annotations_to_image_media",
    "is_squidle_image_media",
    "squidle_media_type_name",
]

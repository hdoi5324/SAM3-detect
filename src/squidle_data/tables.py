"""Load Squidle annotation / collection tables from CSV exports."""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple
from urllib.parse import urlparse

import pandas as pd

from squidle_data.columns import (
    COL_COLLECTION_ID,
    COL_FILE_NAME,
    COL_IMAGE_ID,
    COL_PIXEL_HEIGHT,
    COL_PIXEL_WIDTH,
    COL_POINT_MEDIA_ID,
    COL_POINT_MEDIA_PATH,
    COL_URI,
)


def clean_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop = [c for c in out.columns if str(c).startswith("Unnamed")]
    if drop:
        out = out.drop(columns=drop, errors="ignore")
    return out


def load_table(path: pathlib.Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return clean_export_columns(pd.read_parquet(path))
    if suf in {".csv", ".tsv"}:
        sep = "\t" if suf == ".tsv" else ","
        return clean_export_columns(pd.read_csv(path, sep=sep))
    raise ValueError(f"Unsupported table format: {path} (use .csv, .tsv, or .parquet)")


def file_name_from_uri(uri: str) -> str:
    s = str(uri).strip()
    if not s:
        return "image.jpg"
    if s.startswith("http://") or s.startswith("https://"):
        name = pathlib.Path(urlparse(s).path).name
    else:
        name = pathlib.Path(s).name
    return name if name and name not in (".", "..") else "image.jpg"


def normalize_squidle_collection_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Squidle collection export for COCO image metadata."""
    out = df.copy()
    if COL_IMAGE_ID not in out.columns:
        id_col = COL_COLLECTION_ID if COL_COLLECTION_ID in out.columns else COL_POINT_MEDIA_ID
        if id_col not in out.columns:
            raise KeyError(
                f"Expected '{COL_COLLECTION_ID}', '{COL_POINT_MEDIA_ID}', or '{COL_IMAGE_ID}' "
                "in collection dataframe"
            )
        out[COL_IMAGE_ID] = pd.to_numeric(out[id_col], errors="coerce")
        if out[COL_IMAGE_ID].isna().any():
            raise ValueError("Non-numeric media id in collection table")
        out[COL_IMAGE_ID] = out[COL_IMAGE_ID].astype(int)

    uri_col = COL_URI if COL_URI in out.columns else COL_POINT_MEDIA_PATH
    if COL_FILE_NAME not in out.columns:
        if uri_col not in out.columns:
            raise KeyError(f"Need '{COL_URI}' or '{COL_POINT_MEDIA_PATH}' to build '{COL_FILE_NAME}'")
        out[COL_FILE_NAME] = out[uri_col].map(
            lambda u: file_name_from_uri("" if pd.isna(u) else str(u))
        )

    if COL_URI not in out.columns and COL_POINT_MEDIA_PATH in out.columns:
        out[COL_URI] = out[COL_POINT_MEDIA_PATH]

    return out


def media_size_lookup_from_collection_df(df: pd.DataFrame) -> Dict[int, Tuple[int, int]]:
    if COL_PIXEL_WIDTH not in df.columns or COL_PIXEL_HEIGHT not in df.columns:
        return {}
    if COL_IMAGE_ID not in df.columns:
        return {}
    out: Dict[int, Tuple[int, int]] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get(COL_PIXEL_WIDTH)) or pd.isna(row.get(COL_PIXEL_HEIGHT)):
            continue
        out[int(row[COL_IMAGE_ID])] = (int(row[COL_PIXEL_WIDTH]), int(row[COL_PIXEL_HEIGHT]))
    return out

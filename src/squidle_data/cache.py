"""Image fetch/cache helpers for Squidle media URIs."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse
import urllib.request

from PIL import Image
from sqapi.media import SQMediaObject

_IMAGE_CACHE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_remote_uri(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def cache_path_for_url(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    ext = Path(urlparse(url).path).suffix.lower()
    if ext not in _IMAGE_CACHE_EXTS:
        ext = ".jpg"
    return cache_dir / f"{digest}{ext}"


def fetch_url_to_cache(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_path_for_url(url, cache_dir)
    if dest.is_file():
        return dest

    req = urllib.request.Request(url, headers={"User-Agent": "squidle-data/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    tmp = dest.with_name(dest.name + ".part")
    with tmp.open("wb") as f:
        f.write(data)
    tmp.replace(dest)
    return dest


def resolve_image_cache_dir(cache_dir: Path | str | None) -> Path:
    if cache_dir is None:
        return Path(".cache/squidle_images").resolve()
    p = Path(cache_dir).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def open_pil_from_uri_or_path(uri_or_path: str, image_cache_dir: Path | None = None) -> Image.Image:
    if is_remote_uri(uri_or_path):
        if image_cache_dir is not None:
            local = fetch_url_to_cache(uri_or_path, image_cache_dir)
            with Image.open(local) as im:
                return im.convert("RGB")
        req = urllib.request.Request(uri_or_path, headers={"User-Agent": "squidle-data/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")

    p = Path(uri_or_path)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    with Image.open(p) as im:
        return im.convert("RGB")


def image_size_from_uri(uri: str, image_cache_dir: Path | None = None) -> Tuple[int, int]:
    try:
        img = open_pil_from_uri_or_path(uri, image_cache_dir)
        try:
            return int(img.size[0]), int(img.size[1])
        finally:
            try:
                img.close()
            except Exception:
                pass
    except Exception:
        m = SQMediaObject(uri)
        m.data()
        return int(m.width), int(m.height)

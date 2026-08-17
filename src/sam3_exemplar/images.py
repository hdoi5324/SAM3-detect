"""Image loading and inference autocast helpers for SAM3 exemplar workflows."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from sqapi import SQMediaObject

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".heic", ".svg",
}
Kind = Literal["url_image_like", "url_non_image_like", "path_image_like", "path_non_image_like"]


@contextmanager
def autocast_scope(
    device: str = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    enabled: bool = True,
):
    """Scoped autocast for CUDA/CPU. No-ops if disabled or CUDA is unavailable."""
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if not enabled or dtype is None:
        yield
    elif use_cuda:
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        with torch.autocast(device_type="cpu", dtype=dtype):
            yield


def numpy_bgr_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """Convert BGR/RGBA buffers (e.g. from ``SQMediaObject``) to PIL RGB."""
    x = np.asarray(arr)
    if x.ndim == 2:
        return Image.fromarray(x, mode="L").convert("RGB")
    if x.ndim != 3:
        raise ValueError(f"Unexpected image array shape: {x.shape}")
    c = x.shape[2]
    if c == 3:
        rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    elif c == 4:
        rgb = cv2.cvtColor(x, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unexpected channel count: {c}")
    return Image.fromarray(rgb)


def classify_image_string(s: str) -> Kind:
    """Classify a string as URL or path and image-likeness by suffix."""
    p = urlparse(s)
    if p.scheme:
        web_schemes = {"http", "https"}
        if p.scheme in web_schemes and not p.netloc:
            return "url_non_image_like"
        ext = Path(p.path).suffix.lower()
        return "url_image_like" if ext in IMAGE_EXTS else "url_non_image_like"
    ext = Path(s).suffix.lower()
    return "path_image_like" if ext in IMAGE_EXTS else "path_non_image_like"


def load_image(path_or_url, images_dir=None) -> Image.Image:
    kind = classify_image_string(path_or_url)
    if kind == "url_image_like":
        sq_media = SQMediaObject(path_or_url)
        return numpy_bgr_to_pil_rgb(sq_media.data())
    full_path = Path(images_dir if images_dir is not None else "./", path_or_url)
    if classify_image_string(str(full_path)) == "path_image_like":
        return Image.open(full_path).convert("RGB")
    raise FileNotFoundError(f"Not an image path or URL: {path_or_url!r}")


def maybe_downscale_pil(image: Image.Image, max_side: int | None) -> Image.Image:
    """Downscale so the longest side is at most ``max_side``; no-op if unset or smaller."""
    if max_side is None:
        return image
    max_side = int(max_side)
    if max_side <= 0:
        raise ValueError(f"max_side must be positive, got {max_side}")
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.BILINEAR)

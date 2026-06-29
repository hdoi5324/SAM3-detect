from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Literal, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from sqapi import SQMediaObject


@contextmanager
def autocast_scope(
    device: str = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,  # None → disable autocast
    enabled: bool = True
):
    """
    Scoped autocast for CUDA/CPU. No-ops if disabled or if CUDA is unavailable for 'cuda' device.
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if not enabled or dtype is None:
        # Completely disabled
        yield
    elif use_cuda:
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        # Optional: support CPU autocast (PyTorch supports it, but speedups vary)
        # If you don't want CPU autocast, just `yield` here instead.
        with torch.autocast(device_type="cpu", dtype=dtype):
            yield


def numpy_bgr_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """
    ``SQMediaObject.data()`` (and typical OpenCV-style buffers) use BGR order; PIL/SAM3
    expect RGB. Match ``sam3_segment_bot/sam3_segmentation.py`` which uses
    ``cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`` before ``set_image``.
    """
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


def load_image(path_or_url, images_dir=None):
    type_of_path = classify_image_string(path_or_url)
    if type_of_path == "url_image_like":
        sq_media = SQMediaObject(path_or_url)
        return numpy_bgr_to_pil_rgb(sq_media.data())
    else:
        full_path = Path(images_dir if images_dir is not None else "./", path_or_url)
        type_of_full_path = classify_image_string(str(full_path))
        if type_of_full_path == "path_image_like":
            return Image.open(full_path).convert("RGB")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp",
              ".bmp", ".tiff", ".tif", ".heic", ".svg"}
Kind = Literal["url_image_like", "url_non_image_like", "path_image_like", "path_non_image_like"]


def classify_image_string(s: str) -> Tuple[Kind, bool]:
    """
    Classify a string as URL-or-Path and image-likeness by suffix only.
    Returns (kind, is_image_like).
    """
    p = urlparse(s)

    # Case A: Looks like a URL (has a scheme and, for web URLs, a netloc)
    if p.scheme:
        web_schemes = {"http", "https"}
        if p.scheme in web_schemes and not p.netloc:
            # Something like 'http:/foo' is malformed; treat as non-image URL-ish
            return "url_non_image_like"
        # Check the path part's extension for image-ness
        ext = Path(p.path).suffix.lower()
        is_img = ext in IMAGE_EXTS
        return "url_image_like" if is_img else "url_non_image_like"

    # Case B: No scheme -> treat as filesystem path
    path = Path(s)
    ext = path.suffix.lower()
    is_img = ext in IMAGE_EXTS
    return "path_image_like" if is_img else "path_non_image_like"

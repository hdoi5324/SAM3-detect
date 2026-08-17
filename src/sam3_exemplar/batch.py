from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Union, Optional, Any, Dict

from PIL import Image

from sam3_exemplar.viz import save_heatmap_per_image

# -------------------------------
# Helpers: listing and sampling
# -------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_paths(
    images_dir: Union[str, Path],
    sample_every_n: int = 1,
    recursive: bool = False,
) -> Iterator[Path]:
    """
    Yield every nth image path in sorted order from images_dir.
    """
    root = Path(images_dir)
    print(os.getcwd())
    if not root.exists():
        raise FileNotFoundError(f"Images directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")

    if recursive:
        candidates = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    if sample_every_n <= 0:
        raise ValueError("sample_every_n must be >= 1")

    # Take every nth file
    for idx, p in enumerate(candidates):
        if idx % sample_every_n == 0:
            yield p


# -------------------------------
# Transform: center crop (square or rectangular)
# -------------------------------
def center_crop(
    img: Image.Image,
    crop_size: Union[int, Tuple[int, int]],
) -> Image.Image:
    """
    Center-crop a PIL image to (h, w) or square `crop_size`.
    If crop_size is larger than the image on any side, it will be clamped to the image size.
    """
    W, H = img.size
    if isinstance(crop_size, int):
        cw = ch = crop_size
    else:
        ch, cw = crop_size  # expect (h, w)

    # Clamp to image size
    ch = min(ch, H)
    cw = min(cw, W)

    left = (W - cw) // 2
    top = (H - ch) // 2
    right = left + cw
    bottom = top + ch
    return img.crop((left, top, right, bottom))


# -------------------------------
# Main processing pipeline
# -------------------------------
def process_images_with_center_crop(
    images_dir: Union[str, Path],
    segmenter,  # instance of ExemplarSeg or subclass with segment_image(image, **kwargs)
    sample_every_n: int = 1,
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,  # <-- now Optional
    convert_mode: Optional[str] = "RGB",
    recursive: bool = False,
    segment_image_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Iterate images, sample every n-th, center-crop, and apply `segmenter.segment_image`.

    Args:
        images_dir: folder with images
        segmenter: object exposing `segment_image(image_or_path, **kwargs)`; typically your ExemplarSeg subclass
        sample_every_n: take every n-th image (sorted)
        crop_size: int for square crop, or (height, width)
        convert_mode: optional PIL convert mode (e.g., "RGB"); set to None to keep original
        recursive: whether to search subdirectories
        segment_image_kwargs: extra kwargs passed to `segmenter.segment_image(...)`

    Returns:
        A list of results from `segment_image` per processed image.
    """
    results: List[Any] = []
    segment_image_kwargs = segment_image_kwargs or {}

    for img_path in iter_image_paths(images_dir, sample_every_n=sample_every_n, recursive=recursive):
        try:
            with Image.open(img_path) as img:
                if convert_mode:
                    img = img.convert(convert_mode)

                if crop_size is not None:
                    processed_img = center_crop(img, crop_size)
                else:
                    processed_img = img  # no crop applied

                heatmap = segmenter.segment_image(
                    image_path=str(img_path),           # keep original path for traceability
                    image=processed_img,                # pass PIL image in-memory
                    original_size=img.size,             # (W, H) before crop
                    crop_size=(processed_img.size if crop_size is not None else None),
                    **segment_image_kwargs
                )

        except Exception as e:
            # Log and continue; you can raise if you prefer hard-fail
            print(f"[warn] Failed processing {img_path}: {e}")
            continue

    return results
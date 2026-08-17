"""Lift GT bboxes to SAM3-derived polygons for support exemplar encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam3.model.box_ops import box_xywh_to_cxcywh


def select_best_mask_index(scores: torch.Tensor) -> int | None:
    """Return index of the highest score, or None if ``scores`` is empty."""
    if scores is None:
        return None
    flat = scores.reshape(-1)
    if flat.numel() == 0:
        return None
    return int(flat.argmax().item())


def _mask_to_numpy_2d(mask: torch.Tensor) -> np.ndarray:
    arr = mask.detach().cpu().numpy()
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    return np.asarray(arr)


def _box_xywh_norm_to_xyxy_px(
    box_xywh_norm: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, bw, bh = [float(v) for v in box_xywh_norm]
    x0 = int(round(x * width))
    y0 = int(round(y * height))
    x1 = int(round((x + bw) * width))
    y1 = int(round((y + bh) * height))
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def mask_fraction_inside_bbox(
    mask_2d: np.ndarray,
    box_xywh_norm: Sequence[float],
) -> float:
    """Fraction of foreground mask pixels that lie inside the normalized xywh box."""
    m = np.asarray(mask_2d).astype(bool)
    if m.ndim != 2 or not m.any():
        return 0.0
    h, w = m.shape
    x0, y0, x1, y1 = _box_xywh_norm_to_xyxy_px(box_xywh_norm, w, h)
    ys, xs = np.where(m)
    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    return float(inside.mean()) if inside.size else 0.0


def select_best_contained_mask_index(
    masks: torch.Tensor,
    scores: torch.Tensor,
    box_xywh_norm: Sequence[float],
    *,
    min_frac_inside: float = 1.0,
) -> int | None:
    """
    Highest-score mask whose foreground is contained in the GT bbox.

    ``min_frac_inside=1.0`` requires every mask pixel inside the box (strict).
    """
    if masks is None or scores is None:
        return None
    flat_scores = scores.reshape(-1)
    n = int(flat_scores.numel())
    if n == 0 or int(masks.shape[0]) == 0:
        return None

    best_i: int | None = None
    best_score = float("-inf")
    for i in range(min(n, int(masks.shape[0]))):
        frac = mask_fraction_inside_bbox(_mask_to_numpy_2d(masks[i]), box_xywh_norm)
        if frac + 1e-9 < float(min_frac_inside):
            continue
        sc = float(flat_scores[i].item())
        if sc > best_score:
            best_score = sc
            best_i = i
    return best_i


def save_polygon_from_bbox_debug(
    image: Image.Image,
    box_xywh_norm: Sequence[float],
    mask_2d: np.ndarray | None,
    out_path: Path,
    *,
    score: float | None = None,
    status: str = "ok",
) -> None:
    """Save RGB overlay: support image, red GT bbox, green selected mask."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    w, h = rgb.size
    overlay = rgb.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    x0, y0, x1, y1 = _box_xywh_norm_to_xyxy_px(box_xywh_norm, w, h)
    draw.rectangle([x0, y0, max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)], outline=(255, 0, 0, 255), width=3)

    if mask_2d is not None:
        m = np.asarray(mask_2d).astype(bool)
        if m.shape[:2] != (h, w):
            # Resize mask to image size if SAM state resolution differs.
            from PIL import Image as PILImage

            m_img = PILImage.fromarray((m.astype(np.uint8) * 255))
            m = np.asarray(m_img.resize((w, h), PILImage.Resampling.NEAREST)) > 0
        tint = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        tint_px = tint.load()
        ys, xs = np.where(m)
        for y, x in zip(ys.tolist(), xs.tolist()):
            tint_px[x, y] = (0, 255, 0, 110)
        overlay = Image.alpha_composite(overlay, tint)

    label = f"{status}"
    if score is not None:
        label += f" score={score:.3f}"
    draw = ImageDraw.Draw(overlay)
    draw.text((8, 8), label, fill=(255, 255, 0, 255))
    overlay.convert("RGB").save(out_path)


def refine_box_to_polygon_via_sam3(
    processor,
    state: Dict[str, Any],
    box_xywh_norm: Sequence[float],
    *,
    mask_threshold: float = 0.5,
    min_frac_inside: float = 1.0,
    debug_image: Image.Image | None = None,
    debug_path: Path | None = None,
) -> list[tuple[float, float]] | None:
    """Ground one normalized xywh box on ``state`` and return the best contained-mask polygon."""
    # Lazy import avoids detect ↔ prompts ↔ polygon_from_bbox cycles.
    from sam3_exemplar.detect import binary_mask_to_norm_polygon

    if box_xywh_norm is None or len(box_xywh_norm) != 4:
        return None

    processor.reset_all_prompts(state)
    cxcywh = box_xywh_to_cxcywh(
        torch.tensor(list(box_xywh_norm), dtype=torch.float32).view(1, 4)
    )
    box_list = [cxcywh.view(-1).tolist()]
    state = processor.forward_grounding_with_geometry_prompts(
        state,
        box_list,
        [True],
        [],
        [],
        mask_threshold=mask_threshold,
    )
    masks = state.get("masks")
    scores = state.get("scores")
    best_idx = select_best_contained_mask_index(
        masks, scores, box_xywh_norm, min_frac_inside=min_frac_inside
    )

    mask_np: np.ndarray | None = None
    score_f: float | None = None
    status = "fallback_no_contained_mask"
    verts: list[tuple[float, float]] | None = None

    if best_idx is not None and masks is not None and best_idx < int(masks.shape[0]):
        mask_np = _mask_to_numpy_2d(masks[best_idx])
        if scores is not None:
            score_f = float(scores.reshape(-1)[best_idx].item())
        verts = binary_mask_to_norm_polygon(mask_np)
        status = "ok" if verts is not None else "fallback_empty_polygon"

    if debug_path is not None and debug_image is not None:
        save_polygon_from_bbox_debug(
            debug_image,
            box_xywh_norm,
            mask_np,
            debug_path,
            score=score_f,
            status=status,
        )

    return verts


def refine_boxes_to_polygons_via_sam3(
    processor,
    state: Dict[str, Any],
    boxes_xywh_norm: Sequence[Optional[Sequence[float]]],
    *,
    mask_threshold: float = 0.5,
    min_frac_inside: float = 1.0,
    debug_image: Image.Image | None = None,
    debug_dir: Path | None = None,
    debug_stem: str = "support",
) -> List[Optional[list[tuple[float, float]]]]:
    """
    For each normalized xywh box (or ``None``), return a polygon or ``None``.

    Only masks fully contained in the GT bbox are eligible (see ``min_frac_inside``).
    Empty / non-contained SAM output yields ``None`` (caller falls back to raw bbox).
    """
    out: List[Optional[list[tuple[float, float]]]] = []
    box_i = 0
    for box in boxes_xywh_norm:
        if box is None:
            out.append(None)
            continue
        dbg = None
        if debug_dir is not None:
            dbg = Path(debug_dir) / f"{debug_stem}_box{box_i}.jpg"
        out.append(
            refine_box_to_polygon_via_sam3(
                processor,
                state,
                box,
                mask_threshold=mask_threshold,
                min_frac_inside=min_frac_inside,
                debug_image=debug_image,
                debug_path=dbg,
            )
        )
        box_i += 1
    return out


def apply_polygon_from_bbox_to_lists(
    processor,
    state: Dict[str, Any],
    boxes: list,
    polygons: list,
    *,
    mask_threshold: float = 0.5,
    min_frac_inside: float = 1.0,
    keep_bbox: bool = False,
    debug_image: Image.Image | None = None,
    debug_dir: Path | None = None,
    debug_stem: str = "support",
) -> Tuple[list, list, int]:
    """
    Replace successful box slots with SAM-derived polygons.

    When ``keep_bbox`` is True, successful lifts retain the original box alongside
    the polygon (for modes that prompt with both). Otherwise the box slot is cleared.

    Returns ``(new_boxes, new_polygons, n_bbox_fallback)``.
    """
    refined = refine_boxes_to_polygons_via_sam3(
        processor,
        state,
        boxes,
        mask_threshold=mask_threshold,
        min_frac_inside=min_frac_inside,
        debug_image=debug_image,
        debug_dir=debug_dir,
        debug_stem=debug_stem,
    )
    new_boxes: list = []
    new_polygons: list = []
    n_fallback = 0
    polys = list(polygons) + [None] * max(0, len(boxes) - len(polygons))
    for box, verts, existing_poly in zip(boxes, refined, polys):
        if box is None:
            new_boxes.append(None)
            new_polygons.append(existing_poly)
            continue
        if verts is not None:
            new_boxes.append(box if keep_bbox else None)
            new_polygons.append(verts)
        else:
            new_boxes.append(box)
            new_polygons.append(None)
            n_fallback += 1
    return new_boxes, new_polygons, n_fallback

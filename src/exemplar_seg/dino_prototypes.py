"""DINOv2 prototype scoring for SAM3 exemplar detections (bot copy).

SYNCED from ExemplarSegmentation ``src/fsod_eval/dino_prototypes.py`` by
``scripts/export_to_sam3_detect.py`` — do not edit here; edit the canonical
file and re-export. Bot-repo differences: COCO episode helpers removed,
exemplar-dict field parsing inlined, images loaded via
``sam3_override.utils.load_image``.

Journal fusion (alpha=0.5): score = alpha * SAM3 + (1 - alpha) * DINO cosine.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Journal method: score = alpha * SAM3 + (1 - alpha) * DINO cosine.
DEFAULT_DINO_SCORE_ALPHA = 0.5
DEFAULT_DINOV2_NAME = "dinov2_vitl14"
DEFAULT_DINO_LONG_SIDE = 630


def parse_exemplar_media_fields(exemplar_fields):
    """Return pos/neg boxes, points, polygons from a 6-field exemplar dict value."""
    def _slot(i, default_len=0):
        return list(exemplar_fields[i]) if len(exemplar_fields) > i else [None] * default_len

    boxes = _slot(0)
    points = _slot(1)
    neg_boxes = _slot(2)
    neg_points = _slot(3)
    polygons = _slot(4, len(points))
    neg_polygons = _slot(5, len(neg_points))
    if len(polygons) < len(points):
        polygons.extend([None] * (len(points) - len(polygons)))
    if len(neg_polygons) < len(neg_points):
        neg_polygons.extend([None] * (len(neg_points) - len(neg_polygons)))
    return boxes, points, neg_boxes, neg_points, polygons, neg_polygons


def load_dinov2(
    device: str | torch.device = "cuda",
    model_name: str = "dinov2_vitl14",
) -> tuple[torch.nn.Module, Any]:
    """Load a pretrained DINOv2 model from torch.hub (facebookresearch/dinov2)."""
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model.eval()
    model.to(device)
    return model, None

def resize_with_aspect_ratio(
    img_pil: Image.Image,
    target_long_side: int = 630,
    patch_size: int = 14,
) -> Image.Image:
    """Resize so the long side is ``target_long_side``, multiples of ``patch_size``."""
    orig_width, orig_height = img_pil.size
    aspect_ratio = orig_width / max(orig_height, 1)
    if orig_width >= orig_height:
        new_width = target_long_side
        new_height = int(target_long_side / max(aspect_ratio, 1e-6))
    else:
        new_height = target_long_side
        new_width = int(target_long_side * aspect_ratio)
    new_width = max(new_width // patch_size, 1) * patch_size
    new_height = max(new_height // patch_size, 1) * patch_size
    return img_pil.resize((new_width, new_height), resample=Image.BICUBIC)

def _dinov2_normalize(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return tfm(pil_img)[None].to(device)

def get_dinov2_features(
    model: torch.nn.Module,
    pil_img: Image.Image,
    *,
    device: str | torch.device = "cuda",
    target_long_side: int = 630,
    patch_size: int = 14,
) -> torch.Tensor:
    """Return spatial DINOv2 features ``(1, C, Hf, Wf)`` for a PIL RGB image."""
    device = torch.device(device)
    resized = resize_with_aspect_ratio(
        pil_img.convert("RGB"),
        target_long_side=target_long_side,
        patch_size=patch_size,
    )
    image_tensor = _dinov2_normalize(resized, device)
    # Prefer intermediate layers with reshape (matches FSOD-VFM); fall back to forward.
    if hasattr(model, "get_intermediate_layers"):
        output = model.get_intermediate_layers(
            image_tensor, n=1, reshape=True, return_class_token=True, norm=False
        )
        # Each entry is (patch_tokens, class_token); use patch map only.
        feat = torch.stack([out[0] for out in output], dim=0).sum(dim=0)
        return feat
    tokens = model.forward_features(image_tensor)
    if isinstance(tokens, dict):
        x = tokens.get("x_norm_patchtokens", tokens.get("x_prenorm"))
        if x is None:
            raise RuntimeError("Unexpected DINOv2 forward_features output")
        # Drop CLS if present: [B, 1+N, C] vs [B, N, C]
        b, n, c = x.shape
        h = w = int(n**0.5)
        if h * w != n:
            x = x[:, 1:, :]
            n = x.shape[1]
            h = w = int(n**0.5)
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    raise RuntimeError("Unsupported DINOv2 model API")

def resize_mask_to_features(
    mask_np: np.ndarray,
    feature_hw: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbour resize of a 2D binary mask to ``(Hf, Wf)``.

    If nearest resize empties a small object, fall back to marking every feature
    cell that intersects the original mask's bounding box (covers sub-patch objs).
    """
    from PIL import Image as PILImage

    m = np.asarray(mask_np)
    if m.ndim == 3:
        m = m.squeeze()
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {m.shape}")
    hf, wf = int(feature_hw[0]), int(feature_hw[1])
    mh, mw = m.shape
    binary = (m > 0).astype(np.uint8)
    if binary.shape == (hf, wf):
        return binary.astype(np.float32)
    img = PILImage.fromarray(binary * 255)
    resized = np.asarray(img.resize((wf, hf), resample=PILImage.Resampling.NEAREST))
    out = (resized > 0).astype(np.float32)
    if out.sum() > 0:
        return out
    # Sub-patch object: paint cells overlapping the mask bbox.
    ys, xs = np.where(binary > 0)
    if ys.size == 0:
        return out
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # Map image pixel ranges → feature cells (inclusive).
    fy0 = max(0, min(hf - 1, int(np.floor(y0 * hf / mh))))
    fy1 = max(0, min(hf - 1, int(np.floor(y1 * hf / mh))))
    fx0 = max(0, min(wf - 1, int(np.floor(x0 * wf / mw))))
    fx1 = max(0, min(wf - 1, int(np.floor(x1 * wf / mw))))
    out[fy0 : fy1 + 1, fx0 : fx1 + 1] = 1.0
    return out

def pool_features(
    feat_map: torch.Tensor,
    mask_np: np.ndarray,
    *,
    eps: float = 1e-7,
) -> torch.Tensor | None:
    """Masked average-pool ``feat_map`` (1,C,H,W) under ``mask_np``; L2-normalize.

    Returns a 1D tensor of shape ``(C,)`` or None if the mask is empty.
    """
    if feat_map.ndim != 4 or feat_map.shape[0] != 1:
        raise ValueError(f"Expected feat_map (1,C,H,W), got {tuple(feat_map.shape)}")
    hf, wf = int(feat_map.shape[2]), int(feat_map.shape[3])
    mask_r = resize_mask_to_features(mask_np, (hf, wf))
    mask_t = torch.from_numpy(mask_r).to(device=feat_map.device, dtype=feat_map.dtype)
    valid = float(mask_t.sum().item())
    if valid <= 0:
        return None
    masked = feat_map * mask_t.view(1, 1, hf, wf)
    vec = masked.sum(dim=(2, 3)) / (valid + eps)  # (1, C)
    return F.normalize(vec, dim=1, eps=1e-2).squeeze(0)

def build_prototypes(
    features_by_class: Mapping[Any, Sequence[torch.Tensor]],
    *,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, list[Any]]:
    """Mean-pool shot vectors per class; return L2-normalized ``(C, N_cls)`` and class ids."""
    proto_cls: list[Any] = []
    proto_cols: list[torch.Tensor] = []
    for cls_id, feats in features_by_class.items():
        usable = [f for f in feats if f is not None]
        if not usable:
            continue
        stacked = torch.stack([f.detach().float().cpu() for f in usable], dim=0)
        proto = stacked.mean(dim=0)
        proto_cols.append(proto)
        proto_cls.append(cls_id)
    if not proto_cols:
        raise RuntimeError("No valid exemplar features to build prototypes")
    mat = torch.stack(proto_cols, dim=1)  # (C, N)
    mat = F.normalize(mat, dim=0)
    if device is not None:
        mat = mat.to(device)
    return mat, proto_cls

def cosine_to_prototypes(
    feat_vec: torch.Tensor,
    proto_feat: torch.Tensor,
) -> tuple[torch.Tensor, int, float]:
    """Cosine similarities of a L2-normalized vector to prototype columns.

    Returns ``(sims[N], best_index, best_score)``.
    """
    v = F.normalize(feat_vec.float().reshape(1, -1), dim=1, eps=1e-2)
    proto = proto_feat.float()
    if proto.ndim != 2:
        raise ValueError(f"Expected proto_feat (C, N), got {tuple(proto.shape)}")
    # Columns may already be normalized; re-normalize for safety.
    proto = F.normalize(proto, dim=0)
    sims = (v @ proto).squeeze(0)  # (N,)
    best_i = int(torch.argmax(sims).item())
    best_s = float(sims[best_i].item())
    return sims, best_i, best_s

def fuse_sam3_dino_score(
    sam3_score: float,
    dino_cosine_pred: float,
    *,
    alpha: float = 0.5,
) -> float:
    """Additive fusion: ``alpha * sam3 + (1 - alpha) * dino``."""
    a = float(alpha)
    return float(a * float(sam3_score) + (1.0 - a) * float(dino_cosine_pred))

def norm_geometry_mask(
    *,
    width: int,
    height: int,
    polygon: Any = None,
    bbox: Any = None,
    point: Any = None,
) -> np.ndarray:
    """Rasterize normalised polygon/bbox/point geometry to a binary mask.

    Geometry is in relative (0..1) image coordinates: polygon vertices are
    ``(x, y)`` pairs, bbox is ``(x, y, w, h)``, point is ``(x, y)`` (drawn as
    a small disc, for point-only exemplars).
    """
    from PIL import Image as PILImage
    from PIL import ImageDraw

    img = PILImage.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(img)
    if polygon:
        pts = [(float(x) * width, float(y) * height) for x, y in polygon]
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
            return np.asarray(img, dtype=np.uint8)
    if bbox is not None:
        x, y, w, h = (float(v) for v in bbox)
        draw.rectangle(
            [x * width, y * height, (x + w) * width, (y + h) * height], fill=1
        )
        return np.asarray(img, dtype=np.uint8)
    if point is not None:
        px, py = float(point[0]) * width, float(point[1]) * height
        r = max(4.0, 0.01 * min(width, height))
        draw.ellipse([px - r, py - r, px + r, py + r], fill=1)
    return np.asarray(img, dtype=np.uint8)

def build_prototypes_from_exemplar_dict(
    exemplars_by_label_id: Mapping[Any, Mapping[str, Any]],
    dino_model: torch.nn.Module,
    *,
    device: str | torch.device = "cuda",
    target_long_side: int = DEFAULT_DINO_LONG_SIDE,
    image_loader=None,
) -> tuple[torch.Tensor, list[Any]]:
    """Build DINO prototypes from a bot-style exemplar dict.

    ``exemplars_by_label_id`` is ``{label_id: {media_key: six parallel lists
    (bboxes, points, neg_*, polygons, neg_polygons)}}`` with normalised
    geometry — the layout built by ``build_exemplars_by_label_id_from_df``
    and consumed by SAM3 prompt encoding. ``image_loader(media_key) -> PIL``
    defaults to ``sam3_override.utils.load_image`` (paths and URLs).
    """
    from collections import defaultdict


    if image_loader is None:
        from sam3_override.utils import load_image as image_loader  # type: ignore[assignment]

    feats_by_cls: dict[Any, list[torch.Tensor]] = defaultdict(list)
    for label_id, media_dict in exemplars_by_label_id.items():
        for media_key, fields in media_dict.items():
            bboxes, points, _nb, _np, polygons, _npoly = parse_exemplar_media_fields(fields)
            n = max(len(points), len(bboxes), len(polygons))
            if n == 0:
                continue
            pil = image_loader(media_key).convert("RGB")  # type: ignore[operator]
            w, h = pil.size
            feat_map = get_dinov2_features(
                dino_model, pil, device=device, target_long_side=int(target_long_side)
            )
            for i in range(n):
                mask = norm_geometry_mask(
                    width=w,
                    height=h,
                    polygon=polygons[i] if i < len(polygons) else None,
                    bbox=bboxes[i] if i < len(bboxes) else None,
                    point=points[i] if i < len(points) else None,
                )
                vec = pool_features(feat_map, mask)
                if vec is not None:
                    feats_by_cls[label_id].append(vec.detach().cpu())
    return build_prototypes(feats_by_cls, device=device)

def _masks_to_feature_grid(
    masks: torch.Tensor,
    *,
    feature_hw: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device | str,
    chunk: int = 16,
) -> torch.Tensor:
    """Nearest-resize ``(N, H, W)`` masks onto ``feature_hw`` and threshold at 0.5.

    Processes ``chunk`` masks at a time so the full-resolution float32 copy stays
    bounded (16 × 1024² × 4 bytes ≈ 64 MB).
    """
    n = int(masks.shape[0])
    hf, wf = int(feature_hw[0]), int(feature_hw[1])
    out = torch.empty((n, hf, wf), device=device, dtype=dtype)
    if n == 0:
        return out
    src_hw = (int(masks.shape[-2]), int(masks.shape[-1]))
    step = max(int(chunk), 1)
    for start in range(0, n, step):
        end = min(n, start + step)
        plane = masks[start:end].detach().to(device=device, dtype=torch.float32)
        if src_hw != (hf, wf):
            plane = F.interpolate(
                plane.unsqueeze(1), size=(hf, wf), mode="nearest"
            ).squeeze(1)
        out[start:end] = (plane > 0.5).to(dtype=dtype)
        del plane
    return out

def score_masks_against_prototypes(
    feat_map: torch.Tensor,
    masks: torch.Tensor,
    proto_feat: torch.Tensor,
    proto_cls_ids: Sequence[int],
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Cosine of each mask vs class prototypes.

    Args:
        feat_map: ``(1, C, Hf, Wf)`` DINO map for the query image
        masks: ``(N, H, W)`` boolean / float masks (any spatial size)
        proto_feat: ``(C, N_cls)`` L2-normalized prototype columns
        proto_cls_ids: length ``N_cls`` category ids aligned with columns

    Returns:
        ``cosines`` ``(N, N_cls)`` and a list of ``{str(cat_id): float}`` rows.
    """
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    n = int(masks.shape[0])
    n_cls = len(proto_cls_ids)
    if n == 0:
        empty = torch.zeros((0, n_cls), dtype=torch.float32, device=feat_map.device)
        return empty, []

    hf, wf = int(feat_map.shape[2]), int(feat_map.shape[3])
    # Resize in chunks. One float32 copy of every full-res mask is multi-GB
    # on a 20-class image (max_masks_per_class × classes × H × W).
    m = _masks_to_feature_grid(
        masks,
        feature_hw=(hf, wf),
        dtype=feat_map.dtype,
        device=feat_map.device,
    )
    row_sum = m.reshape(n, -1).sum(dim=-1)
    # Sub-patch objects vanish under nearest resize. Refill those cells from the
    # original mask, one plane at a time — a float32 copy of the full stack OOMs.
    empty_idx = (row_sum <= 0).nonzero(as_tuple=False).view(-1)
    for i in empty_idx.tolist():
        plane = masks[int(i)].detach()
        if plane.dtype.is_floating_point:
            plane = plane.float()
        filled = resize_mask_to_features(plane.cpu().numpy() > 0, (hf, wf))
        m[int(i)] = torch.from_numpy(filled).to(device=m.device, dtype=m.dtype)

    denom = m.reshape(n, -1).sum(dim=-1).clamp_min(1e-6)
    feat_flat = feat_map[0].reshape(feat_map.shape[1], -1)  # C, HW
    m_flat = m.reshape(n, -1)  # N, HW
    pooled = (m_flat @ feat_flat.t()) / denom.unsqueeze(1)  # N, C
    pooled = F.normalize(pooled.float(), dim=1, eps=1e-2)
    proto = F.normalize(proto_feat.float().to(device=pooled.device), dim=0)
    sims = pooled @ proto  # N, N_cls

    rows: list[dict[str, float]] = []
    for i in range(n):
        rows.append(
            {
                str(int(cid)): float(sims[i, j].item())
                for j, cid in enumerate(proto_cls_ids)
            }
        )
    return sims, rows

def dino_row_extras(
    cosine_row: Mapping[str, float],
    category_id: int,
    *,
    alpha: float,
    sam3_score: float,
) -> dict[str, Any]:
    """Build prediction-file DINO fields + fused score for one detection."""
    cat_key = str(int(category_id))
    dino_pred = float(cosine_row.get(cat_key, 0.0))
    best_cat = None
    best_val = float("-inf")
    for k, v in cosine_row.items():
        fv = float(v)
        if fv > best_val:
            best_val = fv
            best_cat = int(k)
    fused = fuse_sam3_dino_score(sam3_score, dino_pred, alpha=alpha)
    return {
        "sam3_score": float(sam3_score),
        "dino_cosine": {str(k): float(v) for k, v in cosine_row.items()},
        "dino_cosine_pred": dino_pred,
        "dino_best_category_id": best_cat,
        "score": fused,
    }

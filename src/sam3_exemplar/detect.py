"""Shared SAM3 exemplar detect API for fsod_eval and fss_eval.

Both evaluation tracks encode support exemplars into prompt embeddings and run
detection on query images. This module owns that shared path so the only
difference between FSOD (COCO AP) and FSS (mIoU) is post-processing of the
returned instance masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence

import torch
from PIL import Image
from sam3_exemplar.masks import mask_to_polygon

from sam3_exemplar.exemplar_data import build_exemplar_data_dict
from sam3_exemplar.images import autocast_scope, load_image, maybe_downscale_pil
from sam3_exemplar.model import (
    Sam3InferenceConfig,
    Sam3ModelConfig,
    load_sam3_model,
    make_processor,
    resolve_autocast_dtype,
)
from sam3_exemplar.prompts import encode_exemplar_prompts

SupportGeometry = Literal[
    "auto", "polygon", "bbox", "polygon_from_bbox", "polygon_from_bbox_and_box"
]


def resolve_image_path(file_name: str, images_dir: Path | None) -> Path:
    """Resolve a COCO ``file_name`` under ``images_dir``.

    Tries ``images_dir / file_name``, then basename, then one-level subdirs
    (e.g. Squidle/CVAT exports under ``images/default/<basename>``).
    """
    p = Path(file_name)
    if p.is_file():
        return p
    if images_dir is None:
        raise FileNotFoundError(
            f"Could not resolve image path for file_name={file_name!r} "
            f"(images_dir={images_dir})"
        )
    cand = images_dir / file_name
    if cand.is_file():
        return cand
    base = Path(file_name).name
    cand2 = images_dir / base
    if cand2.is_file():
        return cand2
    try:
        for child in images_dir.iterdir():
            if not child.is_dir():
                continue
            cand3 = child / base
            if cand3.is_file():
                return cand3
    except OSError:
        pass
    raise FileNotFoundError(
        f"Could not resolve image path for file_name={file_name!r} "
        f"(images_dir={images_dir})"
    )


def polygon_ring_to_norm_vertices(
    segmentation: Any,
    width: float,
    height: float,
) -> list[tuple[float, float]] | None:
    """Parse COCO polygon segmentation into normalized vertex pairs."""
    if not segmentation or not isinstance(segmentation, list):
        return None
    ring = segmentation[0]
    if isinstance(ring, (list, tuple)) and ring and isinstance(ring[0], (list, tuple)):
        pairs = [(float(pt[0]), float(pt[1])) for pt in ring if len(pt) >= 2]
    elif isinstance(ring, (list, tuple)) and ring and isinstance(ring[0], (int, float)):
        flat = list(ring)
        pairs = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]
    else:
        return None
    if len(pairs) < 3:
        return None
    return [(x / width, y / height) for x, y in pairs]


def binary_mask_to_norm_polygon(
    mask: Any,
    *,
    width: float | None = None,
    height: float | None = None,
) -> list[tuple[float, float]] | None:
    """Largest contour of a binary mask as normalized (x,y) vertices."""
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim != 2 or not arr.any():
        return None
    h, w = arr.shape
    polys = mask_to_polygon((arr > 0).astype("float32"))
    if not polys:
        return None
    best = max(polys, key=lambda ring: len(ring))
    if len(best) < 3:
        return None
    out_w = float(width if width is not None else w)
    out_h = float(height if height is not None else h)
    return [(float(x) / out_w, float(y) / out_h) for x, y in best]


def coco_support_to_exemplar_lists(
    support_coco: Dict[str, Any],
    *,
    images_dir: Path | None,
    support_geometry: SupportGeometry = "auto",
) -> tuple[list[str], list[list], list[list], list[list]]:
    """Convert COCO support annotations into per-image exemplar lists for encoding.

    ``support_geometry``:
      - ``auto``: prefer polygon segmentation when present, else bbox (default)
      - ``polygon``: require / prefer polygons; fall back to bbox only if missing
      - ``bbox``: always use bbox, ignore segmentation
      - ``polygon_from_bbox``: emit bboxes like ``bbox``; encode path lifts each box
        to a SAM3-derived polygon before prompting
      - ``polygon_from_bbox_and_box``: same lift as ``polygon_from_bbox``, but also
        keeps the original bbox alongside sampled polygon points
    """
    images_by_id = {int(img["id"]): img for img in support_coco.get("images", [])}
    anns_by_img: dict[int, list] = {}
    for ann in support_coco.get("annotations", []):
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    exemplar_images: list[str] = []
    exemplar_bboxes: list[list] = []
    exemplar_points: list[list] = []
    exemplar_polygons: list[list] = []

    bbox_like = support_geometry in (
        "bbox",
        "polygon_from_bbox",
        "polygon_from_bbox_and_box",
    )

    for img_id in sorted(anns_by_img.keys()):
        img = images_by_id[img_id]
        w, h = float(img["width"]), float(img["height"])
        path = str(resolve_image_path(img["file_name"], images_dir))
        boxes: list = []
        points: list = []
        polygons: list = []
        for ann in anns_by_img[img_id]:
            verts = None
            if not bbox_like:
                verts = polygon_ring_to_norm_vertices(ann.get("segmentation"), w, h)
            if verts is not None and support_geometry in ("auto", "polygon"):
                polygons.append(verts)
                boxes.append(None)
                points.append(None)
                continue
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4 and support_geometry in (
                "auto",
                "bbox",
                "polygon",
                "polygon_from_bbox",
                "polygon_from_bbox_and_box",
            ):
                # polygon mode falls back to bbox when no segmentation exists
                x, y, bw, bh = bbox
                boxes.append((x / w, y / h, bw / w, bh / h))
                polygons.append(None)
                points.append(None)
        if not boxes and not any(polygons):
            continue
        exemplar_images.append(path)
        exemplar_bboxes.append(boxes)
        exemplar_points.append(points)
        exemplar_polygons.append(polygons)

    return exemplar_images, exemplar_bboxes, exemplar_points, exemplar_polygons


def masks_scores_to_coco_annotations(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    image_id: int,
    category_id: int,
    image_width: int,
    image_height: int,
    score_thresh: float,
    start_ann_id: int,
) -> List[Dict[str, Any]]:
    """Convert SAM3 instance masks/scores into COCO annotation dicts."""
    polygons, kept_scores = masks_scores_to_polygons(
        masks,
        scores,
        score_thresh=score_thresh,
        output_width=image_width,
        output_height=image_height,
    )

    anns: List[Dict[str, Any]] = []
    for offset, (flat, score) in enumerate(zip(polygons, kept_scores)):
        xs = flat[0::2]
        ys = flat[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        bbox = [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]
        anns.append(
            {
                "id": start_ann_id + offset,
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": bbox,
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
                "score": score,
                "segmentation": [flat],
            }
        )
    return anns


def masks_scores_to_polygons(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    score_thresh: float = 0.0,
    output_width: int | None = None,
    output_height: int | None = None,
) -> tuple[List[List[float]], List[float]]:
    """Convert instance masks to largest-contour polygons and aligned scores.

    Polygons are flat pixel-coordinate rings, one per retained instance. When
    output dimensions are supplied, vertices are scaled from mask resolution to
    that image resolution.
    """
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if scores.dim() == 1:
        scores = scores.unsqueeze(1)
    scores_1d = scores.view(-1).float().cpu()
    masks_np = masks.float().cpu().numpy()

    polygons: List[List[float]] = []
    kept_scores: List[float] = []
    for i in range(masks_np.shape[0]):
        score = float(scores_1d[i].item())
        if score < score_thresh:
            continue
        polys = mask_to_polygon(masks_np[i])
        if not polys:
            continue
        best = max(polys, key=lambda ring: len(ring))
        flat: list[float] = []
        mask_height, mask_width = masks_np[i].shape[-2:]
        scale_x = float(output_width) / mask_width if output_width is not None else 1.0
        scale_y = float(output_height) / mask_height if output_height is not None else 1.0
        for x, y in best:
            flat.extend([float(x) * scale_x, float(y) * scale_y])
        if len(flat) < 6:
            continue
        polygons.append(flat)
        kept_scores.append(score)
    return polygons, kept_scores


@dataclass
class DetectResult:
    """Raw SAM3 instance outputs for one query image."""

    masks: torch.Tensor  # [N, H, W] or empty
    scores: torch.Tensor  # [N] or [N, 1]
    features: torch.Tensor | None = None  # [N, C] L2-normalized mask-pooled feats


@dataclass
class PolygonDetectResult:
    """Polygonized SAM3 instance outputs for one query image."""

    polygons: List[List[float]]
    scores: List[float]


def _class_name_from_support_coco(support_coco: Dict[str, Any]) -> str:
    """Resolve the COCO category name used when ``text_prompt`` is ``class_name``."""
    categories = support_coco.get("categories") or []
    if not categories:
        raise ValueError(
            "text_prompt='class_name' requires support_coco['categories'] "
            "with a non-empty name"
        )
    name = str(categories[0].get("name") or "").strip()
    if not name:
        raise ValueError(
            "text_prompt='class_name' requires a non-empty category name "
            "in support_coco['categories'][0]['name']"
        )
    return name


class Sam3ExemplarDetector:
    """Load SAM3 once, encode support exemplars, detect on query images."""

    def __init__(self, method_config: Dict[str, Any]) -> None:
        s = method_config.get("sam3", {}) or {}
        self.device = method_config.get("device", "cuda")
        self.use_autocast = bool(s.get("use_autocast", True))
        self.autocast_dtype = resolve_autocast_dtype(str(s.get("autocast_dtype", "bf16")))
        self.confidence_threshold = float(s.get("confidence_threshold", 0.5))
        self.combine_prompts = s.get(
            "combine_prompts", s.get("embed_merge", "mean_ex_embed_per_image")
        )
        self.encode_text = bool(s.get("encode_text", True))
        self.use_img_pos_embed = bool(s.get("use_img_pos_embed", True))
        self.text_prompt = str(s.get("text_prompt", "visual"))
        self.polygon_sample_points = int(s.get("polygon_sample_points", 8))
        self.sam_resolution = int(s.get("sam_resolution") or 1008)
        raw_max_query_side = s.get("max_query_side")
        self.max_query_side = (
            int(raw_max_query_side) if raw_max_query_side is not None else None
        )
        raw_max_masks = s.get("max_masks_per_class", 100)
        self.max_masks_per_class = (
            int(raw_max_masks) if raw_max_masks is not None else None
        )

        model_config = Sam3ModelConfig(
            device=self.device,
            checkpoint_path=s.get("checkpoint_path", "./models/sam3.pt"),
            use_autocast=self.use_autocast,
            autocast_dtype=str(s.get("autocast_dtype", "bf16")),
        )
        self.inference_config = Sam3InferenceConfig(
            sam_resolution=self.sam_resolution,
            confidence_threshold=self.confidence_threshold,
            polygon_sample_points=self.polygon_sample_points,
            combine_prompts=self.combine_prompts,
            max_masks_per_class=self.max_masks_per_class,
        )
        self.model = load_sam3_model(model_config)
        self.processor = make_processor(self.model, self.inference_config)
        self._prompt_data: list | None = None

    def encode_exemplars(
        self,
        exemplar_images: Sequence[str],
        exemplar_bboxes: Sequence[list],
        exemplar_points: Sequence[list],
        exemplar_polygons: Sequence[list],
        *,
        support_geometry: SupportGeometry = "auto",
        text_prompt: str | None = None,
        debug_dir: Path | None = None,
    ) -> list:
        """Encode support exemplars into prompt tensors; store and return them."""
        if not exemplar_images:
            raise ValueError("No exemplar images/annotations to encode.")
        resolved_text_prompt = (
            self.text_prompt if text_prompt is None else str(text_prompt)
        )
        if resolved_text_prompt == "class_name":
            raise ValueError(
                "text_prompt='class_name' requires encode_from_coco_support "
                "(or pass the resolved class name string explicitly)."
            )
        exemplar_data_dict = build_exemplar_data_dict(
            list(exemplar_images),
            list(exemplar_bboxes),
            list(exemplar_points),
            list(exemplar_polygons),
        )
        with autocast_scope(
            device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast
        ):
            self._prompt_data = encode_exemplar_prompts(
                self.processor,
                exemplar_data_dict,
                combine_prompts=self.combine_prompts,
                encode_text=self.encode_text,
                use_img_pos_embed=self.use_img_pos_embed,
                n_polygon_sample_points=self.polygon_sample_points,
                support_geometry=support_geometry,
                text_prompt=resolved_text_prompt,
                debug_dir=debug_dir,
            )
        return self._prompt_data

    def encode_from_coco_support(
        self,
        support_coco: Dict[str, Any],
        *,
        images_dir: Path | None,
        support_geometry: SupportGeometry = "auto",
        debug_dir: Path | None = None,
    ) -> list:
        lists = coco_support_to_exemplar_lists(
            support_coco,
            images_dir=images_dir,
            support_geometry=support_geometry,
        )
        text_prompt = self.text_prompt
        if text_prompt == "class_name":
            text_prompt = _class_name_from_support_coco(support_coco)
        return self.encode_exemplars(
            *lists,
            support_geometry=support_geometry,
            text_prompt=text_prompt,
            debug_dir=debug_dir,
        )

    def _detect_result_from_state(self, state: Dict[str, Any]) -> DetectResult:
        """Copy masks/scores/features to CPU and drop GPU tensors from ``state``.

        Multiclass prompt banks run sequentially on one image; keeping prior-class
        full-res masks on CUDA stacks VRAM across labels.
        """
        masks = state.get("masks")
        scores = state.get("scores")
        features = state.get("obj_features")
        if masks is None:
            empty_m = torch.zeros((0, 1, 1), device="cpu")
            empty_s = torch.zeros((0,), device="cpu")
            empty_f = (
                None
                if features is None
                else torch.zeros((0, int(features.shape[-1])), device="cpu")
            )
            state.pop("obj_features", None)
            return DetectResult(masks=empty_m, scores=empty_s, features=empty_f)
        masks = masks.squeeze()
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if scores is None:
            scores = torch.ones((masks.shape[0],), device=masks.device)
        feat_cpu = None
        if features is not None:
            feat_cpu = features.detach().float().cpu()
            if feat_cpu.ndim == 1:
                feat_cpu = feat_cpu.unsqueeze(0)
        result = DetectResult(
            masks=masks.detach().cpu(),
            scores=scores.detach().cpu(),
            features=feat_cpu,
        )
        # Free GPU copies before the next class grounds on the same image.
        state.pop("masks", None)
        state.pop("boxes", None)
        state.pop("scores", None)
        state.pop("obj_features", None)
        return result

    def detect(
        self,
        image: Image.Image | str | Path,
        *,
        prompt_data: list | None = None,
        return_obj_features: bool = False,
    ) -> DetectResult:
        """Run encoded prompts on one query image; return instance masks + scores."""
        prompts = prompt_data if prompt_data is not None else self._prompt_data
        if prompts is None:
            raise RuntimeError("Call encode_exemplars(...) before detect(), or pass prompt_data.")
        results = self.detect_with_prompt_bank(
            image,
            {0: prompts},
            return_obj_features=return_obj_features,
        )
        return results[0]

    def detect_with_prompt_bank(
        self,
        image: Image.Image | str | Path,
        prompts_by_label: Mapping[Any, list],
        *,
        return_obj_features: bool = False,
    ) -> Dict[Any, DetectResult]:
        """Encode the query image once, then ground with each pre-encoded prompt bank.

        Matches the segmenter pattern: support prompts are encoded once per label and
        reused across images; ``set_image`` runs once per query image.
        """
        if not prompts_by_label:
            return {}
        if isinstance(image, (str, Path)):
            pil = load_image(str(image))
        else:
            pil = image
        pil = maybe_downscale_pil(pil, self.max_query_side)

        results: Dict[Any, DetectResult] = {}
        with autocast_scope(
            device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast
        ):
            state = self.processor.set_image(pil)
            for label_id, prompts in prompts_by_label.items():
                self.processor.reset_all_prompts(state)
                prompt, prompt_mask = prompts
                state = self.processor.forward_grounding_with_prompt_embeddings(
                    state,
                    prompt,
                    prompt_mask,
                    return_obj_features=return_obj_features,
                )
                results[label_id] = self._detect_result_from_state(state)
        return results

    def detect_polygons(
        self,
        image: Image.Image | str | Path,
        *,
        prompt_data: list | None = None,
        score_thresh: float = 0.0,
        output_size: tuple[int, int] | None = None,
    ) -> PolygonDetectResult:
        """Detect instances and return polygons instead of raw masks.

        ``output_size`` is ``(width, height)``. It defaults to the query image
        dimensions so polygon coordinates align with the original image.
        """
        if isinstance(image, (str, Path)):
            pil = load_image(str(image))
        else:
            pil = image
        width, height = output_size if output_size is not None else pil.size
        result = self.detect(pil, prompt_data=prompt_data)
        polygons, scores = masks_scores_to_polygons(
            result.masks,
            result.scores,
            score_thresh=score_thresh,
            output_width=width,
            output_height=height,
        )
        return PolygonDetectResult(polygons=polygons, scores=scores)

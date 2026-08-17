"""Geometry prompt builders and exemplar prompt encoding for SAM3."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import shapely
import torch
from shapely.geometry import Point, Polygon as ShapelyPolygon
from sam3_exemplar.geometry import is_axis_aligned_bbox
from sam3.model.box_ops import box_xywh_to_cxcywh

from sam3_exemplar.exemplar_data import build_exemplar_data_dict, parse_exemplar_media_fields
from sam3_exemplar.images import load_image
from sam3_exemplar.polygon_from_bbox import apply_polygon_from_bbox_to_lists

__all__ = [
    "absolute_polygon_from_point",
    "bbox_xywh_from_vertices",
    "build_exemplar_data_dict",
    "build_geometry_prompts",
    "build_query_geometry_prompts",
    "encode_exemplar_prompts",
    "feature_grid_points_in_polygon",
    "feature_map_size",
    "is_axis_aligned_bbox",
    "parse_exemplar_media_fields",
    "sample_points_in_polygon",
]


def sample_points_in_polygon(
    polygon_xy,
    *,
    n_points: int = 8,
    include_click: tuple[float, float] | list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[list[float]]:
    """Sample normalized [x, y] points inside a polygon for SAM3 point prompts."""
    if n_points <= 0:
        return []

    poly = ShapelyPolygon(polygon_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        if include_click is not None:
            return [[float(include_click[0]), float(include_click[1])]]
        return []

    coords: list[tuple[float, float]] = []
    if include_click is not None:
        coords.append((float(include_click[0]), float(include_click[1])))

    minx, miny, maxx, maxy = poly.bounds
    rng = rng or np.random.default_rng(42)
    max_attempts = max(n_points * 50, 100)
    attempts = 0
    while len(coords) < n_points and attempts < max_attempts:
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if poly.contains(Point(x, y)):
            coords.append((x, y))
        attempts += 1

    if len(coords) < n_points:
        rep = poly.representative_point()
        coords.append((float(rep.x), float(rep.y)))

    result = [[x, y] for x, y in coords[:n_points]]
    print(
        f"[prompts] polygon random sampling: {len(result)} point(s) "
        f"(requested n_points={n_points}"
        + (", LIMIT reached)" if len(result) >= n_points else ")")
    )
    return result


def feature_map_size(processor, state) -> Tuple[int, int]:
    """Return ``(H, W)`` of SAM3's finest feature level for the image in ``state``.

    This is ``vis_feat_sizes[-1]`` -- the level the geometry encoder pools point
    prompts from via ``grid_sample`` -- and is used by
    :func:`feature_grid_points_in_polygon` to place one prompt point per feature
    cell so that each token inside a polygon is gathered exactly once.
    """
    _backbone_out, _img_feats, _img_pos, vis_feat_sizes = processor.model._get_img_feats(
        state["backbone_out"], processor.find_stage.img_ids
    )
    h, w = vis_feat_sizes[-1]
    return int(h), int(w)


def feature_grid_points_in_polygon(
    polygon_xy,
    feat_hw: Tuple[int, int],
    *,
    include_click: tuple[float, float] | list[float] | None = None,
    max_points: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[list[float]]:
    """Normalized ``[x, y]`` centers of feature cells whose centers fall in a polygon.

    Instead of randomly sampling interior points, this deterministically enumerates
    the SAM3 feature-grid cells (at the finest level, ``feat_hw = vis_feat_sizes[-1]``)
    whose cell centers lie within ``polygon_xy`` (normalized ``[0, 1]`` image coords).

    Each returned point maps 1:1 onto a feature token when the geometry encoder pools
    features with ``grid_sample(align_corners=False)``: for a grid of size ``S``, the
    normalized center ``(i + 0.5) / S`` lands exactly on feature cell ``i``. The
    resulting exemplar prompt therefore gathers exactly the encoded features contained
    within the polygon.

    ``max_points`` optionally caps the number of tokens (a random subset is kept, but
    the ``include_click`` anchor is always retained); pass ``None`` for all cells.
    """
    H, W = int(feat_hw[0]), int(feat_hw[1])
    coords: list[list[float]] = []
    if include_click is not None:
        coords.append([float(include_click[0]), float(include_click[1])])

    poly = ShapelyPolygon(polygon_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)

    if H > 0 and W > 0 and not poly.is_empty and poly.area > 0:
        # Cell-center grid in normalized [0, 1]; (i + 0.5) / size maps exactly onto
        # feature cell i under grid_sample(align_corners=False).
        xs = (np.arange(W) + 0.5) / W
        ys = (np.arange(H) + 0.5) / H
        gx, gy = (arr.ravel() for arr in np.meshgrid(xs, ys))
        inside = shapely.contains_xy(poly, gx, gy)
        coords.extend([float(x), float(y)] for x, y in zip(gx[inside], gy[inside]))

    if not coords and not poly.is_empty:
        # Tiny polygon whose interior misses every cell center: fall back to a
        # representative interior point so the exemplar still contributes a token.
        rep = poly.representative_point()
        coords.append([float(rep.x), float(rep.y)])

    n_candidates = len(coords)
    capped = max_points is not None and max_points > 0 and n_candidates > max_points
    if capped:
        rng = rng or np.random.default_rng(42)
        if include_click is not None:
            head, tail, keep = coords[:1], coords[1:], max_points - 1
            if keep <= 0:
                coords = head
            else:
                idx = rng.choice(len(tail), size=min(keep, len(tail)), replace=False)
                coords = head + [tail[i] for i in sorted(idx.tolist())]
        else:
            idx = rng.choice(len(coords), size=max_points, replace=False)
            coords = [coords[i] for i in sorted(idx.tolist())]

    print(
        f"[prompts] polygon feature-grid tokens: {n_candidates} cell(s) inside "
        f"(feat_hw={H}x{W}) -> {len(coords)} sampled"
        + (f" (CAPPED at polygon_sample_points={max_points})" if capped else "")
    )
    return [[float(x), float(y)] for x, y in coords]


def absolute_polygon_from_point(x: float, y: float, relative_polygon) -> list[list[float]]:
    """Convert Squidle point-relative polygon vertices to absolute normalized coords."""
    return [[float(vx + x), float(vy + y)] for vx, vy in relative_polygon]


def bbox_xywh_from_vertices(vertices) -> tuple[float, float, float, float]:
    """Bounding box as normalized top-left xywh from absolute polygon vertices."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x1, y1 = float(min(xs)), float(min(ys))
    x2, y2 = float(max(xs)), float(max(ys))
    return x1, y1, (x2 - x1), (y2 - y1)


def build_geometry_prompts(
    points,
    boxes,
    polygons,
    *,
    n_polygon_sample_points: int,
    positive: bool,
    feat_hw: Tuple[int, int] | None = None,
):
    """Build SAM3 box/point prompt lists from parallel click, box, and polygon annotations.

    When ``feat_hw`` (the finest feature-map ``(H, W)``) is provided, polygon prompts
    extract one point per feature cell inside the polygon (see
    :func:`feature_grid_points_in_polygon`) instead of random sampling; in that mode
    ``n_polygon_sample_points`` acts as an optional cap (``0`` means all cells).
    """
    box_inputs = []
    box_labels = []
    point_inputs = []
    point_labels = []

    n = max(len(points), len(boxes), len(polygons))
    if n == 0:
        return box_inputs, box_labels, point_inputs, point_labels

    points = list(points) + [None] * (n - len(points))
    boxes = list(boxes) + [None] * (n - len(boxes))
    polygons = list(polygons) + [None] * (n - len(polygons))

    for pt, box, poly in zip(points, boxes, polygons):
        use_polygon = (
            poly is not None
            and len(poly) >= 3
            and not is_axis_aligned_bbox(poly)
        )
        if use_polygon:
            if feat_hw is not None:
                sampled = feature_grid_points_in_polygon(
                    poly,
                    feat_hw,
                    include_click=pt,
                    max_points=n_polygon_sample_points or None,
                )
            else:
                sampled = sample_points_in_polygon(
                    poly,
                    n_points=n_polygon_sample_points,
                    include_click=pt,
                )
            point_inputs.extend(sampled)
            point_labels.extend([positive] * len(sampled))
            # When a box is also present (e.g. polygon_from_bbox_and_box), emit both.
            if box:
                cxcywh = box_xywh_to_cxcywh(torch.tensor(box).view(1, 4)).tolist()[0]
                box_inputs.append(cxcywh)
                box_labels.append(positive)
        else:
            if box:
                cxcywh = box_xywh_to_cxcywh(torch.tensor(box).view(1, 4)).tolist()[0]
                box_inputs.append(cxcywh)
                box_labels.append(positive)
            if pt is not None:
                point_inputs.append(pt)
                point_labels.append(positive)

    return box_inputs, box_labels, point_inputs, point_labels


def build_query_geometry_prompts(
    label_points,
    *,
    n_polygon_sample_points: int,
    positive: bool = True,
    feat_hw: Tuple[int, int] | None = None,
):
    """Build SAM3 geometry prompts from target-image Squidle points.

    ``feat_hw`` is forwarded to :func:`build_geometry_prompts` to enable
    feature-grid polygon token extraction on the query image.
    """
    points, boxes, polygons = [], [], []
    for p in label_points:
        x, y = p["x"], p["y"]
        rel_poly = p.get("polygon")
        box = None
        poly = None
        if rel_poly:
            abs_poly = absolute_polygon_from_point(x, y, rel_poly)
            if is_axis_aligned_bbox(abs_poly):
                box = bbox_xywh_from_vertices(abs_poly)
            else:
                poly = abs_poly
        points.append([x, y])
        boxes.append(box)
        polygons.append(poly)
    return build_geometry_prompts(
        points,
        boxes,
        polygons,
        n_polygon_sample_points=n_polygon_sample_points,
        positive=positive,
        feat_hw=feat_hw,
    )


@torch.inference_mode()
def encode_exemplar_prompts(
    processor,
    exemplar_data_dict: Dict[str, Any],
    *,
    combine_prompts: str = "mean_ex_embed_per_image",
    encode_text: bool = True,
    use_img_pos_embed = True,
    n_polygon_sample_points: int = 8,
    support_geometry: str = "auto",
    text_prompt: str = "visual",
    debug_dir: Path | None = None,
) -> list:
    """Encode all media entries in ``exemplar_data_dict`` into combined prompt tensors.

    ``combine_prompts`` modes:
      - ``all_ex_embeds``: concatenate all geometric/visual tokens across exemplars
      - ``mean_of_all_ex_embeds``: concatenate all tokens, then mean to one embedding
      - ``mean_ex_embed_per_image``: mean geometric tokens per exemplar image, then keep one per image
      - ``mean_of_mean_ex_embed_per_image``: mean per image, then mean those into one embedding
      - anything else (e.g. ``none``): use the first exemplar only

    When ``support_geometry`` is ``polygon_from_bbox`` or
    ``polygon_from_bbox_and_box``, each box is lifted to a SAM3-derived polygon
    (mask must be contained in the GT bbox) on the same ``set_image`` state
    before encoding; boxes with empty/non-contained SAM output fall back to the
    raw bbox. ``polygon_from_bbox_and_box`` also keeps the original bbox alongside
    the sampled polygon points. If ``debug_dir`` is set, overlays are written
    under that directory.

    ``text_prompt`` seeds the text tower via ``forward_text`` (default
    ``"visual"``). Callers that want the COCO class name should resolve it
    before calling this function.

    k-shot limiting is applied when building the dict (annotation selection), not here.
    """
    if not exemplar_data_dict:
        raise ValueError("exemplar_data_dict is empty; nothing to encode")

    lift_from_bbox = support_geometry in (
        "polygon_from_bbox",
        "polygon_from_bbox_and_box",
    )
    prompt_data = []
    n_bbox_fallback_total = 0
    for img_i, (img, exemplar_fields) in enumerate(exemplar_data_dict.items()):
        boxes, points, neg_boxes, neg_points, polygons, neg_polygons = parse_exemplar_media_fields(
            exemplar_fields
        )
        exemplar_image = load_image(img)
        exemplar_state = processor.set_image(exemplar_image)
        processor.reset_all_prompts(exemplar_state)

        if lift_from_bbox:
            stem = Path(str(img)).stem if img is not None else f"img{img_i}"
            boxes, polygons, n_fb = apply_polygon_from_bbox_to_lists(
                processor,
                exemplar_state,
                boxes,
                polygons,
                keep_bbox=(support_geometry == "polygon_from_bbox_and_box"),
                debug_image=exemplar_image,
                debug_dir=debug_dir,
                debug_stem=f"{img_i:03d}_{stem}",
            )
            n_bbox_fallback_total += n_fb

        # Extract encoded feature tokens inside polygons (Approach A) using this
        # exemplar image's finest feature-map resolution.
        feat_hw = feature_map_size(processor, exemplar_state)

        pos_boxes, pos_box_labels, pos_points, pos_point_labels = build_geometry_prompts(
            points,
            boxes,
            polygons,
            n_polygon_sample_points=n_polygon_sample_points,
            positive=True,
            feat_hw=feat_hw,
        )
        neg_b, neg_box_labels, neg_pts, neg_point_labels = build_geometry_prompts(
            neg_points,
            neg_boxes,
            neg_polygons,
            n_polygon_sample_points=n_polygon_sample_points,
            positive=False,
            feat_hw=feat_hw,
        )
        box_inputs = pos_boxes + neg_b
        box_labels = pos_box_labels + neg_box_labels
        point_inputs = pos_points + neg_pts
        point_labels = pos_point_labels + neg_point_labels

        processor.reset_all_prompts(exemplar_state)
        exemplar_state = processor.add_geometric_prompts_to_state(
            box_inputs,
            box_labels,
            point_inputs,
            point_labels,
            exemplar_state,
            text_prompt=text_prompt,
        )

        (
            prompt,
            prompt_mask,
            _,
            txt_feats,
            txt_masks,
            geo_feats,
            geo_masks,
            visual_prompt_embed,
            visual_prompt_mask,
        ) = processor.model.get_prompt_embeddings(
            backbone_out=exemplar_state["backbone_out"],
            find_input=processor.find_stage,
            geometric_prompt=exemplar_state["geometric_prompt"].clone(),
            encode_text=encode_text,
            use_img_pos_embed=use_img_pos_embed,
        )
        prompt_data.append(
            [
                prompt,
                prompt_mask,
                txt_feats,
                txt_masks,
                geo_feats,
                geo_masks,
                visual_prompt_embed,
                visual_prompt_mask,
            ]
        )

    if n_bbox_fallback_total:
        print(
            f"[sam3_exemplar] {support_geometry}: fell back to raw bbox for "
            f"{n_bbox_fallback_total} support annotation(s) with empty/non-contained SAM masks"
        )
    if debug_dir is not None and lift_from_bbox:
        print(f"[sam3_exemplar] {support_geometry} debug overlays → {debug_dir}")

    (
        _prompts,
        _prompt_masks,
        txt_featss,
        txt_maskss,
        geo_featss,
        geo_maskss,
        visual_prompt_embeds,
        visual_prompt_masks,
    ) = map(list, zip(*prompt_data))
    txt_feats, txt_masks = txt_featss[0], txt_maskss[0]

    if combine_prompts == "all_ex_embeds":
        # Concatenate all embeddings across exemplars.
        geo_feats = torch.cat(geo_featss, dim=0)
        geo_masks = torch.cat(geo_maskss, dim=1)
        visual_prompt_embed = torch.cat(visual_prompt_embeds, dim=0)
        visual_prompt_mask = torch.cat(visual_prompt_masks, dim=1)
    elif combine_prompts == "mean_of_all_ex_embeds":
        # Average every geometric token across all exemplars into one embedding.
        # keepdim=True preserves the [tokens, batch, channels] rank so the result is a
        # single geo token with the same shape as mean_of_mean_ex_embed_per_image (a
        # bare mean(dim=0) drops to 2-D and breaks the downstream torch.cat).
        geo_feats = torch.cat(geo_featss, dim=0).mean(dim=0, keepdim=True)
        geo_masks = torch.zeros((1, 1), device=txt_masks.device, dtype=txt_masks.dtype)
        visual_prompt_embed = torch.cat(visual_prompt_embeds, dim=0)
        visual_prompt_mask = torch.cat(visual_prompt_masks, dim=1)
    elif combine_prompts == "mean_ex_embed_per_image":
        # One mean geometric embedding per exemplar image.
        geo_feats = torch.stack([t.mean(dim=0) for t in geo_featss], dim=0)
        geo_masks = torch.zeros(
            (1, geo_feats.shape[0]),
            device=txt_masks.device,
            dtype=txt_masks.dtype,
        )
        visual_prompt_embed = torch.cat(visual_prompt_embeds, dim=0)
        visual_prompt_mask = torch.cat(visual_prompt_masks, dim=1)
    elif combine_prompts == "mean_of_mean_ex_embed_per_image":
        # Mean per image, then mean those → one overall geometric embedding.
        geo_feats = torch.stack([t.mean(dim=0) for t in geo_featss], dim=0).mean(dim=0)
        geo_feats = geo_feats.unsqueeze(0)
        geo_masks = torch.zeros((1, 1), device=txt_masks.device, dtype=txt_masks.dtype)
        visual_prompt_embed = torch.cat(visual_prompt_embeds, dim=0)
        visual_prompt_mask = torch.cat(visual_prompt_masks, dim=1)
    else:
        geo_feats, geo_masks = geo_featss[0], geo_maskss[0]
        visual_prompt_embed, visual_prompt_mask = visual_prompt_embeds[0], visual_prompt_masks[0]

    if encode_text:
        prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
        prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
    else:
        prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
        prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)

    result = [prompt.clone(), prompt_mask.clone()]
    gc.collect()
    return result

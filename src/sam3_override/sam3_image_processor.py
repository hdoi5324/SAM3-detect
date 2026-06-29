import torch
import gc

from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Processor
from sam3.model.geometry_encoders import concat_padded_sequences
from typing import Dict, List
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from .utils import load_image
from sam3.model.box_ops import box_xywh_to_cxcywh
from .polygon_prompts import (
    absolute_polygon_from_point,
    bbox_xywh_from_vertices,
    is_axis_aligned_bbox,
    sample_points_in_polygon,
)


def build_geometry_prompts(
    points,
    boxes,
    polygons,
    *,
    n_polygon_sample_points: int,
    positive: bool,
):
    """Build SAM3 box/point prompt lists from parallel click, box, and polygon annotations."""
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
            sampled = sample_points_in_polygon(
                poly,
                n_points=n_polygon_sample_points,
                include_click=pt,
            )
            point_inputs.extend(sampled)
            point_labels.extend([positive] * len(sampled))
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
):
    """
    Build SAM3 geometry prompts from target-image Squidle points.

    Each entry may include a polygon stored relative to the point anchor (x, y).
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
    )


def _parse_exemplar_media_fields(exemplar_fields):
    """Return (pos boxes/points/polygons, neg boxes/points/polygons) from exemplar dict value."""
    boxes = list(exemplar_fields[0]) if len(exemplar_fields) > 0 else []
    points = list(exemplar_fields[1]) if len(exemplar_fields) > 1 else []
    neg_boxes = list(exemplar_fields[2]) if len(exemplar_fields) > 2 else []
    neg_points = list(exemplar_fields[3]) if len(exemplar_fields) > 3 else []
    polygons = (
        list(exemplar_fields[4])
        if len(exemplar_fields) > 4
        else [None] * len(points)
    )
    neg_polygons = (
        list(exemplar_fields[5])
        if len(exemplar_fields) > 5
        else [None] * len(neg_points)
    )
    if len(polygons) < len(points):
        polygons.extend([None] * (len(points) - len(polygons)))
    if len(neg_polygons) < len(neg_points):
        neg_polygons.extend([None] * (len(neg_points) - len(neg_polygons)))
    return boxes, points, neg_boxes, neg_points, polygons, neg_polygons


class Sam3Processor(_Sam3Processor):
    def _exemplar_geometry_prompts(
        self,
        points,
        boxes,
        polygons,
        *,
        n_polygon_sample_points: int,
        positive: bool,
    ):
        """Build SAM3 box/point lists for one exemplar image (positive or negative)."""
        return build_geometry_prompts(
            points,
            boxes,
            polygons,
            n_polygon_sample_points=n_polygon_sample_points,
            positive=positive,
        )

    @staticmethod
    def build_query_geometry_prompts(label_points, *, n_polygon_sample_points: int, positive: bool = True):
        return build_query_geometry_prompts(
            label_points,
            n_polygon_sample_points=n_polygon_sample_points,
            positive=positive,
        )

    @torch.inference_mode()
    def add_geometric_prompts_to_state(self, boxes: List, box_labels: List, points: List, point_labels: List,
                                       state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, device=self.device, dtype=torch.float32).view(-1, 1, 4)
            labels = torch.tensor([box_labels], device=self.device, dtype=torch.bool).view(-1, 1)
            state["geometric_prompt"].append_boxes(boxes, labels)

        # adding a batch and sequence dimension
        if len(points) > 0:
            points = torch.tensor(points, device=self.device, dtype=torch.float32).view(-1, 1, 2)
            labels = torch.tensor([point_labels], device=self.device, dtype=torch.bool).view(-1, 1)
            state["geometric_prompt"].append_points(points, labels)

        return state

    @torch.inference_mode()
    def forward_grounding_with_prompt_embeddings(self, state: Dict, prompt, prompt_mask, backbone_out=None, mask_threshold=0.5):
        # Based on Sam3Processor._forward_grounding method.  Splits model.forward_grounding into prompt encoding and inference.

        #outputs = self.model.forward_grounding(
        #    backbone_out=state["backbone_out"],
        #    find_input=self.find_stage,
        #    geometric_prompt=state["geometric_prompt"],
        #    find_target=None,
        #)

        backbone_out = backbone_out if backbone_out is not None else state["backbone_out"]

        # Inference - encode,decode,segment from Sam3Image.forward_grounding
        outputs = self.model.forward_with_prompt_encoding(
            backbone_out,
            self.find_stage,
            prompt,
            prompt_mask,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > mask_threshold
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    @torch.inference_mode()
    def forward_grounding_with_exemplar_and_point_prompts(
        self,
        state: Dict,
        exemplar_prompt: torch.Tensor,
        exemplar_prompt_mask: torch.Tensor,
        mask_threshold = 0.5,
    ) -> Dict:
        """
        Run grounding with exemplar prompt tensors plus geometry from the **target** image.

        Call ``set_image`` first, then ``add_geometric_prompts_to_state`` with normalized
        point(s) ``[x, y]`` in ``[0, 1]`` (and empty box lists). This encodes those points
        with the target backbone and appends only the target geometry tokens (not duplicate
        text features) after the exemplar prompt along the sequence dimension.
        """
        if "geometric_prompt" not in state:
            raise ValueError(
                "Missing geometric_prompt: call add_geometric_prompts_to_state with query points first."
            )
        (
            _,
            _,
            _,
            _,
            _,
            geo_feats,
            geo_masks,
            visual_prompt_embed,
            visual_prompt_mask,
        ) = self.model.get_prompt_embeddings(
            state["backbone_out"],
            self.find_stage,
            state["geometric_prompt"].clone(),
            encode_text=True,
        )
        target_geo_parts = [geo_feats]
        target_mask_parts = [geo_masks]
        if visual_prompt_embed.shape[0] > 0:
            target_geo_parts.append(visual_prompt_embed)
            target_mask_parts.append(visual_prompt_mask)
        target_geo = torch.cat(target_geo_parts, dim=0)
        target_geo_mask = torch.cat(target_mask_parts, dim=1)
        prompt = torch.cat([target_geo, exemplar_prompt], dim=0)
        prompt_mask = torch.cat([target_geo_mask, exemplar_prompt_mask], dim=1)
        #prompt, prompt_mask = concat_padded_sequences(exemplar_prompt, exemplar_prompt_mask, target_geo, target_geo_mask)
        return self.forward_grounding_with_prompt_embeddings(state, prompt, prompt_mask, mask_threshold=mask_threshold)

    @torch.inference_mode()
    def forward_grounding_with_geometry_prompts(
        self,
        state: Dict,
        boxes: List,
        box_labels: List,
        points: List,
        point_labels: List,
        mask_threshold: float = 0.5,
    ) -> Dict:
        """
        Run grounding using geometric box and/or point prompts on the current image.

        Call ``set_image`` first. Boxes are normalized cxcywh; points are normalized ``[x, y]``.
        """
        if not boxes and not points:
            raise ValueError("At least one box or point prompt is required")
        state = self.add_geometric_prompts_to_state(
            boxes, box_labels, points, point_labels, state
        )
        prompt, prompt_mask, *_ = self.model.get_prompt_embeddings(
            state["backbone_out"],
            self.find_stage,
            state["geometric_prompt"].clone(),
            encode_text=True,
        )
        return self.forward_grounding_with_prompt_embeddings(
            state, prompt, prompt_mask, mask_threshold=mask_threshold
        )

    @torch.inference_mode()
    def forward_grounding_with_point_prompts(
        self,
        state: Dict,
        points: List,
        point_labels: List | None = None,
        mask_threshold: float = 0.5,
    ) -> Dict:
        """
        Run grounding using only geometric point prompts on the current image.

        Call ``set_image`` first. Points are normalized ``[x, y]`` in ``[0, 1]``.
        """
        if not points:
            raise ValueError("points must be non-empty")
        if point_labels is None:
            point_labels = [True] * len(points)
        return self.forward_grounding_with_geometry_prompts(
            state, [], [], points, point_labels, mask_threshold=mask_threshold
        )

    @torch.inference_mode()
    def get_exemplar_prompts(
        self,
        exemplar_data_dict,
        combine_prompts="avg_cat",
        encode_text=True,
        n_exemplars=3,
        n_polygon_sample_points=8,
    ):
        prompt_data = []
        for img, exemplar_fields in exemplar_data_dict.items():
            boxes, points, neg_boxes, neg_points, polygons, neg_polygons = _parse_exemplar_media_fields(
                exemplar_fields
            )
            exemplar_image = load_image(img)
            exemplar_state = self.set_image(exemplar_image)
            self.reset_all_prompts(exemplar_state)

            pos_boxes, pos_box_labels, pos_points, pos_point_labels = self._exemplar_geometry_prompts(
                points,
                boxes,
                polygons,
                n_polygon_sample_points=n_polygon_sample_points,
                positive=True,
            )
            neg_b, neg_box_labels, neg_pts, neg_point_labels = self._exemplar_geometry_prompts(
                neg_points,
                neg_boxes,
                neg_polygons,
                n_polygon_sample_points=n_polygon_sample_points,
                positive=False,
            )
            box_inputs = pos_boxes + neg_b
            box_labels = pos_box_labels + neg_box_labels
            point_inputs = pos_points + neg_pts
            point_labels = pos_point_labels + neg_point_labels

            self.reset_all_prompts(exemplar_state)
            exemplar_state = self.add_geometric_prompts_to_state(
                box_inputs, box_labels, point_inputs, point_labels, exemplar_state
            )

            # Prompt encoding from Sam3Image.forward_grounding
            prompt, prompt_mask, _, txt_feats, txt_masks, geo_feats, geo_masks, visual_prompt_embed, visual_prompt_mask = self.model.get_prompt_embeddings(
                backbone_out=exemplar_state["backbone_out"],
                find_input=self.find_stage,
                geometric_prompt=exemplar_state["geometric_prompt"].clone(),
                encode_text=encode_text,
            )
            prompt_data.append([prompt, prompt_mask, txt_feats, txt_masks, geo_feats, geo_masks, visual_prompt_embed,
                                visual_prompt_mask])
            if len(prompt_data) == n_exemplars:
                break

        prompts, prompt_masks, txt_featss, txt_maskss, geo_featss, geo_maskss, visual_prompt_embeds, visual_prompt_masks = map(
            list, zip(*prompt_data))
        txt_feats, txt_masks = txt_featss[0], txt_maskss[0]

        # Variations of merging geo_feats from different images which each have different prompts
        if combine_prompts == "merge":
            geo_feats = torch.stack(geo_featss, dim=0).mean(dim=0)
            geo_masks = geo_maskss[0]
            visual_prompt_embed = torch.stack(visual_prompt_embeds, dim=0).mean(dim=0)
            visual_prompt_mask = visual_prompt_masks[0]
        elif combine_prompts == "cat":
            geo_feats = torch.cat(geo_featss, dim=0)
            geo_masks = torch.cat(geo_maskss, dim=1)
            visual_prompt_embed = torch.cat(visual_prompt_embeds, dim=0)
            visual_prompt_mask = torch.cat(visual_prompt_masks, dim=1)
        elif combine_prompts == "meancat":
            geo_feats = torch.stack([t.mean(dim=0) for t in geo_featss], dim=0).mean(dim=0)
            geo_feats = geo_feats.unsqueeze(0)
            geo_masks = torch.zeros(
                (1, 1),
                device=txt_masks.device,
                dtype=txt_masks.dtype,
            )
            visual_prompt_embed = torch.cat(visual_prompt_embeds, dim=0)
            visual_prompt_mask = torch.cat(visual_prompt_masks, dim=1)
        elif combine_prompts == "avg_cat":
            geo_feats = torch.stack([t.mean(dim=0) for t in geo_featss], dim=0)
            geo_masks = torch.zeros(
                (1, geo_feats.shape[0]),
                device=txt_masks.device,
                dtype=txt_masks.dtype,
            )
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

        prompt_data = [prompt.clone(), prompt_mask.clone()]
        del prompts, prompt_masks, txt_featss, txt_maskss, geo_featss, geo_maskss, visual_prompt_embeds, visual_prompt_masks
        gc.collect()
        return prompt_data
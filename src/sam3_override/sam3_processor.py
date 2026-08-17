"""Sam3Processor subclass using patched Sam3Image inference hooks."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from sam3.model import box_ops
from sam3.model.data_misc import interpolate
from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Processor

from sam3_exemplar.prompts import build_query_geometry_prompts


class Sam3Processor(_Sam3Processor):
    @staticmethod
    def build_query_geometry_prompts(
        label_points,
        *,
        n_polygon_sample_points: int,
        positive: bool = True,
        feat_hw=None,
    ):
        return build_query_geometry_prompts(
            label_points,
            n_polygon_sample_points=n_polygon_sample_points,
            positive=positive,
            feat_hw=feat_hw,
        )

    @torch.inference_mode()
    def add_geometric_prompts_to_state(
        self,
        boxes: List,
        box_labels: List,
        points: List,
        point_labels: List,
        state: Dict,
        text_prompt: str = "visual",
    ):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(
                [text_prompt], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        if len(boxes) > 0:
            boxes_t = torch.tensor(boxes, device=self.device, dtype=torch.float32).view(-1, 1, 4)
            labels = torch.tensor([box_labels], device=self.device, dtype=torch.bool).view(-1, 1)
            state["geometric_prompt"].append_boxes(boxes_t, labels)

        if len(points) > 0:
            points_t = torch.tensor(points, device=self.device, dtype=torch.float32).view(-1, 1, 2)
            labels = torch.tensor([point_labels], device=self.device, dtype=torch.bool).view(-1, 1)
            state["geometric_prompt"].append_points(points_t, labels)

        return state

    @torch.inference_mode()
    def forward_grounding_with_prompt_embeddings(
        self,
        state: Dict,
        prompt,
        prompt_mask,
        backbone_out=None,
        mask_threshold=0.5,
        return_obj_features: bool = False,
    ):
        backbone_out = backbone_out if backbone_out is not None else state["backbone_out"]
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

        # Cap like COCO maxDets=100: keep highest scores before full-res upsample.
        max_masks = getattr(self, "max_masks_per_class", None)
        if max_masks is not None and out_probs.numel() > int(max_masks):
            scores_1d = out_probs.reshape(-1)
            _, top_idx = torch.topk(scores_1d, k=int(max_masks))
            out_probs = scores_1d[top_idx]
            out_masks = out_masks.reshape(-1, *out_masks.shape[1:])[top_idx]
            out_bbox = out_bbox.reshape(-1, *out_bbox.shape[1:])[top_idx]

        # Mask-pool finest FPN features at pred_masks resolution (typically 288×288)
        # before upsample — NTTT-style object features for semantic soft-merge.
        state.pop("obj_features", None)
        if return_obj_features:
            feat = backbone_out["backbone_fpn"][0]  # [1, C, Hf, Wf]
            c_dim = int(feat.shape[1])
            n_keep = int(out_masks.shape[0]) if out_masks.ndim >= 1 else 0
            if n_keep == 0:
                state["obj_features"] = torch.zeros(
                    (0, c_dim), device=feat.device, dtype=torch.float32
                )
            else:
                # out_masks are logits at feature resolution; match later mask_threshold.
                m = (out_masks.sigmoid() > float(mask_threshold)).to(dtype=feat.dtype)
                if m.ndim == 2:
                    m = m.unsqueeze(0)
                # Align spatial size if FPN / mask grids ever differ.
                if m.shape[-2:] != feat.shape[-2:]:
                    m = F.interpolate(
                        m.unsqueeze(1),
                        size=feat.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                denom = m.flatten(1).sum(-1).clamp_min(1e-6)
                # Avoid materializing [N,C,H,W] (OOM: N≈200, 288², C≈256 → multi‑GB).
                # Mask-pool via matmul: [N,HW] @ [HW,C] → [N,C].
                feat_hw = feat[0].reshape(c_dim, -1)  # [C, HW]
                m_hw = m.reshape(n_keep, -1)  # [N, HW]
                obj = (m_hw @ feat_hw.t()) / denom.unsqueeze(-1)
                state["obj_features"] = F.normalize(obj.float(), p=2, dim=-1)

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

        # state["masks_logits"] = out_masks  # skipped: unused full-res float tensor (saves GPU memory)
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
        mask_threshold=0.5,
    ) -> Dict:
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
        return self.forward_grounding_with_prompt_embeddings(
            state, prompt, prompt_mask, mask_threshold=mask_threshold
        )

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
        if not points:
            raise ValueError("points must be non-empty")
        if point_labels is None:
            point_labels = [True] * len(points)
        return self.forward_grounding_with_geometry_prompts(
            state, [], [], points, point_labels, mask_threshold=mask_threshold
        )

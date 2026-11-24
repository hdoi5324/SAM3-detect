"""SAM3 segmenter using the sqbot_framework orchestration layer."""

from __future__ import annotations

import time
import gc
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sqbot_framework.context import AnnotationContext, MediaContext, UnitContext
from sqbot_framework.handlers import MediaHandlerConfig, wms_config_from_bot_params
from sqbot_framework.matching import build_unit_detection, points_inside_mask
from sqbot_framework.media_annotator import MediaAnnotator
from sqbot_framework.masks import check_polygon, clean_polygon, mask_to_polygon
from sqbot_framework.results import UnitResult

from sam3_exemplar import (
    Sam3InferenceConfig,
    Sam3ModelConfig,
    autocast_scope,
    build_exemplars_by_label_id_from_df,
    configure_torch_for_sam3,
    encode_exemplar_prompts,
    feature_map_size,
    load_sam3_model,
    make_processor,
    resolve_autocast_dtype,
)
from squidle_data import load_squidle_annotations_to_df

configure_torch_for_sam3()


class SAM3Segmenter(MediaAnnotator):
    default_supported_media_types = frozenset({"image", "wms"})

    def __init__(
        self,
        device="cuda",
        use_autocast=True,
        autocast_dtype="bf16",
        model_path="models/sam3.pt",
        add_new_annotations=True,
        add_new_threshold=0.5,
        replace_existing_polygons=True,
        iou_threshold=0.5,
        k_shot=5,
        n_negative=0,
        n_images_to_process=0,
        sam_resolution=1008,
        embed_merge="mean_ex_embed_per_image",
        encode_text=True,
        use_img_pos_embed=True,
        point_prompt_mode="combined",
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        polygon_sample_points=8,
        wms_tile_size_px=1024,
        wms_max_mosaic_px=2048,
        wms_fetch_via_api=True,
        wms_point_size_m=1.0,
        wms_skip_blank_tiles=True,
        wms_grid_stride_m=None,
        wms_preflight_tile_size_px=128,
        wms_use_occupancy_cull=True,
        wms_alpha_threshold=16,
        wms_occupancy_min_fraction=0.0,
        wms_max_tiles=None,
        supported_media_types=None,
        dino_enabled=True,
        dino_score_alpha=0.5,
        dinov2_name="dinov2_vitl14",
        dino_long_side=630,
        dino_device=None,
        **annotator_args,
    ):
        super().__init__(supported_media_types=supported_media_types, **annotator_args)
        self.media_handler_config = MediaHandlerConfig(
            wms=wms_config_from_bot_params(
                wms_tile_size_px=wms_tile_size_px,
                wms_max_mosaic_px=wms_max_mosaic_px,
                wms_fetch_via_api=wms_fetch_via_api,
                wms_point_size_m=wms_point_size_m,
                wms_skip_blank_tiles=wms_skip_blank_tiles,
                wms_grid_stride_m=wms_grid_stride_m,
                wms_preflight_tile_size_px=wms_preflight_tile_size_px,
                wms_use_occupancy_cull=wms_use_occupancy_cull,
                wms_alpha_threshold=wms_alpha_threshold,
                wms_occupancy_min_fraction=wms_occupancy_min_fraction,
                wms_max_tiles=wms_max_tiles,
            ),
        )
        self.device = device
        self.use_autocast = use_autocast
        self.autocast_dtype = resolve_autocast_dtype(autocast_dtype)
        self.confidence_threshold = self.prob_thresh
        self.add_new_annotations = add_new_annotations
        self.add_new_threshold = add_new_threshold
        self.replace_existing_polygons = replace_existing_polygons
        self.sam_resolution = sam_resolution if sam_resolution is not None else 1008
        self.k_shot = k_shot if k_shot is not None else 5
        self.n_negative = n_negative if n_negative is not None else 0
        self.n_images_to_process = n_images_to_process if n_images_to_process is not None else 0
        self.iou_threshold = iou_threshold
        self.embed_merge = (
            embed_merge if embed_merge is not None else "mean_ex_embed_per_image"
        )
        self.encode_text = encode_text if encode_text is not None else True
        self.use_img_pos_embed = use_img_pos_embed if use_img_pos_embed is not None else True
        self.polygon_sample_points = (
            polygon_sample_points if polygon_sample_points is not None else 8
        )
        self.point_prompt_mode = (point_prompt_mode or "combined").lower()
        if self.point_prompt_mode not in ("combined", "point_only"):
            raise ValueError(
                f"point_prompt_mode must be 'combined' or 'point_only'; "
                f"got {point_prompt_mode!r}"
            )

        self.model_config = Sam3ModelConfig(
            device=device,
            checkpoint_path=model_path,
            use_autocast=use_autocast,
            autocast_dtype=autocast_dtype,
            points_direct_project=points_direct_project,
            points_pool=points_pool,
            points_pos_enc=points_pos_enc,
            boxes_direct_project=boxes_direct_project,
            boxes_pool=boxes_pool,
            boxes_pos_enc=boxes_pos_enc,
        )
        self.inference_config = Sam3InferenceConfig(
            sam_resolution=self.sam_resolution,
            confidence_threshold=self.confidence_threshold,
            polygon_sample_points=self.polygon_sample_points,
            combine_prompts=self.embed_merge,
        )

        self.exemplars_by_label_id = {}
        self.prompts_by_label_id = {}
        self.model = None

        # DINOv2 prototype score refinement (journal method, alpha=0.5):
        # fused score = alpha * SAM3 + (1 - alpha) * DINO cosine, stored as
        # the detection score / Squidle likelihood. Set dino_enabled=False
        # (or dino_score_alpha=1.0) for SAM3-only behaviour.
        self.dino_enabled = bool(dino_enabled) if dino_enabled is not None else True
        self.dino_score_alpha = float(dino_score_alpha) if dino_score_alpha is not None else 0.5
        if not 0.0 <= self.dino_score_alpha <= 1.0:
            raise ValueError(f"dino_score_alpha must be in [0, 1], got {self.dino_score_alpha}")
        self.dinov2_name = dinov2_name or "dinov2_vitl14"
        self.dino_long_side = int(dino_long_side) if dino_long_side else 630
        self.dino_device = dino_device  # None/auto -> follow SAM3 device
        self.dino_model = None
        self.dino_proto_feat = None
        self.dino_proto_label_ids: list = []

    def _ensure_model(self) -> None:
        if self.model is None:
            print("Loading SAM model...")
            self.model = load_sam3_model(self.model_config)

    def on_annotation_start(self, ctx: AnnotationContext) -> None:
        self._ensure_model()
        self.exemplars_by_label_id = {}
        self.prompts_by_label_id = {}
        if self.point_prompt_mode == "point_only":
            print("point_prompt_mode=point_only: skipping exemplar loading and encoding")
            return
        self.exemplars_by_label_id = self.load_exemplar_annotations(ctx.base_annotation_set_id)
        self.register_identity_label_codes(self.exemplars_by_label_id.keys())
        processor = make_processor(self.model, self.inference_config)
        with autocast_scope(
            device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast
        ):
            for label_id in self.exemplars_by_label_id.keys():
                self.prompts_by_label_id[label_id] = encode_exemplar_prompts(
                    processor,
                    self.exemplars_by_label_id[label_id],
                    combine_prompts=self.embed_merge,
                    n_polygon_sample_points=self.polygon_sample_points,
                    use_img_pos_embed=self.use_img_pos_embed,
                    encode_text=self.encode_text,
                )
        self._ensure_dino_prototypes()

    # -- DINOv2 prototype scoring -------------------------------------------
    def _resolved_dino_device(self) -> str:
        if self.dino_device and str(self.dino_device).strip().lower() not in ("", "auto"):
            return str(self.dino_device)
        return str(self.device)

    def _ensure_dino_prototypes(self) -> None:
        """Load DINOv2 and build per-label prototypes from the exemplars.

        Failures are non-fatal: the bot falls back to SAM3-only scores so a
        DINO download/GPU issue never kills a Squidle run.
        """
        self.dino_proto_feat = None
        self.dino_proto_label_ids = []
        if not self.dino_enabled or not self.exemplars_by_label_id:
            return
        try:
            from exemplar_seg.dino_prototypes import (
                build_prototypes_from_exemplar_dict,
                load_dinov2,
            )

            dino_device = self._resolved_dino_device()
            if self.dino_model is None:
                self.dino_model, _ = load_dinov2(device=dino_device, model_name=self.dinov2_name)
            proto_feat, proto_labels = build_prototypes_from_exemplar_dict(
                self.exemplars_by_label_id,
                self.dino_model,
                device=dino_device,
                target_long_side=self.dino_long_side,
            )
            self.dino_proto_feat = proto_feat.detach().cpu()
            self.dino_proto_label_ids = list(proto_labels)
            # Free GPU for SAM3 grounding; the model is staged back briefly
            # per media unit in _dino_feat_map_for_image.
            self._release_dino_gpu()
            print(
                f"DINO prototypes: {len(self.dino_proto_label_ids)} labels, "
                f"alpha={self.dino_score_alpha:.2f}, model={self.dinov2_name}"
            )
        except Exception as exc:  # noqa: BLE001 - fall back to SAM3-only scores
            print(f"DINO prototype build failed ({exc}); continuing with SAM3-only scores")
            self.dino_proto_feat = None
            self.dino_proto_label_ids = []

    def _release_dino_gpu(self) -> None:
        if self.dino_model is None:
            return
        try:
            self.dino_model.to("cpu")
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _dino_feat_map_for_image(self, pil_image):
        """DINO feature map for one query image, or None if unavailable."""
        if self.dino_model is None or self.dino_proto_feat is None:
            return None
        try:
            from exemplar_seg.dino_prototypes import get_dinov2_features

            dino_device = self._resolved_dino_device()
            if str(dino_device).startswith("cuda"):
                self.dino_model.to(dino_device)
            feat = get_dinov2_features(
                self.dino_model,
                pil_image,
                device=dino_device,
                target_long_side=self.dino_long_side,
            )
            if str(dino_device).startswith("cuda") and str(self.device).startswith("cuda"):
                self._release_dino_gpu()
            return feat
        except Exception as exc:  # noqa: BLE001
            print(f"DINO feature extraction failed ({exc}); SAM3-only scores for this unit")
            return None

    def _clear_dino(self) -> None:
        self.dino_proto_feat = None
        self.dino_proto_label_ids = []
        if self.dino_model is not None:
            try:
                self.dino_model.to("cpu")
            except Exception:
                pass
            self.dino_model = None

    def on_annotation_end(self, ctx: AnnotationContext) -> None:
        def _gpu_mem_mb():
            if not torch.cuda.is_available():
                return None
            torch.cuda.synchronize()
            return {
                "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            }

        before = _gpu_mem_mb()
        self.log.debug(
            "on_annotation_end: before teardown annotation_set_id=%s model_loaded=%s gpu=%s",
            ctx.annotation_set_id,
            self.model is not None,
            before,
        )

        self.exemplars_by_label_id = {}
        self.prompts_by_label_id = {}
        self._clear_dino()
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        time.sleep(3) # Allow GPU memory time to be released and defragment

        after = _gpu_mem_mb()
        self.log.debug(
            "on_annotation_end: after teardown annotation_set_id=%s gpu=%s",
            ctx.annotation_set_id,
            after,
        )

    def on_media_start(self, ctx: MediaContext) -> None:
        handler = ctx.handler
        if handler.merge_mode != "spatial":
            return
        occ_stats = getattr(handler, "occupancy_cull_stats", lambda: None)()
        n_specs = len(handler._point_tile_specs())
        occ_txt = ""
        if occ_stats is not None:
            occ_txt = (
                f", occupancy_cull={occ_stats['after_cull']}/{occ_stats['candidates']} "
                f"({occ_stats['mask_method']})"
            )
        wms = self.media_handler_config.wms
        print(
            f"WMS grid tiling: {n_specs} candidate tiles{occ_txt}, "
            f"mosaic {handler.storage_width}x{handler.storage_height}px, "
            f"tile_size={wms.tile_size_px}px, "
            f"point_window={wms.point_size_m}m, "
            f"grid_stride={wms.effective_grid_stride_m()}m, "
            f"skip_blank={wms.skip_blank_tiles}, "
            f"preflight={wms.preflight_tile_size_px}px, "
            f"max_tiles={wms.max_tiles}, "
            f"prompt_mode={'point_only' if self.point_prompt_mode == 'point_only' else 'combined (exemplars)'}"
        )

    def should_process_unit(self, unit_ctx: UnitContext) -> bool:
        if self.point_prompt_mode == "point_only":
            return bool(unit_ctx.points)
        return True

    def unit_update_and_create(self, ctx: UnitContext) -> UnitResult:
        detections = self._collect_unit_detections(ctx)
        if ctx.media.handler.merge_mode == "spatial":
            for det in detections:
                det.unit = ctx.unit
        return UnitResult(detections=detections)

    def load_exemplar_annotations(self, annotation_set_id, label_ids_to_ignore=None):
        if label_ids_to_ignore is None:
            label_ids_to_ignore = [6835, 636, 458]
        df = load_squidle_annotations_to_df(
            self.sqapi,
            annotation_set_id,
            label_ids_to_ignore,
            require_polygon=True,
        )
        exemplars_by_label_id = build_exemplars_by_label_id_from_df(
            self.sqapi,
            df,
            wms_config=self.media_handler_config.wms,
            label_ids_to_ignore=label_ids_to_ignore,
            n_negative=self.n_negative,
            k_shot=self.k_shot,
        )
        if not exemplars_by_label_id:
            raise ValueError(
                "Error: The loaded dataframe is empty. No available exemplar annotations. "
                "Does the squidle user have access to this annotation set?"
            )
        return exemplars_by_label_id

    def email_annotation_set_user(self, a, counts):
        user_ids = [a.get("user", {}).get("id")]
        annotation_set_url = f"{self.sqapi.host}/geodata/annotation_set/{a.get('id')}"
        skipped_media = counts.get("skipped_media", 0)
        skipped_line = (
            f'Skipped: {skipped_media} media items (unsupported media type).<br>\n'
            if skipped_media
            else ""
        )
        message = (
            f'Hi {a.get("user", {}).get("first_name")}, <br><br>\n'
            f'Your annotation set "<b>{a.get("media_collection", {}).get("name")} / {a.get("name")}</b>" has been '
            f"processed by {self.annotator_info.get('name')}.<br><br>\n"
            f'Analysed: {counts["media"]} media items, {counts["points"]} points and {counts["annotations"]} annotations.<br>\n'
            f"{skipped_line}"
            f'Created/updated: {counts["new_points"]} points and {counts["new_annotations"]} annotations.<br><br>\n'
            f'Any Label suggestions will appear as "Magical Suggestions" '
            f'in the annotation window and can be validated using the QA/QC tool.<br><br>\n'
            f'To see results, click: <a href="{annotation_set_url}">{annotation_set_url}</a>'
        )
        self.sqapi.send_user_email(
            "SQ+ BOT: your Annotation Set has been processed", message, user_ids=user_ids
        )

    def _collect_unit_detections(self, ctx: UnitContext):
        handler = ctx.media.handler
        unit = ctx.unit
        points = ctx.points
        force_point_only = self.point_prompt_mode == "point_only"

        points_by_label_id, _unlabeled = self._partition_points_by_label(points or [])
        if force_point_only:
            self.register_identity_label_codes(points_by_label_id.keys())
        all_detections = []

        image_data = unit.image_rgb
        dino_feat_map = self._dino_feat_map_for_image(Image.fromarray(image_data))
        processor = make_processor(self.model, self.inference_config)
        with autocast_scope(
            device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast
        ):
            inference_state = processor.set_image(Image.fromarray(image_data))

        for label_id, prompt, prompt_mask, label_points in self._iter_label_jobs(
            points_by_label_id, force_point_only=force_point_only
        ):
            label_pts_coords, point_indices = self._label_point_pixel_coords(
                label_points, handler, unit
            )
            if label_pts_coords.size == 0 and not (prompt is not None and prompt_mask is not None):
                continue
            all_detections.extend(
                self._collect_label_detections(
                    processor,
                    inference_state,
                    prompt,
                    prompt_mask,
                    label_id,
                    label_points,
                    label_pts_coords,
                    point_indices,
                    unit.frame_width,
                    unit.frame_height,
                    dino_feat_map=dino_feat_map,
                )
            )
        return all_detections

    def _partition_points_by_label(self, points):
        points_by_label_id = defaultdict(list)
        for idx, p in enumerate(points):
            annotations = p.get("annotations")
            if not annotations:
                continue
            label_id = annotations[0].get("label_id")
            if label_id is None:
                continue
            entry = {"index": p.get("_source_index", idx), "x": p["x"], "y": p["y"]}
            if p.get("data") is not None and p.get("data").get("polygon") is not None:
                entry["polygon"] = p.get("data").get("polygon")
            points_by_label_id[label_id].append(entry)
        return points_by_label_id, []

    def _label_point_pixel_coords(self, label_points, handler, unit=None):
        if not label_points:
            return np.empty((0, 2), dtype=int), []
        rows, cols, indices = [], [], []
        max_row = (unit.frame_height if unit is not None else handler.storage_height) - 1
        max_col = (unit.frame_width if unit is not None else handler.storage_width) - 1
        for p in label_points:
            if unit is not None:
                rc = handler.point_to_unit_pixels(p["x"], p["y"], unit)
                if rc is None:
                    continue
                row, col = rc
            else:
                row, col = handler.point_to_storage_pixels(p["x"], p["y"])
            rows.append(int(np.clip(row, 0, max_row)))
            cols.append(int(np.clip(col, 0, max_col)))
            indices.append(p["index"])
        if not indices:
            return np.empty((0, 2), dtype=int), []
        return np.stack([rows, cols], axis=1), indices

    def _iter_label_jobs(self, points_by_label_id, *, force_point_only=False):
        if force_point_only or self.point_prompt_mode == "point_only":
            for label_id, label_points in points_by_label_id.items():
                if label_points:
                    yield label_id, None, None, label_points
            return
        for label_id, prompt_pair in self.prompts_by_label_id.items():
            if label_id is None:
                continue
            prompt, prompt_mask = prompt_pair
            yield label_id, prompt, prompt_mask, points_by_label_id.get(label_id) or []

    def _masks_to_detections(
        self,
        scores,
        masks,
        label_id,
        label_pts_coords,
        point_indices,
        frame_width,
        frame_height,
        dino_feat_map=None,
    ):
        # Fused scores (alpha * SAM3 + (1 - alpha) * DINO cosine) replace the
        # SAM3 scores everywhere downstream, including the stored likelihood.
        fused_scores = None
        if dino_feat_map is not None and self.dino_proto_feat is not None:
            try:
                from exemplar_seg.dino_prototypes import (
                    dino_row_extras,
                    score_masks_against_prototypes,
                )

                masks_3d = masks.squeeze(1) if masks.ndim == 4 else masks
                _sims, rows = score_masks_against_prototypes(
                    dino_feat_map,
                    masks_3d,
                    self.dino_proto_feat,
                    self.dino_proto_label_ids,
                )
                fused_scores = [
                    float(
                        dino_row_extras(
                            rows[i],
                            int(label_id),
                            alpha=self.dino_score_alpha,
                            sam3_score=float(scores[i].item()),
                        )["score"]
                    )
                    for i in range(len(scores))
                ]
            except Exception as exc:  # noqa: BLE001
                print(f"DINO mask scoring failed ({exc}); SAM3-only scores")
                fused_scores = None
        detections = []
        for i in range(len(scores)):
            mask_np = masks[i].squeeze(0).cpu().numpy()
            prob = fused_scores[i] if fused_scores is not None else scores[i].item()
            hit_local = points_inside_mask(mask_np, label_pts_coords)
            hit_indices = [point_indices[j] for j in hit_local]
            raw_polys = mask_to_polygon(mask_np)
            if not raw_polys:
                continue
            poly = clean_polygon(raw_polys[0], frame_width, frame_height)
            if not check_polygon(poly, frame_width, frame_height):
                continue
            det = build_unit_detection(
                score=prob,
                class_code=int(label_id),
                polygon=poly,
                hit_point_indices=hit_indices,
            )
            if det is not None:
                detections.append(det)
        return detections

    def _run_label_grounding(
        self,
        processor,
        inference_state,
        prompt,
        prompt_mask,
        box_inputs,
        box_labels,
        point_inputs,
        point_labels,
    ):
        if self.point_prompt_mode == "combined" and (box_inputs or point_inputs):
            inference_state = processor.add_geometric_prompts_to_state(
                box_inputs, box_labels, point_inputs, point_labels, inference_state
            )
            return processor.forward_grounding_with_exemplar_and_point_prompts(
                inference_state, prompt, prompt_mask
            )
        return processor.forward_grounding_with_prompt_embeddings(
            inference_state, prompt, prompt_mask
        )

    def _collect_label_detections(
        self,
        processor,
        inference_state,
        prompt,
        prompt_mask,
        label_id,
        label_points,
        label_pts_coords,
        point_indices,
        frame_width,
        frame_height,
        dino_feat_map=None,
    ):
        box_inputs, box_labels, point_inputs, point_labels = processor.build_query_geometry_prompts(
            label_points,
            n_polygon_sample_points=self.polygon_sample_points,
            feat_hw=feature_map_size(processor, inference_state),
        )
        has_query_geometry = bool(box_inputs or point_inputs)
        has_exemplar_prompt = prompt is not None and prompt_mask is not None
        if not has_query_geometry and (
            self.point_prompt_mode == "point_only" or not has_exemplar_prompt
        ):
            return []
        processor.reset_all_prompts(inference_state)
        with autocast_scope(
            device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast
        ):
            if self.point_prompt_mode == "point_only":
                inference_state = processor.forward_grounding_with_geometry_prompts(
                    inference_state,
                    box_inputs,
                    box_labels,
                    point_inputs,
                    point_labels,
                )
            else:
                inference_state = self._run_label_grounding(
                    processor,
                    inference_state,
                    prompt,
                    prompt_mask,
                    box_inputs,
                    box_labels,
                    point_inputs,
                    point_labels,
                )
        scores = inference_state["scores"]
        masks = inference_state["masks"]
        if scores.numel() == 0:
            return []
        return self._masks_to_detections(
            scores,
            masks,
            label_id,
            label_pts_coords,
            point_indices,
            frame_width,
            frame_height,
            dino_feat_map=dino_feat_map,
        )

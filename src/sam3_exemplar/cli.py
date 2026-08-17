"""Offline SAM3 exemplar segmentation CLI (YAML config)."""

from __future__ import annotations

import argparse
import os
import pathlib
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from sam3_exemplar.images import autocast_scope
from sam3_exemplar.model import (
    Sam3InferenceConfig,
    Sam3ModelConfig,
    configure_torch_for_sam3,
    load_sam3_model,
    make_processor,
    resolve_autocast_dtype,
)
from sam3_exemplar.prompts import build_exemplar_data_dict, encode_exemplar_prompts
from sam3_exemplar.batch import process_images_with_center_crop
from sam3_exemplar.cli_config import (
    load_config_file,
    parse_kv_override,
    set_by_dotted_key,
    unique_run_dir,
    write_cli_invocation,
)
from sam3_exemplar.file_exemplars import (
    load_exemplar_annotations,
    load_squidle_annotation_df,
    select_k_shot_annotations,
    squidle_negative_exemplars_for_image,
    squidle_positive_label_ids_on_image,
)
from sam3_exemplar.viz import plot_exemplars_side_by_side, save_heatmap_per_image, show_three_panels

configure_torch_for_sam3()


class Sam3CliSegmenter:
    """Run SAM3 exemplar segmentation over a directory of images from a YAML config."""

    def __init__(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        self.config = config
        self.output_dir = unique_run_dir(config["output_dir"], config["run_id"])
        self.images_dir = config.get("images_dir", "./")
        self.sample_every_n = config.get("sample_every_n", 1)
        self.centre_crop_size = config.get("centre_crop_size", None)
        self.exemplar_annotation_file = config["exemplar_annotation_file"]
        self.exemplar_format = config["exemplar_format"]
        self.exemplar_dir = config.get("exemplar_dir", None)
        self.relative_coords = config.get("relative_coords", True)
        self.device = config.get("device", "cuda")

        with open(os.path.join(self.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

        s = config.get("sam3", {})
        self.use_autocast = bool(s.get("use_autocast", True))
        self.autocast_dtype = resolve_autocast_dtype(str(s.get("autocast_dtype", "bf16")))
        self.checkpoint_path = s.get("checkpoint_path", "../models/sam3.pt")
        self.confidence_threshold = float(s.get("confidence_threshold", 0.5))
        self.combine_prompts = s.get(
            "combine_prompts", s.get("embed_merge", "mean_ex_embed_per_image")
        )
        self.encode_text = s.get("encode_text", True)
        self.use_img_pos_embed = s.get("use_img_pos_embed", True)
        self.k_shot = int(s.get("k_shot", 5))
        self.n_negative = int(s.get("n_negative", 0))
        self.exemplar_seed = int(s.get("exemplar_seed", 42))
        self.sam_resolution = int(s.get("sam_resolution") or 1008)
        self.polygon_sample_points = int(s.get("polygon_sample_points", 8))

        self.model_config = Sam3ModelConfig(
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            use_autocast=self.use_autocast,
            autocast_dtype=str(s.get("autocast_dtype", "bf16")),
        )
        self.inference_config = Sam3InferenceConfig(
            sam_resolution=self.sam_resolution,
            confidence_threshold=self.confidence_threshold,
            polygon_sample_points=self.polygon_sample_points,
            combine_prompts=self.combine_prompts,
        )

    def _neg_fields_for_image(self, squidle_df: Optional[pd.DataFrame]):
        if squidle_df is None or self.n_negative <= 0:
            return lambda _media_path: ([], [], [])

        def _fn(media_path: str):
            positive_label_ids = squidle_positive_label_ids_on_image(squidle_df, media_path)
            return squidle_negative_exemplars_for_image(
                squidle_df,
                media_path,
                positive_label_ids,
                self.n_negative,
                seed=self.exemplar_seed,
            )

        return _fn

    def build_segmenter(self) -> None:
        exemplar_images, exemplar_bboxes, exemplar_points, exemplar_polygons, _sizes = (
            load_exemplar_annotations(
                self.exemplar_annotation_file,
                self.exemplar_format,
                exemplar_dir=self.exemplar_dir,
                relative_coords=self.relative_coords,
            )
        )
        if self.exemplar_format == "coco":
            exemplar_images = [os.path.join(self.exemplar_dir, img) for img in exemplar_images]

        squidle_df = None
        if self.exemplar_format == "squidle" and self.n_negative > 0:
            squidle_df = load_squidle_annotation_df(self.exemplar_annotation_file)

        if len(exemplar_images) == 0:
            raise ValueError("No exemplar images found in exemplar annotations.")

        sel_images, sel_bboxes, sel_points, sel_polygons, selected = select_k_shot_annotations(
            exemplar_images,
            exemplar_bboxes,
            exemplar_points,
            exemplar_polygons,
            k_shot=self.k_shot,
            seed=self.exemplar_seed,
        )
        print(
            f"Using {len(selected)} annotation(s) across {len(sel_images)} media "
            f"(k_shot={self.k_shot}, seed={self.exemplar_seed}); "
            f"selected={selected}"
        )

        exemplar_img_file = Path(os.path.basename(self.exemplar_annotation_file)).with_suffix(".png")
        out_path = plot_exemplars_side_by_side(
            sel_images,
            sel_bboxes,
            sel_points,
            polygons_per_image=sel_polygons,
            outputs_dir=self.output_dir,
            exemplar_img_file=exemplar_img_file,
        )
        print(f"Saved exemplar images to {out_path}")

        self.model = load_sam3_model(self.model_config)
        processor = make_processor(self.model, self.inference_config)
        exemplar_data_dict = build_exemplar_data_dict(
            sel_images,
            sel_bboxes,
            sel_points,
            sel_polygons,
            neg_fields_fn=self._neg_fields_for_image(squidle_df),
        )
        with autocast_scope(
            device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast
        ):
            self.prompt_data = encode_exemplar_prompts(
                processor,
                exemplar_data_dict,
                combine_prompts=self.combine_prompts,
                encode_text=self.encode_text,
                use_img_pos_embed=self.use_img_pos_embed,
                n_polygon_sample_points=self.polygon_sample_points,
            )

    def segment_image(self, image_path: str, **kwargs) -> Any:
        target_image = kwargs.get("image")
        processor = make_processor(self.model, self.inference_config)
        with autocast_scope(
            device=self.device, dtype=self.autocast_dtype, enabled=self.use_autocast
        ):
            inference_state = processor.set_image(target_image)
            processor.reset_all_prompts(inference_state)
            prompt, prompt_mask = self.prompt_data
            inference_state = processor.forward_grounding_with_prompt_embeddings(
                inference_state, prompt, prompt_mask
            )
        masks = inference_state["masks"].squeeze()
        scores = inference_state["scores"]

        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.shape[0] == 0:
            heatmap_max = np.zeros(masks.shape[-2:], dtype=np.float32)
        else:
            if scores.dim() == 1:
                scores = scores.unsqueeze(1)
            heatmap_max = (masks * scores.unsqueeze(-1)).max(dim=0).values.float().cpu().numpy()

        results_file = os.path.basename(image_path)
        out_path = show_three_panels(
            target_image,
            inference_state,
            heatmap_max,
            output_dir=self.output_dir,
            results_filename=results_file,
        )
        print(f"Saved results to {out_path}")

        save_heatmap_per_image(heatmap_max, image_path, target_image, out_root=self.output_dir)
        return heatmap_max


def score_weighted_heatmap_torch(masks: torch.Tensor, scores: torch.Tensor):
    masks_f = masks.float()
    weighted = masks_f * scores.view(-1, 1, 1)
    return weighted.sum(dim=0) / (masks_f.sum(dim=0) + 1e-12)


def run(
    config: Dict[str, Any],
    *,
    args: Optional[argparse.Namespace] = None,
    argv: Optional[Sequence[str]] = None,
) -> None:
    print("[sam3-exemplar] Starting with config:")
    pprint(config)

    segmenter = Sam3CliSegmenter(config)
    segmenter.build_segmenter()
    if args is not None:
        write_cli_invocation(
            pathlib.Path(segmenter.output_dir),
            args,
            argv,
            script="sam3_exemplar.cli",
            log_prefix="[sam3-exemplar]",
        )

    process_images_with_center_crop(
        segmenter.images_dir,
        segmenter,
        sample_every_n=segmenter.sample_every_n,
        crop_size=segmenter.centre_crop_size,
    )
    print("[sam3-exemplar] Finished.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment images using SAM3 exemplar prompts (YAML config).",
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with key=value (supports dotted keys; JSON for lists).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print merged config then exit.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    config = load_config_file(args.config)
    for raw in args.set:
        key, val = parse_kv_override(raw)
        set_by_dotted_key(config, key, val)

    if args.dry_run:
        print("Merged config:")
        pprint(config)
        return 0

    run(config, args=args, argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

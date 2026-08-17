"""SAM3 model load, inference config, and processor factory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sam3
import torch

from sam3_exemplar.images import autocast_scope
from sam3_override.model_builder import build_sam3_image_model
from sam3_override.sam3_processor import Sam3Processor

SAM3_BPE_PATH = Path(sam3.__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "none": None,
}


def resolve_autocast_dtype(name: str) -> torch.dtype | None:
    return _DTYPE_MAP.get(str(name).lower(), torch.bfloat16)


@dataclass
class Sam3ModelConfig:
    device: str = "cuda"
    checkpoint_path: str = "models/sam3.pt"
    use_autocast: bool = True
    autocast_dtype: str = "bf16"
    points_direct_project: bool = True
    points_pool: bool = True
    points_pos_enc: bool = True
    boxes_direct_project: bool = True
    boxes_pool: bool = True
    boxes_pos_enc: bool = True


@dataclass
class Sam3InferenceConfig:
    sam_resolution: int = 1008
    confidence_threshold: float = 0.5
    polygon_sample_points: int = 8
    combine_prompts: str = "mean_ex_embed_per_image"
    max_masks_per_class: int | None = 100


def configure_torch_for_sam3() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_sam3_model(config: Sam3ModelConfig):
    """Build and eval-load the SAM3 image model."""
    dtype = resolve_autocast_dtype(config.autocast_dtype)
    with autocast_scope(device=config.device, dtype=dtype, enabled=config.use_autocast):
        model = build_sam3_image_model(
            bpe_path=SAM3_BPE_PATH,
            device=config.device,
            checkpoint_path=config.checkpoint_path,
            points_direct_project=config.points_direct_project,
            points_pool=config.points_pool,
            points_pos_enc=config.points_pos_enc,
            boxes_direct_project=config.boxes_direct_project,
            boxes_pool=config.boxes_pool,
            boxes_pos_enc=config.boxes_pos_enc,
        )
        model.to(config.device)
        model.eval()
    return model


def make_processor(model, config: Sam3InferenceConfig) -> Sam3Processor:
    processor = Sam3Processor(
        model,
        resolution=config.sam_resolution,
        confidence_threshold=config.confidence_threshold,
    )
    processor.max_masks_per_class = config.max_masks_per_class
    return processor

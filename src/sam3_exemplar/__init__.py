"""SAM3 exemplar segmentation — shared by sqbot and offline CLI."""

from sam3_exemplar.cli_config import load_config_file, set_by_dotted_key, unique_run_dir
from sam3_exemplar.file_exemplars import (
    DEFAULT_IMAGE_CACHE_DIR,
    annotated_image_indices,
    annotation_indices,
    load_exemplar_annotations,
    load_squidle_annotation_df,
    select_k_shot_annotations,
    squidle_negative_exemplars_for_image,
    squidle_positive_label_ids_on_image,
)
from sam3_exemplar.images import autocast_scope, load_image, numpy_bgr_to_pil_rgb
from sam3_exemplar.model import (
    SAM3_BPE_PATH,
    Sam3InferenceConfig,
    Sam3ModelConfig,
    configure_torch_for_sam3,
    load_sam3_model,
    make_processor,
    resolve_autocast_dtype,
)
from sam3_exemplar.prompts import (
    build_exemplar_data_dict,
    build_geometry_prompts,
    build_query_geometry_prompts,
    encode_exemplar_prompts,
    feature_grid_points_in_polygon,
    feature_map_size,
    parse_exemplar_media_fields,
)

try:
    from sam3_exemplar.squidle import build_exemplars_by_label_id_from_df
except ImportError:  # sqbot-framework optional extra not installed
    build_exemplars_by_label_id_from_df = None  # type: ignore[assignment]

from sam3_exemplar.batch import process_images_with_center_crop
from sam3_exemplar.cli import Sam3CliSegmenter, main as cli_main

__all__ = [
    "SAM3_BPE_PATH",
    "Sam3CliSegmenter",
    "Sam3InferenceConfig",
    "Sam3ModelConfig",
    "DEFAULT_IMAGE_CACHE_DIR",
    "annotated_image_indices",
    "annotation_indices",
    "autocast_scope",
    "build_exemplar_data_dict",
    "build_exemplars_by_label_id_from_df",
    "build_geometry_prompts",
    "build_query_geometry_prompts",
    "cli_main",
    "configure_torch_for_sam3",
    "encode_exemplar_prompts",
    "feature_grid_points_in_polygon",
    "feature_map_size",
    "load_config_file",
    "load_exemplar_annotations",
    "load_image",
    "load_sam3_model",
    "load_squidle_annotation_df",
    "make_processor",
    "numpy_bgr_to_pil_rgb",
    "parse_exemplar_media_fields",
    "process_images_with_center_crop",
    "resolve_autocast_dtype",
    "select_k_shot_annotations",
    "set_by_dotted_key",
    "squidle_negative_exemplars_for_image",
    "squidle_positive_label_ids_on_image",
    "unique_run_dir",
]

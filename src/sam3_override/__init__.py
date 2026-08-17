"""Patches to upstream SAM3 classes (model + processor)."""

from sam3_override.model_builder import build_sam3_image_model
from sam3_override.sam3_image import Sam3Image
from sam3_override.sam3_processor import Sam3Processor

__all__ = ["Sam3Image", "Sam3Processor", "build_sam3_image_model"]

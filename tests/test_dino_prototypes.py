"""Tests for DINOv2 prototype score refinement (synced scoring module)."""

import pytest
import torch
from PIL import Image

import exemplar_seg.dino_prototypes as dp


def test_fuse_defaults_to_journal_alpha():
    assert dp.DEFAULT_DINO_SCORE_ALPHA == 0.5
    assert dp.fuse_sam3_dino_score(0.8, 0.4) == pytest.approx(0.6)
    assert dp.fuse_sam3_dino_score(0.8, 0.4, alpha=1.0) == pytest.approx(0.8)
    # Fused score is what the bot stores as the Squidle likelihood.
    extras = dp.dino_row_extras({"7": 0.4}, 7, alpha=0.5, sam3_score=0.8)
    assert extras["score"] == pytest.approx(0.6)
    assert extras["sam3_score"] == pytest.approx(0.8)
    assert extras["dino_cosine_pred"] == pytest.approx(0.4)


def test_norm_geometry_mask_polygon_and_bbox():
    poly = dp.norm_geometry_mask(
        width=10, height=10, polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
    )
    assert poly.sum() > 20
    assert poly[9, 9] == 0
    bbox = dp.norm_geometry_mask(width=10, height=10, bbox=(0.5, 0.5, 0.5, 0.5))
    assert bbox[9, 9] == 1
    assert bbox[0, 0] == 0


def test_build_prototypes_from_exemplar_dict(monkeypatch):
    monkeypatch.setattr(
        dp, "get_dinov2_features", lambda model, pil, *, device, target_long_side: torch.ones(1, 4, 2, 2)
    )
    exemplars = {
        7: {
            "img1": [
                [(0.0, 0.0, 0.5, 0.5)],
                [(0.25, 0.25)],
                [],
                [],
                [[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]],
                [],
            ]
        }
    }
    proto, labels = dp.build_prototypes_from_exemplar_dict(
        exemplars, object(), device="cpu", image_loader=lambda key: Image.new("RGB", (10, 10))
    )
    assert labels == [7]
    assert proto.shape == (4, 1)


def test_bot_candidates_store_fused_score():
    import numpy as np
    from types import SimpleNamespace

    from sam3_segment_bot.sam3_segmentation import SAM3Segmenter

    bot = object.__new__(SAM3Segmenter)
    proto, labels = dp.build_prototypes({7: [torch.ones(4)]}, device="cpu")
    bot.dino_proto_feat = proto
    bot.dino_proto_label_ids = labels
    bot.dino_score_alpha = 0.5

    masks = torch.zeros(1, 1, 20, 20)
    masks[0, 0, 2:10, 2:10] = 1.0
    scores = torch.tensor([0.8])
    mediaobj = SimpleNamespace(width=20, height=20)
    empty = np.empty((0, 2), dtype=int)
    cands = bot._masks_to_candidates(
        scores,
        masks,
        7,
        empty,
        [],
        empty,
        [],
        mediaobj,
        dino_feat_map=torch.ones(1, 4, 2, 2),
    )
    assert len(cands) == 1
    # One-hot-ish mask pools to the (all-ones) prototype: cosine ~1 -> fused ~0.9
    assert cands[0].score == pytest.approx(0.5 * 0.8 + 0.5 * 1.0, abs=0.05)


def test_score_masks_against_prototypes():
    feat = torch.ones(1, 4, 2, 2)
    proto, labels = dp.build_prototypes({7: [torch.ones(4)]}, device="cpu")
    masks = torch.ones(2, 4, 4)
    sims, rows = dp.score_masks_against_prototypes(feat, masks, proto, labels)
    assert sims.shape == (2, 1)
    assert rows[0]["7"] == pytest.approx(1.0, abs=1e-5)

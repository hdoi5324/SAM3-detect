# SAM3-detect

SAM3 exemplar segmentation bot for Squidle+ (sqbot). Given a small set of
exemplar annotations in a base Squidle annotation set, the bot segments the
same objects in the target media and creates/updates point annotations with
polygons.

Installed by sqbot as a git dependency:

```toml
sam3-exemplar-detect = { git = "https://github.com/hdoi5324/SAM3-detect" }
```

## Method

SAM3 exemplar prompts propose candidate polygons. From this branch, candidate
scores are refined with DINOv2 class prototypes (built once per label from
the exemplar images/polygons):

```
score = alpha * SAM3 + (1 - alpha) * DINO cosine     (alpha = 0.5, journal default)
```

The **fused score** is stored as the Squidle `likelihood` and is also what
drives overlapping-candidate suppression and point assignment/updates. Set
`dino_enabled=False` (or `dino_score_alpha=1.0`) for SAM3-only behaviour.

Bot parameters (sqbot `create_parser` arguments):

| Parameter | Default | Meaning |
|---|---|---|
| `dino_enabled` | `True` | Enable DINOv2 prototype refinement |
| `dino_score_alpha` | `0.5` | Fusion weight (1.0 = SAM3-only) |
| `dinov2_name` | `dinov2_vitl14` | torch.hub DINOv2 model (weights auto-download) |
| `dino_long_side` | `630` | DINO input long side |
| `dino_device` | `None` (auto) | DINO device; auto follows the SAM3 device |

DINO failures (download, GPU memory) are non-fatal: the bot logs a warning
and continues with SAM3-only scores. DINOv2 runs on the GPU and stays
resident alongside SAM3 (target machines have enough VRAM); overlapping
candidates are suppressed with global, class-agnostic polygon NMS
(`exemplar_seg.postprocess.suppress_overlapping_candidates`).

## Layout

| Path | Contents |
|---|---|
| `src/sam3_segment_bot/` | The Squidle `Annotator` bot |
| `src/exemplar_seg/dino_prototypes.py` | DINOv2 prototypes + fusion — **synced** from ExemplarSegmentation (`fsod_eval/dino_prototypes.py`) via `scripts/export_to_sam3_detect.py`; do not edit here |
| `src/exemplar_seg/postprocess.py` | Candidate NMS / point assignment |
| `src/sam3_override/` | Local SAM3 model/processor overrides |

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

## Citation

Please cite this software using [`CITATION.cff`](CITATION.cff) (synced from
the canonical ExemplarSegmentation repository).

## Licence

Licensed under the Apache License, Version 2.0 (see `LICENSE`).

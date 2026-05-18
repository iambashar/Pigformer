# PigFormer

End-to-end two-stage system for regressing pig body-condition measurements
(backfat, loin muscle depth, total tissue depth at the last rib) from a
ceiling-mounted Azure Kinect / Orbbec depth camera.

- **Stage 1 (geometric front-end)** — depth-only segmentation (SAM3-to-MaskDINO
  distillation), RANSAC ground-plane removal, BEV projection, and
  orientation normalization. Produces a standardized 96×224 height map.
- **Stage 2 (Slice Attention Encoder)** — a single RoPE transformer layer
  over 224 cross-sectional slice tokens, dual mean+max pooling, MLP head
  to three regression targets.

Three interchangeable Stage 1 segmenters are supported:

| Stage 1 | Backbone | Stage 1 (ms / frame, A100) | End-to-end MAE |
|---|---|---:|---:|
| MaskDINO (paper) | R50, 300q, 9 dec | 106.92 | 3.87 mm |
| Pruned MaskDINO | R18, 50q, 5 dec | 52.73 | 3.94 mm |
| UNet | MobileNetV3-Small | 6.58 | 3.95 mm |

Stage 2 takes ≈0.50 ms / frame on top. End-to-end with the UNet front-end
is ≈7 ms / frame, fast enough for real-time monitoring on a single A100.

## Repo layout

```
├── dataset.py            # PigDataset + AllFramesIterator (HDF5 height-map loader)
├── models.py             # PigFormer + MLP / CNN Stage-2 alternatives
├── split.py              # Identity-level train / val / test split
├── train.py              # Fold-0 training (AdamW + cosine + IQR-weighted L1 / Huber)
├── evaluate.py           # Per-bag evaluation from a checkpoint
├── evaluate_ensemble.py  # 4-fold cross-validation ensemble evaluation
├── preprocessing/        # ROS bag → MaskDINO → height map → dataset.h5
│   ├── rosbag_to_h5.py        # Stage 0: extract depth + camera intrinsics
│   ├── maskdino/              # Stage 1a: v1 MaskDINO inference (R50+300q+9L)
│   ├── maskdino_v2/           # Stage 1b: pruned MaskDINO (R18+50q+5L)
│   ├── unet_depth.py          # Stage 1c: UNet segmenter
│   ├── build_height_dataset.py# Stage 2: ground-plane + BEV height map
│   ├── msu_ground_plane.py    # Per-date plane caching
│   ├── parse_labels.py        # Slaughter-lab CSV → label.h5
│   └── camera_params/         # Per-recording Orbbec intrinsics
├── scripts/              # Auxiliary scripts (inference profiling, viz, baselines)
├── data/                 # dataset.h5, label.h5, split.json (not in git)
└── weights/              # pretrained checkpoints (not in git)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# Optional dev deps for visualization and classical-ML baseline:
# pip install -e ".[dev]"
```

## Reproduce the paper's test numbers

The bundled `weights/pigformer_fold0.pt` checkpoint regenerates the paper's
fold-0 test numbers (3.91 mm overall on single-fold-0 / input-aggregation;
3.87 mm on 4-fold ensemble / output-aggregation):

```bash
# Single-fold-0 (matches paper Table 3.91 mm overall)
python evaluate.py \
    --checkpoint weights/pigformer_fold0.pt \
    --dataset data/dataset.h5 --labels data/label.h5 --split_json data/split.json

# 4-fold ensemble (matches paper Table 1 headline 3.87 mm overall)
python evaluate_ensemble.py \
    --checkpoints results/fold0/best.pt results/fold1/best.pt results/fold2/best.pt results/fold3/best.pt \
    --dataset data/dataset.h5 --labels data/label.h5 --split_json data/split.json \
    --aggregation output
```

`--aggregation input` averages height maps before one forward pass (3.91 mm).
`--aggregation output` forwards every frame and averages predictions
(3.87 mm). The paper headline uses output aggregation across the 4-fold
ensemble; the bundled single-fold checkpoint reproduces the 3.91 mm number.

## Train from scratch (paper protocol)

```bash
python train.py --arch pigformer \
    --dataset data/dataset.h5 --labels data/label.h5 --split_json data/split.json \
    --results_dir results/pigformer_fold0 \
    --epochs 5000 --warmup_epochs 10 --lr 3e-4 --weight_decay 0.05 \
    --batch_size 32 --moderate_aug \
    --loss huber --huber_delta 1.0 \
    --selection_metric overall_mae --val_aggregation output \
    --fold 0
```

Run for folds 0–3 to assemble the ensemble. Each fold takes ≈50 min on an A100.

Stage-2 architecture baselines (consume the same height map):
- MLP encoder: `--arch mlp`
- CNN encoder (auto-switches to 3-channel `height + valid mask + gradient`): `--arch cnn`

## Preprocessing pipeline

End-to-end path from ROS2 bags to `data/dataset.h5` + `data/label.h5`:

1. `preprocessing/rosbag_to_h5.py` — extract synced color + depth + intrinsics.
2. `preprocessing/maskdino/infer_pig_depth_h5.py` (or `maskdino_v2/` for the
   pruned variant, or `unet_depth.py` for the UNet) — predict pig / upper-body
   masks from depth alone.
3. `preprocessing/build_height_dataset.py` — RANSAC ground-plane removal,
   BEV projection at 1 cm × 1 cm, min-area-rectangle long-axis + upper-body
   centroid for heading, lateral crop to 96 × 224.
4. `preprocessing/parse_labels.py` — aggregate slaughter-lab CSV into
   `label.h5`.

See `preprocessing/README.md` for full details and flags. Stage 1 alternatives
share the same pipeline downstream of segmentation — switch by passing
`--maskdino_config`, `--maskdino_weights`, or `--unet_weights` to
`build_height_dataset.py`.

## Citation

If you use this code or the trained checkpoints, please cite:

```bibtex
@inproceedings{bashar2026pigformer,
  title={What's Under the Skin? Estimating Swine Body Condition},
  author={Bashar, Mk and ...},
  booktitle={CV4Animals Workshop, CVPR},
  year={2026}
}
```

See `CITATION.cff` for the canonical machine-readable form.

## License

MIT (see `LICENSE`).

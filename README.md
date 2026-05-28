# PigFormer

End-to-end two-stage system for regressing pig body-condition measurements
(backfat thickness, loin muscle depth, total tissue depth at the last rib)
from a ceiling-mounted Azure Kinect / Orbbec depth camera.

**Project page:** <https://pigformer.github.io>

![PigFormer pipeline](https://pigformer.github.io/pipeline.png)

- **Stage 1 (geometric front-end)** — depth-only segmentation (SAM3-to-MaskDINO
  distillation), RANSAC ground-plane removal, BEV projection, and orientation
  normalization. Produces a standardized 96×224 height map.
- **Stage 2 (Slice Attention Encoder)** — a single RoPE transformer layer
  over 224 cross-sectional slice tokens, dual mean+max pooling, MLP head to
  three regression targets.

## Results

Held-out test results on 79 sow / gilt instances. MAE in mm. Per-frame
inference measured on A100 with batch = 1 (MaskDINO Stage 1 in fp16; UNet
Stage 1, single-stage backbones, and PigFormer Stage 2 in fp32). Single-stage
baselines feed raw depth directly to an ImageNet-pretrained backbone and
predict fat and loin only (total is $\hat{y}_f + \hat{y}_l$ at evaluation).
PigFormer numbers are 4-fold cross-validation ensembles with output
aggregation. Best MAE in **bold**.

| Method | Backbone | Stage 1 (ms) | Stage 2 (ms) | Fat ↓ | Loin ↓ | Total ↓ | Overall ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| ViT-small (single-stage) | ViT-S/16 | — | 4.98 | 3.57 | 7.29 | 8.16 | 6.34 |
| ResNet-18 (single-stage) | ResNet-18 | — | 2.88 | 2.88 | 6.10 | 5.81 | 4.93 |
| **PigFormer** | MaskDINO (R50-300q-9L) | 106.92 | 0.50 | 2.43 | **5.01** | **4.19** | **3.87** |
| **PigFormer** | Pruned MaskDINO (R18-50q-5L) | 52.73 | 0.50 | **2.34** | 5.27 | 4.20 | 3.94 |
| **PigFormer** | UNet (MobileNetV3-Small) | 6.58 | 0.50 | 2.40 | 5.20 | 4.26 | 3.95 |
| Human Ultrasound Std | — | — | — | 1.30 | 2.02 | 2.29 | 1.87 |

End-to-end with the UNet front-end is ≈7 ms / frame, fast enough for
real-time monitoring on a single A100. The pruned MaskDINO retains the
detection-style inductive bias for out-of-distribution content (handlers,
empty pens) at half the latency of the original.

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

## Reproduce the paper's headline result (3.87 mm overall MAE)

The headline number is a 4-fold ensemble with output aggregation:

```bash
python evaluate_ensemble.py \
    --checkpoints results/fold0/best.pt results/fold1/best.pt results/fold2/best.pt results/fold3/best.pt \
    --dataset data/dataset.h5 --labels data/label.h5 --split_json data/split.json \
    --aggregation output
```

Single-fold evaluation from one checkpoint:

```bash
python evaluate.py \
    --checkpoint weights/pigformer_fold0.pt \
    --dataset data/dataset.h5 --labels data/label.h5 --split_json data/split.json
```

`--aggregation input` averages height maps before one forward pass.
`--aggregation output` forwards every frame and averages predictions.

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

```bibtex
@inproceedings{bashar2026pigformer,
  title     = {What's Under the Skin? Estimating Swine Body Condition},
  author    = {Bashar, Mk and Bhatti, Kuljit and Rohrer, Gary
               and Benjamin, Madonna and Brown-Brandl, Tami
               and Morris, Daniel},
  booktitle = {CV4Animals Workshop, IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

See `CITATION.cff` for the canonical machine-readable form.

## License

GNU General Public License v3.0 (GPLv3). See `LICENSE`.

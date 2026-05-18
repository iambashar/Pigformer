# MaskDINO v2 — efficient pig + upper-body depth segmenter

A retrain of the Stage-1 MaskDINO segmenter targeting **~16 ms/frame** on
H200 (vs 59 ms for v1), without changing what it produces (pig mask +
upper-body mask + endpoints, 2 classes).

## What changed and why

**Train-time augmentations** (see [mapper.py](mapper.py) `_build_transform_gen`):
- horizontal flip (head↔tail) p=0.5
- vertical flip (left↔right body side) p=0.5
- random rotation ±30° (range sample style; pigs appear at varied yaw in the depth FOV)
- additive depth offset ±0.3 m (per-image scalar, simulates camera-height variation; valid pixels only)
- `ResizeShortestEdge` to 576/640 (no-op since input is already that size)

`INPUT.DEPTH_OFFSET_MAX_M` controls the depth-offset range (default 0.3); set to 0 to disable.

| Knob | v1 | v2 | Why |
|---|---|---|---|
| Decoder layers (`MaskDINO.DEC_LAYERS`) | 9 | **3** | Decoder dominates 80% of S1 time; depth is COCO-tuned overkill for a 2-instance task. |
| Object queries (`NUM_OBJECT_QUERIES`) | 300 | **10** | 2 GTs per image (pig + upper-body); 5× matching slack keeps Hungarian + DN viable. |
| DN noisy queries (`DN_NUM`) | 100 | **20** | Scaled with the query budget. |
| Input channels | 3 (depth + valid + grad) | **1** (raw depth) | Encoded RGB hack from when conv1 needed 3-ch ImageNet init; backbone can learn equivalents. Stem `in_channels=1` is derived automatically from `len(PIXEL_MEAN)` — no `STEM_IN_CHANNELS` field. |
| Input range | per-frame 2/98-percentile to [0, 255] uint8 | **z-score by training μ/σ in meters** (μ=2.0191, σ=0.4183) | Per-frame rescale erases absolute depth (a body-condition signal). |
| Conv1 init | 3×64×7×7 ImageNet | **mean-collapsed 1×64×7×7** | Standard timm convention for n-channel surgery. |
| Param count | 52.15 M | **42.87 M** | Smaller decoder + smaller queries. |

Expected stage-1 ms / frame on H200, batch=1, 576×640:

| | v1 | v2 |
|---|---:|---:|
| FP32 | 59.0 | ~16 |
| TF32 | 53 | ~14 |
| TF32 + `torch.compile` | 42 | ~11 |

## Files in this directory

| File | Lives where | Purpose |
|---|---|---|
| `maskdino_R50_depth_v2.yaml` | here | Detectron2 config. Inherits the COCO base, overrides what we changed. |
| `compute_depth_stats.py` | here | One-shot utility: training-fold μ/σ in meters → paste into the YAML. |
| `convert_r50_conv1_to_1ch.py` | here | Average pretrained R50 conv1 (3→1 channel) for a 1-channel warm start. |
| `export_pig_depth_coco_v2.py` | here | Re-export the COCO dataset as 1-channel uint16 mm PNGs. |
| `infer_pig_depth_h5_v2.py` | here | Inference on raw depth HDF5 (drop-in replacement for the v1 script). |
| `mapper.py` (`Depth1ChannelDatasetMapper`) | **copy into MaskDINO repo** | Custom DatasetMapper — `utils.read_image` can't load uint16 grayscale. |
| `register_pig_depth_v2.py` | **copy into MaskDINO repo** | Detectron2 dataset registration for the v2 export root. |
| `train_net_patch.diff` | apply to MaskDINO repo | Wires `DATASET_MAPPER_NAME: "depth_1ch_instance"` into `train_net.py`. |

## End-to-end retrain procedure

All paths assume:
- pigformer release repo at `/mnt/gs21/scratch/basharmk/data/unl/pigformer_release`
- MaskDINO repo at `/mnt/gs21/scratch/basharmk/data/unl/MaskDINO`
- existing v1 dataset root at `$PIG_DEPTH_DATASET_ROOT` (default `MaskDINO/datasets/pig_depth_combined`)

### 1. Compute depth normalization stats (training fold only)

```bash
. run.sh
python preprocessing/maskdino_v2/compute_depth_stats.py \
    --source-h5 /path/to/combined_upadted_upper_body.h5 \
    --splits-json $PIG_DEPTH_DATASET_ROOT/splits.json
```

Paste the printed `PIXEL_MEAN` / `PIXEL_STD` into `maskdino_R50_depth_v2.yaml`
(replacing the placeholder values 2.10 / 0.49).

### 2. Re-export the COCO dataset as 1-channel uint16 PNGs

```bash
python preprocessing/maskdino_v2/export_pig_depth_coco_v2.py \
    --source-h5 /path/to/combined_upadted_upper_body.h5 \
    --src-coco $PIG_DEPTH_DATASET_ROOT \
    --output-root /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined_v2
```

Annotations and split JSON are copied verbatim — only the image PNGs change.

### 3. Convert R50 conv1 to 1-channel for a warm start

```bash
python preprocessing/maskdino_v2/convert_r50_conv1_to_1ch.py \
    --in-pkl  ~/.torch/iopath_cache/detectron2/ImageNetPretrained/torchvision/R-50.pkl \
    --out-pkl /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/weights/r50_conv1_1ch.pkl
```

Detectron2 caches the catalog URI to that path on first download. If you
haven't downloaded R50 yet, run any v1 training command once to trigger
the fetch.

### 4. Wire the new mapper + dataset into the MaskDINO repo

```bash
cd /mnt/gs21/scratch/basharmk/data/unl/MaskDINO

cp /mnt/gs21/scratch/basharmk/data/unl/pigformer_release/preprocessing/maskdino_v2/mapper.py \
   maskdino/data/dataset_mappers/depth_1ch_dataset_mapper.py

cp /mnt/gs21/scratch/basharmk/data/unl/pigformer_release/preprocessing/maskdino_v2/register_pig_depth_v2.py \
   maskdino/data/datasets/register_pig_depth_v2.py

# Add `from . import register_pig_depth_v2` at the bottom of
# maskdino/data/datasets/__init__.py.

patch -p1 < /mnt/gs21/scratch/basharmk/data/unl/pigformer_release/preprocessing/maskdino_v2/train_net_patch.diff
```

### 5. Train

```bash
cd /mnt/gs21/scratch/basharmk/data/unl/MaskDINO

export PIG_DEPTH_V2_DATASET_ROOT=$PWD/datasets/pig_depth_combined_v2
export DETECTRON2_DATASETS=$PWD/datasets

python train_net.py \
    --config-file /mnt/gs21/scratch/basharmk/data/unl/pigformer_release/preprocessing/maskdino_v2/maskdino_R50_depth_v2.yaml \
    --num-gpus 1 \
    OUTPUT_DIR ./output/pig_depth_v2_endpoint \
    MODEL.WEIGHTS ./weights/r50_conv1_1ch.pkl
```

10k iters at IMS_PER_BATCH=4 ≈ 6 h on a single H200 (smaller decoder is
faster at training too). Watch val mask AP — it should land within
~1–2 AP of v1; if it drops more than that, bump `DEC_LAYERS` to 4–5
before adding queries back.

### 6. Inference

```bash
python preprocessing/maskdino_v2/infer_pig_depth_h5_v2.py \
    --config-file preprocessing/maskdino_v2/maskdino_R50_depth_v2.yaml \
    --weights /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/output/pig_depth_v2_endpoint/model_final.pth \
    --input-h5  /path/to/Recording*.h5 \
    --output-h5 /path/to/out_v2.h5 \
    --torch-compile
```

`--torch-compile` adds the H200 ~1.3× win on top of the architectural
gains. AMP is **off** by default — the deformable-attn op is FP32-locked
so autocast doesn't help; leaving it on just adds dtype-conversion churn.

## Validation gates

Before declaring v2 the new default, verify:

1. **Mask quality on the existing test split**: AP@[.5:.95] within ~1–2 of v1.
2. **Downstream MAE**: feed v2 masks into `build_height_dataset.py` and
   train one PigFormer fold-0 from scratch. MAE should be within
   ~0.1 mm of the v1 (original MaskDINO) result.
3. **Inference latency**: re-run `scripts/benchmark_maskdino_quick.py`
   pointed at the v2 config + weights. Target ~16 ms FP32 / ~11 ms with
   compile.

## Rollback

v2 is fully isolated:
- Different config file, different OUTPUT_DIR, different dataset root,
  different dataset names (`pig_depth_v2_*`).
- The v1 pipeline is untouched.

If v2 underperforms on quality, fall back by simply running v1 inference.
The patch to `train_net.py` is additive (a new `elif` branch); v1
training paths are unaffected.

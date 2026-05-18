"""MaskDINO v2 inference on raw depth HDF5 — 1-channel input + smaller decoder.

Mirrors preprocessing/maskdino/infer_pig_depth_h5.py but consumes raw
uint16 depth directly:
  uint16 mm  →  float32 / 1000 (meters)  →  z-score by training-fold μ/σ
  →  feed as (1, H, W) into MaskDINO with STEM_IN_CHANNELS=1.

No per-frame percentile rescale, no gradient channel, no valid-mask channel.

Usage:
    python preprocessing/maskdino_v2/infer_pig_depth_h5_v2.py \\
        --config-file preprocessing/maskdino_v2/maskdino_R50_depth_v2.yaml \\
        --weights /path/to/maskdino_v2_final.pth \\
        --input-h5  /path/to/Recording*.h5 \\
        --output-h5 /path/to/out.h5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# MaskDINO config registration (assumes the MaskDINO repo is on PYTHONPATH).
from maskdino import add_maskdino_config


DEFAULT_IMAGE_KEY_CANDIDATES = (
    "depth_maps_undistorted",
    "depth_maps_distorted",
    "depth_maps",
    "depth",
    "depth/images",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--input-h5", required=True)
    p.add_argument("--output-h5", required=True)
    p.add_argument("--image-key", default="")
    p.add_argument("--output-mask-key", default="pig_masks")
    p.add_argument("--output-score-key", default="pig_scores")
    p.add_argument("--output-endpoint-key", default="pig_endpoints_xy")
    p.add_argument("--output-endpoint-valid-key", default="pig_endpoint_valid")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--pig-class-index", type=int, default=0)
    p.add_argument("--depth-scale", type=float, default=1000.0,
                   help="Divisor mapping raw depth → meters. uint16 mm → 1000.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                   help="Autocast off by default; deformable-attn op is FP32-locked so AMP doesn't help. "
                        "Leave off unless verifying.")
    p.add_argument("--torch-compile", action="store_true",
                   help="Wrap model with torch.compile(mode='reduce-overhead') for ~1.3× speedup.")
    p.add_argument("--compression", default="gzip", choices=("gzip", "none"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def setup_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = args.score_threshold
    cfg.freeze()
    return cfg


def resolve_image_key(handle: h5py.File, requested_key: str) -> str:
    if requested_key:
        if requested_key in handle:
            return requested_key
        raise KeyError(f"Image key '{requested_key}' not found.")
    for key in DEFAULT_IMAGE_KEY_CANDIDATES:
        if key in handle and isinstance(handle[key], h5py.Dataset) and handle[key].ndim == 3:
            return key
    raise KeyError("Could not auto-detect a 3D depth dataset.")


def build_model_and_augment(cfg):
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    augment = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    return model, augment


def prepare_batch_inputs(depth_batch: np.ndarray, augment, device: torch.device,
                         depth_scale: float, undistort_maps=None):
    """depth_batch: (B, H, W) uint16/float in raw units. Returns Detectron2 input list.

    If `undistort_maps=(map_x, map_y)` is provided, depth is undistorted before
    encoding (v2 was trained on undistorted GT, so input must match)."""
    import cv2
    batch_inputs = []
    for raw in depth_batch:
        h, w = raw.shape
        if undistort_maps is not None:
            mx, my = undistort_maps
            raw = cv2.remap(raw.astype(np.uint16), mx, my,
                            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        depth_m = raw.astype(np.float32) / depth_scale
        depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0.0)
        # (H, W, 1) so detectron2 transforms preserve the channel axis.
        depth_3d = depth_m[..., None]
        transformed = augment.get_transform(depth_3d).apply_image(depth_3d)
        # → (1, H', W') float32. Mean/std normalization happens inside the
        # MaskDINO forward via PIXEL_MEAN/STD from cfg.
        image_tensor = torch.as_tensor(transformed.astype("float32").transpose(2, 0, 1))
        if device.type == "cuda":
            image_tensor = image_tensor.to(device, non_blocking=True)
        batch_inputs.append({"image": image_tensor, "height": int(h), "width": int(w)})
    return batch_inputs


def select_pig_prediction(output, shape, pig_class_index: int, score_threshold: float):
    instances = output["instances"].to("cpu")
    if len(instances) == 0:
        return np.zeros(shape, np.uint8), 0.0, np.zeros((4,), np.float32), 0
    classes = instances.pred_classes.numpy()
    pig = np.where(classes == pig_class_index)[0]
    if pig.size == 0:
        return np.zeros(shape, np.uint8), 0.0, np.zeros((4,), np.float32), 0
    scores = instances.scores.numpy()
    best = pig[int(np.argmax(scores[pig]))]
    if float(scores[best]) < score_threshold:
        return np.zeros(shape, np.uint8), 0.0, np.zeros((4,), np.float32), 0
    masks = instances.pred_masks.numpy().astype(np.uint8)
    ep = np.zeros((4,), np.float32)
    ep_valid = 0
    if instances.has("pred_endpoints"):
        ep_arr = instances.pred_endpoints.numpy().astype(np.float32)
        if 0 <= best < ep_arr.shape[0]:
            ep = ep_arr[best]
            ep_valid = 1
    return masks[best], float(scores[best]), ep, ep_valid


def compression_kwargs(kind: str) -> dict:
    if kind == "none":
        return {}
    return {"compression": "gzip", "compression_opts": 1, "shuffle": True}


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_h5).expanduser().resolve()
    out_path = Path(args.output_h5).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(out_path)
    if out_path.exists():
        out_path.unlink()

    cfg = setup_cfg(args)
    device = torch.device(cfg.MODEL.DEVICE)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, augment = build_model_and_augment(cfg)
    if args.torch_compile and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    with h5py.File(in_path, "r") as fin:
        image_key = resolve_image_key(fin, args.image_key)
        ds = fin[image_key]
        if ds.ndim != 3:
            raise ValueError(f"Expected (N,H,W), got {ds.shape}")
        N, H, W = int(ds.shape[0]), int(ds.shape[1]), int(ds.shape[2])
        start = max(int(args.start_index), 0)
        end = N if args.max_frames <= 0 else min(N, start + int(args.max_frames))
        if start >= end:
            raise ValueError(f"Empty range: {start}..{end}")
        num = end - start
        frame_indices = np.arange(start, end, dtype=np.int32)

        with h5py.File(out_path, "w") as fout:
            mask_ds = fout.create_dataset(args.output_mask_key, shape=(num, H, W), dtype=np.uint8,
                                          chunks=(1, H, W), **compression_kwargs(args.compression))
            score_ds = fout.create_dataset(args.output_score_key, shape=(num,), dtype=np.float32)
            ep_ds = fout.create_dataset(args.output_endpoint_key, shape=(num, 4), dtype=np.float32)
            epv_ds = fout.create_dataset(args.output_endpoint_valid_key, shape=(num,), dtype=np.uint8)
            fout.create_dataset("frame_indices", data=frame_indices, dtype=np.int32)

            t0 = time.time()
            positives = 0
            for bs in range(start, end, args.batch_size):
                be = min(bs + args.batch_size, end)
                ws = bs - start
                we = be - start
                depth_batch = np.asarray(ds[bs:be])
                inputs = prepare_batch_inputs(depth_batch, augment, device, args.depth_scale)
                with torch.inference_mode():
                    if args.amp and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outs = model(inputs)
                    else:
                        outs = model(inputs)
                masks_b = np.zeros((be - bs, H, W), np.uint8)
                scores_b = np.zeros((be - bs,), np.float32)
                eps_b = np.zeros((be - bs, 4), np.float32)
                epv_b = np.zeros((be - bs,), np.uint8)
                for j, o in enumerate(outs):
                    m, s, e, v = select_pig_prediction(o, (H, W), args.pig_class_index, args.score_threshold)
                    masks_b[j], scores_b[j], eps_b[j], epv_b[j] = m, s, e, v
                    if s > 0:
                        positives += 1
                mask_ds[ws:we] = masks_b
                score_ds[ws:we] = scores_b
                ep_ds[ws:we] = eps_b
                epv_ds[ws:we] = epv_b
                done = we
                fps = done / max(time.time() - t0, 1e-6)
                print(f"[v2] {done}/{num}  {fps:.2f} fps", flush=True)

    elapsed = time.time() - t0
    summary = {
        "input_h5": str(in_path), "output_h5": str(out_path), "image_key": image_key,
        "weights": args.weights, "config_file": args.config_file,
        "num_frames": num, "positive_frames": positives,
        "elapsed_seconds": float(elapsed),
        "frames_per_second": float(num / max(elapsed, 1e-6)),
        "depth_scale": args.depth_scale, "score_threshold": args.score_threshold,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

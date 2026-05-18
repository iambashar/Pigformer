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

from maskdino import add_maskdino_config


DEFAULT_IMAGE_KEY_CANDIDATES = (
    "depth_maps_undistorted",
    "depth_maps_distorted",
    "depth_maps",
    "depth",
)
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs/pig_depth/instance-segmentation/maskdino_R50_depth_instance_endpoint_pigonly.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "output/pig_depth_endpoint_pigonly_full_20260319_seed43_gpu1/model_best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batched MaskDINO inference on raw depth frames stored in a single HDF5 file "
            "and export pig masks, confidence scores, and ordered pig endpoints."
        )
    )
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_FILE),
        help=f"Model config file. Defaults to {DEFAULT_CONFIG_FILE}",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help=f"Checkpoint path. Defaults to {DEFAULT_WEIGHTS}",
    )
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument(
        "--image-key",
        default="",
        help="Depth dataset key inside the input HDF5. Empty means auto-detect.",
    )
    parser.add_argument("--output-mask-key", default="pig_masks")
    parser.add_argument("--output-score-key", default="pig_scores")
    parser.add_argument("--output-endpoint-key", default="pig_endpoints_xy")
    parser.add_argument("--output-endpoint-valid-key", default="pig_endpoint_valid")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on the number of frames to process. 0 means all frames from start-index.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument(
        "--pig-class-index",
        type=int,
        default=0,
        help="Contiguous class index for pig. Pig-only checkpoints also use 0.",
    )
    parser.add_argument(
        "--encoding",
        default="depth_valid_gradient",
        choices=("depth_valid_gradient", "depth_repeat", "raw_meters_1ch"),
    )
    parser.add_argument("--device", default="", help="Override cfg.MODEL.DEVICE, for example cuda:0 or cpu.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=("gzip", "none"),
        help="Compression for the output mask dataset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    if args.device:
        cfg.MODEL.DEVICE = args.device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = args.score_threshold
    cfg.freeze()
    return cfg


def resolve_image_key(handle: h5py.File, requested_key: str) -> str:
    if requested_key:
        if requested_key not in handle:
            available = ", ".join(sorted(handle.keys()))
            raise KeyError(f"Image key '{requested_key}' was not found. Available keys: {available}")
        return requested_key

    for key in DEFAULT_IMAGE_KEY_CANDIDATES:
        if key in handle and isinstance(handle[key], h5py.Dataset) and handle[key].ndim == 3:
            return key

    candidate_keys = [
        key
        for key, value in handle.items()
        if isinstance(value, h5py.Dataset)
        and value.ndim == 3
        and np.issubdtype(value.dtype, np.number)
    ]
    if len(candidate_keys) == 1:
        return candidate_keys[0]

    available = ", ".join(sorted(handle.keys()))
    raise KeyError(f"Could not auto-detect a 3D depth dataset from keys: {available}")


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros_like(depth, dtype=np.float32)
    if not np.any(valid):
        return normalized

    valid_values = depth[valid]
    lo = float(np.percentile(valid_values, 2.0))
    hi = float(np.percentile(valid_values, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(valid_values.min())
        hi = float(valid_values.max())
    if hi <= lo:
        return normalized

    normalized[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    return normalized


def encode_depth_image(depth_map: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "raw_meters_1ch":
        # MaskDINO v2 input: (H, W, 1) float32 meters. PIXEL_MEAN/STD
        # standardization happens inside the model. Source `depth_map`
        # is float32 mm (uint16 mm cast to float by the caller).
        depth_m = depth_map.astype(np.float32) / 1000.0
        depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0.0)
        return depth_m[..., None]

    normalized = normalize_depth(depth_map)
    valid_mask = (normalized > 0).astype(np.float32)
    if encoding == "depth_repeat":
        repeated = np.round(normalized * 255.0).astype(np.uint8)
        return np.stack([repeated, repeated, repeated], axis=-1)

    grad_y, grad_x = np.gradient(normalized)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_valid = grad_mag[valid_mask > 0]
    grad_scaled = np.zeros_like(grad_mag, dtype=np.float32)
    if grad_valid.size > 0:
        hi = float(np.percentile(grad_valid, 98.0))
        if hi > 0:
            grad_scaled = np.clip(grad_mag / hi, 0.0, 1.0)

    depth_channel = np.round(normalized * 255.0).astype(np.uint8)
    valid_channel = np.round(valid_mask * 255.0).astype(np.uint8)
    gradient_channel = np.round(grad_scaled * 255.0).astype(np.uint8)
    return np.stack([depth_channel, valid_channel, gradient_channel], axis=-1)


def build_model_and_augment(cfg):
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    augment = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    return model, augment


def prepare_batch_inputs(
    depth_batch: np.ndarray,
    augment,
    device: torch.device,
    encoding: str,
) -> list[dict[str, object]]:
    batch_inputs: list[dict[str, object]] = []
    for depth_map in depth_batch:
        height, width = depth_map.shape
        encoded_rgb = encode_depth_image(depth_map, encoding)
        transformed = augment.get_transform(encoded_rgb).apply_image(encoded_rgb)
        image_tensor = torch.as_tensor(transformed.astype("float32").transpose(2, 0, 1))
        if device.type == "cuda":
            image_tensor = image_tensor.to(device, non_blocking=True)
        batch_inputs.append(
            {
                "image": image_tensor,
                "height": int(height),
                "width": int(width),
            }
        )
    return batch_inputs


def select_pig_prediction(
    output: dict[str, object],
    shape: tuple[int, int],
    pig_class_index: int,
    score_threshold: float,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    instances = output["instances"].to("cpu")
    if len(instances) == 0:
        return np.zeros(shape, dtype=np.uint8), 0.0, np.zeros((4,), dtype=np.float32), 0

    pred_classes = instances.pred_classes.numpy()
    pig_indices = np.where(pred_classes == pig_class_index)[0]
    if pig_indices.size == 0:
        return np.zeros(shape, dtype=np.uint8), 0.0, np.zeros((4,), dtype=np.float32), 0

    pred_scores = instances.scores.numpy()
    best_position = pig_indices[int(np.argmax(pred_scores[pig_indices]))]
    best_score = float(pred_scores[best_position])
    if best_score < score_threshold:
        return np.zeros(shape, dtype=np.uint8), 0.0, np.zeros((4,), dtype=np.float32), 0

    pred_masks = instances.pred_masks.numpy().astype(np.uint8)
    endpoint = np.zeros((4,), dtype=np.float32)
    endpoint_valid = 0
    if instances.has("pred_endpoints"):
        pred_endpoints = instances.pred_endpoints.numpy().astype(np.float32)
        if 0 <= best_position < pred_endpoints.shape[0]:
            endpoint = pred_endpoints[best_position]
            endpoint_valid = 1

    return pred_masks[best_position], best_score, endpoint, endpoint_valid


def compression_kwargs(kind: str) -> dict[str, object]:
    if kind == "none":
        return {}
    return {"compression": "gzip", "compression_opts": 1, "shuffle": True}


def write_summary(summary_path: Path, payload: dict[str, object]) -> None:
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    input_h5 = Path(args.input_h5).expanduser().resolve()
    output_h5 = Path(args.output_h5).expanduser().resolve()
    summary_path = output_h5.with_suffix(".summary.json")
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    if output_h5.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_h5}")
    if output_h5.exists():
        output_h5.unlink()
    if summary_path.exists() and args.overwrite:
        summary_path.unlink()

    cfg = setup_cfg(args)
    device = torch.device(cfg.MODEL.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cfg requests CUDA, but CUDA is not available.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, augment = build_model_and_augment(cfg)

    with h5py.File(input_h5, "r") as input_handle:
        image_key = resolve_image_key(input_handle, args.image_key)
        depth_dataset = input_handle[image_key]
        if depth_dataset.ndim != 3:
            raise ValueError(f"Expected a [N, H, W] depth dataset, got shape {depth_dataset.shape}")

        total_frames = int(depth_dataset.shape[0])
        height = int(depth_dataset.shape[1])
        width = int(depth_dataset.shape[2])
        start_index = max(int(args.start_index), 0)
        end_index = total_frames
        if args.max_frames > 0:
            end_index = min(end_index, start_index + int(args.max_frames))
        if start_index >= end_index:
            raise ValueError(
                f"No frames selected: start_index={start_index}, end_index={end_index}, total_frames={total_frames}"
            )

        num_frames = end_index - start_index
        frame_indices = np.arange(start_index, end_index, dtype=np.int32)

        mask_chunks = (1, height, width)
        summary_payload: dict[str, object] = {
            "input_h5": str(input_h5),
            "output_h5": str(output_h5),
            "config_file": str(Path(args.config_file).expanduser().resolve()),
            "weights": str(Path(args.weights).expanduser().resolve()),
            "image_key": image_key,
            "output_mask_key": args.output_mask_key,
            "output_score_key": args.output_score_key,
            "output_endpoint_key": args.output_endpoint_key,
            "output_endpoint_valid_key": args.output_endpoint_valid_key,
            "encoding": args.encoding,
            "device": str(device),
            "batch_size": int(args.batch_size),
            "score_threshold": float(args.score_threshold),
            "pig_class_index": int(args.pig_class_index),
            "start_index": int(start_index),
            "end_index": int(end_index),
            "num_frames": int(num_frames),
            "height": int(height),
            "width": int(width),
        }

        started_at = time.time()
        positive_frames = 0
        endpoint_frames = 0
        score_sum = 0.0
        score_max = 0.0
        endpoint_length_sum = 0.0

        with h5py.File(output_h5, "w") as output_handle:
            mask_dataset = output_handle.create_dataset(
                args.output_mask_key,
                shape=(num_frames, height, width),
                dtype=np.uint8,
                chunks=mask_chunks,
                **compression_kwargs(args.compression),
            )
            score_dataset = output_handle.create_dataset(
                args.output_score_key,
                shape=(num_frames,),
                dtype=np.float32,
            )
            endpoint_dataset = output_handle.create_dataset(
                args.output_endpoint_key,
                shape=(num_frames, 4),
                dtype=np.float32,
            )
            endpoint_valid_dataset = output_handle.create_dataset(
                args.output_endpoint_valid_key,
                shape=(num_frames,),
                dtype=np.uint8,
            )
            output_handle.create_dataset("frame_indices", data=frame_indices, dtype=np.int32)

            output_handle.attrs["input_h5"] = str(input_h5)
            output_handle.attrs["image_key"] = image_key
            output_handle.attrs["weights"] = str(Path(args.weights).expanduser().resolve())
            output_handle.attrs["score_threshold"] = float(args.score_threshold)
            output_handle.attrs["pig_class_index"] = int(args.pig_class_index)
            output_handle.attrs["encoding"] = args.encoding
            output_handle.attrs["output_endpoint_key"] = args.output_endpoint_key
            output_handle.attrs["output_endpoint_valid_key"] = args.output_endpoint_valid_key

            for batch_start in range(start_index, end_index, args.batch_size):
                batch_end = min(batch_start + args.batch_size, end_index)
                write_start = batch_start - start_index
                write_end = batch_end - start_index

                depth_batch = np.asarray(depth_dataset[batch_start:batch_end], dtype=np.float32)
                batch_inputs = prepare_batch_inputs(
                    depth_batch=depth_batch,
                    augment=augment,
                    device=device,
                    encoding=args.encoding,
                )

                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=bool(args.amp and device.type == "cuda"),
                    ):
                        batch_outputs = model(batch_inputs)

                batch_masks = np.zeros((batch_end - batch_start, height, width), dtype=np.uint8)
                batch_scores = np.zeros((batch_end - batch_start,), dtype=np.float32)
                batch_endpoints = np.zeros((batch_end - batch_start, 4), dtype=np.float32)
                batch_endpoint_valid = np.zeros((batch_end - batch_start,), dtype=np.uint8)
                for index, output in enumerate(batch_outputs):
                    mask, score, endpoint, endpoint_valid = select_pig_prediction(
                        output=output,
                        shape=(height, width),
                        pig_class_index=args.pig_class_index,
                        score_threshold=args.score_threshold,
                    )
                    batch_masks[index] = mask
                    batch_scores[index] = score
                    batch_endpoints[index] = endpoint
                    batch_endpoint_valid[index] = endpoint_valid
                    if score > 0.0:
                        positive_frames += 1
                    if endpoint_valid:
                        endpoint_frames += 1
                        endpoint_length_sum += float(np.linalg.norm(endpoint[2:] - endpoint[:2]))
                    score_sum += float(score)
                    score_max = max(score_max, float(score))

                mask_dataset[write_start:write_end] = batch_masks
                score_dataset[write_start:write_end] = batch_scores
                endpoint_dataset[write_start:write_end] = batch_endpoints
                endpoint_valid_dataset[write_start:write_end] = batch_endpoint_valid

                elapsed = time.time() - started_at
                processed = write_end
                frames_per_second = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[infer_pig_depth_h5] processed {processed}/{num_frames} frames "
                    f"({frames_per_second:.2f} frames/s)",
                    flush=True,
                )

        elapsed = time.time() - started_at
        summary_payload.update(
            {
                "elapsed_seconds": float(elapsed),
                "frames_per_second": float(num_frames / elapsed if elapsed > 0 else 0.0),
                "positive_frames": int(positive_frames),
                "endpoint_frames": int(endpoint_frames),
                "mean_endpoint_length_px": float(endpoint_length_sum / max(endpoint_frames, 1)),
                "mean_score": float(score_sum / max(num_frames, 1)),
                "max_score": float(score_max),
                "summary_json": str(summary_path),
            }
        )
        write_summary(summary_path, summary_payload)
        print(json.dumps(summary_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

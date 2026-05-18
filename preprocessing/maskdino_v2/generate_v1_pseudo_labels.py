"""Generate pseudo-label COCO annotations using v1's production MaskDINO predictions.

For every image in pig_depth_combined_v2/annotations/instances_*.json,
we look up the corresponding raw bag depth, run v1 (production
pig_upper_body weights, NUM_CLASSES=2) on the raw distorted depth,
undistort the predicted masks so they line up with v2's already-
undistorted PNG inputs, and emit COCO-style RLE annotations.

Output: a new dataset directory mirroring pig_depth_combined_v2 but
with `annotations/instances_*.json` containing v1-pseudo-labels
instead of the (too-tight) original GT polygons.

Usage:
    python preprocessing/maskdino_v2/generate_v1_pseudo_labels.py \\
        --src-coco /mnt/.../pig_depth_combined_v2 \\
        --output-root /mnt/.../pig_depth_combined_v2_distill \\
        [--max-images 0]   # 0 = all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from pycocotools import mask as coco_mask

REPO = Path("/mnt/gs21/scratch/basharmk/data/unl/pigformer_release")
MASKDINO = Path("/mnt/gs21/scratch/basharmk/data/unl/MaskDINO")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MASKDINO))
sys.path.insert(0, str(REPO / "preprocessing"))

from msu_ground_plane import (  # noqa: E402
    get_camera_params_for_bag, load_depth_camera_info_from_hdf5, distortion_array_to_dict,
)
from preprocessing.maskdino.infer_pig_depth_h5 import (  # noqa: E402
    setup_cfg as v1_setup, build_model_and_augment as v1_build, prepare_batch_inputs as v1_prep,
)

V1_CFG = str(MASKDINO / "configs/pig_depth/instance-segmentation/maskdino_R50_depth_instance_endpoint_pig_upper_body.yaml")
V1_WEIGHTS = str(MASKDINO / "output/pig_depth_endpoint_pig_upper_body_full_20260321_4gpu/model_best.pth")

UNL_BAG_DIR = Path("/mnt/gs21/scratch/basharmk/data/unl/hdf5_out")
MSU_BAG_DIR = Path("/mnt/scratch/basharmk/data/body_condition/msu/bagfiles")
FRAME_RE = re.compile(r"_frame(\d+)$")


def load_v1():
    class _NS:
        config_file = V1_CFG; weights = V1_WEIGHTS; device = "cuda"; score_threshold = 0.3

    orig = os.getcwd(); os.chdir(MASKDINO)
    try:
        cfg = v1_setup(_NS())
    finally:
        os.chdir(orig)
    m, aug = v1_build(cfg)
    m.eval()
    return m, aug


def run_v1(model, augment, depth_distorted_uint16: np.ndarray):
    batch = v1_prep(depth_distorted_uint16[None].astype(np.float32), augment,
                    torch.device("cuda"), "depth_valid_gradient")
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            o = model(batch)[0]
    inst = o["instances"].to("cpu")
    return {
        "classes": inst.pred_classes.numpy() if len(inst) else np.array([], dtype=int),
        "scores": inst.scores.numpy() if len(inst) else np.array([]),
        "masks": inst.pred_masks.numpy().astype(bool) if len(inst) else np.zeros((0, *depth_distorted_uint16.shape), bool),
    }


def best_mask(out, ci, smin=0.3):
    if out["classes"].size == 0:
        return None, None
    sel = np.where(out["classes"] == ci)[0]
    if sel.size == 0:
        return None, None
    sel = sel[out["scores"][sel] >= smin]
    if sel.size == 0:
        return None, None
    j = sel[int(np.argmax(out["scores"][sel]))]
    return out["masks"][j], float(out["scores"][j])


def get_cam(bag_basename: str, hdf5_path: str):
    cam = get_camera_params_for_bag(bag_basename)
    if cam is not None:
        return cam
    intr, dist = load_depth_camera_info_from_hdf5(hdf5_path)
    return {"DepthIntrinsic": intr, "DepthDistortion": distortion_array_to_dict(dist)}


def encode_rle(bool_mask: np.ndarray) -> dict:
    """COCO-RLE in the dict-with-counts-string format (compatible with pycocotools)."""
    rle = coco_mask.encode(np.asfortranarray(bool_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    rle["size"] = list(rle["size"])
    return rle


def bbox_xywh_from_mask(bool_mask: np.ndarray):
    ys, xs = np.where(bool_mask)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    return [x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0]


def resolve_raw_path(image_rec):
    domain = image_rec.get("domain", "msu")
    bag = image_rec["bag_name"]
    if domain == "unl":
        return UNL_BAG_DIR / f"{bag}.h5"
    return MSU_BAG_DIR / f"{bag}.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-coco", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src_coco).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")
    if out.exists():
        shutil.rmtree(out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    # Symlink images from src — we don't change the depth PNGs.
    if not (out / "images").exists():
        os.symlink(src / "images", out / "images")
    # Carry over splits.json if present.
    if (src / "splits.json").is_file():
        shutil.copy2(src / "splits.json", out / "splits.json")

    print("loading v1 (production pig_upper_body)...")
    model, augment = load_v1()

    # Cache raw bag handle + undistortion remap per bag (faster than reopening).
    bag_cache = {}

    def get_remap(bag_basename, raw_path, H, W):
        if bag_basename in bag_cache:
            return bag_cache[bag_basename]
        cam = get_cam(bag_basename, raw_path)
        di, dd = cam["DepthIntrinsic"], cam["DepthDistortion"]
        K = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1]], dtype=np.float64)
        D = np.array([dd[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")], dtype=np.float64)
        mx, my = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (W, H), cv2.CV_32FC1)
        bag_cache[bag_basename] = (mx, my)
        return mx, my

    raw_handles: dict[str, h5py.File] = {}

    def open_bag(raw_path):
        sp = str(raw_path)
        if sp not in raw_handles:
            raw_handles[sp] = h5py.File(sp, "r")
        return raw_handles[sp]

    try:
        for split in ("train", "val", "test"):
            in_json = src / "annotations" / f"instances_{split}.json"
            if not in_json.is_file():
                continue
            payload = json.loads(in_json.read_text())
            images = payload["images"]
            if args.max_images > 0:
                images = images[: args.max_images]

            cats_by_id = {c["id"]: c["name"] for c in payload["categories"]}
            pig_id = next(c["id"] for c in payload["categories"] if c["name"] == "pig")
            up_id = next((c["id"] for c in payload["categories"] if c["name"] == "pig_upper_body"), None)

            kept_images = []
            kept_anns = []
            ann_id = 1
            n_skip = 0
            t0 = time.time()
            for i, im in enumerate(images):
                file_name = im["file_name"]
                domain = im.get("domain", "msu")
                bag_basename = im["bag_name"]
                source_name = im["source_name"]
                m = FRAME_RE.search(source_name)
                if not m:
                    n_skip += 1
                    continue
                frame_idx = int(m.group(1))

                raw_path = resolve_raw_path(im)
                if not raw_path.is_file():
                    n_skip += 1
                    continue
                f = open_bag(raw_path)
                depth_ds = f["depth/images"]
                if not (0 <= frame_idx < depth_ds.shape[0]):
                    n_skip += 1
                    continue
                depth_raw = np.asarray(depth_ds[frame_idx], dtype=np.uint16)
                H, W = depth_raw.shape

                # Run v1 on raw distorted.
                out_v1 = run_v1(model, augment, depth_raw)
                v1_pig_mask, v1_pig_score = best_mask(out_v1, 0, args.score_threshold)
                v1_up_mask, v1_up_score = best_mask(out_v1, 1, args.score_threshold)

                # Need at least the pig mask to keep the frame.
                if v1_pig_mask is None:
                    n_skip += 1
                    continue

                # Undistort the masks to match v2 PNG coords.
                mx, my = get_remap(bag_basename, str(raw_path), H, W)
                pig_und = cv2.remap(v1_pig_mask.astype(np.uint8), mx, my,
                                     cv2.INTER_NEAREST, borderValue=0).astype(bool)
                up_und = (cv2.remap(v1_up_mask.astype(np.uint8), mx, my,
                                    cv2.INTER_NEAREST, borderValue=0).astype(bool)
                          if v1_up_mask is not None else None)

                kept_images.append(im)
                # Pig annotation
                rle = encode_rle(pig_und)
                ann = {
                    "id": ann_id,
                    "image_id": int(im["id"]),
                    "category_id": int(pig_id),
                    "iscrowd": 0,
                    "bbox": bbox_xywh_from_mask(pig_und),
                    "bbox_mode": 1,  # XYWH_ABS
                    "area": float(int(pig_und.sum())),
                    "segmentation": rle,
                    "v1_score": float(v1_pig_score),
                }
                kept_anns.append(ann); ann_id += 1
                if up_und is not None and int(up_und.sum()) > 0 and up_id is not None:
                    rle_u = encode_rle(up_und)
                    kept_anns.append({
                        "id": ann_id,
                        "image_id": int(im["id"]),
                        "category_id": int(up_id),
                        "iscrowd": 0,
                        "bbox": bbox_xywh_from_mask(up_und),
                        "bbox_mode": 1,
                        "area": float(int(up_und.sum())),
                        "segmentation": rle_u,
                        "v1_score": float(v1_up_score),
                    })
                    ann_id += 1

                if (i + 1) % 100 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"  [{split}] {i+1}/{len(images)}  ({rate:.1f} img/s)  skipped={n_skip}")

            new_payload = dict(payload)
            new_payload["images"] = kept_images
            new_payload["annotations"] = kept_anns
            new_payload["info"] = payload.get("info", {})
            new_payload["info"]["pseudo_label_source"] = "v1_production_pig_upper_body"
            (out / "annotations" / f"instances_{split}.json").write_text(json.dumps(new_payload))
            print(f"[{split}] kept {len(kept_images)}/{len(images)} images, "
                  f"{len(kept_anns)} annotations, skipped={n_skip}")
    finally:
        for h in raw_handles.values():
            h.close()

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

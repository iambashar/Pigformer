"""Distill v2 from v1 production: generate pseudo-labels by running v1
(full pig_upper_body weights) on the aligned undistorted PNGs.

Output: a new dataset directory with the same images but with the GT
polygons replaced by v1's predictions (encoded as COCO RLE).

Usage:
    python preprocessing/maskdino_v2/generate_v1_distill_labels.py \\
        --src-coco    /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined_v2_aligned \\
        --output-root /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined_v2_distill
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask

REPO = Path("/mnt/gs21/scratch/basharmk/data/unl/pigformer_release")
MASKDINO = Path("/mnt/gs21/scratch/basharmk/data/unl/MaskDINO")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MASKDINO))

from preprocessing.maskdino.infer_pig_depth_h5 import (  # noqa: E402
    setup_cfg as v1_setup, build_model_and_augment as v1_build,
    prepare_batch_inputs as v1_prep,
)

V1_CFG = str(MASKDINO / "configs/pig_depth/instance-segmentation/maskdino_R50_depth_instance_endpoint_pig_upper_body.yaml")
V1_WEIGHTS = str(MASKDINO / "output/pig_depth_endpoint_pig_upper_body_full_20260321_4gpu/model_best.pth")


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


def run_v1(model, augment, depth_uint16: np.ndarray):
    """v1 normalizes per-frame internally, so the input units don't matter."""
    batch = v1_prep(depth_uint16[None].astype(np.float32), augment,
                    torch.device("cuda"), "depth_valid_gradient")
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            o = model(batch)[0]
    inst = o["instances"].to("cpu")
    return {
        "classes": inst.pred_classes.numpy() if len(inst) else np.array([], dtype=int),
        "scores": inst.scores.numpy() if len(inst) else np.array([]),
        "masks": inst.pred_masks.numpy().astype(bool) if len(inst) else np.zeros((0, *depth_uint16.shape), bool),
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


def encode_rle(bool_mask: np.ndarray) -> dict:
    rle = coco_mask.encode(np.asfortranarray(bool_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    rle["size"] = list(rle["size"])
    return rle


def bbox_xywh(bool_mask):
    ys, xs = np.where(bool_mask)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()),
            float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-coco", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src_coco).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")
    if out.exists():
        shutil.rmtree(out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    # Symlink images (same depth PNGs as the aligned source).
    if not (out / "images").exists():
        os.symlink(src / "images", out / "images")
    if (src / "splits.json").is_file():
        shutil.copy2(src / "splits.json", out / "splits.json")

    print("loading v1 (production pig_upper_body)...")
    model, augment = load_v1()

    for split in ("train", "val", "test"):
        in_json = src / "annotations" / f"instances_{split}.json"
        if not in_json.is_file():
            continue
        payload = json.loads(in_json.read_text())
        images = payload["images"]
        if args.max_images > 0:
            images = images[: args.max_images]

        cats = {c["id"]: c["name"] for c in payload["categories"]}
        pig_id = next(c["id"] for c in payload["categories"] if c["name"] == "pig")
        up_id = next((c["id"] for c in payload["categories"] if c["name"] == "pig_upper_body"), None)

        kept_images = []
        kept_anns = []
        ann_id = 1
        n_skip = 0
        t0 = time.time()
        for i, im in enumerate(images):
            png_path = src / "images" / split / im["file_name"]
            if not png_path.is_file():
                n_skip += 1
                continue
            depth = np.array(Image.open(png_path)).astype(np.uint16)
            if depth.ndim == 3:
                depth = depth[..., 0]

            out_v1 = run_v1(model, augment, depth)
            v1_pig, v1_pig_s = best_mask(out_v1, 0, args.score_threshold)
            v1_up, v1_up_s = best_mask(out_v1, 1, args.score_threshold)
            if v1_pig is None:
                n_skip += 1
                continue

            kept_images.append(im)
            kept_anns.append({
                "id": ann_id, "image_id": int(im["id"]), "category_id": int(pig_id),
                "iscrowd": 0, "bbox": bbox_xywh(v1_pig), "bbox_mode": 1,
                "area": float(int(v1_pig.sum())),
                "segmentation": encode_rle(v1_pig),
                "v1_score": float(v1_pig_s),
            })
            ann_id += 1
            if v1_up is not None and int(v1_up.sum()) > 0 and up_id is not None:
                kept_anns.append({
                    "id": ann_id, "image_id": int(im["id"]), "category_id": int(up_id),
                    "iscrowd": 0, "bbox": bbox_xywh(v1_up), "bbox_mode": 1,
                    "area": float(int(v1_up.sum())),
                    "segmentation": encode_rle(v1_up),
                    "v1_score": float(v1_up_s),
                })
                ann_id += 1

            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [{split}] {i+1}/{len(images)} ({rate:.1f} img/s) skipped={n_skip}")

        new_payload = dict(payload)
        new_payload["images"] = kept_images
        new_payload["annotations"] = kept_anns
        new_payload["info"] = payload.get("info", {})
        new_payload["info"]["pseudo_label_source"] = "v1_production_pig_upper_body_distill"
        (out / "annotations" / f"instances_{split}.json").write_text(json.dumps(new_payload))
        print(f"[{split}] kept {len(kept_images)}/{len(images)} images, "
              f"{len(kept_anns)} annotations, skipped={n_skip}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

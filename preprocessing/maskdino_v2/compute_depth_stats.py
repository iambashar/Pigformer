"""Training-fold μ/σ in meters for MaskDINO v2 PIXEL_MEAN/STD, computed on
per-bag raw HDF5s + cv2.undistort (matching the v2 export pipeline).

Per-bag source roots:
  UNL: /mnt/gs21/scratch/basharmk/data/unl/hdf5_out/<bag>.h5
  MSU: /mnt/scratch/basharmk/data/body_condition/msu/bagfiles/<bag>.h5

Usage:
    python preprocessing/maskdino_v2/compute_depth_stats.py \\
        --src-coco /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

_PREP = str(Path(__file__).resolve().parents[1])
if _PREP not in sys.path:
    sys.path.insert(0, _PREP)
from msu_ground_plane import get_camera_params_for_bag as _get_ini_camera_params  # noqa: E402


UNL_BAG_DIR = Path("/mnt/gs21/scratch/basharmk/data/unl/hdf5_out")
MSU_BAG_DIR = Path("/mnt/scratch/basharmk/data/body_condition/msu/bagfiles")
FRAME_RE = re.compile(r"_frame(\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-coco", required=True)
    ap.add_argument("--max-bags", type=int, default=0)
    ap.add_argument("--no-undistort", action="store_true",
                    help="Skip undistortion. Default: undistort using HDF5 camera_info or per-date "
                         "INI fallback. Stats should match what training uses.")
    args = ap.parse_args()

    train_json = Path(args.src_coco).expanduser().resolve() / "annotations" / "instances_train.json"
    payload = json.loads(train_json.read_text())
    print(f"training images: {len(payload['images'])}")

    # Group by bag.
    by_bag: dict[tuple[str, str], list[int]] = {}
    for img in payload["images"]:
        idx = int(FRAME_RE.search(img["source_name"]).group(1))
        by_bag.setdefault((img["domain"], img["bag_name"]), []).append(idx)

    bags = sorted(by_bag.items())
    if args.max_bags > 0:
        bags = bags[: args.max_bags]
    print(f"bags: {len(bags)}")

    n = 0
    mean = 0.0
    m2 = 0.0
    seen_bags = 0
    for (domain, bag), idxs in bags:
        base = UNL_BAG_DIR if domain == "unl" else MSU_BAG_DIR
        bp = base / f"{bag}.h5"
        if not bp.is_file():
            print(f"  miss: {bp}")
            continue
        with h5py.File(bp, "r") as f:
            depth_ds = f["depth/images"]
            N, H, W = depth_ds.shape
            map_x = map_y = None
            if not args.no_undistort:
                if "depth/camera_info/K" in f:
                    K = np.asarray(f["depth/camera_info/K"], dtype=np.float64).reshape(3, 3)
                    D = np.asarray(f["depth/camera_info/D"], dtype=np.float64).reshape(-1)
                else:
                    cam = _get_ini_camera_params(bag)
                    if cam is None:
                        print(f"  no calib for {bag}; skipping")
                        continue
                    di = cam["DepthIntrinsic"]; dd = cam["DepthDistortion"]
                    K = np.array([[di["fx"], 0, di["cx"]],
                                  [0, di["fy"], di["cy"]],
                                  [0, 0, 1]], dtype=np.float64)
                    D = np.array([dd[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")],
                                 dtype=np.float64)
                map_x, map_y = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (W, H), cv2.CV_32FC1)
            for idx in idxs:
                if not (0 <= idx < N):
                    continue
                raw = np.asarray(depth_ds[idx], dtype=np.uint16)
                if map_x is not None:
                    depth = cv2.remap(raw, map_x, map_y, cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                else:
                    depth = raw
                depth_m = depth.astype(np.float32) / 1000.0
                valid = (depth_m > 0) & np.isfinite(depth_m)
                x = depth_m[valid]
                if x.size:
                    new_n = n + x.size
                    delta = x - mean
                    mean = mean + (delta.sum() / new_n)
                    delta2 = x - mean
                    m2 = m2 + float((delta * delta2).sum())
                    n = new_n
        seen_bags += 1
        if seen_bags % 25 == 0:
            std_now = (m2 / max(n - 1, 1)) ** 0.5
            print(f"  [{seen_bags}/{len(bags)}] μ={mean:.4f} σ={std_now:.4f} m  valid_px={n:,}")

    std = (m2 / max(n - 1, 1)) ** 0.5
    print(f"\nFinal: μ={mean:.6f} m   σ={std:.6f} m   over {n:,} valid pixels")
    print(f"\n  PIXEL_MEAN: [{mean:.4f}]")
    print(f"  PIXEL_STD:  [{std:.4f}]")


if __name__ == "__main__":
    main()

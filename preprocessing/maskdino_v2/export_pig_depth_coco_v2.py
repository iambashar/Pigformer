"""Re-export pig-depth COCO dataset as 1-channel uint16 PNGs (depth in mm).

Pulls depth from per-bag raw HDF5s (full-coverage source), applies the
same distortion correction v1 used (`depth_maps_undistorted`), and saves
as uint16 mm PNGs. The COCO annotations from v1 apply verbatim because
file_name + image_id are preserved.

Per-bag sources:
  UNL: /mnt/gs21/scratch/basharmk/data/unl/hdf5_out/<bag_name>.h5
  MSU: /mnt/scratch/basharmk/data/body_condition/msu/bagfiles/<bag_name>.h5

Each per-bag HDF5 has:
  depth/images          (N, H, W) uint16 mm (raw, distorted)
  depth/camera_info/K   (3, 3) float64
  depth/camera_info/D   (8,)   float64

source_name format is `<bag_name>_frame<idx>` → `depth/images[idx]`.

Usage:
    python preprocessing/maskdino_v2/export_pig_depth_coco_v2.py \\
        --src-coco    /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined \\
        --output-root /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined_v2
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

# msu_ground_plane lives next to this file's grandparent (preprocessing/).
_PREP = str(Path(__file__).resolve().parents[1])
if _PREP not in sys.path:
    sys.path.insert(0, _PREP)
from msu_ground_plane import get_camera_params_for_bag as _get_ini_camera_params  # noqa: E402


UNL_BAG_DIR = Path("/mnt/gs21/scratch/basharmk/data/unl/hdf5_out")
MSU_BAG_DIR = Path("/mnt/scratch/basharmk/data/body_condition/msu/bagfiles")
FRAME_RE = re.compile(r"_frame(\d+)$")


def bag_path(domain: str, bag_name: str) -> Path:
    base = UNL_BAG_DIR if domain == "unl" else MSU_BAG_DIR
    return base / f"{bag_name}.h5"


def parse_frame_idx(source_name: str) -> int:
    m = FRAME_RE.search(source_name)
    if not m:
        raise ValueError(f"could not parse frame index from {source_name!r}")
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-coco", required=True,
                    help="v1 COCO root; provides annotations + splits.json.")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--clip-mm", type=int, default=6000)
    ap.add_argument("--no-undistort", action="store_true",
                    help="Skip undistortion. By default we undistort to match the GT mask coordinate "
                         "system (the GT polygons were drawn on undistorted depth; training on raw "
                         "creates a 3-5 px mean misalignment that systematically erodes the predicted "
                         "boundary). UNL bag HDF5s carry camera_info; MSU bags use the per-date INI.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_coco = Path(args.src_coco).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()

    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to rebuild.")
    if out.exists():
        shutil.rmtree(out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    splits_json = src_coco / "splits.json"
    if splits_json.is_file():
        shutil.copy2(splits_json, out / "splits.json")

    # Group all records by (domain, bag_name) so we open each HDF5 once.
    work: dict[tuple[str, str], list[tuple[str, int, str, Path]]] = {}
    # (domain, bag) → list of (file_name, frame_idx, split, image_id)
    for split in ("train", "val", "test"):
        ann_in = src_coco / "annotations" / f"instances_{split}.json"
        if not ann_in.is_file():
            continue
        payload = json.loads(ann_in.read_text())
        for img in payload["images"]:
            domain = img["domain"]
            bag = img["bag_name"]
            file_name = img["file_name"]
            idx = parse_frame_idx(img["source_name"])
            work.setdefault((domain, bag), []).append((file_name, idx, split, img["id"]))

    print(f"Total bags to read: {len(work)}")

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)

    kept_ids_per_split: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    misses: list[tuple[str, str, str]] = []

    for bag_no, ((domain, bag), entries) in enumerate(sorted(work.items())):
        bp = bag_path(domain, bag)
        if not bp.is_file():
            for _, _, split, image_id in entries:
                misses.append((domain, bag, "no h5"))
            print(f"  [{bag_no+1}/{len(work)}] MISS bag {domain}/{bag}: file not found")
            continue
        try:
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
                            raise RuntimeError(f"No HDF5 camera_info and no INI for bag {bag}")
                        di = cam["DepthIntrinsic"]
                        dd = cam["DepthDistortion"]
                        K = np.array([[di["fx"], 0, di["cx"]],
                                      [0, di["fy"], di["cy"]],
                                      [0, 0, 1]], dtype=np.float64)
                        D = np.array([dd[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")],
                                     dtype=np.float64)
                    map_x, map_y = cv2.initUndistortRectifyMap(
                        K, D, np.eye(3), K, (W, H), cv2.CV_32FC1
                    )

                for file_name, idx, split, image_id in entries:
                    if not (0 <= idx < N):
                        misses.append((domain, bag, f"idx {idx} oob (N={N})"))
                        continue
                    raw = np.asarray(depth_ds[idx], dtype=np.uint16)  # mm
                    if map_x is not None:
                        depth_mm = cv2.remap(raw, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    else:
                        depth_mm = raw
                    if args.clip_mm:
                        depth_mm = np.clip(depth_mm, 0, args.clip_mm).astype(np.uint16)
                    target = out / "images" / split / file_name
                    Image.fromarray(depth_mm, mode="I;16").save(target, optimize=True)
                    kept_ids_per_split[split].add(int(image_id))
        except Exception as e:
            print(f"  [{bag_no+1}/{len(work)}] ERROR on {domain}/{bag}: {e}")
            for _, _, split, image_id in entries:
                misses.append((domain, bag, f"error: {e}"))
            continue

        if (bag_no + 1) % 25 == 0:
            print(f"  [{bag_no+1}/{len(work)}] processed {domain}/{bag} ({len(entries)} frames)")

    # Filter and write annotations JSON per split.
    for split in ("train", "val", "test"):
        ann_in = src_coco / "annotations" / f"instances_{split}.json"
        if not ann_in.is_file():
            continue
        payload = json.loads(ann_in.read_text())
        n0 = len(payload["images"])
        a0 = len(payload["annotations"])
        kept = kept_ids_per_split[split]
        kept_imgs = [im for im in payload["images"] if int(im["id"]) in kept]
        kept_anns = [a for a in payload["annotations"] if int(a["image_id"]) in kept]
        new_payload = dict(payload)
        new_payload["images"] = kept_imgs
        new_payload["annotations"] = kept_anns
        (out / "annotations" / f"instances_{split}.json").write_text(json.dumps(new_payload))
        print(f"[{split}] kept {len(kept_imgs)}/{n0} images, {len(kept_anns)}/{a0} annotations")

    if misses:
        print(f"\nTotal misses: {len(misses)}")
        for m in misses[:10]:
            print(f"  miss: {m}")
    print(f"\nDone. v2 root: {out}")


if __name__ == "__main__":
    main()

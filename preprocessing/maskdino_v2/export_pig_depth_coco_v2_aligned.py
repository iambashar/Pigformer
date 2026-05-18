"""Re-export v2 PNGs from the consolidated source's depth_maps_undistorted field.

This avoids the alignment bug where my previous export re-undistorted raw
depth using per-bag camera params that differed slightly from whatever
process produced the consolidated source's `depth_maps_undistorted`. By
reading depth_maps_undistorted directly, the depth is already in the
exact same pixel grid that the GT polygons were drawn against.

Usage:
    python preprocessing/maskdino_v2/export_pig_depth_coco_v2_aligned.py \\
        --src-coco    /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/combined_upadted_upper_body_coco_endpoint_full_20260321 \\
        --src-source  /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/combined_upadted_upper_body \\
        --output-root /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined_v2_aligned
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-coco", required=True,
                    help="Source COCO dataset (provides splits.json + annotations + image records).")
    ap.add_argument("--src-source", required=True,
                    help="Consolidated source root with {msu,unl}/<date>/dataset.h5 containing "
                         "depth_maps_undistorted. This is the SAME source the GT polygons were "
                         "drawn against, so depth and polygons share the same pixel grid.")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--image-key", default="depth_maps_undistorted")
    ap.add_argument("--clip-mm", type=int, default=6000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_coco = Path(args.src_coco).expanduser().resolve()
    src_source = Path(args.src_source).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()

    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")
    if out.exists():
        shutil.rmtree(out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    # Carry over splits.json
    if (src_coco / "splits.json").is_file():
        shutil.copy2(src_coco / "splits.json", out / "splits.json")

    # Build per-(domain, date) source_name → source_index map
    source_index: dict[tuple, dict] = {}
    print("indexing consolidated source...")
    for domain in ("msu", "unl"):
        for ds_path in sorted((src_source / domain).glob("*/dataset.h5")):
            date = ds_path.parent.name
            with h5py.File(ds_path, "r") as f:
                names = [n.decode() for n in f["source_names"][:]]
            source_index[(domain, date)] = {n: i for i, n in enumerate(names)}
            print(f"  {domain}/{date}: {len(names)} frames")

    # Open one HDF5 handle per (domain, date) lazily
    handles: dict[tuple, h5py.File] = {}

    def open_h(domain, date):
        key = (domain, date)
        if key not in handles:
            handles[key] = h5py.File(src_source / domain / date / "dataset.h5", "r")
        return handles[key]

    try:
        for split in ("train", "val", "test"):
            ann_in = src_coco / "annotations" / f"instances_{split}.json"
            if not ann_in.is_file():
                continue
            payload = json.loads(ann_in.read_text())
            images_dir = out / "images" / split
            images_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{split}] {len(payload['images'])} images")
            kept_ids = set()
            kept_images = []
            misses = 0
            for i, im in enumerate(payload["images"]):
                domain = im["domain"]; date = im["date"]
                source_name = im["source_name"]
                file_name = im["file_name"]
                idx_map = source_index.get((domain, date))
                src_idx = idx_map.get(source_name) if idx_map is not None else None
                if src_idx is None:
                    misses += 1
                    continue

                f = open_h(domain, date)
                # depth_maps_undistorted is stored as float32 meters (we verified earlier).
                depth_m = np.asarray(f[args.image_key][src_idx], dtype=np.float32)
                # Convert to uint16 mm, clip to clip_mm.
                depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0.0)
                depth_mm = np.clip(depth_m * 1000.0, 0, args.clip_mm).astype(np.uint16)
                Image.fromarray(depth_mm, mode="I;16").save(images_dir / file_name, optimize=True)
                kept_ids.add(int(im["id"]))
                kept_images.append(im)
                if (i + 1) % 250 == 0:
                    print(f"  [{split}] {i+1}/{len(payload['images'])}")

            kept_anns = [a for a in payload["annotations"] if int(a["image_id"]) in kept_ids]
            new_payload = dict(payload)
            new_payload["images"] = kept_images
            new_payload["annotations"] = kept_anns
            (out / "annotations" / f"instances_{split}.json").write_text(json.dumps(new_payload))
            print(f"  kept {len(kept_images)}/{len(payload['images'])} images, "
                  f"{len(kept_anns)} annotations, misses={misses}")
    finally:
        for h in handles.values():
            h.close()

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

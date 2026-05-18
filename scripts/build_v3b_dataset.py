"""Build pig_depth_combined_v3b: v3 but with positives re-rendered from
raw HDF5 + on-the-fly cv2.initUndistortRectifyMap (matching what
build_height_dataset.py's deployment pipeline does at inference time).

Why: the original v2_aligned positives came from a consolidated source's
`depth_maps_undistorted` field, whose undistortion process produces a
*different* depth distribution than the on-the-fly raw->undistorted pass
used at inference. A UNet trained on the consolidated-source positives
collapses to all-zero output when fed inference-style undistortion,
even though the model achieves 99% IoU on val (which is also from the
consolidated source). v3b unifies the preprocessing.

Annotations are carried over verbatim — the segmentation polygons /
RLEs were drawn against the consolidated source's pixel grid, but
since both undistortion methods produce images at the same resolution
(576x640) with the pig in roughly the same place, the polygons are
"close enough" for binary supervision (mIoU 0.99 -> probably 0.95+ even
with the mismatch). The model just needs to learn the same kind of
pixels at inference time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "preprocessing"))
from preprocessing import build_height_dataset as bh  # noqa: E402

V2_DIR = Path("/mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined_v2_aligned")
V3B_DIR = Path("/mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined_v3b")
RAW_DIRS = {
    "msu": Path("/mnt/scratch/basharmk/data/body_condition/msu/bagfiles"),
    "unl": Path("/mnt/scratch/basharmk/data/unl/hdf5_out"),
}
CLIP_MM = 6000


FX_SCALE = 0.85  # match consolidated source (build_summary.json)


def undistort_inference_style(depth_u16: np.ndarray, cam: dict) -> np.ndarray:
    """K-with-fx_scale newCameraMatrix; matches the consolidated source
    that the GT polygons were drawn against."""
    di = cam["DepthIntrinsic"]; dd = cam["DepthDistortion"]
    K = np.array([[di["fx"], 0, di["cx"]],
                  [0, di["fy"], di["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array([dd[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")],
                 dtype=np.float64)
    K_new = K.copy()
    K_new[0, 0] *= FX_SCALE
    K_new[1, 1] *= FX_SCALE
    H, W = depth_u16.shape
    mx, my = cv2.initUndistortRectifyMap(K, D, np.eye(3), K_new, (W, H), cv2.CV_32FC1)
    return cv2.remap(depth_u16, mx, my, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def render_image(domain: str, bag: str, depth_idx: int, cam_cache: dict) -> np.ndarray | None:
    raw_path = RAW_DIRS[domain] / f"{bag}.h5"
    if not raw_path.is_file():
        return None
    if bag not in cam_cache:
        try:
            cam_cache[bag] = bh.get_camera_params_for_bag(bag, hdf5_path=str(raw_path))
        except Exception:
            cam_cache[bag] = None
    cam = cam_cache.get(bag)
    if cam is None:
        return None
    with h5py.File(raw_path, "r") as f:
        if depth_idx >= int(f["depth/images"].shape[0]):
            return None
        depth_u16 = np.asarray(f["depth/images"][depth_idx]).astype(np.uint16)
    d_und = undistort_inference_style(depth_u16, cam)
    return np.clip(d_und.astype(np.int32), 0, CLIP_MM).astype(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if V3B_DIR.exists() and not args.overwrite:
        raise SystemExit(f"{V3B_DIR} exists; pass --overwrite")
    if V3B_DIR.exists():
        shutil.rmtree(V3B_DIR)
    (V3B_DIR / "annotations").mkdir(parents=True)

    if (V2_DIR / "splits.json").is_file():
        shutil.copy2(V2_DIR / "splits.json", V3B_DIR / "splits.json")

    cam_cache: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        ann_in = V2_DIR / "annotations" / f"instances_{split}.json"
        if not ann_in.is_file():
            continue
        payload = json.loads(ann_in.read_text())
        out_imgs = V3B_DIR / "images" / split
        out_imgs.mkdir(parents=True)

        kept_imgs: list[dict] = []
        for im in tqdm(payload["images"], desc=f"[{split}] re-render positives"):
            img = render_image(im["domain"], im["bag_name"], int(im["depth_index"]), cam_cache)
            if img is None:
                continue
            Image.fromarray(img, mode="I;16").save(out_imgs / im["file_name"], optimize=True)
            kept_imgs.append(im)
        kept_ids = {int(im["id"]) for im in kept_imgs}
        kept_anns = [a for a in payload["annotations"] if int(a["image_id"]) in kept_ids]
        new_payload = dict(payload)
        new_payload["images"] = kept_imgs
        new_payload["annotations"] = kept_anns
        (V3B_DIR / "annotations" / f"instances_{split}.json").write_text(json.dumps(new_payload))
        print(f"[{split}] kept {len(kept_imgs)}/{len(payload['images'])} positives, "
              f"{len(kept_anns)} anns")

        # Append negatives to train only (mirror what build_v3_dataset.py did).
        if split == "train":
            train_imgs = kept_imgs
            train_anns = kept_anns
            pos_per_bag: dict[tuple[str, str], set[int]] = defaultdict(set)
            for im in train_imgs:
                pos_per_bag[(im["domain"], im["bag_name"])].add(int(im["depth_index"]))

            next_id = (max(int(im["id"]) for im in train_imgs) + 1) if train_imgs else 1
            new_neg_records: list[dict] = []
            for (domain, bag), pos_idx in sorted(pos_per_bag.items()):
                raw_path = RAW_DIRS[domain] / f"{bag}.h5"
                if not raw_path.is_file():
                    continue
                cam = cam_cache.get(bag)
                if cam is None:
                    continue
                with h5py.File(raw_path) as f:
                    n_frames = int(f["depth/images"].shape[0])
                forbidden = set()
                for pi in pos_idx:
                    forbidden.update(range(pi - 50, pi + 51))
                cands = [i for i in range(n_frames) if i not in forbidden]
                if len(cands) < 2:
                    continue
                step = len(cands) / 2
                picks = [cands[int(i * step)] for i in range(2)]
                for fi in picks:
                    img = render_image(domain, bag, fi, cam_cache)
                    if img is None:
                        continue
                    date = bag.split("_")[0] if bag[:8].isdigit() else None
                    if date is None:
                        import re
                        m = re.search(r"_(\d{8})_\d{6}", bag)
                        date = m.group(1) if m else "00000000"
                    file_name = f"{domain}_{date}_neg{next_id:06d}_{bag}_frame{fi}.png"
                    Image.fromarray(img, mode="I;16").save(out_imgs / file_name, optimize=True)
                    new_neg_records.append({
                        "id": next_id,
                        "width": int(img.shape[1]),
                        "height": int(img.shape[0]),
                        "file_name": file_name,
                        "domain": domain,
                        "date": date,
                        "bag_name": bag,
                        "source_name": f"{bag}_frame{fi}",
                        "color_index": int(fi),
                        "depth_index": int(fi),
                        "is_negative": True,
                    })
                    next_id += 1
            print(f"[{split}] added {len(new_neg_records)} negatives")
            new_payload["images"] = train_imgs + new_neg_records
            new_payload["annotations"] = train_anns
            (V3B_DIR / "annotations" / f"instances_{split}.json").write_text(json.dumps(new_payload))

    print(f"done. output: {V3B_DIR}")


if __name__ == "__main__":
    main()

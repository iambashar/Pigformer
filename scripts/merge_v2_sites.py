"""Concatenate the MSU and UNL per-site v2 HDF5s into a full dataset.

Schema matches what dataset.py expects: height_maps (N, 96, 224) float32,
bag_names (N,) str, source_names (N,) str.

Usage:
    /mnt/home/basharmk/.conda/envs/swine-rgbd/bin/python scripts/merge_v2_sites.py \\
        --msu /tmp/dataset_v2_msu_correct.h5 \\
        --unl /tmp/dataset_v2_unl_correct.h5 \\
        --out /tmp/dataset_v2_full_correct.h5
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msu", required=True)
    ap.add_argument("--unl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for p in (args.msu, args.unl):
        if not os.path.isfile(p):
            raise SystemExit(f"missing input: {p}")

    arrays = {"height_maps": [], "bag_names": [], "source_names": []}
    counts = {}
    for tag, path in (("msu", args.msu), ("unl", args.unl)):
        with h5py.File(path, "r") as f:
            for k in arrays:
                arrays[k].append(np.asarray(f[k]))
            counts[tag] = int(f["height_maps"].shape[0])
            bag_count = len(set(np.asarray(f["bag_names"]).tolist()))
            nz_frac = float((np.asarray(f["height_maps"][:1000]) > 0).mean())
            print(f"  {tag}: n={counts[tag]}  unique_bags={bag_count}  "
                  f"nz_frac(first1000)={nz_frac:.4f}")

    merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    n = merged["height_maps"].shape[0]
    bag_count = len(set(merged["bag_names"].tolist()))
    print(f"  merged: n={n}  unique_bags={bag_count}")
    assert n == counts["msu"] + counts["unl"]

    with h5py.File(args.out, "w") as f:
        f.create_dataset("height_maps", data=merged["height_maps"],
                         compression="lzf", chunks=(1, 96, 224))
        f.create_dataset("bag_names", data=merged["bag_names"])
        f.create_dataset("source_names", data=merged["source_names"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

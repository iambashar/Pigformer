"""
Parse the slaughter-lab CSV into the ``label.h5`` format expected by
``dataset.py``.

Input CSV columns (case-sensitive):
  RosBagPath, UniqueID, Fat_r, Loin_r

Multiple rows per UniqueID are averaged. The resulting HDF5 has one row per
pig identity with datasets: ``bag_names``, ``unique_ids``, ``fat_rib12``,
``loin_rib12``, ``total_rib``. Missing values become NaN.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import h5py
import numpy as np


def _float_or_nan(val: str | None) -> float:
    if val is None:
        return float("nan")
    val = val.strip()
    if not val:
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def parse_csv(csv_path: str, hdf5_dir: str | None) -> list[dict]:
    """Aggregate measurements by UniqueID, keeping only bags present on disk."""
    valid_bags: set[str] | None = None
    if hdf5_dir and os.path.exists(hdf5_dir):
        valid_bags = {f.replace(".h5", "") for f in os.listdir(hdf5_dir) if f.endswith(".h5")}

    pig_data: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bag = row.get("RosBagPath", "").strip()
            if valid_bags is not None and bag and bag not in valid_bags:
                continue
            uid = row.get("UniqueID", "").strip()
            if not uid:
                continue
            pig_data[uid].append({
                "bag_name": bag,
                "fat_rib12": _float_or_nan(row.get("Fat_r")),
                "loin_rib12": _float_or_nan(row.get("Loin_r")),
            })

    records = []
    for uid, entries in pig_data.items():
        def avg(key: str) -> float:
            vals = [e[key] for e in entries if not np.isnan(e[key])]
            return float(np.mean(vals)) if vals else float("nan")

        fat, loin = avg("fat_rib12"), avg("loin_rib12")
        records.append({
            "unique_id": uid,
            "bag_name": entries[0]["bag_name"],
            "fat_rib12": fat,
            "loin_rib12": loin,
            "total_rib": fat + loin,
        })
    return records


def write_h5(records: list[dict], output: str) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    dt_str = h5py.string_dtype()
    with h5py.File(output, "w") as f:
        f.create_dataset("bag_names", data=[r["bag_name"].encode() for r in records], dtype=dt_str)
        f.create_dataset("unique_ids", data=[r["unique_id"].encode() for r in records], dtype=dt_str)
        for key in ("fat_rib12", "loin_rib12", "total_rib"):
            f.create_dataset(key, data=np.array([r[key] for r in records], dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Slaughter-lab CSV with UniqueID + measurements.")
    parser.add_argument("--hdf5_dir", default=None, help="Directory of per-bag HDF5 files used to filter rows.")
    parser.add_argument("--output", required=True, help="Destination label.h5 path.")
    args = parser.parse_args()

    records = parse_csv(args.csv, args.hdf5_dir)
    print(f"Extracted {len(records)} label records")
    if not records:
        raise SystemExit("No labels found — check paths and CSV headers.")
    write_h5(records, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

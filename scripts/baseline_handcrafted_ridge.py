"""Classical-ML lower bound: ridge regression on handcrafted features
extracted from input-aggregated height maps.

Addresses cv4animal2026 reviewer concern (4) ("compare with simpler
machine learning approaches based on global features or classical
regressors to verify whether the transformer is necessary") and bears on
concern (3) ("the model may rely mostly on global statistics rather than
detailed shape information").

Two feature sets:
  - global_stats: 4 features the reviewer explicitly names (mean height,
                  body volume, body area, max height). This is the most
                  direct test of "does global statistics suffice?".
  - handcrafted : 14 features = global_stats + length, width, std, p50,
                  p97, length/width aspect, 5 per-zone spine means.
                  Stronger handcrafted lower bound.

Protocol matches tab:supp_arch in workshop_paper/sec/X_suppl.tex:
  - Fold 0 train_indices to fit.
  - Fold 0 val_indices to pick ridge alpha per target.
  - Test_indices to report.
  - Input-aggregation: mean of all frames per bag -> one feature vector.

Run:
    /mnt/home/basharmk/.conda/envs/swine-rgbd/bin/python \\
        scripts/baseline_handcrafted_ridge.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

TARGET_NAMES = ("fat_rib12", "loin_rib12", "total_rib")


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else v for v in values]


def load_targets(labels_path: str) -> tuple[list[str], np.ndarray]:
    with h5py.File(labels_path, "r") as f:
        bag_names = _decode(f["bag_names"][:])
        fat = f["fat_rib12"][:].astype(np.float32)
        loin = f["loin_rib12"][:].astype(np.float32)
        total = fat + loin
    y = np.stack([fat, loin, total], axis=-1)
    return bag_names, y


def aggregate_bag_height_maps(dataset_path: str) -> tuple[list[str], np.ndarray]:
    """Return (bag_names_unique, mean_height_maps).
    One row per bag = mean of all frames in that bag."""
    with h5py.File(dataset_path, "r") as f:
        hm = f["height_maps"][:].astype(np.float32)
        bag_names = _decode(f["bag_names"][:])
    bag_to_rows = defaultdict(list)
    for i, b in enumerate(bag_names):
        bag_to_rows[b].append(i)
    bags = sorted(bag_to_rows.keys())
    out = np.stack([hm[bag_to_rows[b]].mean(axis=0) for b in bags], axis=0)
    return bags, out


def _zone_mean(h: np.ndarray, lo: int, hi: int) -> float:
    sl = h[:, lo:hi]
    nz = sl > 0
    if not nz.any():
        return 0.0
    return float(sl[nz].mean())


def extract_features(h: np.ndarray, kind: str = "handcrafted") -> np.ndarray:
    """h shape (96, 224) float32 meters. Grid is 1 cm/pixel.
    axis 0 = lateral, axis 1 = spine (pig faces right)."""
    nz = h > 0
    n_nz = int(nz.sum())
    if n_nz == 0:
        n_feat = 4 if kind == "global_stats" else 14
        return np.zeros(n_feat, dtype=np.float32)
    vals = h[nz]
    ys, xs = np.where(nz)

    mean_h = float(vals.mean())
    max_h = float(vals.max())
    volume = float(vals.sum())  # cm^3 (px area 1 cm^2 * height)
    area = float(n_nz)          # cm^2

    if kind == "global_stats":
        return np.array([mean_h, volume, area, max_h], dtype=np.float32)

    # Extended handcrafted set.
    length = float(xs.max() - xs.min() + 1)
    width = float(ys.max() - ys.min() + 1)
    std_h = float(vals.std())
    p50_h = float(np.percentile(vals, 50))
    p97_h = float(np.percentile(vals, 97))
    aspect = length / max(width, 1.0)

    # Five spine zones from tail (left) to head (right) on the pig bbox.
    x_lo, x_hi = int(xs.min()), int(xs.max()) + 1
    zone_edges = np.linspace(x_lo, x_hi, 6, dtype=int)
    zone_means = [_zone_mean(h, zone_edges[k], zone_edges[k + 1]) for k in range(5)]

    return np.array([
        mean_h, volume, area, max_h,
        length, width, std_h, p50_h, p97_h, aspect,
        *zone_means,
    ], dtype=np.float32)


def featurize_subset(bags: list[str], hmaps: np.ndarray,
                     index_to_bag: list[str], wanted_indices: list[int],
                     kind: str) -> tuple[np.ndarray, list[int]]:
    """Return (X, used_label_indices) for label indices that have a height map."""
    bag_to_row = {b: i for i, b in enumerate(bags)}
    X, used = [], []
    for li in wanted_indices:
        b = index_to_bag[li]
        if b not in bag_to_row:
            continue
        X.append(extract_features(hmaps[bag_to_row[b]], kind=kind))
        used.append(li)
    return np.stack(X, axis=0), used


def fit_ridge_per_target(X_tr: np.ndarray, y_tr: np.ndarray,
                          X_va: np.ndarray, y_va: np.ndarray,
                          alphas: list[float]) -> tuple[list[Ridge], list[float]]:
    """Fit one Ridge per target. Pick alpha by val MAE."""
    n_targets = y_tr.shape[1]
    models, picked_alphas = [], []
    for ti in range(n_targets):
        best_mae, best_a, best_m = np.inf, alphas[0], None
        for a in alphas:
            m = Ridge(alpha=a, fit_intercept=True)
            m.fit(X_tr, y_tr[:, ti])
            p = m.predict(X_va)
            mae = float(np.mean(np.abs(p - y_va[:, ti])))
            if mae < best_mae:
                best_mae, best_a, best_m = mae, a, m
        models.append(best_m)
        picked_alphas.append(best_a)
    return models, picked_alphas


def predict_targets(models: list[Ridge], X: np.ndarray) -> np.ndarray:
    return np.stack([m.predict(X) for m in models], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.h5")
    ap.add_argument("--labels", default="data/label.h5")
    ap.add_argument("--split_json", default="data/split.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--output", default="results/classical_ridge.json")
    args = ap.parse_args()

    split = json.loads(Path(args.split_json).read_text())
    fold = split["folds"][args.fold]
    train_li = list(fold["train"])
    val_li = list(fold["val"])
    test_li = list(split["test_indices"])

    label_bags, y_all = load_targets(args.labels)
    bags, hmaps = aggregate_bag_height_maps(args.dataset)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    results: dict = {"fold": args.fold, "feature_sets": {}, "mean_targets_train": {}}

    # Train-set per-target mean as the rock-bottom predict-the-mean baseline.
    y_tr_all = y_all[train_li]
    for ti, name in enumerate(TARGET_NAMES):
        results["mean_targets_train"][name] = float(np.mean(y_tr_all[:, ti]))

    for kind in ["global_stats", "handcrafted"]:
        X_tr, used_tr = featurize_subset(bags, hmaps, label_bags, train_li, kind)
        X_va, used_va = featurize_subset(bags, hmaps, label_bags, val_li, kind)
        X_te, used_te = featurize_subset(bags, hmaps, label_bags, test_li, kind)
        y_tr = y_all[used_tr]; y_va = y_all[used_va]; y_te = y_all[used_te]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        models, picked = fit_ridge_per_target(X_tr_s, y_tr, X_va_s, y_va, alphas)
        preds = predict_targets(models, X_te_s)

        per_t_mae = [float(np.mean(np.abs(preds[:, ti] - y_te[:, ti])))
                     for ti in range(len(TARGET_NAMES))]
        overall = float(np.mean(per_t_mae))
        per_t_mape = [per_t_mae[ti] / results["mean_targets_train"][TARGET_NAMES[ti]] * 100
                      for ti in range(len(TARGET_NAMES))]
        overall_mape = float(np.mean(per_t_mape))

        print(f"\n=== {kind} (n_features={X_tr.shape[1]}, "
              f"n_train={len(used_tr)}, n_val={len(used_va)}, n_test={len(used_te)}) ===")
        for ti, name in enumerate(TARGET_NAMES):
            print(f"  {name:12s}  MAE={per_t_mae[ti]:.3f}  "
                  f"MAPE={per_t_mape[ti]:.2f}%  alpha={picked[ti]}")
        print(f"  Overall MAE      = {overall:.3f}")
        print(f"  Overall MAPE     = {overall_mape:.2f}%")

        results["feature_sets"][kind] = {
            "n_features": int(X_tr.shape[1]),
            "n_train": len(used_tr),
            "n_val": len(used_va),
            "n_test": len(used_te),
            "alphas": picked,
            "per_target_mae": dict(zip(TARGET_NAMES, per_t_mae)),
            "per_target_mape": dict(zip(TARGET_NAMES, per_t_mape)),
            "overall_mae": overall,
            "overall_mape": overall_mape,
        }

    # Predict-the-mean lower bound (no features at all).
    preds_mean = np.tile(np.array([results["mean_targets_train"][n] for n in TARGET_NAMES]),
                          (len(test_li), 1))
    y_te_all = y_all[test_li]
    per_t_mae_mean = [float(np.mean(np.abs(preds_mean[:, ti] - y_te_all[:, ti])))
                      for ti in range(len(TARGET_NAMES))]
    overall_mean = float(np.mean(per_t_mae_mean))
    print(f"\n=== predict-the-train-mean (no model) ===")
    for ti, name in enumerate(TARGET_NAMES):
        print(f"  {name:12s}  MAE={per_t_mae_mean[ti]:.3f}")
    print(f"  Overall MAE      = {overall_mean:.3f}")
    results["predict_train_mean"] = {
        "per_target_mae": dict(zip(TARGET_NAMES, per_t_mae_mean)),
        "overall_mae": overall_mean,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

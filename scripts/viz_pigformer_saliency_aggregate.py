"""Aggregate column-importance curve across all test bags, plus
per-bag overlays with anatomical landmarks marked.

For each test bag:
  1. Average all bag frames to one height map (paper test protocol).
  2. SmoothGrad saliency for fat / loin / total.
  3. Column importance = sum saliency along the lateral axis.
  4. Locate pig body bbox along spine axis (head end / tail end), use
     upper-body centroid (= front) as reference; rib-12 sits ~60% from
     head toward tail per pig anatomy references.
  5. Resample importance to relative body coordinate [0, 1] where 0 =
     head end, 1 = tail end.

Then average resampled curves over all test bags -> population curve
with std shading.

Per-bag panels (top): height map with importance heatmap overlay +
markers at head / rib-12 / tail positions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import create_model

TARGET_LABELS = {"fat_rib12": "Fat", "loin_rib12": "Loin", "total_rib": "Total"}
TARGET_COLORS = {"fat_rib12": "#e6194b", "loin_rib12": "#3cb44b", "total_rib": "#4363d8"}
# In the height-map convention, pigs face right after rotation (CLAUDE.md):
# head is at the rightmost pig column, tail at the leftmost. The last rib
# sits ~60% along the body axis measured from head to tail.
LAST_RIB_REL = 0.60


def load_pigformer(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ta = ckpt["args"]
    target_names = ckpt["target_names"]
    model = create_model(
        ta["arch"], seq_len=224, feature_dim=96, dim_out=len(target_names),
        nhead=ta.get("nhead", 8), num_layers=ta.get("num_layers", 1),
        dim_feedforward=ta.get("dim_feedforward", 512),
        dropout=ta.get("dropout", 0.0),
        head_dropout=ta.get("head_dropout", 0.0),
        in_channels=3 if ta["arch"] == "cnn" else 1,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, target_names


def smoothgrad_column_importance(model, x_seq: torch.Tensor, target_idx: int,
                                  n_samples: int = 20, noise_std: float = 0.04
                                  ) -> np.ndarray:
    """Returns (224,) per-spine-column importance using SmoothGrad x Input
    attribution: attribution_i = |x_i| * mean_k(|grad_k at x + noise_k|).
    Multiplying by the input value zeros out the background contribution,
    so this measures what the model uses in the *current* input rather
    than counterfactual sensitivity to non-existent pixels."""
    valid = (x_seq != 0).float()
    accum = None
    for _ in range(n_samples):
        xi = (x_seq + torch.randn_like(x_seq) * noise_std * valid).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        y = model(xi)
        y[0, target_idx].backward()
        g = xi.grad.detach().abs()           # (1, 224, 96)
        accum = g if accum is None else accum + g
    grad_mean = accum / n_samples
    attribution = grad_mean * x_seq.detach().abs()  # input x gradient
    col = attribution.sum(dim=2)[0].cpu().numpy()   # (224,)
    return col


def pig_extent_along_spine(height_map_96x224: np.ndarray, top_pct: int = 5
                            ) -> tuple[int, int]:
    """Return (tail_col, head_col) — the LEFT and RIGHT spine-axis edges of
    the pig. Pigs face right, so the leftmost pig column is the tail tip
    and the rightmost is the head tip.
    """
    col_mass = (height_map_96x224 > 0).sum(axis=0)
    cutoff = max(col_mass.max() * top_pct / 100.0, 1)
    cols = np.where(col_mass >= cutoff)[0]
    if cols.size == 0:
        return 0, height_map_96x224.shape[1] - 1
    return int(cols[0]), int(cols[-1])  # (tail_col_left, head_col_right)


def resample_to_relative(curve: np.ndarray, tail: int, head: int,
                          n_bins: int = 100) -> np.ndarray:
    """Resample curve[tail:head+1] onto a fixed n_bins grid of [0, 1]
    where 0 = TAIL end (left of the height map) and 1 = HEAD end (right)."""
    segment = curve[tail:head + 1]
    if segment.size < 2:
        return np.zeros(n_bins, dtype=np.float32)
    x_src = np.linspace(0, 1, segment.size)
    x_dst = np.linspace(0, 1, n_bins)
    return np.interp(x_dst, x_src, segment).astype(np.float32)


def mean_pig_profile(height_map: np.ndarray, tail: int, head: int,
                     n_bins: int = 100) -> np.ndarray:
    """Average top-profile of a pig: max height per spine column, then
    resample to relative-body coords [0=tail, 1=head]."""
    top = height_map.max(axis=0)
    return resample_to_relative(top, tail, head, n_bins)


def pig_mask_to_relative(height_map: np.ndarray, tail: int, head: int,
                          n_x: int = 200, n_y: int = 80) -> np.ndarray:
    """Resample a single height map's binary pig mask (h > 0) to a
    fixed (n_y, n_x) grid in relative-body coords. Uses cv2.resize
    with NEAREST so the shape is preserved."""
    mask = (height_map[:, tail:head + 1] > 0).astype(np.uint8)
    if mask.size == 0:
        return np.zeros((n_y, n_x), dtype=np.float32)
    resized = cv2.resize(mask, (n_x, n_y), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32)


def build_average_pig_outline(masks_rel: list[np.ndarray], thresh: float = 0.4
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Given a list of (n_y, n_x) binary masks all in the same relative
    coord grid, average + threshold, then return the largest-contour as
    (x, y) arrays in normalized [0, 1] coordinates."""
    avg = np.stack(masks_rel, axis=0).mean(axis=0)
    binary = (avg >= thresh).astype(np.uint8)
    n_y, n_x = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([]), np.array([])
    big = max(contours, key=cv2.contourArea)
    xs = big[:, 0, 0].astype(np.float32) / max(n_x - 1, 1)
    # Image y goes 0=top -> n_y-1=bottom. We want plot y to go bottom->up,
    # but the pig viewed from above is symmetric so this orientation
    # doesn't matter; we'll just center it. Normalize y to [-1, 1] around
    # the pig's mid-line.
    ys_pix = big[:, 0, 1].astype(np.float32)
    pig_ys = np.where(binary.any(axis=1))[0]
    y_mid = (pig_ys.min() + pig_ys.max()) / 2.0 if pig_ys.size else (n_y - 1) / 2.0
    y_half = max((pig_ys.max() - pig_ys.min()) / 2.0, 1.0)
    ys = (ys_pix - y_mid) / y_half
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="weights/pigformer_fold0.pt")
    ap.add_argument("--dataset", default="data/dataset.h5")
    ap.add_argument("--labels", default="data/label.h5")
    ap.add_argument("--split_json", default="data/split.json")
    ap.add_argument("--out_dir", default="verification_fix/pigformer_saliency_aggregate")
    ap.add_argument("--n_panel_bags", type=int, default=6)
    ap.add_argument("--smooth_samples", type=int, default=20)
    ap.add_argument("--noise_std", type=float, default=0.04)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target_names = load_pigformer(args.checkpoint, device)
    print(f"target_names: {target_names}")

    split = json.loads(Path(args.split_json).read_text())
    test_label_indices = list(split["test_indices"])
    # Map label-row indices to bag names via label.h5.
    with h5py.File(args.labels, "r") as f:
        label_bag_names = np.asarray(f["bag_names"]).astype(str)
    test_bag_set = {label_bag_names[i] for i in test_label_indices}

    n_bins = 100
    aggregated: dict[str, list[np.ndarray]] = {t: [] for t in target_names}
    pig_profiles: list[np.ndarray] = []
    pig_masks_rel: list[np.ndarray] = []
    bag_data: list[dict] = []

    with h5py.File(args.dataset, "r") as f:
        height_maps = f["height_maps"]
        bag_names_arr = np.asarray(f["bag_names"]).astype(str)
        bag_to_rows = defaultdict(list)
        for ri, b in enumerate(bag_names_arr):
            if b in test_bag_set:
                bag_to_rows[b].append(ri)
        unique_bags = sorted(bag_to_rows.keys())
        print(f"processing {len(unique_bags)} test bags...")

        for bi, bag in enumerate(unique_bags):
            rows = sorted(bag_to_rows[bag])
            hmaps = np.stack([np.asarray(height_maps[r]) for r in rows], axis=0)
            mean_hmap = hmaps.mean(axis=0).astype(np.float32)  # (96, 224)
            tail_col, head_col = pig_extent_along_spine(mean_hmap)
            if (head_col - tail_col) < 50:
                # pig too small / partial; skip
                continue
            x_seq = torch.from_numpy(mean_hmap.T).unsqueeze(0).to(device)  # (1, 224, 96)

            with torch.no_grad():
                pred = model(x_seq).cpu().numpy()[0]
            col_imps = {}
            for ti, tname in enumerate(target_names):
                col_imp = smoothgrad_column_importance(
                    model, x_seq.clone(), ti,
                    n_samples=args.smooth_samples, noise_std=args.noise_std)
                col_imps[tname] = col_imp
                resampled = resample_to_relative(col_imp, tail_col, head_col, n_bins=n_bins)
                # normalize each bag's curve to peak=1 so we don't weight by
                # the absolute saliency magnitude
                resampled = resampled / max(resampled.max(), 1e-9)
                aggregated[tname].append(resampled)
            pig_profiles.append(mean_pig_profile(mean_hmap, tail_col, head_col, n_bins))
            pig_masks_rel.append(pig_mask_to_relative(mean_hmap, tail_col, head_col))
            bag_data.append({"bag": bag, "mean_hmap": mean_hmap,
                             "tail_col": tail_col, "head_col": head_col,
                             "col_imps": col_imps, "pred": pred})
            if (bi + 1) % 10 == 0:
                print(f"  ...{bi + 1}/{len(unique_bags)}")

    print(f"aggregated over {len(aggregated[target_names[0]])} bags")

    # === Figure A: aggregate curve over an "ideal pig" silhouette ===
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor((237/255,)*3); ax.set_facecolor((237/255,)*3)
    rel_x = np.linspace(0, 1, n_bins)
    # 0 = tail (left), 1 = head (right). Last rib is 0.6 of the way from
    # head toward tail, i.e. (1 - 0.6) = 0.4 in left-origin coords.
    last_rib_x = 1.0 - LAST_RIB_REL
    # Pig silhouette: use the user-provided hand-drawn sketch as background.
    # Make white pixels transparent so only the outline shows through.
    extra_handles: list = []
    sketch_path = out / "sketch.png"
    if sketch_path.is_file():
        from PIL import Image
        sk = np.asarray(Image.open(sketch_path).convert("RGB"), dtype=np.float32) / 255.0
        # White -> transparent, black outline -> opaque. Alpha = 1 - brightness.
        alpha = 1.0 - sk.mean(axis=2)
        # Boost contrast so the outline reads clearly at low overall alpha.
        alpha = np.clip(alpha * 2.5, 0, 1)
        rgba = np.dstack([np.zeros_like(alpha), np.zeros_like(alpha),
                          np.zeros_like(alpha), alpha * 0.55])
        # Pig faces right in the sketch (head on right) -> matches our
        # convention (0 = tail at left, 1 = head at right). No flip needed.
        ax.imshow(rgba, extent=(0, 1, 0.1, 0.9), aspect="auto",
                  zorder=0, interpolation="bilinear")
        # Proxy artist for the legend. The silhouette is rendered with
        # alpha=0.55 over the gray axes facecolor (237/255), so its visible
        # color on the plot is ~rgb(107,107,107). Match that here.
        extra_handles.append(Line2D([0], [0], color="#6b6b6b", linewidth=2,
                                     label="Ideal Pig Silhouette"))
        print(f"  using background sketch from {sketch_path}")
    else:
        print(f"  [warn] {sketch_path} not found; no background sketch")
    for tname in target_names:
        arr = np.stack(aggregated[tname], axis=0)  # (n_bags, n_bins)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = TARGET_COLORS.get(tname, "k")
        ax.plot(rel_x, mean, color=color, linewidth=3,
                label=TARGET_LABELS.get(tname, tname))
        ax.fill_between(rel_x, mean - std, mean + std, color=color, alpha=0.13)
    ax.axvline(last_rib_x, color="k", linewidth=2, linestyle="--", alpha=0.7,
               label=f"Anatomical last rib")
    ax.set_xlabel("Relative body position  (0 = tail end,  1 = head end)", fontsize=15)
    ax.set_ylabel("Normalized column\nimportance", fontsize=15)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", labelsize=13)
    cur_handles, cur_labels = ax.get_legend_handles_labels()
    ax.legend(extra_handles + cur_handles,
              [h.get_label() for h in extra_handles] + cur_labels,
              fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    agg_path = out / "aggregate_curve.png"
    fig.savefig(agg_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {agg_path}")

    # === Figure B: per-bag overlays for a few representative bags ===
    if bag_data:
        step = max(1, len(bag_data) // args.n_panel_bags)
        picks = [bag_data[i * step] for i in range(args.n_panel_bags) if i * step < len(bag_data)]
        for entry in picks:
            bag = entry["bag"]
            mean_hmap = entry["mean_hmap"]
            tail_col, head_col = entry["tail_col"], entry["head_col"]
            # last rib at LAST_RIB_REL of the way from head toward tail; head
            # is on the right, so last_rib_col = head_col - LAST_RIB_REL * length
            length = head_col - tail_col
            last_rib_col = int(head_col - LAST_RIB_REL * length)

            fig, axes = plt.subplots(2, 1, figsize=(11, 5.0),
                                      gridspec_kw={"height_ratios": [1.0, 0.7]})
            # top: height map with vertical lines at tail/last-rib/head
            ax = axes[0]
            hm_disp = np.ma.masked_where(mean_hmap <= 0, mean_hmap)
            cmap_jet = plt.cm.jet.copy(); cmap_jet.set_bad(color="white")
            ax.imshow(hm_disp, cmap=cmap_jet, vmin=0, vmax=1.0, aspect="auto")
            ax.axvline(tail_col, color="#666", linewidth=2, linestyle=":")
            ax.axvline(last_rib_col, color="black", linewidth=3, linestyle="--")
            ax.axvline(head_col, color="#666", linewidth=2, linestyle=":")
            # Place text at the bottom of the height map (y near the image's
            # bottom edge) so it doesn't collide with the bag-name title.
            y_text = mean_hmap.shape[0] + 5
            ax.text(tail_col, y_text, "tail", fontsize=10, ha="center", color="#444")
            ax.text(last_rib_col, y_text, "last rib", fontsize=10, ha="center")
            ax.text(head_col, y_text, "head", fontsize=10, ha="center", color="#444")
            ax.set_yticks([]); ax.set_xticks([])

            # bottom: per-target column importance with same vertical lines
            ax2 = axes[1]
            ax2.set_facecolor((237/255,)*3)
            x_cols = np.arange(mean_hmap.shape[1])
            for tname in target_names:
                ci = entry["col_imps"][tname]
                ci_norm = ci / max(ci.max(), 1e-9)
                ax2.plot(x_cols, ci_norm, color=TARGET_COLORS.get(tname, "k"),
                          linewidth=2.5, label=TARGET_LABELS.get(tname, tname))
            ax2.axvline(tail_col, color="#666", linewidth=2, linestyle=":")
            ax2.axvline(last_rib_col, color="black", linewidth=3, linestyle="--")
            ax2.axvline(head_col, color="#666", linewidth=2, linestyle=":")
            ax2.set_xlim(0, mean_hmap.shape[1])
            ax2.set_xticks([0, 40, 80, 120, 160, 200])
            ax2.set_xlabel("Spine column (cm)", fontsize=12)
            ax2.set_ylabel("Importance", fontsize=12)
            ax2.tick_params(axis="both", labelsize=11)
            ax2.grid(True, alpha=0.4)
            ax2.legend(fontsize=10, loc="upper right")
            fig.tight_layout()
            fig.savefig(out / f"bag_{bag.replace('/', '_')}.png",
                        dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote bag overlay: {bag}")

    print(f"\nDone. Output dir: {out}")


if __name__ == "__main__":
    main()

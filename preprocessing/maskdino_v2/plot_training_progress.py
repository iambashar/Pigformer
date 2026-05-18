"""Live training progress plot for the MaskDINO v2 run.

Polls `<OUTPUT_DIR>/metrics.json` (Detectron2 writes one JSON object per
line, append-only) and re-renders a 3-panel PNG whenever new lines
appear: training total_loss, mask & box loss components, and val mask /
bbox AP from the periodic COCO evals.

Usage:
    python preprocessing/maskdino_v2/plot_training_progress.py \\
        --output-dir /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/output/pig_depth_v2_endpoint \\
        --interval 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def render(rows: list[dict], out_png: Path, title_suffix: str = "") -> None:
    train = [r for r in rows if "total_loss" in r and "iteration" in r]
    eval_rows = [r for r in rows if "segm/AP" in r or "bbox/AP" in r]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # --- Panel 1: total_loss
    if train:
        x = [r["iteration"] for r in train]
        y = [r["total_loss"] for r in train]
        axes[0].plot(x, y, lw=1.0, color="C0")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("total_loss")
        axes[0].set_title(f"training total_loss  ({len(train)} pts)")
        axes[0].grid(True, alpha=0.3)
        if y:
            axes[0].set_ylim(0, max(min(max(y), y[0] * 1.05), 1.0))

    # --- Panel 2: loss components (mask, dice, bbox, ce, endpoint)
    if train:
        comps = ("loss_mask", "loss_dice", "loss_bbox", "loss_ce", "loss_endpoint")
        for c in comps:
            ys = [(r["iteration"], r[c]) for r in train if c in r]
            if not ys:
                continue
            xs, vs = zip(*ys)
            axes[1].plot(xs, vs, lw=0.9, label=c)
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("component loss")
        axes[1].set_title("loss components")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    # --- Panel 3: val AP (mask + bbox) per eval
    if eval_rows:
        x = [r.get("iteration", 0) for r in eval_rows]
        for key, label, color in (("segm/AP", "mask AP", "C2"), ("bbox/AP", "bbox AP", "C3")):
            ys = [(r.get("iteration", 0), r[key]) for r in eval_rows if key in r]
            if not ys:
                continue
            xs, vs = zip(*ys)
            axes[2].plot(xs, vs, "-o", lw=1.2, ms=4, label=label, color=color)
        # Also break out per-category mask AP if present
        for key, label, color in (("segm/AP-pig", "pig mask AP", "C2"),
                                  ("segm/AP-pig_upper_body", "upper mask AP", "C4")):
            ys = [(r.get("iteration", 0), r[key]) for r in eval_rows if key in r]
            if not ys:
                continue
            xs, vs = zip(*ys)
            axes[2].plot(xs, vs, "--o", lw=1.0, ms=3, label=label, color=color, alpha=0.7)
        axes[2].set_xlabel("iteration")
        axes[2].set_ylabel("AP")
        axes[2].set_title(f"val AP  ({len(eval_rows)} evals)")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "no eval rows yet", transform=axes[2].transAxes,
                     ha="center", va="center", color="gray")
        axes[2].set_title("val AP")

    last_iter = train[-1]["iteration"] if train else "-"
    fig.suptitle(f"MaskDINO v2 training progress — last iter {last_iter}{title_suffix}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="Detectron2 OUTPUT_DIR (contains metrics.json after iter 0).")
    ap.add_argument("--png", default="",
                    help="Output PNG. Default: <output-dir>/training_progress.png")
    ap.add_argument("--interval", type=float, default=30.0,
                    help="Poll interval in seconds. Re-renders only when metrics.json grew.")
    ap.add_argument("--once", action="store_true",
                    help="Render once and exit (no polling).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    metrics_path = out_dir / "metrics.json"
    out_png = Path(args.png) if args.png else out_dir / "training_progress.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    last_size = -1
    while True:
        size = metrics_path.stat().st_size if metrics_path.is_file() else 0
        if size != last_size:
            rows = read_metrics(metrics_path)
            if rows:
                render(rows, out_png, title_suffix=f"  ({size//1024} KB metrics)")
                print(f"[plot] wrote {out_png} from {len(rows)} rows  ({size} B)", flush=True)
            last_size = size
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

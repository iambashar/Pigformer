"""
Hold-out test evaluation.

Loads a checkpoint saved by train.py and reports per-bag MAE / NMAE-IQR on the
test set. Two aggregation modes are supported:

- ``input`` (default, paper Table 2): average height maps across all frames of
  a bag into a single input, then one forward pass per bag. This is what
  produces the paper's 3.91 mm overall MAE number.
- ``output``: forward every frame, average the predictions. Slightly worse
  under the paper's checkpoint (~4.00 mm).

Example:

    python evaluate.py \\
        --checkpoint results/pigformer_fold0/best.pt \\
        --dataset data/dataset.h5 \\
        --labels data/label.h5 \\
        --split_json data/split.json

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import AllFramesIterator
from models import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output", default=None, help="Optional JSON file for metrics.")
    parser.add_argument(
        "--aggregation", choices=["input", "output"], default="input",
        help="input: average height maps first, 1 forward per bag. output: forward every frame, average preds.",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    target_names = ckpt["target_names"]
    iqr_scales = np.asarray(ckpt["val_iqr_scales"], dtype=np.float32)

    split = json.loads(Path(args.split_json).read_text())
    test_indices = split["test_indices"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_rep = "height_geom" if train_args["arch"] == "cnn" else "height"

    model = create_model(
        train_args["arch"],
        seq_len=224, feature_dim=96, dim_out=len(target_names),
        nhead=train_args.get("nhead", 8),
        num_layers=train_args.get("num_layers", 1),
        dim_feedforward=train_args.get("dim_feedforward", 512),
        dropout=train_args.get("dropout", 0.0),
        head_dropout=train_args.get("head_dropout", 0.0),
        in_channels=3 if train_args["arch"] == "cnn" else 1,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    it = AllFramesIterator(args.dataset, test_indices, args.labels, target_names, input_rep=input_rep)

    preds, targets, masks = [], [], []
    with torch.no_grad():
        for _, frames, t, m, _tag in it.bags():
            # frames: (F, 224, 96) for height, or (F, 3, 96, 224) for height_geom.
            if args.aggregation == "input":
                x = frames.mean(dim=0, keepdim=True).to(device)
                p = model(x).cpu().numpy()[0]
            else:
                p = model(frames.to(device)).cpu().numpy().mean(axis=0)
            preds.append(p)
            targets.append(t.numpy())
            masks.append(m.numpy())

    preds = np.stack(preds)
    targets = np.stack(targets)
    masks = np.stack(masks).astype(bool)

    per_target_mae = []
    per_target_nmae = []
    for i, name in enumerate(target_names):
        sel = masks[:, i]
        if not sel.any():
            per_target_mae.append(float("nan"))
            per_target_nmae.append(float("nan"))
            continue
        mae = float(np.mean(np.abs(preds[sel, i] - targets[sel, i])))
        per_target_mae.append(mae)
        per_target_nmae.append(mae / float(iqr_scales[i]))

    overall_mae = float(np.nanmean(per_target_mae))
    primary_nmae_iqr = float(np.nanmean(per_target_nmae))

    print(f"Test bags: {len(preds)}")
    print(f"Overall MAE:        {overall_mae:.4f}")
    print(f"Primary NMAE-IQR:   {primary_nmae_iqr:.4f}")
    for name, mae, nmae in zip(target_names, per_target_mae, per_target_nmae):
        print(f"  {name:12s}  MAE={mae:.4f}  NMAE-IQR={nmae:.4f}")

    metrics = {
        "overall_mae": overall_mae,
        "primary_nmae_iqr": primary_nmae_iqr,
        "per_target_mae": dict(zip(target_names, per_target_mae)),
        "per_target_nmae_iqr": dict(zip(target_names, per_target_nmae)),
        "n_test_bags": int(len(preds)),
    }
    out_path = Path(args.output) if args.output else Path(args.checkpoint).parent / "test_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

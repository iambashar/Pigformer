"""
Evaluate an ensemble of PigFormer fold checkpoints on the held-out test set.

Each checkpoint is evaluated per bag, then predictions are averaged across
checkpoints. Aggregation can be output-level (paper text) or input-level
(released-checkpoint README path).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import AllFramesIterator
from models import create_model


def _build_model(ckpt: dict, device: torch.device):
    train_args = ckpt["args"]
    target_names = ckpt["target_names"]
    input_rep = "height_geom" if train_args["arch"] == "cnn" else "height"
    model = create_model(
        train_args["arch"],
        seq_len=224,
        feature_dim=96,
        dim_out=len(target_names),
        nhead=train_args.get("nhead", 8),
        num_layers=train_args.get("num_layers", 1),
        dim_feedforward=train_args.get("dim_feedforward", 512),
        dropout=train_args.get("dropout", 0.0),
        head_dropout=train_args.get("head_dropout", 0.0),
        in_channels=3 if train_args["arch"] == "cnn" else 1,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, input_rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregation", choices=["input", "output"], default="output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in args.checkpoints]
    target_names = ckpts[0]["target_names"]
    iqr_scales = np.asarray(ckpts[0]["val_iqr_scales"], dtype=np.float32)

    for path, ckpt in zip(args.checkpoints, ckpts):
        if ckpt["target_names"] != target_names:
            raise ValueError(f"Target mismatch in {path}: {ckpt['target_names']} != {target_names}")

    models = []
    input_rep = None
    for ckpt in ckpts:
        model, rep = _build_model(ckpt, device)
        models.append(model)
        input_rep = rep if input_rep is None else input_rep
        if rep != input_rep:
            raise ValueError("All ensemble checkpoints must use the same input representation")

    split = json.loads(Path(args.split_json).read_text())
    it = AllFramesIterator(args.dataset, split["test_indices"], args.labels, target_names, input_rep=input_rep)

    preds, targets, masks = [], [], []
    with torch.no_grad():
        for _, frames, t, m, _tag in it.bags():
            model_preds = []
            for model in models:
                if args.aggregation == "input":
                    x = frames.mean(dim=0, keepdim=True).to(device)
                    p = model(x).cpu().numpy()[0]
                else:
                    p = model(frames.to(device)).cpu().numpy().mean(axis=0)
                model_preds.append(p)
            preds.append(np.stack(model_preds, axis=0).mean(axis=0))
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

    metrics = {
        "overall_mae": float(np.nanmean(per_target_mae)),
        "primary_nmae_iqr": float(np.nanmean(per_target_nmae)),
        "per_target_mae": dict(zip(target_names, per_target_mae)),
        "per_target_nmae_iqr": dict(zip(target_names, per_target_nmae)),
        "n_test_bags": int(len(preds)),
        "n_checkpoints": len(args.checkpoints),
        "aggregation": args.aggregation,
        "checkpoints": args.checkpoints,
    }

    print(f"Test bags: {len(preds)}")
    print(f"Checkpoints: {len(args.checkpoints)}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Overall MAE:        {metrics['overall_mae']:.4f}")
    print(f"Primary NMAE-IQR:   {metrics['primary_nmae_iqr']:.4f}")
    for name, mae, nmae in zip(target_names, per_target_mae, per_target_nmae):
        print(f"  {name:12s}  MAE={mae:.4f}  NMAE-IQR={nmae:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

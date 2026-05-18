"""
Train PigFormer or a baseline on the depth height-map dataset.

Reproduces the paper's fold-0 training protocol: per-bag single-frame sampling
during training, frame-level MAE on the validation set, IQR-weighted L1 loss,
AdamW + cosine schedule with warmup.

Example:

    python train.py \\
        --arch pigformer \\
        --dataset data/dataset.h5 \\
        --labels data/label.h5 \\
        --split_json data/split.json \\
        --results_dir results/pigformer_fold0

"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AllFramesIterator, PigDataset, get_available_targets, load_target_column
from models import create_model
from split import load_or_create_split


def resolve_target_names(labels_path: str, targets: list[str] | None = None) -> list[str]:
    available = get_available_targets(labels_path)
    if not targets:
        return available
    missing = [t for t in targets if t not in available]
    if missing:
        raise ValueError(f"Requested targets are unavailable: {missing}. Available targets: {available}")
    return list(targets)


def iqr_weights(labels_h5: str, target_names, train_indices, device) -> torch.Tensor:
    """Per-target weights = 1/IQR, normalized so the mean is 1.0."""
    with h5py.File(labels_h5, "r") as h:
        arrays = {n: load_target_column(h, n) for n in target_names}
    train_indices = np.asarray(train_indices, dtype=int)
    weights = []
    for name in target_names:
        values = arrays[name][train_indices]
        valid = values[np.isfinite(values)]
        q1, q3 = np.percentile(valid, [25, 75])
        scale = max(float(q3 - q1), 1e-6)
        weights.append(1.0 / scale)
    w = np.array(weights, dtype=np.float32)
    w = w / w.mean()
    return torch.tensor(w, device=device)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_target = (F.l1_loss(pred, target, reduction="none") * mask).sum(0) / mask.sum(0).clamp_min(1.0)
    return (per_target * weights).sum()


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Unweighted per-target Huber loss, matching the paper's Eq. 1."""
    err = torch.abs(pred - target)
    d = float(delta)
    loss = torch.where(err <= d, 0.5 * err * err, d * err - 0.5 * d * d)
    per_target = (loss * mask).sum(0) / mask.sum(0).clamp_min(1.0)
    return per_target.sum()


@torch.no_grad()
def validate(model, loader, device, target_names, iqr_scales: np.ndarray) -> dict:
    model.eval()
    preds, targets, masks = [], [], []
    for batch in loader:
        x = batch["data_in"].to(device)
        p = model(x).cpu().numpy()
        preds.append(p)
        targets.append(batch["targets"].numpy())
        masks.append(batch["mask"].numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    masks = np.concatenate(masks).astype(bool)

    per_target_mae = []
    per_target_nmae = []
    for i, name in enumerate(target_names):
        m = masks[:, i]
        if not m.any():
            per_target_mae.append(float("nan"))
            per_target_nmae.append(float("nan"))
            continue
        mae = float(np.mean(np.abs(preds[m, i] - targets[m, i])))
        per_target_mae.append(mae)
        per_target_nmae.append(mae / float(iqr_scales[i]))
    overall_mae = float(np.nanmean(per_target_mae))
    primary_nmae_iqr = float(np.nanmean(per_target_nmae))
    return {
        "overall_mae": overall_mae,
        "primary_nmae_iqr": primary_nmae_iqr,
        "per_target_mae": dict(zip(target_names, per_target_mae)),
    }


@torch.no_grad()
def validate_all_frames(model, iterator: AllFramesIterator, device, target_names, iqr_scales: np.ndarray) -> dict:
    model.eval()
    preds, targets, masks = [], [], []
    for _, frames, t, m, _tag in iterator.bags():
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
        m = masks[:, i]
        if not m.any():
            per_target_mae.append(float("nan"))
            per_target_nmae.append(float("nan"))
            continue
        mae = float(np.mean(np.abs(preds[m, i] - targets[m, i])))
        per_target_mae.append(mae)
        per_target_nmae.append(mae / float(iqr_scales[i]))
    return {
        "overall_mae": float(np.nanmean(per_target_mae)),
        "primary_nmae_iqr": float(np.nanmean(per_target_nmae)),
        "per_target_mae": dict(zip(target_names, per_target_mae)),
    }


def compute_iqr_scales(labels_h5: str, target_names, train_indices) -> np.ndarray:
    with h5py.File(labels_h5, "r") as h:
        arrays = {n: load_target_column(h, n) for n in target_names}
    train_indices = np.asarray(train_indices, dtype=int)
    scales = []
    for name in target_names:
        values = arrays[name][train_indices]
        valid = values[np.isfinite(values)]
        q1, q3 = np.percentile(valid, [25, 75])
        scales.append(max(float(q3 - q1), 1e-6))
    return np.asarray(scales, dtype=np.float32)


def init_output_bias(model, labels_h5, target_names, train_indices):
    with h5py.File(labels_h5, "r") as h:
        arrays = {n: load_target_column(h, n) for n in target_names}
    train_indices = np.asarray(train_indices, dtype=int)
    means = np.array(
        [np.nanmean(arrays[n][train_indices]) for n in target_names], dtype=np.float32
    )
    final_linear = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == len(target_names):
            final_linear = m
    if final_linear is not None and final_linear.bias is not None:
        with torch.no_grad():
            final_linear.bias.copy_(torch.from_numpy(means))


def build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=max(warmup_epochs, 1)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )


def build_optimizer(model, args):
    if args.arch == "pigformer" and (args.encoder_lr is not None or args.head_lr is not None):
        encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
        head_lr = args.head_lr if args.head_lr is not None else args.lr
        return torch.optim.AdamW(
            [
                {"params": model.layers.parameters(), "lr": encoder_lr, "name": "encoder"},
                {"params": model.head.parameters(), "lr": head_lr, "name": "head"},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def load_pretrained_encoder(model, checkpoint_path: str, device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "encoder_state_dict" in ckpt:
        encoder_state = ckpt["encoder_state_dict"]
    elif "model" in ckpt:
        encoder_state = {
            k[len("layers.") :]: v
            for k, v in ckpt["model"].items()
            if k.startswith("layers.")
        }
    else:
        encoder_state = ckpt

    if not hasattr(model, "load_encoder_state_dict"):
        raise ValueError("--pretrained_encoder is only supported for arch=pigformer")
    incompatible = model.load_encoder_state_dict(encoder_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Encoder checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print(f"Loaded pretrained encoder: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="pigformer", choices=["pigformer", "mlp", "cnn"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder_lr", type=float, default=None, help="Optional PigFormer encoder LR for discriminative fine-tuning.")
    parser.add_argument("--head_lr", type=float, default=None, help="Optional PigFormer regression-head LR for discriminative fine-tuning.")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--moderate_aug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional supervised target subset, e.g. --targets fat_rib12 for backfat-only training.",
    )
    parser.add_argument("--loss", choices=["iqr_l1", "huber"], default="iqr_l1")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--selection_metric", choices=["nmae_iqr", "overall_mae"], default="nmae_iqr")
    parser.add_argument(
        "--val_aggregation",
        choices=["middle", "output"],
        default="middle",
        help="middle: validate on middle frame. output: average all-frame predictions per bag.",
    )
    parser.add_argument(
        "--pretrained_encoder",
        default=None,
        help="Optional JEPA checkpoint. Loads only PigFormer encoder weights before supervised training.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    split = load_or_create_split(args.labels, args.split_json, seed=args.seed)
    fold = split["folds"][args.fold]
    target_names = resolve_target_names(args.labels, args.targets)
    print(f"Device: {device}  |  Arch: {args.arch}  |  Targets: {target_names}")
    print(f"Train: {len(fold['train'])}  Val: {len(fold['val'])}  Test: {len(split['test_indices'])}")

    input_rep = "height_geom" if args.arch == "cnn" else "height"
    train_set = PigDataset(
        args.dataset, fold["train"], args.labels, target_names,
        mode="train", frame_agg_count=1, moderate_aug=args.moderate_aug, input_rep=input_rep,
    )
    val_set = PigDataset(
        args.dataset, fold["val"], args.labels, target_names,
        mode="val", frame_agg_count=1, input_rep=input_rep,
    )
    val_all_frames = None
    if args.val_aggregation == "output":
        val_all_frames = AllFramesIterator(args.dataset, fold["val"], args.labels, target_names, input_rep=input_rep)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = create_model(
        args.arch,
        seq_len=224, feature_dim=96, dim_out=len(target_names),
        nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, head_dropout=args.head_dropout,
        in_channels=3 if args.arch == "cnn" else 1,
    ).to(device)
    if args.pretrained_encoder:
        if args.arch != "pigformer":
            raise ValueError("--pretrained_encoder is only supported with --arch pigformer")
        load_pretrained_encoder(model, args.pretrained_encoder, device)
    init_output_bias(model, args.labels, target_names, fold["train"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    optimizer = build_optimizer(model, args)
    if args.encoder_lr is not None or args.head_lr is not None:
        print(f"Optimizer groups: encoder_lr={optimizer.param_groups[0]['lr']:.2e} head_lr={optimizer.param_groups[1]['lr']:.2e}")
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    loss_weights = iqr_weights(args.labels, target_names, fold["train"], device)
    val_iqr_scales = compute_iqr_scales(args.labels, target_names, fold["train"])

    best_score = float("inf")
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["data_in"].to(device)
            y = batch["targets"].to(device)
            m = batch["mask"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            if args.loss == "huber":
                loss = masked_huber(pred, y, m, delta=args.huber_delta)
            else:
                loss = masked_l1(pred, y, m, loss_weights)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss = epoch_loss / max(n_batches, 1)
        if val_all_frames is not None:
            metrics = validate_all_frames(model, val_all_frames, device, target_names, val_iqr_scales)
        else:
            metrics = validate(model, val_loader, device, target_names, val_iqr_scales)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        score = metrics["overall_mae"] if args.selection_metric == "overall_mae" else metrics["primary_nmae_iqr"]
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "target_names": target_names,
                    "val_iqr_scales": val_iqr_scales.tolist(),
                    "epoch": epoch,
                    "val_metrics": metrics,
                },
                results_dir / "best.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            per_t = "  ".join(f"{k}: {v:.2f}" for k, v in metrics["per_target_mae"].items())
            print(
                f"Epoch {epoch:4d} | train_loss: {train_loss:.4f} | "
                f"val_mae: {metrics['overall_mae']:.4f} | nmae_iqr: {metrics['primary_nmae_iqr']:.4f} | "
                f"{per_t} | {int(time.time() - t0)}s"
            )

    (results_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val {args.selection_metric}: {best_score:.4f}")
    print(f"Checkpoint: {results_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

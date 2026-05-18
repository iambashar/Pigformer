"""Train a tiny depth UNet (MobileNetV3-Small encoder) on pig_depth_v3.

Two-channel semantic segmentation: pig vs background, upper_body vs
background. No instances, no queries, no Hungarian matching. Targets are
binary (B, 2, H, W). Negative frames (image present, no annotations)
have an all-zero target on both channels — that's the supervision that
teaches the model to output zero on empty pens, which the v2 MaskDINO
notably failed to learn.

Loss: per-channel BCE-with-logits + Dice (averaged). Equal weight on
both channels.

Run:
    cd /mnt/gs21/scratch/basharmk/data/unl/pigformer_release
    /mnt/home/basharmk/.conda/envs/swine-rgbd/bin/python -u \\
        scripts/train_unet_depth.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py  # noqa: F401  (not used directly but keeps env warm)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader, Dataset

import timm

V3_DIR = Path("/mnt/gs21/scratch/basharmk/data/unl/MaskDINO/datasets/pig_depth_combined_v3b")
PIXEL_MEAN = 2.0208
PIXEL_STD = 0.4158
IMG_H, IMG_W = 576, 640
PIG_CAT_ID = 1
UPPER_CAT_ID = 2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _seg_to_mask(seg, h, w) -> np.ndarray:
    """COCO segmentation -> (H, W) uint8 mask.

    Handles: polygon list, uncompressed RLE (counts: list of ints),
    compressed RLE (counts: bytes or str — v3 dataset uses this).
    """
    if isinstance(seg, list):
        rles = coco_mask.frPyObjects(seg, h, w)
        rle = coco_mask.merge(rles)
    elif isinstance(seg, dict):
        counts = seg.get("counts")
        if isinstance(counts, list):
            rle = coco_mask.frPyObjects(seg, h, w)
        else:
            # Compressed RLE; pycocotools wants bytes for `counts`.
            rle = {
                "size": list(seg["size"]),
                "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
            }
    else:
        return np.zeros((h, w), dtype=np.uint8)
    return coco_mask.decode(rle)


class V3Depth(Dataset):
    def __init__(self, json_path: Path, image_dir: Path, hflip: bool = True):
        coco = json.loads(json_path.read_text())
        self.images = coco["images"]
        anns_by_image: dict[int, list] = {}
        for a in coco["annotations"]:
            anns_by_image.setdefault(int(a["image_id"]), []).append(a)
        self.anns_by_image = anns_by_image
        self.image_dir = image_dir
        self.hflip = hflip

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        rec = self.images[idx]
        img = Image.open(self.image_dir / rec["file_name"])
        depth_mm = np.asarray(img, dtype=np.int32)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        # Normalize using the same μ/σ MaskDINO v3 uses.
        depth_n = (depth_m - PIXEL_MEAN) / PIXEL_STD

        # Build (2, H, W) target.
        target = np.zeros((2, rec["height"], rec["width"]), dtype=np.float32)
        for a in self.anns_by_image.get(int(rec["id"]), []):
            m = _seg_to_mask(a["segmentation"], rec["height"], rec["width"])
            ch = 0 if int(a["category_id"]) == PIG_CAT_ID else 1
            target[ch] = np.maximum(target[ch], m.astype(np.float32))

        if self.hflip and np.random.rand() < 0.5:
            depth_n = depth_n[:, ::-1].copy()
            target = target[:, :, ::-1].copy()

        return (
            torch.from_numpy(depth_n).unsqueeze(0),         # (1, H, W)
            torch.from_numpy(target),                        # (2, H, W)
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class DepthUNet(nn.Module):
    def __init__(self, encoder: str = "mobilenetv3_small_100", n_out: int = 2):
        super().__init__()
        self.encoder = timm.create_model(
            encoder, features_only=True, in_chans=1, pretrained=True
        )
        chs = self.encoder.feature_info.channels()  # length 5 for mobilenetv3
        # chs example for mobilenetv3_small_100: [16, 16, 24, 48, 576]
        assert len(chs) == 5, f"expected 5 feature levels, got {len(chs)}: {chs}"
        self.up3 = UpBlock(chs[4], chs[3], 128)
        self.up2 = UpBlock(128, chs[2], 64)
        self.up1 = UpBlock(64, chs[1], 32)
        self.up0 = UpBlock(32, chs[0], 16)
        self.head = nn.Conv2d(16, n_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        x = feats[4]
        x = self.up3(x, feats[3])
        x = self.up2(x, feats[2])
        x = self.up1(x, feats[1])
        x = self.up0(x, feats[0])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


# ---------------------------------------------------------------------------
# Loss & metric
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel soft Dice (averaged across batch + channel)."""
    probs = torch.sigmoid(logits)
    dims = (-1, -2)
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(logits, target)


@torch.no_grad()
def iou_per_channel(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5):
    pred = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum(dim=(-1, -2))
    union = ((pred + target) > 0).float().sum(dim=(-1, -2))
    iou = (inter + 1e-6) / (union + 1e-6)
    # Only count IoU where there's a real target (avoid 100% from all-zero/all-zero).
    has_target = (target.sum(dim=(-1, -2)) > 0).float()
    return iou, has_target  # (B, C), (B, C)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="/mnt/gs21/scratch/basharmk/data/unl/MaskDINO/output/pig_depth_unet_v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--encoder", default="mobilenetv3_small_100")
    ap.add_argument("--val_every", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"args: {vars(args)}")

    train_set = V3Depth(V3_DIR / "annotations/instances_train.json",
                        V3_DIR / "images/train", hflip=True)
    val_set = V3Depth(V3_DIR / "annotations/instances_val.json",
                      V3_DIR / "images/val", hflip=False)
    log(f"train: {len(train_set)} images  val: {len(val_set)} images")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthUNet(encoder=args.encoder, n_out=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: {args.encoder} + U-Net, {n_params/1e6:.2f}M params")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_b = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = combined_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            train_loss += float(loss.detach())
            n_b += 1
        scheduler.step()
        train_loss /= max(n_b, 1)

        if (ep + 1) % args.val_every == 0 or ep == args.epochs - 1:
            model.eval()
            val_loss = 0.0
            iou_sum = torch.zeros(2, device=device)
            has_sum = torch.zeros(2, device=device)
            n_b = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(x)
                        loss = combined_loss(logits, y)
                    val_loss += float(loss.detach())
                    iou, has = iou_per_channel(logits, y)
                    iou_sum += (iou * has).sum(dim=0)
                    has_sum += has.sum(dim=0)
                    n_b += 1
            val_loss /= max(n_b, 1)
            iou_pig = float(iou_sum[0] / max(float(has_sum[0]), 1.0))
            iou_ub = float(iou_sum[1] / max(float(has_sum[1]), 1.0))
            dt = time.time() - t0
            log(f"epoch {ep+1:>3d}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  iou_pig={iou_pig:.4f}  iou_upper={iou_ub:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s)")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "state_dict": model.state_dict(),
                    "encoder": args.encoder,
                    "pixel_mean": PIXEL_MEAN,
                    "pixel_std": PIXEL_STD,
                    "epoch": ep + 1,
                    "val_loss": val_loss,
                    "iou_pig": iou_pig,
                    "iou_upper": iou_ub,
                }, out / "model_best.pt")
        else:
            dt = time.time() - t0
            log(f"epoch {ep+1:>3d}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s)")

    torch.save({"state_dict": model.state_dict(),
                "encoder": args.encoder,
                "pixel_mean": PIXEL_MEAN,
                "pixel_std": PIXEL_STD,
                "epoch": args.epochs,
                "final": True},
               out / "model_final.pt")
    log(f"done. best val_loss={best_val:.4f}  output={out}")


if __name__ == "__main__":
    main()

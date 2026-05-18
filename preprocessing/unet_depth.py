"""UNet depth segmenter (MobileNetV3-Small encoder, U-Net decoder).

Module split out of scripts/train_unet_depth.py so build_height_dataset.py
can load the trained checkpoint without depending on the training script.
The DepthUNet class definition here is parameter-equivalent to the one
in train_unet_depth.py (same nn.Module structure, same submodule names),
so state_dicts saved by training load cleanly here.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PIXEL_MEAN = 2.0208
PIXEL_STD = 0.4158

# Match the consolidated source's undistortion exactly: newCameraMatrix = K
# with fx/fy scaled by FX_SCALE. The build_summary.json on
# combined_upadted_upper_body records fx_scale: 0.85, and the GT polygons
# were drawn against that grid. Use the same value here so training and
# inference both produce images on the same pixel grid as the labels.
FX_SCALE = 0.85


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
        import timm
        self.encoder = timm.create_model(
            encoder, features_only=True, in_chans=1, pretrained=False
        )
        chs = self.encoder.feature_info.channels()
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


def load_unet(weights_path: str, device: torch.device | str = "cuda") -> tuple[DepthUNet, dict]:
    """Load a trained UNet checkpoint. Returns (model_in_eval_mode, ckpt_meta_dict)."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = DepthUNet(encoder=ckpt.get("encoder", "mobilenetv3_small_100"), n_out=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def undistort_for_unet(depth_u16: np.ndarray, cam: dict,
                       fx_scale: float | None = None) -> np.ndarray:
    """Apply the exact undistortion the consolidated source used: K with
    fx/fy * FX_SCALE as the new camera matrix, INTER_NEAREST, BORDER_CONSTANT 0.
    Returns a uint16 array (mm) of the same shape as input.

    Critical: this is the SAME undistortion the GT polygons were drawn
    against (see build_summary.json on combined_upadted_upper_body where
    fx_scale: 0.85 is recorded), so training images and inference images
    both land on the polygon grid.
    """
    di = cam["DepthIntrinsic"]; dd = cam["DepthDistortion"]
    K = np.array([[di["fx"], 0, di["cx"]],
                  [0, di["fy"], di["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array([dd[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")],
                 dtype=np.float64)
    s = FX_SCALE if fx_scale is None else float(fx_scale)
    K_new = K.copy()
    K_new[0, 0] *= s
    K_new[1, 1] *= s
    H, W = depth_u16.shape
    mx, my = cv2.initUndistortRectifyMap(K, D, np.eye(3), K_new, (W, H), cv2.CV_32FC1)
    return cv2.remap(depth_u16.astype(np.uint16), mx, my, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


@torch.no_grad()
def predict_masks(model: DepthUNet, depth_und_u16_batch: np.ndarray,
                  device: torch.device | str = "cuda") -> np.ndarray:
    """Run UNet on a batch of undistorted uint16 depth frames.

    Args:
        depth_und_u16_batch: (B, H, W) uint16 mm.
    Returns:
        probs: (B, 2, H, W) float32 sigmoid probabilities. Ch 0 = pig,
            ch 1 = upper_body.
    """
    d = depth_und_u16_batch.astype(np.float32) / 1000.0
    d = (d - PIXEL_MEAN) / PIXEL_STD
    x = torch.from_numpy(d).unsqueeze(1).to(device)
    return torch.sigmoid(model(x)).cpu().numpy().astype(np.float32)

"""Average the 3-channel ImageNet-pretrained R50 conv1 weights to 1 channel.

Detectron2 ResNet stem with STEM_IN_CHANNELS=1 expects conv1 weight shape
(64, 1, 7, 7). The standard pretrained pickle has (64, 3, 7, 7). Averaging
along the input-channel dim is the timm convention and gives a sensible
warm start for single-channel depth input.

Usage:
    python preprocessing/maskdino_v2/convert_r50_conv1_to_1ch.py \\
        --in-pkl  detectron2://ImageNetPretrained/torchvision/R-50.pkl \\
        --out-pkl weights/r50_conv1_1ch.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from fvcore.common.file_io import PathManager


CONV1_KEYS = ("stem.conv1.weight", "conv1.weight", "backbone.bottom_up.stem.conv1.weight")


def load_pkl(path: str) -> dict:
    with PathManager.open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def find_conv1(state: dict) -> str:
    for k in state.keys():
        kl = k.lower()
        if kl.endswith("conv1.weight") and "stem" in kl:
            return k
    for k in state.keys():
        if k.lower().endswith("conv1.weight"):
            return k
    raise KeyError(f"conv1 weight not found among {list(state.keys())[:10]}...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pkl", required=True,
                    help="Source pretrained pickle. Detectron2 catalog URI also works after caching.")
    ap.add_argument("--out-pkl", required=True)
    args = ap.parse_args()

    payload = load_pkl(args.in_pkl)
    state = payload.get("model", payload)

    key = find_conv1(state)
    w = state[key]
    if hasattr(w, "detach"):
        w = w.detach().cpu().numpy()
    w = np.asarray(w)
    print(f"conv1 key:   {key}")
    print(f"conv1 shape: {w.shape}  dtype={w.dtype}")
    if w.ndim != 4 or w.shape[1] != 3:
        raise ValueError(f"Expected (out, 3, kh, kw), got {w.shape}")

    w1 = w.mean(axis=1, keepdims=True).astype(w.dtype)
    print(f"new shape:   {w1.shape}")
    state[key] = w1

    out = {"model": state, "__author__": "maskdino_v2_conv1_avg"}
    if "matching_heuristics" in payload:
        out["matching_heuristics"] = payload["matching_heuristics"]
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {args.out_pkl}")


if __name__ == "__main__":
    main()

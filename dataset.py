"""
Dataset for PigFormer: depth height maps from an HDF5 archive.

The preprocessing pipeline writes one row per depth frame into ``dataset.h5``
under ``height_maps`` (shape ``(N, 96, 224)``) and ``bag_names`` (shape
``(N,)``). A separate ``label.h5`` holds one row per pig instance (bag) with
``fat_rib12``, ``loin_rib12`` columns and matching ``bag_names``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Sequence

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


TARGET_NAMES = ("fat_rib12", "loin_rib12", "total_rib")


def load_target_column(handle: h5py.File, name: str) -> np.ndarray:
    if name in handle:
        return handle[name][:]
    if name == "total_rib":
        return load_target_column(handle, "fat_rib12") + load_target_column(handle, "loin_rib12")
    raise KeyError(f"Target '{name}' not found in labels file")


def get_available_targets(labels_path: str, candidates: Sequence[str] = TARGET_NAMES) -> list[str]:
    with h5py.File(labels_path, "r") as h:
        avail = []
        for name in candidates:
            try:
                load_target_column(h, name)
                avail.append(name)
            except KeyError:
                pass
    return avail


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else v for v in values]


class PigDataset(Dataset):
    """Bag-level depth dataset. Returns one height-map tensor per pig instance.

    During training with ``frame_agg_count=1``, a single frame is sampled
    uniformly at random per epoch (implicit augmentation). At evaluation the
    caller should use ``frame_agg_count=-1`` and let the evaluation loop
    average predictions across all frames at the output level.

    Returned tensor shape is ``(seq_len, feature_dim) = (224, 96)`` so the
    sequence axis is the dorsal/longitudinal direction as described in
    Section 3.3 of the paper.
    """

    def __init__(
        self,
        h5_path: str,
        indices: Sequence[int],
        labels_path: str,
        target_names: Sequence[str],
        mode: str = "train",
        frame_agg_count: int = 1,
        moderate_aug: bool = False,
        input_rep: str = "height",
    ):
        assert mode in {"train", "val"}
        assert input_rep in {"height", "height_geom"}
        self.mode = mode
        self.frame_agg_count = frame_agg_count
        self.moderate_aug = moderate_aug and (mode == "train")
        self.target_names = list(target_names)
        self.input_rep = input_rep

        with h5py.File(labels_path, "r") as f:
            self.bag_names = _decode(f["bag_names"][:])
            id_key = "unique_ids" if "unique_ids" in f else "ear_tags"
            self.ids = [str(v) for v in _decode(f[id_key][:])]
            cols = {name: load_target_column(f, name) for name in self.target_names}
        self.targets = np.stack([cols[n] for n in self.target_names], axis=-1).astype(np.float32)

        with h5py.File(h5_path, "r") as f:
            self.height_maps = f["height_maps"][:]
            d_bags = _decode(f["bag_names"][:])
        self.empty_map = np.zeros(self.height_maps.shape[1:], dtype=np.float32)

        bag_to_frames: dict[str, list[int]] = defaultdict(list)
        for i, b in enumerate(d_bags):
            bag_to_frames[b].append(i)
        self.bag_frames = dict(bag_to_frames)

        self.indices = np.array(
            [i for i in indices if self.bag_frames.get(self.bag_names[i], [])]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _select_frames(self, frame_idxs: list[int]) -> list[int]:
        if self.frame_agg_count == 1:
            if self.mode == "train":
                return [int(np.random.choice(frame_idxs))]
            return [frame_idxs[len(frame_idxs) // 2]]
        count = len(frame_idxs) if self.frame_agg_count < 0 else min(self.frame_agg_count, len(frame_idxs))
        if self.mode == "train":
            if count < len(frame_idxs):
                return np.random.choice(frame_idxs, size=count, replace=False).tolist()
            return list(frame_idxs)
        if count == len(frame_idxs):
            return list(frame_idxs)
        offsets = np.linspace(0, len(frame_idxs) - 1, num=count, dtype=int)
        return [frame_idxs[o] for o in offsets]

    def _augment(self, im: np.ndarray) -> np.ndarray:
        im = np.ascontiguousarray(im)
        h, w = im.shape
        if np.random.rand() > 0.5:
            im = np.ascontiguousarray(np.flip(im, axis=0))  # vertical flip (body side swap)
        tx, ty = np.random.randint(-10, 11), np.random.randint(-10, 11)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        angle = float(np.random.uniform(-5.0, 5.0))
        M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        im = cv2.warpAffine(im, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return im

    def _to_geom_channels(self, im: np.ndarray) -> np.ndarray:
        valid = (im > 0).astype(np.float32)
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        return np.stack([im, valid, grad], axis=0)  # (3, H, W)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        frame_idxs = self.bag_frames.get(self.bag_names[real_idx], [])

        if frame_idxs:
            chosen = self._select_frames(frame_idxs)
            im = self.height_maps[chosen].astype(np.float32).mean(axis=0)
        else:
            im = self.empty_map.copy()

        if self.moderate_aug:
            im = self._augment(im)

        im = np.nan_to_num(im, nan=0.0).astype(np.float32)

        if self.input_rep == "height_geom":
            data_in = self._to_geom_channels(im)  # (3, 96, 224) for CNN baseline
        else:
            data_in = im.T  # (224, 96) = (seq_len, feature_dim)

        targets = self.targets[real_idx]
        mask = ~np.isnan(targets)
        targets = np.where(mask, targets, 0.0).astype(np.float32)

        return {
            "data_in": torch.from_numpy(data_in),
            "targets": torch.from_numpy(targets),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "tag": self.ids[real_idx],
        }


class AllFramesIterator:
    """Iterator that yields every frame of every bag as a separate sample,
    for output-level averaging at evaluation time.
    """

    def __init__(self, h5_path: str, indices: Sequence[int], labels_path: str, target_names: Sequence[str], input_rep: str = "height"):
        self.input_rep = input_rep
        self.target_names = list(target_names)

        with h5py.File(labels_path, "r") as f:
            self.bag_names = _decode(f["bag_names"][:])
            id_key = "unique_ids" if "unique_ids" in f else "ear_tags"
            self.ids = [str(v) for v in _decode(f[id_key][:])]
            cols = {name: load_target_column(f, name) for name in self.target_names}
        self.targets = np.stack([cols[n] for n in self.target_names], axis=-1).astype(np.float32)

        with h5py.File(h5_path, "r") as f:
            self.height_maps = f["height_maps"][:]
            d_bags = _decode(f["bag_names"][:])
        bag_to_frames: dict[str, list[int]] = defaultdict(list)
        for i, b in enumerate(d_bags):
            bag_to_frames[b].append(i)
        self.bag_frames = dict(bag_to_frames)
        self.indices = [i for i in indices if self.bag_frames.get(self.bag_names[i], [])]

    def bags(self):
        """Yield (bag_idx, frames_tensor, targets, mask, tag) per bag."""
        for real_idx in self.indices:
            frames = self.bag_frames[self.bag_names[real_idx]]
            batch = np.nan_to_num(self.height_maps[frames].astype(np.float32), nan=0.0)
            if self.input_rep == "height_geom":
                out = []
                for im in batch:
                    valid = (im > 0).astype(np.float32)
                    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
                    grad = np.sqrt(gx * gx + gy * gy)
                    out.append(np.stack([im, valid, grad], axis=0))
                batch = np.stack(out, axis=0)  # (F, 3, 96, 224)
            else:
                batch = batch.transpose(0, 2, 1)  # (F, 224, 96)

            targets = self.targets[real_idx]
            mask = ~np.isnan(targets)
            targets = np.where(mask, targets, 0.0).astype(np.float32)

            yield real_idx, torch.from_numpy(batch), torch.from_numpy(targets), torch.from_numpy(mask.astype(np.float32)), self.ids[real_idx]


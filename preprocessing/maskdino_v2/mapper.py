"""Custom Detectron2 DatasetMapper for 1-channel uint16 depth PNGs.

Drop-in replacement for `DetrDatasetMapper` (the `coco_instance_detr` mapper
in maskdino/data/dataset_mappers/detr_dataset_mapper.py). Differences:

  * Reads the image as a **single-channel uint16** PNG (depth in mm) and
    converts to float32 meters in [0, ~6]. Detectron2's standard
    `utils.read_image` would force 3 channels.
  * Keeps a single channel through the rest of the pipeline. The stem
    conv1 (with STEM_IN_CHANNELS=1) consumes (1, H, W).
  * Preserves the endpoint annotation handling from the original
    `DetrDatasetMapper`.

Install: copy this file to
    /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/maskdino/data/dataset_mappers/depth_1ch_dataset_mapper.py
and add the branch to MaskDINO/train_net.py — see
preprocessing/maskdino_v2/README.md.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, PolygonMasks
from pycocotools import mask as coco_mask


__all__ = ["Depth1ChannelDatasetMapper"]


def _read_depth_meters(path: str, scale: float = 1000.0) -> np.ndarray:
    """Load a uint16 grayscale depth PNG → float32 meters, shape (H, W, 1)."""
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        # Defensive: if a 3-channel PNG sneaks in, take the first channel.
        arr = arr[..., 0]
    if arr.dtype == np.uint16:
        depth = arr.astype(np.float32) / float(scale)
    elif arr.dtype == np.uint8:
        # Legacy 8-bit normalized export. Caller should re-export, but don't crash.
        depth = arr.astype(np.float32) / 255.0
    else:
        depth = arr.astype(np.float32)
    return depth[..., None]  # (H, W, 1) so detectron2 transforms see a "channel" dim


def _convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        m = torch.as_tensor(mask, dtype=torch.uint8).any(dim=2)
        masks.append(m)
    if masks:
        return torch.stack(masks, dim=0)
    return torch.zeros((0, height, width), dtype=torch.uint8)


def _get_endpoints(annotation):
    for key in ("endpoints", "heading_endpoints", "oriented_endpoints"):
        v = annotation.get(key)
        if v is not None:
            return v
    return None


def _transform_annotation_with_endpoints(annotation, transforms, image_shape):
    endpoints = _get_endpoints(annotation)
    annotation = utils.transform_instance_annotations(annotation, transforms, image_shape)
    if endpoints is not None:
        pts = np.asarray(endpoints, dtype=np.float32).reshape(-1, 2)
        pts = transforms.apply_coords(pts)
        annotation["endpoints"] = pts.reshape(-1).tolist()
    return annotation


def _build_endpoint_tensors(annos):
    n = len(annos)
    eps = torch.zeros((n, 4), dtype=torch.float32)
    valid = torch.zeros((n,), dtype=torch.bool)
    for i, a in enumerate(annos):
        v = a.get("endpoints")
        if v is None:
            continue
        pts = np.asarray(v, dtype=np.float32).reshape(-1)
        if pts.size != 4:
            raise ValueError(f"Expected 4 endpoint values, got {pts.size}")
        eps[i] = torch.from_numpy(pts)
        valid[i] = True
    return eps, valid


def _build_transform_gen(cfg, is_train):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"

    tfms = []
    if is_train:
        # Horizontal flip: head↔tail. OK for mask prediction (the upper_body
        # is the dorsal cap, valid in either orientation).
        tfms.append(T.RandomFlip())
        # Vertical flip: left↔right body side. PigFormer found this the
        # single most beneficial aug; should also help mask training.
        tfms.append(T.RandomFlip(horizontal=False, vertical=True))
        # Pigs appear at varied yaw within the depth FOV — rotation
        # augmentation makes the segmenter pose-invariant. Border pixels
        # introduced by rotation become depth=0 (treated as invalid).
        tfms.append(T.RandomRotation(
            angle=[-30.0, 30.0], expand=False, sample_style="range",
        ))
    tfms.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    return tfms


class Depth1ChannelDatasetMapper:
    """1-channel uint16 PNG variant of DetrDatasetMapper."""

    def __init__(self, cfg, is_train: bool = True):
        self.is_train = is_train
        self.tfm_gens = _build_transform_gen(cfg, is_train)
        self.crop_gen = None  # crop disabled for this dataset
        self.depth_scale = float(getattr(cfg.INPUT, "DEPTH_SCALE", 1000.0))
        # Random additive depth offset (meters) — simulates different camera
        # mounting heights so the model isn't tied to one absolute depth band.
        # Only applied at train time; invalid (=0) pixels stay 0.
        self.depth_offset_max = float(getattr(cfg.INPUT, "DEPTH_OFFSET_MAX_M", 0.3))
        self.mask_on = True
        logging.getLogger(__name__).info(
            f"Depth1ChannelDatasetMapper: tfms={self.tfm_gens}  scale={self.depth_scale}  "
            f"depth_offset_max={self.depth_offset_max:.2f} m"
        )

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = _read_depth_meters(dataset_dict["file_name"], self.depth_scale)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # Some Detectron2 transforms (e.g. RandomRotation) collapse (H,W,1) → (H,W).
        if image.ndim == 2:
            image = image[..., None]
        image_shape = image.shape[:2]  # h, w

        # Depth-offset augmentation: shift valid pixels by a random scalar (m).
        # Done after geometric transforms so the offset isn't reshuffled by them.
        if self.is_train and self.depth_offset_max > 0:
            offset = float(np.random.uniform(-self.depth_offset_max, self.depth_offset_max))
            valid = image[..., 0] > 0
            shifted = image[..., 0] + offset
            # Clamp to non-negative so we don't create fake "invalid" zero pixels by going below 0.
            shifted = np.clip(shifted, 1e-6, None)
            image[..., 0] = np.where(valid, shifted, 0.0)

        # (H, W, 1) → (1, H, W); float32 in meters; mean/std normalization
        # is applied by MaskDINO's forward via PIXEL_MEAN/STD.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1).astype(np.float32))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            annos = [
                _transform_annotation_with_endpoints(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            gt_eps, gt_valid = _build_endpoint_tensors(annos)
            instances.gt_endpoints = gt_eps
            instances.gt_endpoint_valid = gt_valid
            instances = utils.filter_empty_instances(instances)
            h, w = instances.image_size
            if hasattr(instances, "gt_masks"):
                gm = instances.gt_masks
                if isinstance(gm, PolygonMasks):
                    gm = _convert_coco_poly_to_mask(gm.polygons, h, w)
                elif isinstance(gm, BitMasks):
                    gm = gm.tensor
                instances.gt_masks = gm
            dataset_dict["instances"] = instances
        return dataset_dict

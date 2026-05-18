"""Register the v2 (1-channel uint16) pig depth dataset in Detectron2.

Mirrors MaskDINO/maskdino/data/datasets/register_pig_depth.py but points
at the v2 export root (uint16 grayscale PNGs).

Install: copy this file to
    /mnt/gs21/scratch/basharmk/data/unl/MaskDINO/maskdino/data/datasets/register_pig_depth_v2.py
and import it from MaskDINO/maskdino/data/datasets/__init__.py.
"""
from __future__ import annotations

import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


_PREDEFINED_SPLITS = {
    "pig_depth_v2_train": ("images/train", "annotations/instances_train.json"),
    "pig_depth_v2_val":   ("images/val",   "annotations/instances_val.json"),
    "pig_depth_v2_test":  ("images/test",  "annotations/instances_test.json"),
}

_EXTRA_ANNOTATION_KEYS = ["endpoints", "heading_endpoint", "obb_corners"]


def _get_meta(json_path: str) -> dict:
    payload = json.loads(open(json_path, "r", encoding="utf-8").read())
    categories = sorted(payload.get("categories", []), key=lambda c: int(c["id"]))
    thing_dataset_id_to_contiguous_id = {int(c["id"]): i for i, c in enumerate(categories)}
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": [str(c["name"]) for c in categories],
    }


def register_all_pig_depth_v2(root: str) -> None:
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        image_dir = os.path.join(root, image_root)
        json_path = os.path.join(root, json_file)
        if not (os.path.isdir(image_dir) and os.path.isfile(json_path)):
            continue
        DatasetCatalog.register(
            key,
            lambda json_path=json_path, image_dir=image_dir, dataset_name=key: load_coco_json(
                json_path, image_dir, dataset_name,
                extra_annotation_keys=_EXTRA_ANNOTATION_KEYS,
            ),
        )
        MetadataCatalog.get(key).set(
            json_file=json_path, image_root=image_dir, evaluator_type="coco",
            **_get_meta(json_path),
        )


_root = os.getenv(
    "PIG_DEPTH_V2_DATASET_ROOT",
    os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "pig_depth_combined_v2"),
)
register_all_pig_depth_v2(_root)

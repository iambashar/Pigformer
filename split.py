"""
Identity-level train/val/test split.

Pigs (bag-level identities) are partitioned so no animal appears in more than
one split. The test set is carved off first, then the remainder is divided
into ``n_folds`` CV folds. The paper reports numbers for fold 0.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else v for v in values]


def create_split(labels_h5: str, test_fraction: float = 0.2, seed: int = 42, n_folds: int = 4):
    """Build an identity-level split dict."""
    with h5py.File(labels_h5, "r") as handle:
        id_key = "unique_ids" if "unique_ids" in handle else "ear_tags"
        ids = _decode(handle[id_key][:])

    tag_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, tag in enumerate(ids):
        tag_to_indices[str(tag)].append(idx)

    unique_tags = sorted(tag_to_indices.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_tags)

    n_test_tags = max(1, int(len(unique_tags) * test_fraction))
    test_tags = set(unique_tags[:n_test_tags])
    train_pool_tags = unique_tags[n_test_tags:]

    test_indices = sorted([i for tag in test_tags for i in tag_to_indices[tag]])

    fold_tags: list[list[str]] = [[] for _ in range(n_folds)]
    for idx, tag in enumerate(train_pool_tags):
        fold_tags[idx % n_folds].append(tag)

    folds = []
    for fold_idx in range(n_folds):
        val_tags = set(fold_tags[fold_idx])
        train_tags = {t for j, group in enumerate(fold_tags) if j != fold_idx for t in group}
        train_indices = sorted([i for tag in train_tags for i in tag_to_indices[tag]])
        val_indices = sorted([i for tag in val_tags for i in tag_to_indices[tag]])
        folds.append({"train": train_indices, "val": val_indices})

    return {
        "seed": seed,
        "test_fraction": test_fraction,
        "num_folds": n_folds,
        "test_indices": test_indices,
        "test_tags": sorted(test_tags),
        "folds": folds,
    }


def load_or_create_split(labels_h5: str, split_json: str, seed: int = 42) -> dict:
    path = Path(split_json)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    split = create_split(labels_h5, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))
    return split

# `data/`

Three artifacts go here; all are gitignored.

| File | Shape / contents | Produced by |
|---|---|---|
| `dataset.h5` | `height_maps (N, 96, 224) float32`, `bag_names (N,)`, `source_names (N,)` | `preprocessing/build_height_dataset.py` |
| `label.h5` | `bag_names (M,)`, `unique_ids (M,)`, `fat_rib12 (M,)`, `loin_rib12 (M,)` | `preprocessing/parse_labels.py` |
| `split.json` | identity-level CV folds + test set | `split.py` |

Run the preprocessing pipeline (see `preprocessing/README.md`) or request a
prepared release archive from the authors.

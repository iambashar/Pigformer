# `weights/`

Trained checkpoints go here; all are gitignored.

| Checkpoint | Description |
|---|---|
| `pigformer_fold0.pt` | Fold-0 PigFormer (Stage 2). Reproduces the paper's 3.91 mm single-fold test MAE via `evaluate.py`. |

Per-fold ensemble checkpoints can be saved under `results/four_fold_*/foldK/best.pt`
and combined with `evaluate_ensemble.py` for the 3.87 mm headline number.

Request the released `pigformer_fold0.pt` checkpoint from the authors, or
retrain via `train.py` (see `README.md` for the protocol).

[日本語版 (Japanese)](README_JA.md) | **English**

# University of Tokyo Deep Learning Course Competition

## Competition Results

- **Final Rank**: **2nd** / 1,263 participants
- **LB Score**: **0.93082**

## Overview

Sentiment analysis on IMDb movie reviews using RNN

## Rules

- Keep every script consolidated into a single file.
- Do not use any training data other than `x_train` and `t_train`.

## Approach

- Data preparation
  - Load `x_train.npy`, `t_train.npy`, and `x_test.npy` from `data/input/`.
  - Split 10% of the training set into validation data with `train_test_split` (`random_state=42`).
  - `collate_batch` applies `text_transform`: truncate long reviews symmetrically, append EOS (id=2), and pad with id=3 after converting to tensors.

- Vocabulary and batching
  - Determine vocabulary size from the union of train/test tokens and pass it to the embedding layer.
  - Use `DataLoader` with shuffling only for training and keep deterministic ordering for validation/test.

- Model (`SequenceTaggingNet` in `src/lecture07_homework.py`)
  - Embedding table with padding id=3 followed by Gaussian noise injection (std=0.2) and Spatial Dropout.
  - Two-layer bidirectional GRU (hidden 256, dropout 0.3) initialized orthogonally for stability.
  - LayerNorm over the concatenated bidirectional states plus masked average pooling and masked max pooling.
  - Concatenate pooled features (dim 512) and project through a final linear head to produce logits.

- Training regimen
  - Train for 18 epochs per seed using AdamW (lr `1.5e-3`, weight_decay `1e-4`) with `OneCycleLR` (pct_start 0.2, cosine anneal).
  - Apply label smoothing (0.05) inside `BCEWithLogitsLoss`, gradient clip at 1.0, and log macro F1 each epoch.
  - Track the best validation macro F1 checkpoint for each run and restore it before inference.

- Ensembling and inference
  - Repeat the full training loop for four seeds `[42, 2025, 777, 1234]`.
  - For each best checkpoint, collect sigmoid probabilities on the test loader, average across seeds, and threshold at 0.5.
  - Save predictions to `data/output/submission_seed_ensemble.csv` with the required Kaggle-style format.

## Tech Stack

- Python 3.9+
- PyTorch (core tensor ops, GRU, optimizers, schedulers)
- NumPy for array wrangling and ensembling
- scikit-learn (`train_test_split`, `f1_score`)
- pandas for submission exports

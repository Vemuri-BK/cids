"""Add out-of-fold columns to configs/splits.csv: fold3_0, fold3_1, fold3_2.

The 1080 training patients are split into 3 folds, stratified by collection (seed 42). In column fold3_k:
  train   -> training patients NOT in fold k (the fold model trains on them, ~720)
  heldout -> training patients in fold k (~360): the fold model never sees them and predicts their maps
  val     -> unchanged (model selection, 120)
  test    -> unchanged (306; not used by the fold runs)
Every training patient is held out exactly once, so each one gets an out-of-fold SegResNet map.
Deterministic: running it again gives the same columns.
  python scripts/make_folds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_FOLDS, SEED = 3, 42


def fold_columns(sp: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    sp = sp.copy()
    fold = pd.Series(-1, index=sp.index)
    rng = np.random.default_rng(seed)
    for _, idx in sp[sp.split == "train"].groupby("collection").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % n_folds
    for k in range(n_folds):
        sp[f"fold{n_folds}_{k}"] = np.where(sp.split != "train", sp.split, np.where(fold == k, "heldout", "train"))
    return sp


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "splits.csv"
    sp = pd.read_csv(path)
    sp = fold_columns(sp.drop(columns=[c for c in sp.columns if c.startswith(f"fold{N_FOLDS}_")]))
    sp.to_csv(path, index=False)
    for k in range(N_FOLDS):
        c = f"fold{N_FOLDS}_{k}"
        print(c, sp[c].value_counts().to_dict(),
              sp[sp[c] == "heldout"].collection.value_counts().to_dict())


if __name__ == "__main__":
    main()

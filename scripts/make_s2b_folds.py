"""Split columns for 5-fold S2b retraining (out-of-fold SAM decoders for S3v).

For outer fold f (the S3k outer folds of the 1,080 training patients) column s2b_f<f> keeps every patient's
original split, except the training patients of fold f, which become 'holdout' (never seen by that decoder).
val stays 'val' (checkpoint selection, as the original S2b), test stays 'test' (never evaluated: run S2a/S2b
with --skip_test).

  python scripts/make_s2b_folds.py --folds ~/CIDS/outputs/s3k_radiomics/folds.csv \\
      --out ~/CIDS/outputs/s2b_folds/splits_s2b_folds.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make(sp: pd.DataFrame, folds: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    tr = set(sp.cache_id[sp[split_col] == "train"])
    f = dict(zip(folds.cache_id, folds.outer_fold))
    if set(f) != tr:
        raise SystemExit(f"folds.csv patients differ from the training split ({len(set(f) ^ tr)} differ)")
    out = sp.copy()
    for k in sorted(set(f.values())):
        hold = {c for c, v in f.items() if v == k}
        out[f"s2b_f{k}"] = [("holdout" if c in hold else s) for c, s in zip(sp.cache_id, sp[split_col])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--folds", required=True, help="S3k folds.csv (cache_id, outer_fold)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = make(pd.read_csv(a.splits), pd.read_csv(Path(a.folds).expanduser()))
    out = Path(a.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    for c in [c for c in df.columns if c.startswith("s2b_f")]:
        print(c, df[c].value_counts().to_dict())
    print("->", out)


if __name__ == "__main__":
    main()

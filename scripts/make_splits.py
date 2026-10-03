"""Create the fixed CIDS split used by every experiment.

official MAMA-MIA train_split (1200) -> train 1080 + val 120 (10% per collection, seed 42)
official MAMA-MIA test_split  (306)  -> test 306 (touched only for final evaluation)

Also adds leave-one-collection-out (LOCO) columns: for each collection C,
loco_<C> = 'test' for patients of C (both official splits), otherwise 'train'/'val'
(10% of the remaining collections, seed 42).

Usage:  python scripts/make_splits.py --splits data/metadata/train_test_splits.csv --out configs/splits.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

COLLECTIONS = ["DUKE", "ISPY1", "ISPY2", "NACT"]


def _val_ids(ids, frac, seed, tag):
    out = []
    by_coll = {}
    for pid in sorted(ids):
        by_coll.setdefault(pid.split("_")[0], []).append(pid)
    for coll, lst in sorted(by_coll.items()):
        rng = random.Random(f"{seed}::{tag}::{coll}")
        lst = lst[:]
        rng.shuffle(lst)
        out += lst[: max(1, round(frac * len(lst)))]
    return set(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="official train_test_splits.csv")
    ap.add_argument("--out", default="configs/splits.csv")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    s = pd.read_csv(a.splits)
    train = [str(x).strip().upper() for x in s["train_split"].dropna()]
    test = [str(x).strip().upper() for x in s["test_split"].dropna()]
    assert len(set(train) & set(test)) == 0
    val = _val_ids(train, a.val_frac, a.seed, "main")

    rows = [dict(patient_id=p, official="train", split="val" if p in val else "train") for p in train]
    rows += [dict(patient_id=p, official="test", split="test") for p in test]
    df = pd.DataFrame(rows)
    df["cache_id"] = df.patient_id.str.lower()
    df["collection"] = df.patient_id.str.split("_").str[0]

    for c in COLLECTIONS:
        rest = df[df.collection != c].patient_id.tolist()
        v = _val_ids(rest, a.val_frac, a.seed, f"loco_{c}")
        df[f"loco_{c}"] = [("test" if col == c else ("val" if p in v else "train"))
                           for p, col in zip(df.patient_id, df.collection)]

    df = df.sort_values(["split", "patient_id"]).reset_index(drop=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(pd.crosstab(df.collection, df.split, margins=True))
    for c in COLLECTIONS:
        print(f"loco_{c}:", df[f"loco_{c}"].value_counts().to_dict())
    print("wrote", out)


if __name__ == "__main__":
    main()

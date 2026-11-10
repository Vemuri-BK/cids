"""S3y: specialise SAM for its backup role in the one-click review (out-of-fold).

Why: S3v (R4) showed SAM is the better tool when the click lands on tumour SegResNet did not segment well;
S3w showed SegResNet-derived prompts do not help there (SAM's value is its independence). So SAM itself is
fine-tuned on the cases it is used for: one click on tumours where SegResNet's blob is poor.

Steps
  --step splits   split columns for s2a_sam_finetune.py (--split_col s3y_f<k> / s3y_full):
                    train  = HARD training patients of the other folds + the same number of random others
                             (HARD = nearest-blob Dice < 0.5 or B0_rad Dice < 0.5 at the S3v deep click;
                              training labels only) -- 'full' uses all 1,080 for the val decoder
                    holdout = every other training patient; val / test unchanged (test never evaluated)
  --step analyse  after re-running S3v with the specialised decoders (out-of-fold for train):
                    R6a (PRIMARY) on the FIXED set of train OOF patients the current S3v-OOF agent sends to SAM
                         (choices_train_oof_deep.csv, tool2 == S): mean Dice(new SAM) - Dice(current SAM),
                         patient-bootstrap 95% CI lower bound > 0
                    R6b  the S3v policy on the new outcomes: R4 passes and its gain on chosen cases >= the
                         current run's (+0.204)
Training recipe (frozen): init = the fold's S2b out-of-fold decoder (full: sam_s2b_best.pt), clicks only
(--p_box 0 --max_clicks 1), 30 epochs, lr 2e-4, warmup 1, val every 5, best.pt by val Dice@1 click
(--select click1; the starting decoder is kept unless an epoch beats it).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
RULE = dict(
    hard="training patients with nearest-blob Dice < 0.5 or B0_rad Dice < 0.5 at the S3v deep click",
    train_set="HARD patients of the other folds + the same number of random non-hard ones (seed 0)",
    recipe=dict(init="fold S2b OOF decoder (full: sam_s2b_best.pt)", p_box=0, max_clicks=1, epochs=30, lr=2e-4,
                warmup=1, select="click1"),
    R6a="fixed set = train OOF patients the current S3v-OOF agent sends to SAM; mean Dice(new) - Dice(current) "
        "bootstrap CI lower bound > 0",
    R6b="S3v policy on the new outcomes: R4 passes and gain_on_chosen >= current (+0.204)",
    bootstrap=dict(n=5000, seed=0))


def hard_flags(outcomes: pd.DataFrame) -> pd.Series:
    d = outcomes[(outcomes.split == "train_oof") & (outcomes.click == "deep")].set_index("cache_id")
    return (d.Cn_dice < 0.5) | (d.B_dice < 0.5)


def make_splits(sp: pd.DataFrame, folds: pd.DataFrame, hard: pd.Series) -> pd.DataFrame:
    f = dict(zip(folds.cache_id, folds.outer_fold))
    tr = sorted(sp.cache_id[sp.split == "train"])
    if set(f) != set(tr) or set(hard.index) != set(tr):
        raise SystemExit("folds / S3v outcomes do not cover exactly the 1,080 training patients")
    out = sp.copy()
    rng = np.random.default_rng(SEED)
    groups = {f"s3y_f{k}": [c for c in tr if f[c] != k] for k in sorted(set(f.values()))}
    groups["s3y_full"] = tr
    for col, pool in groups.items():
        h = [c for c in pool if hard[c]]
        e = [c for c in pool if not hard[c]]
        pick = set(h) | set(rng.choice(e, size=min(len(h), len(e)), replace=False).tolist())
        out[col] = [("train" if c in pick else "holdout") if s == "train" else s
                    for c, s in zip(sp.cache_id, sp.split)]
    return out


def step_splits(a):
    sp = pd.read_csv(a.splits)
    oc = pd.read_csv(Path(a.s3v_out).expanduser() / "outcomes.csv")
    hard = hard_flags(oc)
    df = make_splits(sp, pd.read_csv(Path(a.folds).expanduser()), hard)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before training
    df.to_csv(out / "splits_s3y.csv", index=False)
    print(f"hard training patients: {int(hard.sum())} of {len(hard)}")
    for c in [c for c in df.columns if c.startswith("s3y_")]:
        print(c, df[c].value_counts().to_dict())
    print("->", out / "splits_s3y.csv")


def boot(x, n, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    v = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def step_analyse(a):
    cur, new = Path(a.s3v_out).expanduser(), Path(a.new_s3v_out).expanduser()
    ch = pd.read_csv(cur / "choices_train_oof_deep.csv")
    fixed = ch[ch.tool2 == "S"].cache_id
    oc_cur, oc_new = pd.read_csv(cur / "outcomes.csv"), pd.read_csv(new / "outcomes.csv")
    if not oc_new[oc_new.split == "train_oof"].sam_decoder.astype(str).str.startswith("fold").all():
        raise SystemExit("the new S3v run is not out-of-fold on train (--sam_fold_ckpt missing?)")
    key = ["split", "cache_id", "click"]
    m = oc_cur[key + ["S_dice", "Cn_dice"]].merge(oc_new[key + ["S_dice"]], on=key, suffixes=("_cur", "_new"),
                                                  validate="one_to_one")
    n, seed = RULE["bootstrap"]["n"], RULE["bootstrap"]["seed"]
    rows = []
    for split, pop, sel in (("train_oof", "fixed_sent_to_SAM", m.cache_id.isin(fixed)),
                            ("train_oof", "all", m.split == "train_oof"), ("val", "all", m.split == "val")):
        g = m[(m.split == split) & (m.click == "deep") & sel]
        d = g.S_dice_new - g.S_dice_cur
        lo, hi = boot(d, n, seed) if len(g) > 1 else (np.nan, np.nan)
        rows.append(dict(split=split, population=pop, n=len(g), dice_current=g.S_dice_cur.mean(),
                         dice_new=g.S_dice_new.mean(), nearest_blob=g.Cn_dice.mean(), dice_diff=d.mean(), ci95=[lo, hi],
                         better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum())))
    t = pd.DataFrame(rows)
    p = t.iloc[0]
    res = dict(R6a=dict(n=int(p.n), dice_current=p.dice_current, dice_new=p.dice_new, diff=p.dice_diff, ci95=p.ci95,
                        passes=bool(p.n > 1 and p.ci95[0] > 0)))
    rj = new / "rules.json"
    if rj.exists():
        r4 = json.loads(rj.read_text())["rules"]["train_oof"]["R4"]
        res["R6b"] = dict(R4_passes=bool(r4["passes"]), gain_on_chosen=r4["gain_on_chosen"], current_gain=0.204,
                          passes=bool(r4["passes"] and r4["gain_on_chosen"] >= 0.204))
    else:
        res["R6b"] = "run s3v --step policy on the new outcomes first"
    out = Path(a.out).expanduser()
    t.to_csv(out / "summary.csv", index=False)
    (out / "rules.json").write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    print("\n" + json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["splits", "analyse"], required=True)
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--s3v_out", required=True, help="current OUT-OF-FOLD S3v run (outputs/s3v_review_agent_oof)")
    ap.add_argument("--folds", default="", help="S3k folds.csv (splits)")
    ap.add_argument("--new_s3v_out", default="", help="S3v run with the specialised decoders (analyse)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.step == "splits":
        if not a.folds:
            raise SystemExit("--folds is required")
        step_splits(a)
    else:
        if not a.new_s3v_out:
            raise SystemExit("--new_s3v_out is required")
        step_analyse(a)


if __name__ == "__main__":
    main()

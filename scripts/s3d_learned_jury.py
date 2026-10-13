"""S3d: learned jury -- a small classifier decides which SegResNet candidates to keep (replaces the hand-made
S3b rule segjury_0.7, test Dice 0.699; keeping exactly the right candidates would give ~0.741).

Same fully automatic pipeline as S3a/S3b (SegResNet-100 -> candidates prob > 0.5, >= 0.1 ml, top 5 -> SAM-S2b
with one crop + box per candidate). For every candidate, features that exist at test time (no ground truth):
  log size, size relative to the largest, rank, number of candidates, SegResNet mean/max probability,
  agreement Dice(SAM, SegResNet), SAM hard volume, SAM SOFT volume (S2b ranks true > false with AUC ~0.82),
  ratio SAM / SegResNet volume.
Label (training only): the candidate touches the tumour.

  1. train (1080): features + labels  -> fit logistic regression and gradient boosting
  2. val   (120) : choose the model and the probability threshold t by val Dice
  3. test  (306) : apply once; rule = keep the SegResNet mask of candidates with P(tumour) >= t
                   (none -> keep the most likely one, every case has a tumour)
Dice for any candidate selection is exact from per-candidate tables (components are disjoint:
Dice = 2 sum(inter_k) / (sum(size_k) + |GT|)), so no masks are stored. Test also gets HD95 / NSD for
learned, segjury_0.7 and seg_rel20, a paired Wilcoxon test and per-collection Dice.

Feature tables are cached per split (<out>/<split>_features.csv): re-running skips finished splits.
Example
  python scripts/s3d_learned_jury.py --cache_root <cache> --seg_ckpt segresnet_img_e100_best.pt \\
      --sam_ckpt sam_s2b_best.pt --out /kaggle/working/s3d_jury
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3a_seg2sam as s3  # noqa: E402
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.data.volumes import VolumeDataset, unpad  # noqa: E402
from cids.eval.metrics import all_metrics  # noqa: E402
from cids.models.sam_med3d import download_turbo  # noqa: E402

FEATURES = ["log_seg_ml", "rel_size", "is_largest", "rank", "n_cand", "seg_prob_mean", "seg_prob_max",
            "agree", "log_sam_ml", "log_sam_soft_ml", "log_sam_over_seg"]
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ---------------------------------------------------------------------------- features
def candidate_rows(cid, sprob, lab, cands, refined, gt):
    largest = max(s for _, s in cands)
    by_k = {d["k"]: d for d in refined}
    rows = []
    for rank, (k, size) in enumerate(cands):
        comp = lab == k
        d = by_k[k]
        seg_ml = size / 1000
        rows.append(dict(cache_id=cid, k=k, rank=rank, n_cand=len(cands), seg_ml=seg_ml,
                         size_vox=int(size), inter_vox=int((comp & gt).sum()), gt_vox=int(gt.sum()),
                         log_seg_ml=np.log(seg_ml), rel_size=size / largest, is_largest=int(rank == 0),
                         seg_prob_mean=float(sprob[comp].mean()), seg_prob_max=float(sprob[comp].max()),
                         agree=d["agree"], sam_ml=d["sam_ml"], sam_soft_ml=d["sam_soft_ml"],
                         log_sam_ml=np.log(d["sam_ml"] + 0.01), log_sam_soft_ml=np.log(d["sam_soft_ml"] + 0.01),
                         log_sam_over_seg=np.log((d["sam_soft_ml"] + 0.01) / (seg_ml + 0.01))))
    for r in rows:
        r["on_tumour"] = int(r["inter_vox"] > 0)
    return rows


def run_split(split, cases, seg, sam, box, amp, overlap, out, on_case=None):
    """Pipeline on every case; returns the candidate feature table (cached as CSV)."""
    f = out / f"{split}_features.csv"
    if f.exists() and on_case is None:
        print(f"{split}: using cached {f}")
        return pd.read_csv(f)
    rows, ds = [], VolumeDataset(cases, 96)
    for i in tqdm(range(len(ds)), desc=split, unit="case", file=sys.stdout, mininterval=20):
        item = ds[i]; cid = item["cid"]
        img, lbl = load_case(cases[i][1]); gt = lbl > 0
        sprob = unpad(s3.seg_prob(seg, item["image"], amp, overlap=overlap), item["pads"])
        lab, cands = s3.components(sprob > 0.5, top_k=5)
        refined = s3.sam_refine(sam, box, img, lab, cands, amp) if cands else []
        r = candidate_rows(cid, sprob, lab, cands, refined, gt) if cands else []
        if not r:                                    # no candidate: keep the case for Dice (= 0)
            r = [dict(cache_id=cid, k=-1, size_vox=0, inter_vox=0, gt_vox=int(gt.sum()), on_tumour=0)]
        rows.extend(r)
        if on_case is not None:
            on_case(cid, lab, gt, r)
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False)
    return df


# ---------------------------------------------------------------------------- selection -> Dice
def select(df, keep_col):
    """keep_col: bool column per candidate. Empty selection in a case -> keep its best-scored candidate
    (column 'fallback_score'). Returns a per-case Dice Series."""
    out = {}
    for cid, g in df.groupby("cache_id", sort=False):
        real = g[g.k >= 0]
        kept = real[real[keep_col]]
        if len(kept) == 0 and len(real):
            kept = real.loc[[real["fallback_score"].idxmax()]]
        inter, size, gtv = kept.inter_vox.sum(), kept.size_vox.sum(), g.gt_vox.iloc[0]
        out[cid] = 1.0 if size + gtv == 0 else 2 * inter / (size + gtv)
    return pd.Series(out)


def rule_tables(df, proba=None):
    """Per-case Dice for the reference rules (+ learned if proba given) from a feature table."""
    df = df.copy()
    real = df.k >= 0
    res = {}
    df["keep"] = real & (df.rel_size >= 0.2)                       # seg_rel20 (S1 baseline)
    df["fallback_score"] = df.get("rel_size", 0)
    res["seg_rel20"] = select(df, "keep")
    df["keep"] = real & (df.agree >= 0.7)                          # segjury_0.7 (S3b)
    df["fallback_score"] = df.get("agree", 0)
    res["segjury_0.7"] = select(df, "keep")
    df["keep"] = real & (df.on_tumour == 1)                        # oracle: exactly the right candidates
    df["fallback_score"] = df.get("rel_size", 0)
    res["oracle"] = select(df, "keep")
    if proba is not None:
        df["p"] = proba; df["fallback_score"] = df["p"]
        for t in THRESHOLDS:
            df["keep"] = real & (df.p >= t)
            res[f"learned_t{t:.2f}"] = select(df, "keep")
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------- models
def fit_models(tr):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = tr[FEATURES].to_numpy(float), tr.on_tumour.to_numpy()
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(X, y)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                        class_weight="balanced", random_state=0).fit(X, y)
    return {"logreg": lr, "gboost": gb}


class CrossFitJury:
    """Jury for building council inputs. Training cases get P(tumour) from a model fitted WITHOUT them
    (5-fold by case), so their checker maps are as noisy as on new cases; other cases use the full model.
    Fitted from <out>/train_features.csv in seconds (same settings as the chosen 'gboost' jury)."""

    def __init__(self, train_features_csv, n_folds: int = 5, model: str = "gboost"):
        from sklearn.model_selection import GroupKFold
        tr = pd.read_csv(train_features_csv)
        tr = tr[tr.k >= 0].reset_index(drop=True)
        self.full = fit_models(tr)[model]
        self.fold_of, self.fold_models = {}, []
        for f, (fit_idx, held) in enumerate(GroupKFold(n_folds).split(tr, groups=tr.cache_id)):
            self.fold_models.append(fit_models(tr.iloc[fit_idx])[model])
            self.fold_of.update({c: f for c in tr.cache_id.iloc[held]})

    def predict(self, cid, feats: pd.DataFrame) -> np.ndarray:
        m = self.fold_models[self.fold_of[cid]] if cid in self.fold_of else self.full
        return m.predict_proba(feats[FEATURES].to_numpy(float))[:, 1]


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--sam_ckpt", required=True, help="S2b best.pt (the checker)")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", default="/kaggle/working/s3d_jury")
    ap.add_argument("--limit", type=int, default=0, help="cases per split (tests)")
    ap.add_argument("--train_overlap", type=float, default=0.5)
    a = ap.parse_args()
    amp = s3.DEV == "cuda"
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(a.cache_root)
    turbo = None if a.random_weights else (a.turbo_ckpt or download_turbo("/tmp/sam_ckpt"))
    seg, sam, box = s3.load_models(a.seg_ckpt, a.sam_ckpt, turbo)
    print(f"cache {root}\nfinder {a.seg_ckpt}\nchecker {a.sam_ckpt}\n{s3.DEV}")
    sp = pd.read_csv(a.splits)

    def items(split):
        r = sp[sp.split == split]
        lst = [(c, root / ("train" if o == "train" else "val") / f"{c}.npz") for c, o in zip(r.cache_id, r.official)]
        return [x for x in lst if x[1].exists()][:a.limit or None]

    coll = dict(zip(sp.cache_id, sp.collection))

    # 1. train: features + fit
    tr = run_split("train", items("train"), seg, sam, box, amp, a.train_overlap, out)
    trr = tr[tr.k >= 0]
    print(f"train: {len(trr)} candidates = {int(trr.on_tumour.sum())} true + {int((1 - trr.on_tumour).sum())} false")
    models = fit_models(trr)
    pickle.dump(models, open(out / "jury_models.pkl", "wb"))

    # 2. val: choose model + threshold
    va = run_split("val", items("val"), seg, sam, box, amp, 0.5, out)
    var = va.k >= 0
    summary = {"features": FEATURES, "auc": {}}
    best = None
    for name, m in models.items():
        p = np.zeros(len(va)); p[var.to_numpy()] = m.predict_proba(va.loc[var, FEATURES].to_numpy(float))[:, 1]
        summary["auc"][f"val_{name}"] = auc(p[var.to_numpy()], va.loc[var, "on_tumour"].to_numpy())
        tab = rule_tables(va, p).mean()
        lt = tab[[c for c in tab.index if c.startswith("learned_")]]
        t = float(lt.idxmax().split("_t")[1])
        print(f"val {name}: AUC {summary['auc'][f'val_{name}']:.3f} | best threshold {t:.2f} -> Dice {lt.max():.4f}")
        if best is None or lt.max() > best[2]:
            best = (name, t, float(lt.max()), tab)
    for f_ in ("agree", "sam_soft_ml", "seg_prob_mean"):          # single features, for the paper
        summary["auc"][f"val_{f_}"] = auc(va.loc[var, f_].to_numpy(), va.loc[var, "on_tumour"].to_numpy())
    name, t, vdice, vtab = best
    print(f"\nVALIDATION (120) mean Dice:\n{vtab[['seg_rel20', 'segjury_0.7', 'oracle']].round(4).to_string()}"
          f"\nlearned ({name}, t={t:.2f}): {vdice:.4f}  <- chosen")
    model = models[name]
    if name == "logreg":
        coef = model[-1].coef_[0]
        summary["logreg_coefficients_standardised"] = {f: round(float(c), 3) for f, c in zip(FEATURES, coef)}
        print("logistic regression weights (standardised features):", summary["logreg_coefficients_standardised"])

    # 3. test: apply once; full metrics for learned / segjury_0.7 / seg_rel20
    full_rows = []

    def on_test_case(cid, lab, gt, rows):
        real = [r for r in rows if r["k"] >= 0]
        if not real:
            for rule in ("learned", "segjury_0.7", "seg_rel20"):
                full_rows.append(dict(cache_id=cid, rule=rule, **all_metrics(np.zeros_like(gt), gt)))
            return
        X = pd.DataFrame(real)[FEATURES].to_numpy(float)
        p = model.predict_proba(X)[:, 1]
        ks = [r["k"] for r, pi in zip(real, p) if pi >= t] or [real[int(np.argmax(p))]["k"]]
        kj = [r["k"] for r in real if r["agree"] >= 0.7] or [max(real, key=lambda r: r["agree"])["k"]]
        k20 = [r["k"] for r in real if r["rel_size"] >= 0.2]
        for rule, keep in (("learned", ks), ("segjury_0.7", kj), ("seg_rel20", k20)):
            full_rows.append(dict(cache_id=cid, rule=rule, **all_metrics(np.isin(lab, keep), gt)))

    te = run_split("test", items("test"), seg, sam, box, amp, 0.5, out, on_case=on_test_case)
    ter = (te.k >= 0).to_numpy()
    p = np.zeros(len(te)); p[ter] = model.predict_proba(te.loc[ter, FEATURES].to_numpy(float))[:, 1]
    summary["auc"]["test_jury"] = auc(p[ter], te.loc[ter, "on_tumour"].to_numpy())
    summary["auc"]["test_sam_soft_ml"] = auc(te.loc[ter, "sam_soft_ml"].to_numpy(), te.loc[ter, "on_tumour"].to_numpy())
    ttab = rule_tables(te, p)
    fr = pd.DataFrame(full_rows)
    fr["collection"] = fr.cache_id.map(coll)
    fr.to_csv(out / "test_metrics.csv", index=False)
    agg = fr.groupby("rule").agg(dice=("dice", "mean"), dice_sd=("dice", "std"), hd95_median=("hd95", "median"),
                                 nsd=("nsd", "mean"), missed=("dice", lambda x: int((x < 0.1).sum())))
    agg.loc["oracle", "dice"] = ttab["oracle"].mean()
    from scipy.stats import wilcoxon
    piv = fr.pivot(index="cache_id", columns="rule", values="dice")
    pv = {r: float(wilcoxon(piv["learned"], piv[r]).pvalue) for r in ("segjury_0.7", "seg_rel20")}
    print(f"\nTEST (306), learned jury = {name}, threshold {t:.2f} (chosen on val):")
    print(agg.round(4).to_string())
    print("paired Wilcoxon p (learned vs ...):", {k: round(v, 4) for k, v in pv.items()})
    print("\nper collection Dice:\n" + fr.pivot_table(index="collection", columns="rule", values="dice").round(3).to_string())
    print("\nAUC (true vs false candidates):", {k: round(v, 3) for k, v in summary["auc"].items()})
    summary.update(chosen_model=name, threshold=t, val_dice=vdice, wilcoxon_p=pv,
                   test={r: {k: (round(float(v), 4) if pd.notna(v) else None) for k, v in row.items()}
                         for r, row in agg.iterrows()},
                   minutes=(time.time() - t0) / 60)
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

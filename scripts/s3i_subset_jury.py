"""S3i: conditional CPU pilot of patient-level subset selection (development data only).

Question
    Can patient-level subset scoring select better candidate combinations from imperfect evidence
    than independent candidate classification? Does SAM evidence add information, and is there enough
    patient-level complementarity between a finder-only and a SAM-assisted selector to justify a
    consultation gate?

Scope (frozen)
    * Candidate pool, upstream predictions and candidate masks are the existing S3f tables
      (SegResNet-100 + SAM-S2b). Those upstream models saw the training patients, so this is a
      CONDITIONAL decision-layer pilot, not a fully out-of-fold evaluation of SegResNet / SAM.
    * No new networks, prompts, memory features or radiomics.
    * Test files are never read. Validation is a reused development set, not independent confirmation.

Arms (all newly fitted with the same outer patient folds; complexity and thresholds chosen on inner
patient folds of each outer-training partition)
    B0   binary jury (HistGradientBoostingClassifier), 7 finder features, threshold + highest-score fallback
    B1   singleton-quality regressor, target 2*I_k/(V_k+G), 7 finder features, threshold + fallback
    U0   joint subset utility regressor, permutation-invariant summaries of the 7 finder features
    B11  binary jury, 7 finder + 4 SAM features
    U1   joint subset utility regressor, summaries of the same 11 features
    HIST historical saved scores (finder_only, jury11, jury13, jury13_rad_rel) at documented thresholds,
         reproduction only (their thresholds were chosen globally, so they are not matched comparators)
    ORACLE best nonempty subset (ground truth, NOT deployable)

Usage
    python scripts/s3i_subset_jury.py --train data/train_features.csv --val data/val_features.csv \
        --dev_scores data/dev_scores.csv --config configs/s3i_subset_jury.json --out results
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

EPS_TIE = 1e-12


# ----------------------------------------------------------------------------------------------
# data: features and targets are kept in separate structures
# ----------------------------------------------------------------------------------------------
def check_whitelist(cols, forbidden):
    bad = sorted(set(cols) & set(forbidden))
    if bad:
        raise ValueError(f"forbidden columns in a feature whitelist: {bad}")


def collection_of(cid: str) -> str:
    return cid.split("_")[0].upper()


def load_split(csv: str, feature_cols, forbidden, manifest: str | None = None):
    """Return a list of patients. Each patient has 'feats' (inference inputs only) and 'tgt' (targets).

    Candidates are ordered by size rank (deterministic, independent of the input row order)."""
    check_whitelist(feature_cols, forbidden)
    df = pd.read_csv(csv)
    need = set(feature_cols) | {"cache_id", "k", "rank", "size_vox", "inter_vox", "gt_vox"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv}: missing columns {sorted(missing)}")
    pats = []
    for cid, g in df.groupby("cache_id", sort=True):
        g = g.sort_values(["rank", "k"])
        if g.gt_vox.nunique() != 1:
            raise ValueError(f"{cid}: gt_vox not constant")
        pats.append(dict(
            cid=cid, coll=collection_of(cid), K=len(g), status="ok",
            k=g.k.to_numpy(),
            feats=g[list(feature_cols)].reset_index(drop=True),          # inference inputs
            tgt=dict(I=g.inter_vox.to_numpy(float), V=g.size_vox.to_numpy(float),
                     G=float(g.gt_vox.iloc[0])),                          # training / evaluation only
        ))
    n_expected = None
    if manifest and Path(manifest).exists():
        n_expected = json.loads(Path(manifest).read_text()).get("n_cases")
    if n_expected is not None and n_expected > len(pats):
        for j in range(n_expected - len(pats)):     # K = 0 patients stay in evaluation with Dice 0
            pats.append(dict(cid=f"__no_candidate_{j}", coll="UNKNOWN", K=0, status="no_candidate",
                             k=np.array([], int), feats=pd.DataFrame(columns=list(feature_cols)),
                             tgt=dict(I=np.zeros(0), V=np.zeros(0), G=1.0)))
    return pats


# ----------------------------------------------------------------------------------------------
# subsets and Dice from counts (components are disjoint)
# ----------------------------------------------------------------------------------------------
def enumerate_subsets(K: int):
    """All nonempty subsets as boolean masks, ordered by size then lexicographically."""
    out = []
    for r in range(1, K + 1):
        for S in itertools.combinations(range(K), r):
            m = np.zeros(K, bool)
            m[list(S)] = True
            out.append(m)
    return out


def dice_counts(tgt, mask) -> float:
    if mask.size == 0 or not mask.any():
        return 0.0
    return float(2.0 * tgt["I"][mask].sum() / (tgt["G"] + tgt["V"][mask].sum()))


def oracle_best(p):
    if p["K"] == 0:
        return 0.0, np.zeros(0, bool)
    best, bm = -1.0, None
    for m in enumerate_subsets(p["K"]):
        d = dice_counts(p["tgt"], m)
        if d > best + EPS_TIE:
            best, bm = d, m
    return best, bm


def _stats(x):
    if x.size == 0:
        return [np.nan, np.nan, np.nan]
    return [float(x.mean()), float(x.min()), float(x.max())]


def subset_features(feats: pd.DataFrame, masks, use_sam: bool):
    """Permutation-invariant summaries of one patient's candidates for each subset (rows = subsets)."""
    K = len(feats)
    vol = np.exp(feats["log_seg_ml"].to_numpy(float))         # ml; log_seg_ml = log(seg_ml)
    pm = feats["seg_prob_mean"].to_numpy(float)
    px = feats["seg_prob_max"].to_numpy(float)
    rk = feats["rank"].to_numpy(float)
    rs = feats["rel_size"].to_numpy(float)
    largest = int(np.argmax(vol))
    mostconf = int(np.argmax(pm))
    tot = vol.sum()
    vs = np.sort(vol)[::-1]
    patient = [K, np.log(tot), pm.max() - pm.min(), np.log(vs[0] / vs[1]) if K > 1 else np.nan]
    sam = None
    if use_sam:
        sam = {c: feats[c].to_numpy(float) for c in ["agree", "log_sam_ml", "log_sam_soft_ml", "log_sam_over_seg"]}
        mostagree = int(np.argmax(sam["agree"]))
    rows = []
    for m in masks:
        e = ~m
        r = [int(m.sum()), int(e.sum()), int(not e.any()),
             np.log(vol[m].sum()), np.log(vol[m].max()), np.log(vol[m].min()), vol[m].sum() / tot,
             np.log1p(vol[e].sum())]
        for c in (pm, px):
            r += _stats(c[m]) + [float(c[e].max()) if e.any() else np.nan,
                                 float((c[m] * vol[m]).sum() / vol[m].sum())]
        r += [int(m[largest]), int(m[mostconf]), rk[m].min(), rk[m].max(), rs[m].min(), rs[m].max()]
        r += patient
        if use_sam:
            for c in sam.values():
                r += _stats(c[m]) + [float(c[e].max()) if e.any() else np.nan]
            r += [int(m[mostagree]), float((sam["agree"][m] * vol[m]).sum() / vol[m].sum())]
        rows.append(r)
    return np.asarray(rows, float)


# ----------------------------------------------------------------------------------------------
# folds
# ----------------------------------------------------------------------------------------------
def assign_folds(cids, n_folds: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(cids))
    return {c: int(i % n_folds) for i, c in enumerate(order)}


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
def _hgb(kind, cfg, fixed, seed):
    cls = HistGradientBoostingClassifier if kind == "clf" else HistGradientBoostingRegressor
    return cls(**cfg, **fixed, random_state=seed)


def _grid(g):
    return np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 6)


def keep_threshold(scores, t):
    keep = scores >= t
    if not keep.any():
        keep = np.zeros_like(keep)
        keep[int(np.argmax(scores))] = True          # highest-score fallback (first max)
    return keep


class ThresholdArm:
    """Independent per-candidate model + retention threshold (B0, B1, B11)."""

    def __init__(self, name, kind, cols, C):
        self.name, self.kind, self.cols, self.C = name, kind, list(cols), C
        self.tgrid = _grid(C["prob_threshold_grid"] if kind == "clf" else C["quality_threshold_grid"])

    def _rows(self, pats):
        X, y = [], []
        for p in pats:
            if p["K"] == 0:
                continue
            X.append(p["feats"][self.cols].to_numpy(float))
            t = p["tgt"]
            y.append((t["I"] > 0).astype(int) if self.kind == "clf" else 2 * t["I"] / (t["V"] + t["G"]))
        return np.vstack(X), np.concatenate(y)

    def _fit(self, pats, cfg, seed):
        X, y = self._rows(pats)
        return _hgb(self.kind, cfg, self.C["hgb_fixed"], seed).fit(X, y)

    def scores(self, model, p):
        X = p["feats"][self.cols].to_numpy(float)
        return model.predict_proba(X)[:, 1] if self.kind == "clf" else model.predict(X)

    def tune(self, pats, inner_folds, seed):
        """Choose (config, threshold) by inner-fold patient-level mean Dice."""
        fold = assign_folds([p["cid"] for p in pats], inner_folds, seed)
        best = None
        for ci, cfg in enumerate(self.C["hgb_grid"]):
            sc = {}
            for f in range(inner_folds):
                tr = [p for p in pats if fold[p["cid"]] != f]
                m = self._fit(tr, cfg, seed)
                for p in pats:
                    if fold[p["cid"]] == f and p["K"] > 0:
                        sc[p["cid"]] = self.scores(m, p)
            for t in self.tgrid:
                d = np.mean([dice_counts(p["tgt"], keep_threshold(sc[p["cid"]], t)) if p["K"] > 0 else 0.0
                             for p in pats])
                if best is None or d > best[0] + EPS_TIE:
                    best = (d, ci, float(t))
        return dict(config_index=best[1], config=self.C["hgb_grid"][best[1]], threshold=best[2],
                    inner_dice=float(best[0]))

    def fit_locked(self, pats, inner_folds, seed):
        sel = self.tune(pats, inner_folds, seed)
        self.model = self._fit(pats, sel["config"], seed)
        self.sel = sel
        return sel

    def decide(self, p, ref=None):
        if p["K"] == 0:
            return np.zeros(0, bool), "no_candidate"
        if p["K"] == 1:
            return np.ones(1, bool), "single_candidate"
        return keep_threshold(self.scores(self.model, p), self.sel["threshold"]), "model"


class SubsetArm:
    """Joint subset utility regressor (U0, U1): predicts union Dice of each subset, picks the argmax."""

    def __init__(self, name, use_sam, C):
        self.name, self.use_sam, self.C = name, use_sam, C

    def _sub(self, p):
        key = "sub_sam" if self.use_sam else "sub"
        if key not in p:
            masks = enumerate_subsets(p["K"])
            p[key] = (masks, subset_features(p["feats"], masks, self.use_sam))
        return p[key]

    def _fit(self, pats, cfg, seed):
        X, y, w = [], [], []
        for p in pats:
            if p["K"] == 0:
                continue
            masks, F = self._sub(p)
            X.append(F)
            y.append([dice_counts(p["tgt"], m) for m in masks])
            w.append(np.full(len(masks), 1.0 / len(masks)))      # each patient's subsets sum to 1
        X, y, w = np.vstack(X), np.concatenate(y), np.concatenate(w)
        return _hgb("reg", cfg, self.C["hgb_fixed"], seed).fit(X, y, sample_weight=w)

    def choose(self, model, p, ref=None):
        masks, F = self._sub(p)
        s = model.predict(F)
        top = np.flatnonzero(s >= s.max() - EPS_TIE)
        if ref is not None:
            for i in top:
                if np.array_equal(masks[i], ref):
                    return masks[i], s
        return masks[int(top[0])], s                               # fixed rule: enumeration order

    def tune(self, pats, inner_folds, seed):
        fold = assign_folds([p["cid"] for p in pats], inner_folds, seed)
        best = None
        for ci, cfg in enumerate(self.C["hgb_grid"]):
            ds = []
            for f in range(inner_folds):
                tr = [p for p in pats if fold[p["cid"]] != f]
                m = self._fit(tr, cfg, seed)
                for p in pats:
                    if fold[p["cid"]] != f:
                        continue
                    if p["K"] == 0:
                        ds.append(0.0)
                    elif p["K"] == 1:
                        ds.append(dice_counts(p["tgt"], np.ones(1, bool)))
                    else:
                        ds.append(dice_counts(p["tgt"], self.choose(m, p)[0]))
            d = float(np.mean(ds))
            if best is None or d > best[0] + EPS_TIE:
                best = (d, ci)
        return dict(config_index=best[1], config=self.C["hgb_grid"][best[1]], inner_dice=best[0])

    def fit_locked(self, pats, inner_folds, seed):
        sel = self.tune(pats, inner_folds, seed)
        self.model = self._fit(pats, sel["config"], seed)
        self.sel = sel
        return sel

    def decide(self, p, ref=None):
        if p["K"] == 0:
            return np.zeros(0, bool), "no_candidate"
        if p["K"] == 1:
            return np.ones(1, bool), "single_candidate"
        return self.choose(self.model, p, ref)[0], "model"


# ----------------------------------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------------------------------
def harmful_removal(p, mask):
    """Harmful removal: the decision excluded at least one tumour-touching candidate AND adding the
    excluded touching candidates back would have increased the patient's Dice. Returns (flag, loss)."""
    if p["K"] == 0:
        return False, 0.0
    touch_excl = (~mask) & (p["tgt"]["I"] > 0)
    if not touch_excl.any():
        return False, 0.0
    d0, d1 = dice_counts(p["tgt"], mask), dice_counts(p["tgt"], mask | touch_excl)
    return (d1 > d0 + 1e-9), max(0.0, d1 - d0)


def paired_boot(a, b, n, seed):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))


def mask_str(p, m):
    return "+".join(str(int(x)) for x in sorted(p["k"][m])) if m.size else ""


def run_arm_outer(arm, pats, folds, C, ref_decisions=None):
    """Outer patient-fold OOF decisions; complexity/threshold from inner folds of each training part."""
    rows, sels = [], []
    nf = C["outer_folds"]
    for f in range(nf):
        tr = [p for p in pats if folds[p["cid"]] != f]
        te = [p for p in pats if folds[p["cid"]] == f]
        assert not ({p["cid"] for p in tr} & {p["cid"] for p in te})
        sel = arm.fit_locked(tr, C["inner_folds"], C["seed"] + 101 * (f + 1))
        sels.append(dict(arm=arm.name, fold=f, **{k: v for k, v in sel.items()}))
        for p in te:
            ref = ref_decisions.get(p["cid"]) if ref_decisions else None
            m, st = arm.decide(p, ref)
            rows.append((p["cid"], f, m, st))
    return rows, sels


def run_arm_val(arm, train, val, C, ref_decisions=None):
    sel = arm.fit_locked(train, C["inner_folds"], C["seed"])
    out = []
    for p in val:
        ref = ref_decisions.get(p["cid"]) if ref_decisions else None
        m, st = arm.decide(p, ref)
        out.append((p["cid"], -1, m, st))
    return out, dict(arm=arm.name, fold="final", **sel)


def historical(pats, scores_df, split, var, t):
    s = scores_df[(scores_df.variant == var) & (scores_df.split == split)].set_index(["cache_id", "k"]).p
    out = {}
    for p in pats:
        if p["K"] == 0:
            out[p["cid"]] = np.zeros(0, bool)
            continue
        sc = np.array([s[(p["cid"], int(k))] for k in p["k"]])
        keep = sc >= t
        if not keep.any():
            keep = sc == sc.max()
        out[p["cid"]] = keep
    return out


def summarise(name, pats, dec, oracle, C, group="all"):
    d = np.array([dice_counts(p["tgt"], dec[p["cid"]]) for p in pats])
    o = np.array([oracle[p["cid"]] for p in pats])
    hr = [harmful_removal(p, dec[p["cid"]]) for p in pats]
    r = dict(arm=name, group=group, n=len(pats), dice=d.mean(), dice_median=float(np.median(d)),
             regret=float((o - d).mean()), oracle=o.mean(),
             harmful_removals=int(sum(h for h, _ in hr)), harmful_loss_mean=float(np.mean([l for _, l in hr])))
    for t in C["fail_thresholds"]:
        r[f"dice_lt_{t}"] = int((d < t).sum())
    for coll in sorted({p["coll"] for p in pats}):
        idx = [i for i, p in enumerate(pats) if p["coll"] == coll]
        r[f"dice_{coll}"] = float(d[idx].mean())
    return r, d


def compare(a_name, b_name, da, db, C, group):
    m, lo, hi, p = paired_boot(da, db, C["bootstrap"]["n"], C["bootstrap"]["seed"])
    e = C["change_eps"]
    r = dict(comparison=f"{a_name} - {b_name}", group=group, mean_diff=m, ci95_lo=lo, ci95_hi=hi, p_boot=p,
             better=int((da - db > e).sum()), worse=int((db - da > e).sum()),
             changed=int((np.abs(da - db) > 1e-9).sum()))
    for t in C["fail_thresholds"]:
        r[f"new_fail_lt_{t}"] = int(((db >= t) & (da < t)).sum())
        r[f"rescued_lt_{t}"] = int(((db < t) & (da >= t)).sum())
    return r


def routing_analysis(pats, dec0, dec1, C, split):
    """U0 (no SAM) vs U1 (SAM) complementarity from held-out decisions. No gate is trained."""
    d0 = np.array([dice_counts(p["tgt"], dec0[p["cid"]]) for p in pats])
    d1 = np.array([dice_counts(p["tgt"], dec1[p["cid"]]) for p in pats])
    same = np.array([np.array_equal(dec0[p["cid"]], dec1[p["cid"]]) for p in pats])
    K = np.array([p["K"] for p in pats])
    gain = d1 - d0
    e = C["change_eps"]
    best = np.maximum(d0, d1)
    summ = dict(split=split, n=len(pats), identical_subsets_pct=100 * same.mean(),
                identical_subsets_pct_K2plus=100 * same[K >= 2].mean() if (K >= 2).any() else np.nan,
                helped=int((gain > e).sum()), harmed=int((gain < -e).sum()),
                unchanged=int((np.abs(gain) <= e).sum()),
                helped_any=int((gain > 1e-9).sum()), harmed_any=int((gain < -1e-9).sum()),
                mean_gain_when_helped=float(gain[gain > 1e-9].mean()) if (gain > 1e-9).any() else 0.0,
                mean_loss_when_harmed=float(gain[gain < -1e-9].mean()) if (gain < -1e-9).any() else 0.0,
                max_gain=float(gain.max()), max_loss=float(gain.min()),
                dice_U0=d0.mean(), dice_U1=d1.mean(), oracle_routing=best.mean(),
                oracle_gain_over_U0=best.mean() - d0.mean(), oracle_gain_over_U1=best.mean() - d1.mean())
    # budget curve: cost = K candidate encodings for a consulted patient with K >= 2 (K <= 1 bypass)
    cost = np.where(K >= 2, K, 0).astype(float)
    C_all = cost.sum()
    elig = np.flatnonzero(cost > 0)
    order = elig[np.argsort(-(gain[elig] / cost[elig]))]      # greedy oracle by gain per encoding
    rng = np.random.default_rng(C["seed"])
    rows = []
    for b in C["routing_budgets"]:
        cap = b * C_all
        used, consult = 0.0, np.zeros(len(pats), bool)
        for i in order:
            if gain[i] <= 0:
                break
            if used + cost[i] <= cap + 1e-9:
                consult[i], used = True, used + cost[i]
        orc = np.where(consult, d1, d0).mean()
        rnd = []
        for _ in range(C["random_routing_draws"]):
            used, cons = 0.0, np.zeros(len(pats), bool)
            for i in rng.permutation(elig):
                if used + cost[i] <= cap + 1e-9:
                    cons[i], used = True, used + cost[i]
            rnd.append(np.where(cons, d1, d0).mean())
        rows.append(dict(split=split, budget_frac=b, encodings_allowed=cap, oracle_routing_dice=orc,
                         random_routing_dice_mean=float(np.mean(rnd)),
                         random_routing_dice_p2_5=float(np.percentile(rnd, 2.5)),
                         random_routing_dice_p97_5=float(np.percentile(rnd, 97.5))))
    return summ, pd.DataFrame(rows), gain


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--dev_scores", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_manifest", default="")
    ap.add_argument("--val_manifest", default="")
    a = ap.parse_args()
    for f in (a.train, a.val):
        if "test" in Path(f).name.lower():
            raise SystemExit("refusing to read a test file in development mode")
    C = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = C["finder_features"] + C["sam_features"]
    tm = a.train_manifest or str(Path(a.train).with_suffix(".manifest.json"))
    vm = a.val_manifest or str(Path(a.val).with_suffix(".manifest.json"))
    train = load_split(a.train, cols, C["forbidden_features"], tm)
    val = load_split(a.val, cols, C["forbidden_features"], vm)
    scores = pd.read_csv(a.dev_scores)
    print(f"train {len(train)} patients, val {len(val)} patients; "
          f"K=0: {sum(p['K'] == 0 for p in train)}/{sum(p['K'] == 0 for p in val)}")

    oracle = {p["cid"]: oracle_best(p)[0] for p in train + val}
    folds = assign_folds([p["cid"] for p in train], C["outer_folds"], C["seed"])
    pd.DataFrame(sorted(folds.items()), columns=["cache_id", "outer_fold"]).to_csv(out / "folds.csv", index=False)

    F, S = C["finder_features"], C["finder_features"] + C["sam_features"]
    arms = {
        "B0": ThresholdArm("B0", "clf", F, C),
        "B1": ThresholdArm("B1", "reg", F, C),
        "U0": SubsetArm("U0", False, C),
        "B11": ThresholdArm("B11", "clf", S, C),
        "U1": SubsetArm("U1", True, C),
    }
    dec = {"train_oof": {}, "val": {}}
    fold_of = {"train_oof": {}, "val": {}}
    status = {"train_oof": {}, "val": {}}
    sel_log = []
    for name in ["B0", "B1", "U0", "B11", "U1"]:
        arm = arms[name]
        ref_tr = dec["train_oof"].get("B0") if name in ("U0", "U1") else None
        rows, sels = run_arm_outer(arm, train, folds, C, ref_tr)
        dec["train_oof"][name] = {c: m for c, _, m, _ in rows}
        status["train_oof"][name] = {c: s for c, _, _, s in rows}
        fold_of["train_oof"] = {c: f for c, f, _, _ in rows}
        sel_log += sels
        ref_va = dec["val"].get("B0") if name in ("U0", "U1") else None
        rows, sel = run_arm_val(arm, train, val, C, ref_va)
        dec["val"][name] = {c: m for c, _, m, _ in rows}
        status["val"][name] = {c: s for c, _, _, s in rows}
        sel_log.append(sel)
        print(f"{name}: done")
    (out / "selected_hyperparameters.json").write_text(json.dumps(sel_log, indent=1, default=float))

    # historical references (reproduction only)
    hist = {}
    for split, pats in (("train_oof", train), ("val", val)):
        hist[split] = {f"HIST_{v}": historical(pats, scores, split, v, t)
                       for v, t in C["historical_references"].items()}
    oracle_dec = {p["cid"]: oracle_best(p)[1] for p in train + val}

    summaries, comps, per_patient = [], [], []
    for split, pats in (("train_oof", train), ("val", val)):
        all_dec = {**dec[split], **hist[split], "ORACLE": oracle_dec}
        dice = {}
        for name, dd in all_dec.items():
            for grp, sub in (("all", pats), ("K>=2", [p for p in pats if p["K"] >= 2])):
                r, d = summarise(name, sub, dd, oracle, C, grp)
                r["split"] = split
                summaries.append(r)
                dice[(name, grp)] = d
        for a_, b_ in [("U0", "B0"), ("U0", "B1"), ("B1", "B0"), ("U1", "B11"), ("U1", "U0"), ("B11", "B0")]:
            for grp in ("all", "K>=2"):
                r = compare(a_, b_, dice[(a_, grp)], dice[(b_, grp)], C, grp)
                r["split"] = split
                comps.append(r)
        for p in pats:
            row = dict(split=split, cache_id=p["cid"], collection=p["coll"], K=p["K"],
                       outer_fold=fold_of["train_oof"].get(p["cid"], -1) if split == "train_oof" else -1,
                       oracle_dice=oracle[p["cid"]], oracle_subset=mask_str(p, oracle_dec[p["cid"]]))
            for name, dd in all_dec.items():
                if name == "ORACLE":
                    continue
                row[f"{name}_dice"] = dice_counts(p["tgt"], dd[p["cid"]])
                row[f"{name}_subset"] = mask_str(p, dd[p["cid"]])
            for name in arms:
                row[f"{name}_status"] = status[split][name][p["cid"]]
            per_patient.append(row)

    summ = pd.DataFrame(summaries)
    summ.to_csv(out / "arms_summary.csv", index=False)
    comp = pd.DataFrame(comps)
    comp.to_csv(out / "paired_comparisons.csv", index=False)
    pp = pd.DataFrame(per_patient)
    pp.to_csv(out / "per_patient.csv", index=False)

    rsum, rbud = [], []
    for split, pats in (("train_oof", train), ("val", val)):
        s_, b_, _ = routing_analysis(pats, dec[split]["U0"], dec[split]["U1"], C, split)
        rsum.append(s_)
        rbud.append(b_)
    pd.DataFrame(rsum).to_csv(out / "routing_summary.csv", index=False)
    pd.concat(rbud).to_csv(out / "routing_budget.csv", index=False)

    prov = dict(script="scripts/s3i_subset_jury.py", config=C,
                inputs={k: dict(path=v, md5=md5(v)) for k, v in
                        dict(train=a.train, val=a.val, dev_scores=a.dev_scores).items()},
                note="Conditional decision-layer pilot on S3f tables; upstream SegResNet/SAM saw training "
                     "patients. Validation is a reused development set. Test not read.")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    print(summ[summ.group == "all"][["split", "arm", "dice", "regret", "dice_lt_0.1", "dice_lt_0.5",
                                     "harmful_removals"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

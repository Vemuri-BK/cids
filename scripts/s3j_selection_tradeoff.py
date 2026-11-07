"""S3j: candidate-selection trade-off on OUT-OF-FOLD SegResNet candidates (development data only).

Question (frozen before the run)
    Can patient-level subset selection (U0) prevent near-total loss of the primary tumour while keeping
    better overall segmentation than simpler, fairly tuned rejection rules (binary jury B0 and variants,
    relative-size filtering, keep-all)?

Inputs
    --train  S3g out-of-fold candidate table (outputs/s3g_oof_masks/train_features.csv): every training
             candidate comes from the SegResNet fold model that never saw that patient.
    --val    S3f val candidate table (the pipeline's real val candidates; reused development set).
    Only the 7 finder features are used. No historical scores, no SAM features, no radiomics, test never read.
    Patients without any candidate (k = -1 row) keep their real id, K = 0, Dice 0 under every policy.

Policies (selectors fitted with the S3i outer/inner patient folds; thresholds chosen in inner folds only)
    keep_all     keep every candidate
    rel20        keep candidates with size >= 20% of the largest candidate (SegResNet's rule)
    B0           binary jury, threshold = best inner-fold mean Dice                     (as in S3i)
    B0w          B0 with candidate rows weighted 1/K (each patient total weight 1, as in U0)
    B0_match     B0 model, threshold whose inner-fold mean number of kept candidates is closest to U0's
    B0_safe      B0 model, threshold with the fewest inner-fold Dice<0.1 patients among thresholds whose
                 inner mean Dice is within --margin of B0's best (the permissive binary jury)
    U0           joint subset selector (S3i)
    B0@t         B0 model at every fixed threshold t (trade-off curve, train OOF + val)

Frozen decision rule (train OOF; checked against EVERY fair binary variant B0, B0w, B0_match, B0_safe)
    U0 advances only if, for each variant V:
        mean Dice(U0) >= mean Dice(V) - margin                     (margin 0.005)
        #Dice<0.1(U0) <= (1 - fail_cut) * #Dice<0.1(V)            (fail_cut 0.25)
        #Dice<0.5(U0) <= (1 + mid_tol) * #Dice<0.5(V)             (mid_tol 0.10)
    If any variant matches U0's trade-off, the simpler binary jury is the main method.

Outputs (<out>)
    policies_summary.csv     per split x policy: Dice mean/median, Dice<0.1, Dice<0.5, kept candidates and ml,
                             harmful removals
    tradeoff_curve.csv       B0 at every threshold (mean Dice vs failures vs retention)
    paired_comparisons.csv   U0 vs every other policy: bootstrap CI, better/worse, rescued/new failures
    failure_audit.csv        patients rescued or harmed by U0 vs each fair binary variant (before/after Dice)
    largest_changes.csv      top changes U0 vs B0 for visual mask inspection
    per_patient.csv, folds.csv, selected_thresholds.json, decision_rule.json, provenance.json

Usage (server)
    OMP_NUM_THREADS=1 python -u scripts/s3j_selection_tradeoff.py \\
        --train ~/CIDS/outputs/s3g_oof_masks/train_features.csv \\
        --val ~/CIDS/outputs/s3f_jury_masks/s3f_jury/val_features.csv \\
        --config configs/s3i_subset_jury.json --out ~/CIDS/outputs/s3j_tradeoff
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3i_subset_jury as S  # noqa: E402  (selectors, subsets, Dice from counts, folds, bootstrap)

FAIR_B = ("B0", "B0w", "B0_match", "B0_safe")


# ----------------------------------------------------------------------------------------------
# data (fixes: k = -1 rows -> K = 0 with the real patient id; no manifest padding needed)
# ----------------------------------------------------------------------------------------------
def load_patients(csv, feature_cols, forbidden):
    S.check_whitelist(feature_cols, forbidden)
    df = pd.read_csv(csv)
    need = set(feature_cols) | {"cache_id", "k", "rank", "size_vox", "inter_vox", "gt_vox"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{csv}: missing columns {sorted(miss)}")
    pats = []
    for cid, g in df.groupby("cache_id", sort=True):
        if g.gt_vox.nunique() != 1:
            raise ValueError(f"{cid}: gt_vox not constant")
        G = float(g.gt_vox.iloc[0])
        r = g[g.k >= 0].sort_values(["rank", "k"])
        if len(r) and (g.k < 0).any():
            raise ValueError(f"{cid}: has both real candidates and a k=-1 row")
        pats.append(dict(
            cid=cid, coll=S.collection_of(cid), K=len(r), status="ok" if len(r) else "no_candidate",
            k=r.k.to_numpy(int),
            feats=r[list(feature_cols)].reset_index(drop=True),
            size=r.size_vox.to_numpy(float),
            tgt=dict(I=r.inter_vox.to_numpy(float), V=r.size_vox.to_numpy(float), G=G)))
    return pats


# ----------------------------------------------------------------------------------------------
# simple policies
# ----------------------------------------------------------------------------------------------
def keep_all(p):
    return np.ones(p["K"], bool)


def rel20(p, frac=0.2):
    if p["K"] == 0:
        return np.zeros(0, bool)
    return p["size"] >= frac * p["size"].max()


# ----------------------------------------------------------------------------------------------
# binary jury with an inner-fold table of every (config, threshold) -> several threshold rules
# ----------------------------------------------------------------------------------------------
class BinaryJury(S.ThresholdArm):
    """B0 (weighted=False) or B0w (weighted=True: candidate rows weighted 1/K)."""

    def __init__(self, name, cols, C, weighted=False):
        super().__init__(name, "clf", cols, C)
        self.weighted = weighted

    def _fit(self, pats, cfg, seed):
        X, y, w = [], [], []
        for p in pats:
            if p["K"] == 0:
                continue
            X.append(p["feats"][self.cols].to_numpy(float))
            y.append((p["tgt"]["I"] > 0).astype(int))
            w.append(np.full(p["K"], 1.0 / p["K"] if self.weighted else 1.0))
        return S._hgb("clf", cfg, self.C["hgb_fixed"], seed).fit(np.vstack(X), np.concatenate(y),
                                                                  sample_weight=np.concatenate(w))

    def inner_table(self, pats, inner_folds, seed):
        """Inner-fold held-out decisions for every config x threshold -> DataFrame of
        (config_index, t, dice, fail01, kept_n). Patients with K <= 1 keep their only candidate."""
        fold = S.assign_folds([p["cid"] for p in pats], inner_folds, seed)
        rows = []
        for ci, cfg in enumerate(self.C["hgb_grid"]):
            sc = {}
            for f in range(inner_folds):
                m = self._fit([p for p in pats if fold[p["cid"]] != f], cfg, seed)
                for p in pats:
                    if fold[p["cid"]] == f and p["K"] > 0:
                        sc[p["cid"]] = self.scores(m, p)
            for t in self.tgrid:
                d, kn = [], []
                for p in pats:
                    if p["K"] == 0:
                        d.append(0.0); kn.append(0)
                        continue
                    keep = np.ones(1, bool) if p["K"] == 1 else S.keep_threshold(sc[p["cid"]], t)
                    d.append(S.dice_counts(p["tgt"], keep)); kn.append(int(keep.sum()))
                d = np.asarray(d)
                rows.append(dict(config_index=ci, t=float(t), dice=float(d.mean()),
                                 fail01=int((d < 0.1).sum()), kept_n=float(np.mean(kn))))
        return pd.DataFrame(rows)


def pick_best(tab):
    """Highest inner Dice; ties -> lower config index, lower threshold (enumeration order)."""
    i = tab.sort_values(["dice", "config_index", "t"], ascending=[False, True, True]).index[0]
    return tab.loc[i]


def pick_match(tab, config_index, target_kept):
    """Same config as B0's best; threshold whose inner mean kept candidates is closest to target."""
    t = tab[tab.config_index == config_index].copy()
    t["gap"] = (t.kept_n - target_kept).abs()
    return t.sort_values(["gap", "t"], ascending=[True, False]).iloc[0]


def pick_safe(tab, config_index, margin):
    """Same config as B0's best; fewest inner Dice<0.1 among thresholds within margin of the best Dice;
    ties -> higher Dice, then higher threshold."""
    t = tab[tab.config_index == config_index]
    t = t[t.dice >= t.dice.max() - margin]
    return t.sort_values(["fail01", "dice", "t"], ascending=[True, False, False]).iloc[0]


class SubsetSelector(S.SubsetArm):
    """U0 that also reports its inner-fold mean number of kept candidates (for B0_match)."""

    def inner_kept(self, pats, inner_folds, seed, config):
        fold = S.assign_folds([p["cid"] for p in pats], inner_folds, seed)
        kn = []
        for f in range(inner_folds):
            m = self._fit([p for p in pats if fold[p["cid"]] != f], config, seed)
            for p in pats:
                if fold[p["cid"]] != f:
                    continue
                kn.append(0 if p["K"] == 0 else 1 if p["K"] == 1 else int(self.choose(m, p)[0].sum()))
        return float(np.mean(kn))


def decide_threshold(model_arm, model, p, t):
    if p["K"] == 0:
        return np.zeros(0, bool)
    if p["K"] == 1:
        return np.ones(1, bool)
    return S.keep_threshold(model_arm.scores(model, p), t)


def fit_partition(train_part, C, seed, margin):
    """Fit U0, B0, B0w on one training partition; choose every threshold inside its inner folds.
    Returns callables {policy: p -> mask}, the B0 model (for the curve) and the selection log."""
    F, nf = C["finder_features"], C["inner_folds"]
    u0 = SubsetSelector("U0", False, C)
    usel = u0.fit_locked(train_part, nf, seed)
    u_kept = u0.inner_kept(train_part, nf, seed, usel["config"])
    pols, log = {"U0": lambda p, a=u0: a.decide(p)[0]}, dict(U0=dict(**usel, inner_kept=u_kept))
    b0 = None
    for name, weighted in (("B0", False), ("B0w", True)):
        arm = BinaryJury(name, F, C, weighted)
        tab = arm.inner_table(train_part, nf, seed)
        best = pick_best(tab)
        cfg = C["hgb_grid"][int(best.config_index)]
        model = arm._fit(train_part, cfg, seed)
        pols[name] = lambda p, a=arm, m=model, t=float(best.t): decide_threshold(a, m, p, t)
        log[name] = dict(config_index=int(best.config_index), threshold=float(best.t), inner_dice=float(best.dice),
                         inner_fail01=int(best.fail01), inner_kept=float(best.kept_n))
        if name == "B0":
            b0 = (arm, model)
            for var, row in (("B0_match", pick_match(tab, int(best.config_index), u_kept)),
                             ("B0_safe", pick_safe(tab, int(best.config_index), margin))):
                pols[var] = lambda p, a=arm, m=model, t=float(row.t): decide_threshold(a, m, p, t)
                log[var] = dict(config_index=int(best.config_index), threshold=float(row.t),
                                inner_dice=float(row.dice), inner_fail01=int(row.fail01),
                                inner_kept=float(row.kept_n))
    return pols, b0, log


# ----------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------
def patient_metrics(p, mask):
    d = S.dice_counts(p["tgt"], mask)
    return dict(dice=d, kept_n=int(mask.sum()), kept_ml=float(p["tgt"]["V"][mask].sum()) / 1000 if mask.size else 0.0,
                harmful=bool(S.harmful_removal(p, mask)[0]))


def summarise(pats, dec, split, policy):
    m = pd.DataFrame([patient_metrics(p, dec[p["cid"]]) for p in pats])
    return dict(split=split, policy=policy, n=len(pats), dice=m.dice.mean(), dice_median=m.dice.median(),
                dice_lt_0_1=int((m.dice < 0.1).sum()), dice_lt_0_5=int((m.dice < 0.5).sum()),
                kept_n=m.kept_n.mean(), kept_ml=m.kept_ml.mean(), harmful_removals=int(m.harmful.sum()))


def rule_check(summ_train, margin, fail_cut, mid_tol):
    """Frozen decision rule on train OOF: U0 vs each fair binary variant."""
    s = summ_train.set_index("policy")
    u = s.loc["U0"]
    rows = []
    for v in FAIR_B:
        b = s.loc[v]
        c1 = u.dice >= b.dice - margin
        c2 = u.dice_lt_0_1 <= (1 - fail_cut) * b.dice_lt_0_1
        c3 = u.dice_lt_0_5 <= (1 + mid_tol) * b.dice_lt_0_5
        rows.append(dict(versus=v, dice_U0=u.dice, dice_V=b.dice, dice_ok=bool(c1),
                         fail01_U0=int(u.dice_lt_0_1), fail01_V=int(b.dice_lt_0_1), fail01_ok=bool(c2),
                         fail05_U0=int(u.dice_lt_0_5), fail05_V=int(b.dice_lt_0_5), fail05_ok=bool(c3),
                         passes=bool(c1 and c2 and c3)))
    r = pd.DataFrame(rows)
    return r, bool(r.passes.all())


def audit(pats, dec, split, other):
    """Patients rescued from / pushed into Dice<0.1 by U0 relative to `other` (before = other, after = U0)."""
    rows = []
    for p in pats:
        d0 = S.dice_counts(p["tgt"], dec[other][p["cid"]])
        d1 = S.dice_counts(p["tgt"], dec["U0"][p["cid"]])
        if (d0 < 0.1) != (d1 < 0.1):
            rows.append(dict(split=split, versus=other, cache_id=p["cid"], collection=p["coll"], K=p["K"],
                             change="rescued" if d1 >= 0.1 else "harmed", dice_before=d0, dice_after=d1,
                             reaches_0_5=bool(d1 >= 0.5) if d1 >= 0.1 else None,
                             subset_before=S.mask_str(p, dec[other][p["cid"]]),
                             subset_after=S.mask_str(p, dec["U0"][p["cid"]]),
                             kept_ml_before=float(p["tgt"]["V"][dec[other][p["cid"]]].sum()) / 1000 if p["K"] else 0.0,
                             kept_ml_after=float(p["tgt"]["V"][dec["U0"][p["cid"]]].sum()) / 1000 if p["K"] else 0.0,
                             tumour_ml=p["tgt"]["G"] / 1000))
    return rows


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="S3g out-of-fold train_features.csv")
    ap.add_argument("--val", required=True, help="S3f val_features.csv (development set)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override config seed (fold assignment)")
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--fail_cut", type=float, default=0.25)
    ap.add_argument("--mid_tol", type=float, default=0.10)
    a = ap.parse_args()
    for f in (a.train, a.val):
        if "test" in Path(f).name.lower():
            raise SystemExit("refusing to read a test file in development mode")
    C = json.loads(Path(a.config).read_text())
    if a.seed is not None:
        C["seed"] = a.seed
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    F = C["finder_features"]
    train = load_patients(Path(a.train).expanduser(), F, C["forbidden_features"])
    val = load_patients(Path(a.val).expanduser(), F, C["forbidden_features"])
    rule = dict(margin=a.margin, fail_cut=a.fail_cut, mid_tol=a.mid_tol, versus=list(FAIR_B),
                frozen="U0 advances only if, on train OOF, it passes all three conditions against every variant")
    (out / "decision_rule.json").write_text(json.dumps(rule, indent=1))
    print(f"S3j | train {len(train)} (K=0: {sum(p['K'] == 0 for p in train)}) | val {len(val)} "
          f"(K=0: {sum(p['K'] == 0 for p in val)}) | seed {C['seed']} | rule {rule['margin']}/"
          f"{rule['fail_cut']}/{rule['mid_tol']}", flush=True)

    folds = S.assign_folds([p["cid"] for p in train], C["outer_folds"], C["seed"])
    pd.DataFrame(sorted(folds.items()), columns=["cache_id", "outer_fold"]).to_csv(out / "folds.csv", index=False)
    policies = ("keep_all", "rel20", "B0", "B0w", "B0_match", "B0_safe", "U0")
    dec = {s: {q: {} for q in policies} for s in ("train_oof", "val")}
    tgrid = S._grid(C["prob_threshold_grid"])
    curve = {s: {float(t): {} for t in tgrid} for s in ("train_oof", "val")}
    sel_log = []

    def apply(split, pats, pols, b0):
        arm, model = b0
        for p in pats:
            dec[split]["keep_all"][p["cid"]] = keep_all(p)
            dec[split]["rel20"][p["cid"]] = rel20(p)
            for q, fn in pols.items():
                dec[split][q][p["cid"]] = fn(p)
            for t in tgrid:
                curve[split][float(t)][p["cid"]] = decide_threshold(arm, model, p, float(t))

    for f in range(C["outer_folds"]):
        tr = [p for p in train if folds[p["cid"]] != f]
        te = [p for p in train if folds[p["cid"]] == f]
        assert not ({p["cid"] for p in tr} & {p["cid"] for p in te})
        pols, b0, log = fit_partition(tr, C, C["seed"] + 101 * (f + 1), a.margin)
        apply("train_oof", te, pols, b0)
        sel_log.append(dict(fold=f, **log))
        print(f"outer fold {f}: " + " | ".join(f"{k} t={v.get('threshold', '-')}" for k, v in log.items()
                                                if k != "U0") + f" | U0 kept {log['U0']['inner_kept']:.2f}",
              flush=True)
    pols, b0, log = fit_partition(train, C, C["seed"], a.margin)
    apply("val", val, pols, b0)
    sel_log.append(dict(fold="final_for_val", **log))
    (out / "selected_thresholds.json").write_text(json.dumps(sel_log, indent=1, default=float))

    if {p["cid"] for p in train} & {p["cid"] for p in val}:
        raise SystemExit("train and val share patients")
    summ, comps, aud, pp, crv, big = [], [], [], [], [], []
    for split, pats in (("train_oof", train), ("val", val)):
        oracle = {p["cid"]: S.oracle_best(p) for p in pats}
        dice = {q: np.array([S.dice_counts(p["tgt"], dec[split][q][p["cid"]]) for p in pats]) for q in policies}
        for q in policies:
            summ.append(summarise(pats, dec[split][q], split, q))
        od = {p["cid"]: oracle[p["cid"]][1] for p in pats}
        summ.append(summarise(pats, od, split, "ORACLE"))
        for t in tgrid:
            r = summarise(pats, curve[split][float(t)], split, f"B0@{t:.2f}")
            crv.append(dict(r, threshold=float(t)))
        for q in policies:
            if q == "U0":
                continue
            r = S.compare("U0", q, dice["U0"], dice[q], C, "all")
            r["split"] = split
            comps.append(r)
        for v in FAIR_B + ("keep_all", "rel20"):
            aud += audit(pats, dec[split], split, v)
        diff = dice["U0"] - dice["B0"]
        for i in np.argsort(-np.abs(diff))[:20]:
            if abs(diff[i]) < 1e-9:
                break
            p = pats[i]
            big.append(dict(split=split, cache_id=p["cid"], K=p["K"], dice_B0=dice["B0"][i], dice_U0=dice["U0"][i],
                            change=diff[i], subset_B0=S.mask_str(p, dec[split]["B0"][p["cid"]]),
                            subset_U0=S.mask_str(p, dec[split]["U0"][p["cid"]]),
                            oracle_subset=S.mask_str(p, oracle[p["cid"]][1])))
        for i, p in enumerate(pats):
            row = dict(split=split, cache_id=p["cid"], collection=p["coll"], K=p["K"], status=p["status"],
                       outer_fold=folds.get(p["cid"], -1) if split == "train_oof" else -1,
                       oracle_dice=oracle[p["cid"]][0])
            for q in policies:
                row[f"{q}_dice"] = dice[q][i]
                row[f"{q}_subset"] = S.mask_str(p, dec[split][q][p["cid"]])
            pp.append(row)

    summ = pd.DataFrame(summ)
    summ.to_csv(out / "policies_summary.csv", index=False)
    pd.DataFrame(crv).to_csv(out / "tradeoff_curve.csv", index=False)
    pd.DataFrame(comps).to_csv(out / "paired_comparisons.csv", index=False)
    pd.DataFrame(aud).to_csv(out / "failure_audit.csv", index=False)
    pd.DataFrame(big).to_csv(out / "largest_changes.csv", index=False)
    pd.DataFrame(pp).to_csv(out / "per_patient.csv", index=False)
    rc, verdict = rule_check(summ[summ.split == "train_oof"], a.margin, a.fail_cut, a.mid_tol)
    rc.to_csv(out / "decision_rule_check.csv", index=False)
    prov = dict(script="scripts/s3j_selection_tradeoff.py", config=C, rule=rule,
                inputs={k: dict(path=str(v), md5=md5(Path(v).expanduser())) for k, v in
                        dict(train=a.train, val=a.val).items()},
                note="Train candidates are out-of-fold SegResNet (S3g). Finder features only. Val is a reused "
                     "development set. Test not read.")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))

    pd.set_option("display.width", 200)
    cols = ["split", "policy", "dice", "dice_median", "dice_lt_0_1", "dice_lt_0_5", "kept_n", "kept_ml",
            "harmful_removals"]
    print("\n" + summ[cols].round(4).to_string(index=False))
    a_ = pd.DataFrame(aud)
    if len(a_):
        print("\nU0 vs each policy (train OOF): rescued / harmed at Dice 0.1; rescued reaching 0.5")
        t = a_[a_.split == "train_oof"]
        for v, g in t.groupby("versus"):
            r = g[g.change == "rescued"]
            print(f"  vs {v:9s} rescued {len(r):3d} (reach 0.5: {int(r.reaches_0_5.sum())}) | "
                  f"harmed {int((g.change == 'harmed').sum()):3d}")
    print("\nfrozen decision rule (train OOF):")
    print(rc.round(4).to_string(index=False))
    print(f"\nVERDICT: {'U0 ADVANCES' if verdict else 'U0 does NOT advance (a binary jury matches its trade-off)'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

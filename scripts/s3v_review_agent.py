"""S3v: cost-aware review agent -- per patient, how much human help to ask for, and which tool to use.

Development only: 1,080 train out-of-fold patients (learning + primary result) and 120 val (confirmation).
Test refused. Idea: automatic B0_rad for most patients; one simulated clinician click where it is worth it.
Formulation borrowed from agentic tool use with interaction costs (MedSAM-Agent, arXiv 2602.03320), implemented
as a lightweight two-stage policy (no LLM).

Actions
  accept          B0_rad automatic mask                                              effort 0
  click -> blob   one click; the SegResNet candidate under / nearest the click (Cn)  effort 1
  click -> SAM    one click; SAM-S2b one-click mask (S)                              effort 1  (EXPLORATORY)
  Click = deepest interior voxel of the largest expert component (simulated clinician; s3u.click_point);
  sensitivity: a second click 3-5 mm from it inside the tumour (seeded per patient).

Two-stage agent (no expert information at decision time)
  stage 1 (before any click) -- whether to ask: rank patients by the predicted gain of a click,
          g1 = Dice(Cn) - Dice(B0_rad), from the S3n review features (BASE; S3n train_oof_review.csv /
          val_review.csv). Gain model: gradient boosting regressor, cross-fitted on the S3k outer folds
          (train), fitted on all train for val.
  stage 2 (after the click) -- which tool: SAM instead of Cn if the predicted g2 = Dice(S) - Dice(Cn) > 0, from
          post-click features (click-to-blob distance, blob size / SegResNet probability / kept by B0_rad,
          SAM volume, Dice(SAM, Cn)). EXPLORATORY: the S2b decoder was trained on the 1,080 training
          patients, so SAM outcomes there are optimistic; SAM is judged on val only.

Frozen rules (decision_rule.json written before any policy is fitted)
  R1 (primary)  at a 10% click budget, QC triage (S3n risk) + click->blob removes more Dice<0.1 failures than
                random triage: patient-bootstrap 95% CI lower bound of (removed_QC - expected_removed_random) > 0
                on train OOF; val must point the same way.
  R2            at the same budget the learned stage-1 agent beats QC triage on mean Dice: CI lower bound of
                the paired difference > 0 (train OOF).
  R3 (exploratory) the stage-2 agent chooses SAM for some val patients and those gain over Cn (descriptive).
  Curves (descriptive): budgets 0-100% for accept-all, random, QC, agent, oracle; outcomes mean Dice,
  Dice<0.1 / <0.5 counts, median absolute relative volume error.

Steps
  --step outcomes  GPU (SAM): masks / Dice / measurements / post-click features -> outcomes.csv (resumable)
  --step policy    CPU: decision_rule.json, curves, rules -> curves.csv, rules.json, choices_<split>.csv
Example (server)
  python -u scripts/s3v_review_agent.py --step outcomes --cache_root ~/data \\
      --s3k_out ~/CIDS/outputs/s3k_radiomics --train_cases ~/CIDS/outputs/s3g_oof_masks/cases \\
      --val_cases ~/CIDS/outputs/s3f_jury_masks/s3f_jury/cases \\
      --sam_ckpt ~/CIDS/checkpoints/sam_s2b_best.pt --turbo_ckpt ~/CIDS/checkpoints/sam_med3d_turbo.pth \\
      --out ~/CIDS/outputs/s3v_review_agent
  python scripts/s3v_review_agent.py --step policy --s3k_out ~/CIDS/outputs/s3k_radiomics \\
      --s3n_out ~/CIDS/outputs/s3n_qc --out ~/CIDS/outputs/s3v_review_agent
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3m_contour_study as M  # noqa: E402
import s3u_click_biomarkers as U  # noqa: E402

BASE = ["kept_prob_mean", "kept_prob_max", "margin_kept_min", "margin_drop_max", "n_cand", "n_kept", "log_kept_ml"]
POST = ["cn_dist_mm", "cn_log_ml", "cn_prob", "cn_kept", "s_log_ml", "dice_s_cn", "dice_b_cn"]
MASKS = ["B", "Cn", "S"]
BUDGETS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
RULE = dict(
    cohort="train OOF 1080 (learning + primary) and val 120 (confirmation); test refused",
    click="deepest interior voxel of the largest expert component; sensitivity: 3-5 mm offset inside the tumour",
    actions=dict(accept="B0_rad", click_blob="nearest SegResNet candidate (Cn)", click_sam="SAM-S2b one click"),
    stage1=dict(target="g1 = Dice(Cn) - Dice(B0_rad)", features=BASE,
                model="GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, "
                      "subsample=0.8, random_state=0)", folds="S3k outer folds (train), all train (val)"),
    stage2=dict(target="g2 = Dice(S) - Dice(Cn)", features=POST, rule="SAM if predicted g2 > 0",
                status="EXPLORATORY (S2b trained on the training patients)"),
    R1="10% budget: QC triage + Cn removes more Dice<0.1 failures than random; bootstrap CI lower bound > 0 "
       "(train OOF), same direction on val",
    R2="10% budget: agent ranking beats QC ranking on mean Dice; bootstrap CI lower bound > 0 (train OOF)",
    R3="exploratory: val patients where stage 2 chooses SAM gain over Cn (descriptive)",
    budgets=BUDGETS, bootstrap=dict(n=2000, seed=0), random_draws=200)


# --------------------------------------------------------------------------------------------- outcomes
def offset_click(gt, p, cid, lo=3.0, hi=5.0):
    """A voxel of the expert mask 3-5 mm from p (seeded per patient); p itself if none exists."""
    idx = np.argwhere(gt)
    d = np.sqrt(((idx - np.asarray(p)) ** 2).sum(1))
    ok = idx[(d >= lo) & (d <= hi)]
    if not len(ok):
        return tuple(p)
    rng = np.random.default_rng(zlib.crc32(cid.encode()))
    return tuple(int(v) for v in ok[rng.integers(len(ok))])


def measurements(img, m):
    if not m.any():
        return dict(ml=0.0, diam=np.nan, enh=np.nan)
    return dict(ml=float(m.sum()) / 1000, diam=U.diameter_mm(m), enh=float((img[1] - img[0])[m].mean()))


def patient_outcomes(img, gt, lab, kept, probs, p, sam_fn):
    """Masks for one click -> Dice / measurements / post-click features (no expert information in features)."""
    B = np.isin(lab, kept) if kept else np.zeros(gt.shape, bool)
    Cn, dist = U.nearest_blob(lab, p)
    S = sam_fn(img, p)
    k = int(lab[tuple(np.argwhere(Cn)[0])]) if Cn.any() else 0
    r = dict(cn_k=k, cn_dist_mm=dist, cn_log_ml=float(np.log(Cn.sum() / 1000 + 1e-3)),
             cn_prob=float(probs.get(k, np.nan)), cn_kept=int(k in kept), s_log_ml=float(np.log(S.sum() / 1000 + 1e-3)),
             dice_s_cn=M.dice_count(S, Cn), dice_b_cn=M.dice_count(B, Cn))
    for n, m in (("E", gt), ("B", B), ("Cn", Cn), ("S", S)):
        r[f"{n}_dice"] = M.dice_count(m, gt)
        for q, v in measurements(img, m).items():
            r[f"{n}_{q}"] = v
    return r


def step_outcomes(a, out):
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    pp = M.load_selection(Path(a.s3k_out).expanduser()).set_index(["split", "cache_id"])
    tabs = {s: pd.read_csv(Path(a.s3k_out).expanduser() / f) for s, f in
            (("train_oof", "train_oof_table.csv"), ("val", "val_table.csv"))}
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    jobs = []
    for split, cache_split, cdir in (("train_oof", "train", a.train_cases), ("val", "val", a.val_cases)):
        d = Path(cdir).expanduser()
        d = d if any(d.glob("*.npz")) else next(d.rglob("*.npz")).parent
        cs = require_cases(sp, root, cache_split)
        jobs += [(split, cid, path, d) for cid, path in (cs[:a.limit] if a.limit else cs)]
    part = out / "outcomes.partial.csv"
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {(r["split"], r["cache_id"]) for r in rows}
    runner = M.load_runner(a.sam_ckpt, a.turbo_ckpt, a.random_weights)
    if a.random_weights:
        import torch
        torch.manual_seed(0)

    def sam_fn(img, p):
        return U.sam_one_click(img, p, runner.sam, runner.box, runner.dev, runner.amp)

    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"S3v outcomes | {len(jobs)} patients ({len(jobs) - len(todo)} already done)", flush=True)
    t0 = time.time()
    for i, (split, cid, path, cdir) in enumerate(todo):
        if (split, cid) not in pp.index:
            raise SystemExit(f"{split} {cid}: no B0_rad selection in S3k per_patient.csv")
        img, gt = load_case(path)
        gt = gt > 0
        lab = np.load(cdir / f"{cid}.npz")["lab"]
        kept = M.kept_ids(pp.loc[(split, cid), "B0_rad_subset"])
        t = tabs[split]
        probs = dict(zip(t[t.cache_id == cid].k.astype(int), t[t.cache_id == cid].seg_prob_mean))
        p0 = U.click_point(gt)
        for variant, p in (("deep", p0), ("offset", offset_click(gt, p0, cid))):
            r = patient_outcomes(img, gt, lab, kept, probs, p, sam_fn)
            if abs(r["B_dice"] - float(pp.loc[(split, cid), "B0_rad_dice"])) > 1e-6:
                raise SystemExit(f"PARITY FAILED {split} {cid}: B0_rad Dice {r['B_dice']} vs S3k")
            rows.append(dict(split=split, cache_id=cid, click=variant, click_xyz=list(p), **r))
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"  {i + 1}/{len(todo)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "outcomes.csv", index=False)
    if part.exists():
        part.unlink()
    d = df[df.click == "deep"]
    print("PARITY OK")
    print(d.groupby("split")[["B_dice", "Cn_dice", "S_dice"]].mean().round(4).to_string())
    print("OUTCOMES DONE -> run --step policy")


# --------------------------------------------------------------------------------------------- policy
def gbr():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                                     random_state=0)


def crossfit(df, cols, target, folds):
    pred = pd.Series(np.nan, index=df.index)
    f = df.cache_id.map(folds)
    for k in sorted(f.unique()):
        te = f == k
        m = gbr().fit(df.loc[~te, cols].fillna(-1), df.loc[~te, target])
        pred[te] = m.predict(df.loc[te, cols].fillna(-1))
    return pred


def select(score, cid, frac):
    """Top round(frac*N) by score, ties by cache_id (as S3n)."""
    n = int(round(frac * len(score)))
    order = np.lexsort((np.asarray(cid), -np.asarray(score, float)))
    sel = np.zeros(len(score), bool)
    sel[order[:n]] = True
    return sel


def outcome(df, sel, tool):
    """Final per-patient Dice / volume error when the selected patients get a click with `tool` (array or str)."""
    t = np.where(np.asarray(tool) == "S", "S", "Cn") if not isinstance(tool, str) else np.full(len(df), tool)
    dice = np.where(sel, np.where(t == "S", df.S_dice, df.Cn_dice), df.B_dice)
    ml = np.where(sel, np.where(t == "S", df.S_ml, df.Cn_ml), df.B_ml)
    verr = np.abs(ml - df.E_ml) / df.E_ml.clip(lower=1e-6)
    return dice, verr


def summarise(dice, verr):
    return dict(mean_dice=float(np.mean(dice)), fail_01=int((dice < 0.1).sum()), fail_05=int((dice < 0.5).sum()),
                vol_err_median=float(np.median(verr)))


def boot_ci(fn, n_pat, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n_pat, n_pat)) for _ in range(n)])
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def step_policy(a, out):
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any fit
    oc = pd.read_csv(out / "outcomes.csv")
    s3n = Path(a.s3n_out).expanduser()
    folds = dict(pd.read_csv(Path(a.s3k_out).expanduser() / "folds.csv").values)
    curves, rules, choices = [], {}, {}
    data = {}
    for split, f in (("train_oof", "train_oof_review.csv"), ("val", "val_review.csv")):
        rv = pd.read_csv(s3n / f)
        for variant in ("deep", "offset"):
            d = oc[(oc.split == split) & (oc.click == variant)].merge(
                rv[["cache_id", "is_empty", "risk_baseline"] + BASE], on="cache_id", how="left", validate="one_to_one")
            if d.risk_baseline.isna().any():
                raise SystemExit(f"{split}: S3n review rows missing for some patients")
            d = d.sort_values("cache_id").reset_index(drop=True)
            d["g1"], d["g2"] = d.Cn_dice - d.B_dice, d.S_dice - d.Cn_dice
            data[(split, variant)] = d
    tr = data[("train_oof", "deep")]
    tr["agent1"] = crossfit(tr, BASE, "g1", folds)
    tr["agent2"] = crossfit(tr, POST, "g2", folds)
    m1, m2 = gbr().fit(tr[BASE].fillna(-1), tr.g1), gbr().fit(tr[POST].fillna(-1), tr.g2)
    for key, d in data.items():
        if key != ("train_oof", "deep"):
            src = tr.set_index("cache_id") if key[0] == "train_oof" else None
            if src is not None:                                        # offset click: same stage-1 scores
                d["agent1"] = d.cache_id.map(src.agent1)
                d["agent2"] = crossfit(d, POST, "g2", folds)
            else:
                d["agent1"] = m1.predict(d[BASE].fillna(-1))
                d["agent2"] = m2.predict(d[POST].fillna(-1))
        d["qc"] = np.where(d.is_empty == 1, np.inf, d.risk_baseline)
        d["tool2"] = np.where(d.agent2 > 0, "S", "Cn")
        cid = d.cache_id.to_numpy()
        rng = np.random.default_rng(0)
        for b in BUDGETS:
            for pol, score in (("QC", d.qc), ("agent", d.agent1), ("oracle", d.g1)):
                for tool in ("Cn", "S", "agent2"):
                    sel = select(score, cid, b)
                    dice, verr = outcome(d, sel, d.tool2 if tool == "agent2" else tool)
                    curves.append(dict(split=key[0], click=key[1], budget=b, policy=pol, tool=tool,
                                       n_clicks=int(sel.sum()), **summarise(dice, verr)))
            rs = [summarise(*outcome(d, select(rng.random(len(d)), cid, b), "Cn")) for _ in range(RULE["random_draws"])]
            curves.append(dict(split=key[0], click=key[1], budget=b, policy="random", tool="Cn",
                               n_clicks=int(round(b * len(d))), **pd.DataFrame(rs).mean().to_dict()))
        choices[key] = d
    # ---- frozen rules (deep click)
    for split in ("train_oof", "val"):
        d = data[(split, "deep")]
        cid = d.cache_id.to_numpy()
        fixed = ((d.B_dice < 0.1) & (d.Cn_dice >= 0.1)).to_numpy()            # failures a click->blob removes
        hurt = ((d.B_dice >= 0.1) & (d.Cn_dice < 0.1)).to_numpy()

        def removed(idx, score):
            sel = select(np.asarray(score)[idx], cid[idx], 0.10)
            return int((sel & fixed[idx]).sum() - (sel & hurt[idx]).sum())

        def r1(idx):
            return removed(idx, d.qc) - 0.10 * (fixed[idx].sum() - hurt[idx].sum())

        def r2(idx):
            sa, sq = select(d.agent1.to_numpy()[idx], cid[idx], 0.10), select(d.qc.to_numpy()[idx], cid[idx], 0.10)
            da = np.where(sa, d.Cn_dice.to_numpy()[idx], d.B_dice.to_numpy()[idx])
            dq = np.where(sq, d.Cn_dice.to_numpy()[idx], d.B_dice.to_numpy()[idx])
            return float(da.mean() - dq.mean())
        allidx = np.arange(len(d))
        lo1, hi1 = boot_ci(r1, len(d), RULE["bootstrap"]["n"], RULE["bootstrap"]["seed"])
        lo2, hi2 = boot_ci(r2, len(d), RULE["bootstrap"]["n"], RULE["bootstrap"]["seed"])
        sam = d[d.tool2 == "S"]
        rules[split] = dict(
            R1=dict(removed_QC=removed(allidx, d.qc), expected_random=float(0.10 * (fixed.sum() - hurt.sum())),
                    diff=r1(allidx), ci95=[lo1, hi1], passes=bool(lo1 > 0)),
            R2=dict(dice_agent_minus_qc=r2(allidx), ci95=[lo2, hi2], passes=bool(lo2 > 0)),
            R3=dict(sam_chosen=len(sam), dice_S=float(sam.S_dice.mean()) if len(sam) else None,
                    dice_Cn=float(sam.Cn_dice.mean()) if len(sam) else None,
                    sam_better=int((sam.S_dice > sam.Cn_dice).sum()), cn_better=int((sam.Cn_dice > sam.S_dice).sum())))
    verdict = dict(R1_primary=bool(rules["train_oof"]["R1"]["passes"] and rules["val"]["R1"]["diff"] > 0),
                   R2_agent_beats_qc=rules["train_oof"]["R2"]["passes"], rules=rules)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    (out / "rules.json").write_text(json.dumps(verdict, indent=1, default=float))
    for (split, variant), d in choices.items():
        d.drop(columns=["click_xyz"], errors="ignore").to_csv(out / f"choices_{split}_{variant}.csv", index=False)
    c = pd.DataFrame(curves)
    pd.set_option("display.width", 220)
    show = c[(c.click == "deep") & (c.tool.isin(["Cn"])) & c.budget.isin([0, 0.05, 0.10, 0.20, 1.0])]
    print("effort-accuracy (deep click, click -> nearest blob):")
    print(show[["split", "budget", "policy", "n_clicks", "mean_dice", "fail_01", "fail_05", "vol_err_median"]]
          .round(4).to_string(index=False))
    t10 = c[(c.click == "deep") & (c.budget == 0.10) & (c.policy == "agent")]
    print("\n10% budget, agent ranking, by tool (Cn / always SAM / stage-2 agent):")
    print(t10[["split", "tool", "mean_dice", "fail_01", "fail_05", "vol_err_median"]].round(4).to_string(index=False))
    off = c[(c.click == "offset") & (c.budget == 0.10) & (c.tool == "Cn")]
    print("\nsensitivity, 3-5 mm offset click, 10% budget:")
    print(off[["split", "policy", "mean_dice", "fail_01", "fail_05"]].round(4).to_string(index=False))
    print("\nRULES:", json.dumps(verdict, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["outcomes", "policy"], required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--s3k_out", required=True)
    ap.add_argument("--s3n_out", default="", help="S3n output (train_oof_review.csv, val_review.csv)")
    ap.add_argument("--train_cases", default="", help="S3g OOF component maps")
    ap.add_argument("--val_cases", default="", help="S3f val component maps")
    ap.add_argument("--sam_ckpt", default="")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="tests only")
    ap.add_argument("--limit", type=int, default=0, help="patients per split (tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for f in (a.train_cases, a.val_cases, a.s3n_out):
        if "test" in Path(f).name.lower():
            raise SystemExit("S3v is a development study: test files are refused")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.step == "outcomes":
        if not (a.train_cases and a.val_cases):
            raise SystemExit("--train_cases and --val_cases are required")
        if not a.random_weights and not (a.sam_ckpt and a.turbo_ckpt):
            raise SystemExit("--sam_ckpt (S2b best.pt) and --turbo_ckpt are required")
        step_outcomes(a, out)
    else:
        if not (out / "outcomes.csv").exists():
            raise SystemExit("run --step outcomes first")
        if not a.s3n_out:
            raise SystemExit("--s3n_out is required")
        step_policy(a, out)


if __name__ == "__main__":
    main()

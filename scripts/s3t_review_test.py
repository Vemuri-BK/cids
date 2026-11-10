"""S3t: ONE confirmatory run of the frozen one-click review protocol on the 306 test patients (guide-approved).

Everything is frozen on development data before the test is read; nothing is re-fitted on test.
  automatic mask  B0_rad (outputs/s3o_final/frozen/b0_rad.pkl; must reproduce S3o's test kept sets and Dice)
  QC triage       S3n baseline review score: logistic regression on BASE, fitted on all 1,080 train OOF patients
                  (S3n train_oof_review.csv); empty masks first
  click           deepest interior voxel of the largest expert component (simulated clinician, as S3u / S3v);
                  sensitivity: the S3v 3-5 mm offset click
  tools           Cn (nearest SegResNet blob), S (SAM-S2b one click, sam_s2b_best.pt: never trained on test),
                  F_max = max(P_sam, P_seg) > 0.5 (S3z / S3z-b)
  stage-2 agent   S3v gradient boosting on POST, fitted on all train OOF patients (S3v-OOF choices_train_oof_deep)
Frozen rules (test_plan.json, written by --step freeze, which also checks that every frozen piece reproduces its
development result exactly)
  R1  10% QC budget + click -> blob removes more Dice<0.1 failures than random (S3v R1 formula; CI lower bound > 0)
  R4  (a) test patients the agent sends to SAM: mean Dice(S) - Dice(Cn) CI lower bound > 0, AND (b) at the 10% QC
      budget the agent's tool choice beats always-blob on mean Dice over all test patients, CI lower bound > 0
      (S3v R4 formulas, 2,000 bootstrap resamples, seed 0)
  R9  val + test patients the agent sends to SAM: mean Dice(F_max) - Dice(S), CI lower bound > 0 (S3z-b rule,
      5,000 resamples, seed 0)
  reported  effort-accuracy at 0 / 5 / 10 / 20 / 100% budgets; failures; per hospital; offset-click sensitivity
Steps
  --step freeze  CPU: reproduction checks + test_plan.json (with md5 of every frozen input); test not read
  --step test    GPU: runs once (done.json); refuses if any frozen input changed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3j_selection_tradeoff as J  # noqa: E402,F401  (classes needed to unpickle the frozen jury)
import s3m_contour_study as M  # noqa: E402
import s3n_qc_review as N  # noqa: E402
import s3u_click_biomarkers as U  # noqa: E402
import s3v_review_agent as V  # noqa: E402
import s3z_fusion as Z  # noqa: E402
import s3z_fusion_b as ZB  # noqa: E402

BUDGETS = [0.0, 0.05, 0.10, 0.20, 1.0]
PLAN = dict(
    R1="10% QC budget, click -> nearest blob: removed Dice<0.1 failures minus expected under random triage, "
       "patient-bootstrap (2000, seed 0) CI lower bound > 0",
    R4="(a) agent sends to SAM: mean Dice(S) - Dice(Cn) CI lower bound > 0 AND (b) 10% QC budget, agent tool vs "
       "always-blob, mean Dice over all test patients, CI lower bound > 0 (2000, seed 0)",
    R9="val + test patients the agent sends to SAM: mean Dice(F_max) - Dice(S), CI lower bound > 0 (5000, seed 0)",
    reported="effort-accuracy 0/5/10/20/100%, failures, per hospital, offset click", once=True)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------- rules (S3v formulas)
def rules(d: pd.DataFrame) -> dict:
    """R1 and R4 exactly as s3v_review_agent.step_policy computes them (deep click). d sorted by cache_id."""
    cid = d.cache_id.to_numpy()
    fixed = ((d.B_dice < 0.1) & (d.Cn_dice >= 0.1)).to_numpy()
    hurt = ((d.B_dice >= 0.1) & (d.Cn_dice < 0.1)).to_numpy()
    qc = d.qc.to_numpy()

    def removed(idx):
        sel = V.select(qc[idx], cid[idx], 0.10)
        return int((sel & fixed[idx]).sum() - (sel & hurt[idx]).sum())

    def r1(idx):
        return removed(idx) - 0.10 * (fixed[idx].sum() - hurt[idx].sum())
    dS, dC, t2 = d.S_dice.to_numpy(), d.Cn_dice.to_numpy(), d.tool2.to_numpy()
    ch = np.flatnonzero(t2 == "S")

    def r4a(i):
        return float((dS[ch][i] - dC[ch][i]).mean()) if len(ch) else 0.0

    def r4b(idx):
        q = V.select(qc[idx], cid[idx], 0.10)
        alt = np.where(t2[idx] == "S", dS[idx], dC[idx])
        return float(np.where(q, alt - dC[idx], 0).mean())
    allidx = np.arange(len(d))
    lo1, hi1 = V.boot_ci(r1, len(d), 2000, 0)
    lo4a, hi4a = V.boot_ci(r4a, len(ch), 2000, 0) if len(ch) > 1 else (np.nan, np.nan)
    lo4b, hi4b = V.boot_ci(r4b, len(d), 2000, 0)
    return dict(
        R1=dict(removed_QC=removed(allidx), expected_random=float(0.10 * (fixed.sum() - hurt.sum())),
                diff=float(r1(allidx)), ci95=[lo1, hi1], passes=bool(lo1 > 0)),
        R4=dict(sam_chosen=int(len(ch)), dice_S=float(dS[ch].mean()) if len(ch) else None,
                dice_Cn=float(dC[ch].mean()) if len(ch) else None, gain_on_chosen=r4a(np.arange(len(ch))),
                gain_on_chosen_ci95=[lo4a, hi4a], sam_better=int((dS[ch] > dC[ch]).sum()),
                gain_10pct_vs_always_cn=r4b(allidx), gain_10pct_ci95=[lo4b, hi4b],
                passes=bool(len(ch) > 1 and lo4a > 0 and lo4b > 0)))


def curves(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in BUDGETS:
        sel = V.select(d.qc.to_numpy(), d.cache_id.to_numpy(), b)
        for tool in ("Cn", "agent", "agent_F_max"):
            alt = d.Cn_dice.to_numpy()
            if tool != "Cn":
                alt = np.where(d.tool2 == "S", d.FM_dice if tool == "agent_F_max" else d.S_dice, d.Cn_dice)
            v = np.where(sel, alt, d.B_dice)
            rows.append(dict(budget=b, tool=tool, n_clicks=int(sel.sum()), mean_dice=float(v.mean()),
                             fail_01=int((v < 0.1).sum()), fail_05=int((v < 0.5).sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- frozen pieces
def inputs(a):
    s3v, s3n, s3z = (Path(x).expanduser() for x in (a.s3v_out, a.s3n_out, a.s3z_out))
    so = Path(a.s3o_out).expanduser()
    return {"s3v_train_choices": s3v / "choices_train_oof_deep.csv", "s3v_val_choices": s3v / "choices_val_deep.csv",
            "s3v_rules": s3v / "rules.json", "s3n_train_review": s3n / "train_oof_review.csv",
            "s3n_val_review": s3n / "val_review.csv", "s3z_val_choices": s3z / "choices_val_deep.csv",
            "s3z_b_rule": Path(a.s3z_b).expanduser() / "decision_rule.json",
            "s3n_jury_scores": s3n / "jury_scores.csv", "s3k_val_table": Path(a.s3k_out).expanduser() / "val_table.csv",
            "b0_rad": so / "frozen" / "b0_rad.pkl", "s3o_test_table": so / "test" / "test_table.csv",
            "s3o_test_per_patient": so / "test" / "per_patient.csv", "sam_ckpt": Path(a.sam_ckpt).expanduser()}


def stage2(train_choices: pd.DataFrame):
    tr = train_choices.sort_values("cache_id").reset_index(drop=True)
    return V.gbr().fit(tr[V.POST].fillna(-1), tr.g2)


def checks(f) -> dict:
    """Every frozen piece must reproduce its development output exactly."""
    out = {}
    tr = pd.read_csv(f["s3v_train_choices"])
    ref = json.loads(f["s3v_rules"].read_text())["rules"]["train_oof"]
    mine = rules(tr.sort_values("cache_id").reset_index(drop=True))
    for k in ("removed_QC", "diff"):
        if abs(mine["R1"][k] - ref["R1"][k]) > 1e-9:
            raise SystemExit(f"rule code does not reproduce S3v R1 {k}: {mine['R1'][k]} vs {ref['R1'][k]}")
    for k in ("sam_chosen", "gain_on_chosen", "gain_10pct_vs_always_cn"):
        if abs(mine["R4"][k] - ref["R4"][k]) > 1e-9:
            raise SystemExit(f"rule code does not reproduce S3v R4 {k}: {mine['R4'][k]} vs {ref['R4'][k]}")
    out["rules_reproduce_S3v_train_oof"] = True
    va = pd.read_csv(f["s3v_val_choices"]).sort_values("cache_id")
    pred = stage2(tr).predict(va[V.POST].fillna(-1))
    if not np.allclose(pred, va.agent2.to_numpy(), atol=1e-9) or ((pred > 0) != (va.tool2 == "S")).any():
        raise SystemExit("stage-2 agent fitted on train does not reproduce S3v's val decisions")
    out["stage2_reproduces_S3v_val"] = True
    trn, van = pd.read_csv(f["s3n_train_review"]), pd.read_csv(f["s3n_val_review"])
    r = N.fit_apply(trn, van, N.BASE).to_numpy()
    if not np.allclose(np.where(np.isinf(r), 9, r), np.where(np.isinf(van.risk_baseline), 9, van.risk_baseline),
                       atol=1e-9):
        raise SystemExit("QC model fitted on train does not reproduce S3n's val risk")
    out["qc_reproduces_S3n_val"] = True
    js, _ = jury_scores(f, f["s3k_val_table"])
    ref = pd.read_csv(f["s3n_jury_scores"])
    ref = ref[ref.split == "val"].merge(js, on=["cache_id", "k"], suffixes=("", "_mine"), validate="one_to_one")
    if len(ref) != len(js) or not np.allclose(ref.score, ref.score_mine, atol=1e-12) or (ref.kept != ref.kept_mine).any():
        raise SystemExit("frozen B0_rad does not reproduce S3n's val jury scores (QC features would differ)")
    out["jury_reproduces_S3n_val_scores"] = True
    return out


def step_freeze(a, out):
    plan_f = out / "test_plan.json"
    if plan_f.exists():
        raise SystemExit(f"{plan_f} exists: the plan is frozen")
    f = inputs(a)
    miss = [k for k, p in f.items() if not p.exists()]
    if miss:
        raise SystemExit(f"missing frozen inputs: {miss}")
    ok = checks(f)
    plan = dict(PLAN, frozen_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), checks=ok,
                inputs={k: dict(path=str(p), md5=md5(p)) for k, p in f.items()})
    plan_f.write_text(json.dumps(plan, indent=1))
    print(json.dumps(ok, indent=1))
    print(f"FROZEN -> {plan_f} (test not read)\nnext: --step test (runs once)")


# --------------------------------------------------------------------------------------------- test
def jury_scores(f, table_csv):
    """Frozen B0_rad on a candidate table -> jury scores in the S3n format + each patient's kept set."""
    fr = pickle.loads(f["b0_rad"].read_bytes())
    C = json.loads((ROOT / "configs" / "s3i_subset_jury.json").read_text())
    pats = J.load_patients(table_csv, fr["features"], C["forbidden_features"])
    rows, kept = [], {}
    for p in pats:
        if p["K"]:
            sc = fr["arm"].scores(fr["model"], p)
            keep = J.decide_threshold(fr["arm"], fr["model"], p, fr["threshold"])
            rows += [dict(cache_id=p["cid"], k=int(k), score=float(s), threshold=fr["threshold"], kept=bool(kp))
                     for k, s, kp in zip(p["k"], sc, keep)]
            kept[p["cid"]] = {int(k) for k, kp in zip(p["k"], keep) if kp}
        else:
            kept[p["cid"]] = set()
    return pd.DataFrame(rows), kept


def jury_test(f):
    """Frozen B0_rad on the test candidates; must reproduce S3o's test kept sets."""
    js, kept = jury_scores(f, f["s3o_test_table"])
    ref = pd.read_csv(f["s3o_test_per_patient"], dtype={"B0_rad_subset": str}).set_index("cache_id")
    bad = [c for c, k in kept.items() if k != N.kept_from_subset(ref.loc[c, "B0_rad_subset"])]
    if set(kept) != set(ref.index):
        raise SystemExit("S3o test table and per_patient.csv cover different patients")
    if bad:
        raise SystemExit(f"frozen B0_rad does not reproduce S3o's test kept sets for {len(bad)} patients: {bad[:5]}")
    return js, ref


def step_test(a, out):
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    plan_f, done_f = out / "test_plan.json", out / "done.json"
    if not plan_f.exists():
        raise SystemExit("run --step freeze first")
    if done_f.exists():
        raise SystemExit(f"{done_f} exists: the test split is evaluated once")
    plan = json.loads(plan_f.read_text())
    f = inputs(a)
    for k, v in plan["inputs"].items():
        if md5(f[k]) != v["md5"]:
            raise SystemExit(f"frozen input {k} changed after freezing (md5 mismatch)")
    js, ref = jury_test(f)
    table = pd.read_csv(f["s3o_test_table"])
    pp = ref.reset_index()[["cache_id", "B0_rad_dice", "B0_rad_subset", "collection"]]
    feats = N.patient_features(table, js, pd.DataFrame(dict(cache_id=pp.cache_id, dice_ab=np.nan)), pp)
    feats["risk_baseline"] = N.fit_apply(pd.read_csv(f["s3n_train_review"]), feats, N.BASE).to_numpy()
    m2 = stage2(pd.read_csv(f["s3v_train_choices"]))
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    paths = dict(require_cases(pd.read_csv(a.splits), root, "test"))
    if set(paths) != set(pp.cache_id):
        raise SystemExit("S3o test patients differ from the test split")
    import s3k_radiomics_matched as K
    cdir = K.cases_dir(Path(a.test_cases).expanduser())
    runner = M.load_runner(str(f["sam_ckpt"]), a.turbo_ckpt, a.random_weights)
    rows, t0 = [], time.time()
    probs_all = table[table.k >= 0].groupby("cache_id")
    for i, cid in enumerate(sorted(paths)):
        img, gt = load_case(paths[cid])
        gt = gt > 0
        z = np.load(cdir / f"{cid}.npz")
        lab, p_seg = z["lab"], z["seg"].astype(np.float32) / 255.0
        kept = M.kept_ids(ref.loc[cid, "B0_rad_subset"])
        g = probs_all.get_group(cid) if cid in probs_all.groups else table.iloc[:0]
        probs = dict(zip(g.k.astype(int), g.seg_prob_mean))
        p0 = U.click_point(gt)
        if p0 is None:
            raise SystemExit(f"{cid}: empty expert mask")
        for variant, p in (("deep", p0), ("offset", V.offset_click(gt, p0, cid))):
            r = V.patient_outcomes(img, gt, lab, kept, probs, p,
                                   lambda im, q: U.sam_one_click(im, q, runner.sam, runner.box, runner.dev, runner.amp))
            if abs(r["B_dice"] - float(ref.loc[cid, "B0_rad_dice"])) > 1e-6:
                raise SystemExit(f"PARITY FAILED {cid}: B0_rad Dice {r['B_dice']} vs S3o")
            ms = Z.fuse(Z.sam_prob(img, p, runner), p_seg, p)
            if abs(M.dice_count(ms["S"], gt) - r["S_dice"]) > 1e-6:
                raise SystemExit(f"PARITY FAILED {cid}: SAM mask differs between S3v and S3z code")
            r.update(FM_dice=M.dice_count(ms["FM"], gt), FA_dice=M.dice_count(ms["FA"], gt))
            rows.append(dict(split="test", cache_id=cid, click=variant, click_xyz=list(p), **r))
        if (i + 1) % 50 == 0 or i + 1 == len(paths):
            print(f"  {i + 1}/{len(paths)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    oc = pd.DataFrame(rows).merge(feats[["cache_id", "is_empty", "risk_baseline"] + N.BASE], on="cache_id",
                                  how="left", validate="many_to_one").merge(pp[["cache_id", "collection"]], on="cache_id")
    oc["qc"] = np.where(oc.is_empty == 1, np.inf, oc.risk_baseline)
    oc["agent2"] = m2.predict(oc[V.POST].fillna(-1))
    oc["tool2"] = np.where(oc.agent2 > 0, "S", "Cn")
    oc = oc.sort_values(["click", "cache_id"]).reset_index(drop=True)
    oc.to_csv(out / "outcomes.csv", index=False)
    verdict = analyse(oc, pd.read_csv(f["s3z_val_choices"]), out)
    done_f.write_text(json.dumps(dict(outcomes_md5=md5(out / "outcomes.csv"), plan_md5=md5(plan_f),
                                      n=int((oc.click == "deep").sum()),
                                      utc=datetime.now(timezone.utc).isoformat(timespec="seconds")), indent=1))
    pd.set_option("display.width", 200)
    print("\neffort-accuracy, deep click (test):")
    print(pd.read_csv(out / "curves_deep.csv").round(4).to_string(index=False))
    print("\nper hospital:\n" + pd.read_csv(out / "per_hospital.csv").round(4).to_string(index=False))
    print("\nRESULTS:", json.dumps(verdict, indent=1, default=float))
    print(f"\nTEST DONE (evaluated once) -> {out}")


def analyse(oc: pd.DataFrame, va: pd.DataFrame, out: Path) -> dict:
    """Frozen rules on the test outcomes (+ the val patients for R9) -> results.json, curves, per hospital."""
    res = {}
    for variant in ("deep", "offset"):
        d = oc[oc.click == variant].sort_values("cache_id").reset_index(drop=True)
        res[variant] = rules(d)
        cv = curves(d); cv.insert(0, "click", variant)
        cv.to_csv(out / f"curves_{variant}.csv", index=False)
    d = oc[oc.click == "deep"]
    pool = pd.concat([va[va.tool2 == "S"].assign(split="val"), d[d.tool2 == "S"].assign(split="test")])
    diff = pool.FM_dice - pool.S_dice
    ci = ZB.boot(diff.to_numpy())
    res["R9"] = dict(n=int(len(pool)), n_val=int((pool.split == "val").sum()), n_test=int((pool.split == "test").sum()),
                     dice_S=float(pool.S_dice.mean()), dice_F_max=float(pool.FM_dice.mean()),
                     dice_blob=float(pool.Cn_dice.mean()), diff=float(diff.mean()), ci95=ci,
                     better=int((diff > 1e-9).sum()), worse=int((diff < -1e-9).sum()),
                     passes=bool(len(pool) > 1 and ci[0] > 0))
    pool[["split", "cache_id", "Cn_dice", "S_dice", "FM_dice"]].to_csv(out / "r9_patients.csv", index=False)
    hosp = []
    for c, g in d.groupby("collection"):
        sel = V.select(g.qc.to_numpy(), g.cache_id.to_numpy(), 0.10)
        hosp.append(dict(collection=c, n=len(g), auto=float(g.B_dice.mean()),
                         qc10_agent=float(np.where(sel, np.where(g.tool2 == "S", g.S_dice, g.Cn_dice), g.B_dice).mean())))
    pd.DataFrame(hosp).to_csv(out / "per_hospital.csv", index=False)
    verdict = dict(R1=res["deep"]["R1"]["passes"], R4=res["deep"]["R4"]["passes"], R9=res["R9"]["passes"],
                   results=res)
    (out / "results.json").write_text(json.dumps(verdict, indent=1, default=float))
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["freeze", "test"], required=True)
    ap.add_argument("--s3v_out", required=True, help="outputs/s3v_review_agent_oof")
    ap.add_argument("--s3n_out", required=True)
    ap.add_argument("--s3z_out", required=True, help="outputs/s3z_fusion")
    ap.add_argument("--s3z_b", required=True, help="outputs/s3z_b (frozen R9)")
    ap.add_argument("--s3o_out", required=True, help="outputs/s3o_final")
    ap.add_argument("--s3k_out", required=True, help="outputs/s3k_radiomics (val table for the jury check)")
    ap.add_argument("--sam_ckpt", required=True, help="checkpoints/sam_s2b_best.pt")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--test_cases", default="", help="S3f component maps incl. test (lab, seg)")
    ap.add_argument("--random_weights", action="store_true", help="tests only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.step == "freeze":
        step_freeze(a, out)
    else:
        if not a.test_cases or (not a.random_weights and not a.turbo_ckpt):
            raise SystemExit("--test_cases and --turbo_ckpt are required")
        step_test(a, out)


if __name__ == "__main__":
    main()

"""S3o: freeze the final B0_rad jury, then evaluate it ONCE on the 306-patient test split.

Final pipeline (fully automatic): SegResNet-100 -> candidates (p > 0.5, >= 0.1 ml, top 5) -> B0_rad (24 clues:
7 SegResNet + 15 radiomics + rel_seg_prob + rel_ring_cnr; HistGradientBoosting; threshold chosen in inner folds)
-> union of the kept SegResNet blobs. No SAM. No new deep-learning training.

--step freeze  (test NOT read)
    Fit B0_rad and the 7-clue ablation B0 on all 1,080 out-of-fold training candidates (S3k tables, grid, seed,
    inner-fold threshold rule). Both must reproduce S3k's val decisions exactly (every patient's kept set and Dice).
    Saves frozen/{b0_rad,b0}.pkl, frozen/manifest.json (md5s, thresholds, features, git commit) and test_plan.json
    (the pre-registered test report) BEFORE the test step can run.
--step test    (runs once; refuses if the frozen files changed or a test result already exists)
    Test candidates = S3f test table + component maps (SegResNet-100, as for val). Radiomics extracted with the same
    code (cids.radiomics, RAD_VERSION). The frozen models are applied unchanged. Masks are built at 1 mm from the
    component maps; Dice / HD95 / NSD@2mm against the expert masks; masks saved for figures.

Pre-registered test report (test_plan.json)
    primary     B0_rad: mean Dice + patient-bootstrap 95% CI, median Dice, HD95 median (+ number of jointly
                non-empty pairs), mean NSD@2mm, #Dice<0.1, #Dice<0.5; per hospital (DUKE, ISPY1, ISPY2, NACT)
    ablation    B0_rad vs B0: paired mean Dice difference + CI, better / worse, rescued / harmed at 0.1 and 0.5
    context     rel20 among candidates, keep-all candidates, perfect-selection ceiling (same candidates)
    subgroup    bilateral-cancer patients reported separately (if --clinical is given)
    caveat      the 306-patient benchmark was evaluated by earlier pipeline versions (not untouched)

Usage (server)
    OMP_NUM_THREADS=1 python -u scripts/s3o_final_test.py --step freeze --s3k_out ~/CIDS/outputs/s3k_radiomics \\
        --config configs/s3i_subset_jury.json --out ~/CIDS/outputs/s3o_final
    OMP_NUM_THREADS=1 python -u scripts/s3o_final_test.py --step test --s3k_out ~/CIDS/outputs/s3k_radiomics \\
        --config configs/s3i_subset_jury.json --cache_root ~/data \\
        --test ~/CIDS/outputs/s3f_jury_masks/s3f_jury/test_features.csv \\
        --test_cases ~/CIDS/outputs/s3f_jury_masks/s3f_jury/cases --out ~/CIDS/outputs/s3o_final --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3i_subset_jury as S  # noqa: E402
import s3j_selection_tradeoff as J  # noqa: E402
import s3k_radiomics_matched as K  # noqa: E402
from cids.radiomics import RAD_ALL, RAD_VERSION  # noqa: E402

PLAN = dict(
    primary="B0_rad: mean Dice + patient-bootstrap 95% CI, median Dice, HD95 median (+ n jointly non-empty), mean "
            "NSD@2mm, #Dice<0.1, #Dice<0.5; per hospital",
    ablation="B0_rad vs B0 (7 finder clues): paired mean Dice difference + 95% CI, better/worse, rescued/harmed at "
             "0.1 and 0.5",
    context=["rel20 among candidates", "keep-all candidates", "perfect-selection ceiling"],
    subgroup="bilateral-cancer patients reported separately when --clinical is given",
    bootstrap=5000, seed=0,
    caveat="the 306-patient benchmark was evaluated by earlier pipeline versions (jury13_rad_rel etc.)")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def features(C):
    return dict(B0_rad=list(C["finder_features"]) + RAD_ALL + K.REL2, B0=list(C["finder_features"]))


# ---------------------------------------------------------------------------------------------- freeze
def step_freeze(a, C, out):
    src = Path(a.s3k_out).expanduser()
    F = features(C)
    train = J.load_patients(src / "train_oof_table.csv", F["B0_rad"], C["forbidden_features"])
    val = J.load_patients(src / "val_table.csv", F["B0_rad"], C["forbidden_features"])
    pp = pd.read_csv(src / "per_patient.csv", dtype={"B0_rad_subset": str, "B0_subset": str})
    ref = pp[pp.split == "val"].set_index("cache_id")
    fz = out / "frozen"; fz.mkdir(parents=True, exist_ok=True)
    man = dict(script="scripts/s3o_final_test.py", git_commit=git_commit(), rad_version=RAD_VERSION, config=C,
               inputs={n: dict(path=str(src / n), md5=md5(src / n))
                       for n in ("train_oof_table.csv", "val_table.csv", "per_patient.csv")}, models={})
    for name in ("B0_rad", "B0"):
        arm, model, t, log = K.fit_arm(name, F[name], train, C, C["seed"])
        bad, d = [], []
        for p in val:
            m = J.decide_threshold(arm, model, p, t)
            dd = S.dice_counts(p["tgt"], m)
            d.append(dd)
            sub = ref.loc[p["cid"], f"{name}_subset"]
            sub = "" if not isinstance(sub, str) or sub == "nan" else sub
            if S.mask_str(p, m) != sub or abs(dd - ref.loc[p["cid"], f"{name}_dice"]) > 1e-9:
                bad.append(p["cid"])
        if bad:
            raise SystemExit(f"{name}: val decisions differ from S3k for {len(bad)} patients (e.g. {bad[:5]})")
        f = fz / f"{name.lower()}.pkl"
        f.write_bytes(pickle.dumps(dict(arm=arm, model=model, threshold=t, features=F[name], log=log)))
        man["models"][name] = dict(file=f.name, md5=md5(f), threshold=float(t), features=F[name], log=log,
                                   val_dice=float(np.mean(d)), val_n=len(val))
        print(f"{name}: t={t} | val Dice {np.mean(d):.4f} reproduces S3k for all {len(val)} patients", flush=True)
    (fz / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
    (out / "test_plan.json").write_text(json.dumps(PLAN, indent=1))
    print(f"FROZEN -> {fz} (manifest + test_plan.json written; the test split was not read)")


# ---------------------------------------------------------------------------------------------- test
def oracle_dice(I, V, G):
    """Best Dice over subsets of disjoint candidates (prefix of the precision order is optimal)."""
    if G == 0:
        return np.nan
    if not len(I):
        return 0.0
    o = np.argsort(-(np.asarray(I) / np.maximum(np.asarray(V), 1)))
    ci, cv = np.cumsum(np.asarray(I)[o]), np.cumsum(np.asarray(V)[o])
    return float(max(0.0, (2 * ci / (cv + G)).max()))


def metric_job(job):
    from cids.data.cache import load_case
    from cids.eval.metrics import hd95, nsd
    cid, cache_path, case_path, kept, mask_dir = job
    _, lbl = load_case(cache_path)
    gt = lbl > 0
    lab = np.load(case_path)["lab"]
    if lab.shape != gt.shape:
        raise ValueError(f"{cid}: component map {lab.shape} vs label {gt.shape}")
    out = dict(cache_id=cid)
    for pol, ks in kept.items():
        m = np.isin(lab, ks) if ks else np.zeros(gt.shape, bool)
        s = int(m.sum()) + int(gt.sum())
        out[f"{pol}_dice_vox"] = 0.0 if s == 0 else 2.0 * int((m & gt).sum()) / s
        out[f"{pol}_hd95"] = hd95(m, gt)
        out[f"{pol}_nsd"] = nsd(m, gt, 2.0)
        out[f"{pol}_ml"] = float(m.sum()) / 1000
        if pol == "B0_rad" and mask_dir:
            np.savez_compressed(Path(mask_dir) / f"{cid}.npz", mask=m.astype(np.uint8), kept=np.array(ks, np.int32))
    out["gt_ml"] = float(gt.sum()) / 1000
    return out


def boot_mean(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(df, pol, label):
    d = df[f"{pol}_dice"]
    ne = df[f"{pol}_hd95"].notna()
    lo, hi = boot_mean(d, PLAN["bootstrap"], PLAN["seed"])
    return dict(group=label, policy=pol, n=len(df), dice=d.mean(), dice_ci_lo=lo, dice_ci_hi=hi, dice_median=d.median(),
                hd95_median=df.loc[ne, f"{pol}_hd95"].median(), hd95_n=int(ne.sum()), nsd=df[f"{pol}_nsd"].mean(),
                lt01=int((d < 0.1).sum()), lt05=int((d < 0.5).sum()))


def step_test(a, C, out):
    from cids.data.cache import find_cache_root
    from cids.provenance import require_cases
    fz = out / "frozen"
    man_f, plan_f = fz / "manifest.json", out / "test_plan.json"
    if not man_f.exists() or not plan_f.exists():
        raise SystemExit("run --step freeze first (frozen models and test_plan.json must exist before the test)")
    td = out / "test"
    if (td / "done.json").exists():
        raise SystemExit(f"a test result already exists in {td}: the test split is evaluated once")
    man = json.loads(man_f.read_text())
    models = {}
    for name, m in man["models"].items():
        f = fz / m["file"]
        if md5(f) != m["md5"]:
            raise SystemExit(f"{f} changed after freezing (md5 mismatch)")
        models[name] = pickle.loads(f.read_bytes())
    td.mkdir(exist_ok=True); (td / "masks").mkdir(exist_ok=True)
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    paths = dict(require_cases(sp, root, "test"))
    dev = set(sp[sp.split.isin(["train", "val"])].cache_id)
    df = pd.read_csv(Path(a.test).expanduser())
    if set(df.cache_id) - set(paths) or set(df.cache_id) & dev:
        raise SystemExit("the test table contains patients outside the test split")
    if set(paths) - set(df.cache_id):
        raise SystemExit(f"{len(set(paths) - set(df.cache_id))} test patients have no candidate rows")
    cdir = K.cases_dir(Path(a.test_cases).expanduser())
    rad = K.extract("test", df, paths, cdir, td, a.workers, md5(Path(a.test).expanduser()))
    t = K.assemble(df, rad)
    t.to_csv(td / "test_table.csv", index=False)
    F = features(C)
    pats = J.load_patients(td / "test_table.csv", F["B0_rad"], C["forbidden_features"])
    print(f"S3o test | {len(pats)} patients (K=0: {sum(p['K'] == 0 for p in pats)}) | RAD_VERSION {RAD_VERSION}",
          flush=True)
    rows, jobs = [], []
    for p in pats:
        dec = {n: J.decide_threshold(m["arm"], m["model"], p, m["threshold"]) for n, m in models.items()}
        dec["rel20"] = J.rel20(p)
        dec["keep_all"] = J.keep_all(p)
        r = dict(cache_id=p["cid"], collection=p["coll"], K=p["K"],
                 oracle_dice=oracle_dice(p["tgt"]["I"], p["tgt"]["V"], p["tgt"]["G"]))
        for n, m in dec.items():
            r[f"{n}_dice"] = S.dice_counts(p["tgt"], m)
            r[f"{n}_subset"] = S.mask_str(p, m)
        rows.append(r)
        jobs.append((p["cid"], str(paths[p["cid"]]), str(cdir / f"{p['cid']}.npz"),
                     {n: [int(k) for k in p["k"][m]] for n, m in dec.items()}, str(td / "masks")))
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        met = []
        for i, r in enumerate(pool.imap_unordered(metric_job, jobs, chunksize=2)):
            met.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"  metrics {i + 1}/{len(jobs)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    res = pd.DataFrame(rows).merge(pd.DataFrame(met), on="cache_id")
    pols = ["B0_rad", "B0", "rel20", "keep_all"]
    for pol in pols:                                 # table-count Dice must equal voxel Dice on the rebuilt masks
        dmax = (res[f"{pol}_dice"] - res[f"{pol}_dice_vox"]).abs().max()
        if dmax > 1e-9:
            raise SystemExit(f"{pol}: candidate-table Dice differs from the rebuilt mask Dice (max {dmax})")
    if a.clinical:
        cl = pd.read_excel(Path(a.clinical).expanduser())
        bil = set(cl.loc[cl.bilateral_breast_cancer.fillna(0).astype(int) == 1, "patient_id"].str.lower())
        res["bilateral_cancer"] = res.cache_id.isin(bil)
    res.to_csv(td / "per_patient.csv", index=False)
    summ = []
    for pol in pols:
        summ.append(summarise(res, pol, "all"))
        for c, g in res.groupby("collection"):
            summ.append(summarise(g, pol, c))
        if "bilateral_cancer" in res:
            summ.append(summarise(res[~res.bilateral_cancer], pol, "excluding bilateral cancer"))
            if res.bilateral_cancer.any():
                summ.append(summarise(res[res.bilateral_cancer], pol, "bilateral cancer only"))
    s = pd.DataFrame(summ)
    s.to_csv(td / "summary.csv", index=False)
    da, db = res.B0_dice.to_numpy(), res.B0_rad_dice.to_numpy()
    lo, hi = boot_mean(db - da, PLAN["bootstrap"], PLAN["seed"])
    abl = dict(mean_diff=float((db - da).mean()), ci95=[lo, hi], better=int((db > da + 1e-9).sum()),
               worse=int((db < da - 1e-9).sum()), identical_masks=int((res.B0_subset == res.B0_rad_subset).sum()),
               rescued_lt01=int(((da < 0.1) & (db >= 0.1)).sum()), harmed_lt01=int(((da >= 0.1) & (db < 0.1)).sum()),
               rescued_lt05=int(((da < 0.5) & (db >= 0.5)).sum()), harmed_lt05=int(((da >= 0.5) & (db < 0.5)).sum()),
               oracle_dice=float(res.oracle_dice.mean()))
    (td / "ablation.json").write_text(json.dumps(abl, indent=1))
    (td / "done.json").write_text(json.dumps(dict(per_patient_md5=md5(td / "per_patient.csv"), manifest_md5=md5(man_f),
                                                  plan_md5=md5(plan_f), git_commit=git_commit(), n=len(res)), indent=1))
    pd.set_option("display.width", 220)
    print("\n" + s.round(4).to_string(index=False))
    print("\nablation B0_rad - B0 (paired, patient bootstrap):", json.dumps(abl, indent=1))
    print(f"\nTEST DONE (evaluated once) -> {td}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["freeze", "test"], required=True)
    ap.add_argument("--s3k_out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--test", default="", help="S3f test_features.csv")
    ap.add_argument("--test_cases", default="", help="S3f cases/ (component maps)")
    ap.add_argument("--clinical", default="", help="optional clinical_and_imaging_info.xlsx (bilateral-cancer flag)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    C = json.loads(Path(a.config).read_text())
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.step == "freeze":
        step_freeze(a, C, out)
    else:
        if not (a.test and a.test_cases):
            raise SystemExit("--test and --test_cases are required for --step test")
        step_test(a, C, out)


if __name__ == "__main__":
    main()

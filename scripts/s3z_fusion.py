"""S3z: fuse SAM and SegResNet at the click instead of picking one of them (one-click review, out-of-fold).

Same patients, same simulated clicks and same out-of-fold S2b SAM decoders as S3v (outputs/s3v_review_agent_oof);
only what is done with the two tools' outputs changes. No expert information is used to build any mask. Test refused.

Masks at a click p (1 mm grid; each = the 26-connected component containing p, else the nearest one, as S3u / S3v)
  S      SAM one click: P_sam > 0.5 (parity with S3v's S_dice)
  F_avg  (P_sam + P_seg) / 2 > 0.5   -- a voxel is kept where the two tools are confident together
  F_max  max(P_sam, P_seg) > 0.5     -- a voxel is kept where either tool is confident
  P_sam = SAM-S2b sigmoid on the S3v crop (128^3 post1 at 1.5 mm centred on the click) pasted back to 1 mm
          (0 outside the crop); P_seg = SegResNet's out-of-fold probability map (S3g / S3f 'seg' / 255).
Agents (stage 2, after the click; gradient boosting as S3v, cross-fitted on the S3k outer folds for train OOF, fitted
on all train for val): one regressor per tool t predicts g_t = Dice(t) - Dice(Cn); the tool with the largest
prediction is used if it is > 0, else Cn.
  A2       {Cn, S} with the S3v POST features (= the S3v agent, re-fitted here; parity with S3v's tool2)
  A2_ext   {Cn, S} with POST + the fusion features (control: do the extra features alone help?)
  A4       {Cn, S, F_avg, F_max} with POST + the fusion features
  FUS = fusion features: fa_log_ml, fm_log_ml, dice_fa_cn, dice_fm_cn, dice_fa_s, dice_fm_s

Frozen rules (decision_rule.json written before any SAM call; deep click; train OOF)
  R7 (primary)  at the 10% QC budget (S3n risk, as S3v), A4 beats A2 on mean Dice over all 1,080 patients:
                patient-bootstrap 95% CI lower bound of the paired difference > 0
  R7a           on the 108 QC-clicked patients, F_avg alone beats A2: CI lower bound > 0
  reported      F_max alone; A2_ext (feature control); the fixed 30 patients the S3v-OOF agent sent to SAM;
                failures (Dice < 0.1, < 0.5); val; offset-click sensitivity; how often A4 picks each tool.

Steps
  --step masks    GPU: S / F_avg / F_max for every patient and both S3v clicks -> detail.csv
  --step analyse  CPU: agents + rules -> summary.csv, choices_<split>.csv, rules.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3m_contour_study as M  # noqa: E402
import s3u_click_biomarkers as U  # noqa: E402
import s3v_review_agent as V  # noqa: E402

MASKS = ["S", "FA", "FM"]                      # S, F_avg, F_max
FUS = ["fa_log_ml", "fm_log_ml", "dice_fa_cn", "dice_fm_cn", "dice_fa_s", "dice_fm_s"]
TOOLS4 = ["S", "FA", "FM"]                     # alternatives to Cn
BUDGET = 0.10
RULE = dict(
    cohort="S3v patients and clicks (train OOF 1080 with out-of-fold S2b decoders, val 120); test refused",
    masks=dict(S="P_sam > 0.5", F_avg="(P_sam + P_seg) / 2 > 0.5", F_max="max(P_sam, P_seg) > 0.5",
               component="26-connected component at the click, else the nearest one"),
    inputs="P_sam = SAM-S2b sigmoid on the S3v crop pasted to 1 mm; P_seg = SegResNet out-of-fold probability",
    agents=dict(A2="{Cn,S}, POST features (S3v)", A2_ext="{Cn,S}, POST + FUS", A4="{Cn,S,F_avg,F_max}, POST + FUS",
                rule="per tool regress Dice(t) - Dice(Cn); use argmax if > 0 else Cn; S3v gradient boosting, "
                     "cross-fitted on S3k outer folds (train), all train (val)"),
    features=dict(POST=V.POST, FUS=FUS),
    R7="PRIMARY: 10% QC budget, A4 - A2 mean Dice over all train OOF patients, bootstrap CI lower bound > 0",
    R7a="108 QC-clicked train OOF patients: F_avg - A2 mean Dice, bootstrap CI lower bound > 0",
    bootstrap=dict(n=5000, seed=0))


# --------------------------------------------------------------------------------------------- masks
def sam_prob(img, p, rn):
    """SAM-S2b one-click probability (1 mm grid, 0 outside the crop) -- the same computation as U.sam_one_click."""
    import torch
    import s3a_seg2sam as s3
    from cids.models.sam_lora import decode, upsample
    img15, _ = s3.resample(img, np.zeros(img.shape[1:], np.uint8))
    shape15 = img15.shape[1:]
    c15 = np.clip(M.to_crop(p, np.zeros(3, int)), 0, np.asarray(shape15) - 1)
    post, fg = img15[1], (img15[1] > 0).astype(np.uint8)
    vc, origin = s3.crop_pad(post, c15)
    fc, _ = s3.crop_pad(fg, c15)
    crop = s3.znorm(vc, fc)
    pc = np.clip(c15 - origin, 0, s3.SAM_SIZE - 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=rn.amp):
        x = torch.from_numpy(np.ascontiguousarray(crop[None, None])).float().to(rn.dev)
        emb = rn.sam.image_encoder(x).float()
        co = torch.tensor(np.asarray([pc], np.float32)[:, None], device=rn.dev)
        lb = torch.ones(1, 1, dtype=torch.long, device=rn.dev)
        low = decode(rn.sam, rn.box, emb, co, lb, None, None)
        prob = torch.sigmoid(upsample(low.float()))[0, 0].cpu().numpy()
    return M.paste_back(prob, origin, shape15, img.shape[1:])


def fuse(p_sam, p_seg, p):
    """S / F_avg / F_max masks at click p from the two probability maps (same grid)."""
    return dict(S=U.component_at(p_sam > 0.5, p),
                FA=U.component_at((p_sam + p_seg) / 2 > 0.5, p),
                FM=U.component_at(np.maximum(p_sam, p_seg) > 0.5, p))


def step_masks(a, out):
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any SAM call
    s3v = Path(a.s3v_out).expanduser()
    oc = pd.read_csv(s3v / "outcomes.csv")
    if "sam_decoder" not in oc or not oc[oc.split == "train_oof"].sam_decoder.astype(str).str.startswith("fold").all():
        raise SystemExit("--s3v_out must be the OUT-OF-FOLD S3v run (outputs/s3v_review_agent_oof)")
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    paths, cdirs = {}, {}
    for split, cache_split, cdir in (("train_oof", "train", a.train_cases), ("val", "val", a.val_cases)):
        d = Path(cdir).expanduser()
        d = d if any(d.glob("*.npz")) else next(d.rglob("*.npz")).parent
        for cid, path in require_cases(sp, root, cache_split):
            paths[(split, cid)], cdirs[(split, cid)] = path, d
    if a.random_weights:
        import torch
        torch.manual_seed(0)
    runner = M.load_runner(a.sam_ckpt, a.turbo_ckpt, a.random_weights)
    fold_of = dict(pd.read_csv(Path(a.folds).expanduser()).values)
    fold_runner = {}
    for k in sorted(set(fold_of.values())):
        ck = Path(a.sam_fold_ckpt.format(fold=k)).expanduser()
        if not a.random_weights and not ck.exists():
            raise SystemExit(f"missing fold checkpoint {ck}")
        fold_runner[k] = M.load_runner(str(ck), a.turbo_ckpt, a.random_weights)
    pats = oc[["split", "cache_id"]].drop_duplicates().values.tolist()
    if a.limit:
        pats = [x for s in ("train_oof", "val") for x in [q for q in pats if q[0] == s][:a.limit]]
    part = out / "detail.partial.csv"
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {(r["split"], r["cache_id"]) for r in rows}
    todo = [q for q in pats if tuple(q) not in done]
    print(f"S3z | {len(pats)} patients ({len(pats) - len(todo)} already done) | masks S, F_avg, F_max", flush=True)
    t0 = time.time()
    for i, (split, cid) in enumerate(todo):
        img, gt = load_case(paths[(split, cid)])
        gt = gt > 0
        z = np.load(cdirs[(split, cid)] / f"{cid}.npz")
        p_seg, lab = z["seg"].astype(np.float32) / 255.0, z["lab"]
        if p_seg.shape != gt.shape:
            raise SystemExit(f"{split} {cid}: SegResNet map {p_seg.shape} != image {gt.shape}")
        rn = fold_runner[fold_of[cid]] if split == "train_oof" else runner
        for _, r in oc[(oc.split == split) & (oc.cache_id == cid)].iterrows():
            p = tuple(int(v) for v in json.loads(r.click_xyz))
            ms = fuse(sam_prob(img, p, rn), p_seg, p)
            Cn, _ = U.nearest_blob(lab, p)
            row = dict(split=split, cache_id=cid, click=r.click, S3v_S_dice=r.S_dice)
            for n, m in ms.items():
                row[f"{n}_dice"] = M.dice_count(m, gt)
                for q, val in V.measurements(img, m).items():
                    row[f"{n}_{q}"] = val
                if n != "S":                                    # S's volume / agreement are S3v POST features
                    row[f"{n.lower()}_log_ml"] = float(np.log(m.sum() / 1000 + 1e-3))
                    row[f"dice_{n.lower()}_cn"] = M.dice_count(m, Cn)
            row["dice_fa_s"], row["dice_fm_s"] = M.dice_count(ms["FA"], ms["S"]), M.dice_count(ms["FM"], ms["S"])
            rows.append(row)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"  {i + 1}/{len(todo)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    det = pd.DataFrame(rows)
    det.to_csv(out / "detail.csv", index=False)
    if part.exists():
        part.unlink()
    diff = (det.S_dice - det.S3v_S_dice).abs()
    print(f"parity S vs S3v S_dice: mean |diff| {diff.mean():.5f}, max {diff.max():.4f}, "
          f"> 0.01 in {int((diff > 0.01).sum())} of {len(det)}", flush=True)
    if diff.mean() > 0.005:
        raise SystemExit("PARITY FAILED: S does not reproduce S3v's one-click SAM")
    print(det.groupby(["split", "click"])[[f"{n}_dice" for n in MASKS]].mean().round(4).to_string())
    print("MASKS DONE -> --step analyse")


# --------------------------------------------------------------------------------------------- analyse
def choose(preds: pd.DataFrame) -> np.ndarray:
    """Per row: the tool (column) with the largest predicted gain over Cn if > 0, else 'Cn'."""
    best = preds.idxmax(axis=1).to_numpy()
    return np.where(preds.max(axis=1).to_numpy() > 0, best, "Cn")


def fit_agent(tr, te, tools, cols, folds):
    """Cross-fitted predictions on tr (train OOF), all-train fit on te (val / offset). Returns (tool_tr, tool_te)."""
    ptr, pte = {}, {}
    for t in tools:
        tgt = f"g_{t}"
        ptr[t] = V.crossfit(tr, cols, tgt, folds).to_numpy()
        if te is not None:
            pte[t] = V.gbr().fit(tr[cols].fillna(-1), tr[tgt]).predict(te[cols].fillna(-1))
    return choose(pd.DataFrame(ptr)), (choose(pd.DataFrame(pte)) if te is not None else None)


def tool_dice(d, tool):
    tool = np.asarray(tool)
    out = d.Cn_dice.to_numpy().copy()
    for t in TOOLS4:
        out = np.where(tool == t, d[f"{t}_dice"].to_numpy(), out)
    return out


def boot(x, n, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    v = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def step_analyse(a, out):
    det = pd.read_csv(out / "detail.csv")
    s3v = Path(a.s3v_out).expanduser()
    folds = dict(pd.read_csv(Path(a.folds).expanduser()).values)
    n, seed = RULE["bootstrap"]["n"], RULE["bootstrap"]["seed"]
    data = {}
    for split in ("train_oof", "val"):
        for click in ("deep", "offset"):
            ch = pd.read_csv(s3v / f"choices_{split}_{click}.csv")
            dc = det[(det.split == split) & (det.click == click)].drop(columns=["split", "click"])
            ch = ch.drop(columns=[c for c in dc.columns if c != "cache_id" and c in ch.columns])
            d = ch.merge(dc, on="cache_id", how="inner",
                         validate="one_to_one").sort_values("cache_id").reset_index(drop=True)
            if len(d) != len(ch):
                raise SystemExit(f"{split} {click}: S3z detail does not cover every S3v patient")
            if (d.S_dice - d.S3v_S_dice).abs().mean() > 0.005:
                raise SystemExit(f"{split} {click}: parity with S3v broken")
            for t in TOOLS4:
                d[f"g_{t}"] = d[f"{t}_dice"] - d.Cn_dice
            data[(split, click)] = d
    fixed30 = set(data[("train_oof", "deep")].query("tool2 == 'S'").cache_id)
    agents = dict(A2=(["S"], V.POST), A2_ext=(["S"], V.POST + FUS), A4=(TOOLS4, V.POST + FUS))
    tr = data[("train_oof", "deep")]
    for name, (tools, cols) in agents.items():
        t_tr, t_va = fit_agent(tr, data[("val", "deep")], tools, cols, folds)
        tr[name] = t_tr
        data[("val", "deep")][name] = t_va
        off = data[("train_oof", "offset")]
        off[name], data[("val", "offset")][name] = fit_agent(off, data[("val", "offset")], tools, cols, folds)
    a2_parity = float((tr.A2 == tr.tool2).mean())
    rows, rules = [], {}
    for (split, click), d in data.items():
        sel = V.select(d.qc.to_numpy(), d.cache_id.to_numpy(), BUDGET)
        final = {"no review": d.B_dice.to_numpy(), "Cn": np.where(sel, d.Cn_dice, d.B_dice)}
        for t, lab in (("S", "always S"), ("FA", "always F_avg"), ("FM", "always F_max")):
            final[lab] = np.where(sel, d[f"{t}_dice"], d.B_dice)
        for name in agents:
            final[name] = np.where(sel, tool_dice(d, d[name]), d.B_dice)
        for pol, v in final.items():
            r = dict(split=split, click=click, policy=pol, mean_dice=float(v.mean()), fail_01=int((v < 0.1).sum()),
                     fail_05=int((v < 0.5).sum()), clicked_dice=float(v[sel].mean()))
            if pol in agents:
                r.update({f"n_{t}": int(((d[pol] == t) & sel).sum()) for t in ["Cn"] + TOOLS4})
            rows.append(r)
        if click != "deep":
            continue
        a4, a2 = final["A4"], final["A2"]
        dd = a4 - a2
        lo, hi = boot(dd, n, seed)
        fa = d.FA_dice.to_numpy()[sel] - tool_dice(d, d.A2)[sel]
        lo_a, hi_a = boot(fa, n, seed)
        fm = d.FM_dice.to_numpy()[sel] - tool_dice(d, d.A2)[sel]
        ext = final["A2_ext"] - a2
        f30 = d.cache_id.isin(fixed30).to_numpy()
        rules[split] = dict(
            R7=dict(A4_dice=float(a4.mean()), A2_dice=float(a2.mean()), diff=float(dd.mean()), ci95=[lo, hi],
                    better=int((dd > 1e-9).sum()), worse=int((dd < -1e-9).sum()), passes=bool(lo > 0)),
            R7a=dict(n_clicked=int(sel.sum()), F_avg_dice=float(d.FA_dice[sel].mean()),
                     A2_dice=float(tool_dice(d, d.A2)[sel].mean()), diff=float(fa.mean()), ci95=[lo_a, hi_a],
                     passes=bool(lo_a > 0)),
            F_max_vs_A2_clicked=dict(diff=float(fm.mean()), ci95=list(boot(fm, n, seed))),
            A2_ext_minus_A2=dict(diff=float(ext.mean()), ci95=list(boot(ext, n, seed))),
            fixed30=({t: float(d.loc[f30, f"{t}_dice"].mean()) for t in ["Cn"] + TOOLS4}
                     if split == "train_oof" else None))
    t = pd.DataFrame(rows)
    verdict = dict(R7_primary=rules["train_oof"]["R7"]["passes"], R7a=rules["train_oof"]["R7a"]["passes"],
                   A2_matches_S3v_tool2=a2_parity, rules=rules)
    t.to_csv(out / "summary.csv", index=False)
    for (split, click), d in data.items():
        d.to_csv(out / f"choices_{split}_{click}.csv", index=False)
    (out / "rules.json").write_text(json.dumps(verdict, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(f"10% QC budget (A2 reproduces the S3v agent's choice for {a2_parity:.1%} of train patients)")
    print(t.round(4).fillna("").to_string(index=False))
    print("\nRULES:", json.dumps(verdict, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["masks", "analyse"], required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--s3v_out", required=True, help="the OUT-OF-FOLD S3v run (outcomes.csv, choices_*.csv)")
    ap.add_argument("--train_cases", default="", help="S3g OOF maps (seg, lab)")
    ap.add_argument("--val_cases", default="", help="S3f val maps (seg, lab)")
    ap.add_argument("--sam_ckpt", default="")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--sam_fold_ckpt", default="", help=".../s2b_folds/f{fold}/best.pt")
    ap.add_argument("--folds", required=True, help="S3k folds.csv")
    ap.add_argument("--random_weights", action="store_true", help="tests only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for f in (a.train_cases, a.val_cases, a.s3v_out):
        if "test" in Path(f).name.lower():
            raise SystemExit("S3z is a development study: test files are refused")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.step == "masks":
        if not (a.train_cases and a.val_cases and a.sam_fold_ckpt):
            raise SystemExit("--train_cases, --val_cases and --sam_fold_ckpt are required")
        if not a.random_weights and not (a.sam_ckpt and a.turbo_ckpt):
            raise SystemExit("--sam_ckpt and --turbo_ckpt are required")
        step_masks(a, out)
    else:
        if not (out / "detail.csv").exists():
            raise SystemExit("run --step masks first")
        step_analyse(a, out)


if __name__ == "__main__":
    main()

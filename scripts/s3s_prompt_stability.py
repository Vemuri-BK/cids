"""S3s pre-check: is SAM's mask STABLE across prompts on real tumours and UNSTABLE on false candidates?

Idea (adapted from multi-prompt consistency, Zhang et al., IEEE JBHI 2026, which asks two text prompts of the
same target to give the same mask): prompt SAM several different ways on the same candidate and measure how much
the masks agree. A real tumour should be the same object however it is prompted; a vessel or background
enhancement should not. Used here as an automatic test-time clue, never with the expert mask.

Development only: train out-of-fold candidates (S3g), all of them. Test refused.

Per candidate: one 128^3 post1 crop at 1.5 mm (s3a.candidate_crops, exactly the pipeline crop), ONE encoder pass
(SAM-Med3D-turbo image encoder, frozen; shared by both decoders), then two prompt sets, every prompt with the
zero mask prompt (independent prompts, no chaining):
  turbo (PRIMARY, decoder never fitted to MAMA-MIA, point prompts only; turbo has no box prompt):
      5 single positive points: the deepest interior voxel of the candidate (1.5 mm) + 4 interior voxels drawn
      with a fixed seed per (cache_id, k) from voxels at >= half the maximal depth.
  s2b (SECONDARY, LoRA decoder + box module TRAINED on these training patients -> possibly inflated on tumours):
      7 prompts: box, box shrunk 15% per side, box grown 15% per side, box shifted +2 and -2 voxels on all axes,
      deepest point alone, box + deepest point.
Features (crop space, mask = logit > 0 after trilinear upsampling to 128^3):
  <set>_stab       mean pairwise Dice over the prompt masks (a pair of empty masks counts 0: no support)
  <set>_comp_dice  mean Dice of the prompt masks with the candidate component (agreement with SegResNet)
  <set>_ml_cv      coefficient of variation of the mask volumes;  <set>_empty_frac  share of empty masks

Frozen gate (written to decision_rule.json before any SAM call), train OOF:
  population  B0_rad wrong picks = tumour among the candidates, B0_rad Dice < 0.1 (46 in S3k)
  hits(score) = patients whose highest-scoring candidate overlaps the tumour (ties -> larger candidate)
  PASS (turbo_stab) iff hits >= max(ceil(0.5 * n_wrong), best non-SAM hits + 5)
                    AND, among patients B0_rad already gets right (Dice >= 0.1, tumour present), the top
                        turbo_stab candidate is NOT the tumour in <= 100 of them
                        (S3r: SAM memory had 32/46 hits but 173 elsewhere and did not help the jury)
  non-SAM picks: finder anchor (seg_prob_mean), seg_prob_max, largest. SAM references reported: agree, mem_knn.
  s2b_stab is judged by the same rule but reported only (training-patient leakage).
  PASS -> full matched S3s (B0_rad vs + stability vs + non-SAM SegResNet test-time-augmentation stability).
  FAIL -> stop.

Steps
  --step precheck  GPU, ~2,700 candidates (resumable: partial file every 100 patients)
  --step gate      CPU, reads <out>/candidates_stability.csv
Example (server)
  python -u scripts/s3s_prompt_stability.py --step precheck --cache_root ~/data \\
      --s3k_out ~/CIDS/outputs/s3k_radiomics --train_cases ~/CIDS/outputs/s3g_oof_masks/cases \\
      --sam_ckpt ~/CIDS/checkpoints/sam_s2b_best.pt --turbo_ckpt ~/CIDS/checkpoints/sam_med3d_turbo.pth \\
      --out ~/CIDS/outputs/s3s_precheck
  python scripts/s3s_prompt_stability.py --step gate --s3k_out ~/CIDS/outputs/s3k_radiomics \\
      --memory_csv ~/CIDS/outputs/s3r_sam_memory/train_oof_memory_features.csv --out ~/CIDS/outputs/s3s_precheck
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

N = 128
CFG = dict(n_turbo_points=5, depth_frac=0.5, box_scale=0.15, box_shift=2, empty_pair=0.0)
RULE = dict(
    population="train OOF B0_rad wrong picks: tumour among candidates and B0_rad Dice < 0.1",
    primary_score="turbo_stab", secondary_score="s2b_stab (reported only: S2b decoder trained on these patients)",
    hits="patients whose top-scoring candidate overlaps the tumour (ties -> larger candidate)",
    pass_hits="hits >= max(ceil(0.5 * n_wrong), best non-SAM hits + 5)",
    pass_harm="top candidate is not the tumour in <= 100 of the B0_rad-correct patients",
    harm_max=100, non_sam=["seg_prob_mean", "seg_prob_max", "size_vox"], sam_reference=["agree", "mem_knn"],
    next_if_pass="full matched S3s vs non-SAM SegResNet TTA-stability control", config=CFG)
FEATS = ["stab", "comp_dice", "ml_cv", "empty_frac"]


# --------------------------------------------------------------------------------------------- prompts
def deepest_point(comp):
    """Deepest interior voxel of a (1.5 mm, crop) component; ties -> nearest to its centroid, then lexicographic.
    Empty component -> None."""
    if not comp.any():
        return None, None
    dt = ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1, 1:-1]
    cand = np.argwhere(dt == dt.max())
    c = np.argwhere(comp).mean(0)
    d = np.linalg.norm(cand - c, axis=1)
    best = cand[d == d.min()]
    return tuple(int(v) for v in sorted(map(tuple, best))[0]), dt


def interior_points(comp, seed_key, n, frac=CFG["depth_frac"]):
    """Deepest point + (n - 1) voxels drawn (fixed seed per candidate) from voxels at >= frac * max depth."""
    p0, dt = deepest_point(comp)
    if p0 is None:
        return []
    pool = np.argwhere(dt >= max(1.0, frac * dt.max()))
    pool = pool[~(pool == np.asarray(p0)).all(1)]
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
    pick = pool[rng.choice(len(pool), size=min(n - 1, len(pool)), replace=False)] if len(pool) else np.zeros((0, 3))
    pts = [p0] + [tuple(int(v) for v in q) for q in pick]
    while len(pts) < n:                                                  # tiny blob: repeat the deepest point
        pts.append(p0)
    return pts


def box_variants(box, scale=CFG["box_scale"], shift=CFG["box_shift"]):
    """box (2,3) [lo, hi] in crop voxels -> [box, shrunk, grown, shifted +, shifted -], clipped to the crop."""
    lo, hi = np.asarray(box, float)
    ext = hi - lo + 1
    m = np.maximum(1.0, np.round(scale * ext))
    out = [(lo, hi), (lo + m, hi - m), (lo - m, hi + m), (lo + shift, hi + shift), (lo - shift, hi - shift)]
    res = []
    for a, b in out:
        a, b = np.clip(a, 0, N - 1), np.clip(b, 0, N - 1)
        if (a > b).any():                                               # shrink collapsed a thin axis
            mid = np.clip(np.floor((a + b) / 2), 0, N - 1)
            a, b = np.where(a > b, mid, a), np.where(a > b, mid, b)
        res.append(np.stack([a, b]))
    return np.stack(res)


# --------------------------------------------------------------------------------------------- stability
def mask_stats(masks, comp):
    """masks (n, ...) bool numpy or torch; comp (...) bool. Returns stab, comp_dice, ml_cv, empty_frac."""
    import torch
    m = torch.as_tensor(masks).reshape(len(masks), -1).float()
    c = torch.as_tensor(comp, device=m.device).reshape(-1).float()
    s = m.sum(1)
    inter = m @ m.T
    den = s[:, None] + s[None, :]
    dice = torch.where(den > 0, 2 * inter / den.clamp(min=1), torch.full_like(den, CFG["empty_pair"]))
    iu = torch.triu_indices(len(m), len(m), 1)
    stab = float(dice[iu[0], iu[1]].mean()) if len(m) > 1 else float("nan")
    cs = c.sum()
    cd = torch.where(s + cs > 0, 2 * (m @ c) / (s + cs).clamp(min=1), torch.zeros_like(s))
    ml = s.cpu().numpy() * 1.5 ** 3 / 1000
    cv = float(ml.std() / ml.mean()) if ml.mean() > 0 else 0.0
    return dict(stab=stab, comp_dice=float(cd.mean()), ml_cv=cv, empty_frac=float((s == 0).float().mean()))


class Prompter:
    """One encoder pass per crop; turbo point set and S2b box/point set; masks stay on the device."""

    def __init__(self, turbo, s2b, box_mod, dev, amp):
        self.turbo, self.s2b, self.box, self.dev, self.amp = turbo, s2b, box_mod, dev, amp

    def __call__(self, crop, box, comp, seed_key):
        import torch
        from cids.models.sam_lora import decode, upsample
        pts = interior_points(comp, seed_key, CFG["n_turbo_points"])
        if not pts:                                                     # component vanished at 1.5 mm
            c = tuple(int(v) for v in ((np.asarray(box[0]) + np.asarray(box[1])) // 2))
            pts = [c] * CFG["n_turbo_points"]
        dp = pts[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            x = torch.from_numpy(np.ascontiguousarray(crop[None, None])).float().to(self.dev)
            emb = self.turbo.image_encoder(x).float()
            n = len(pts)
            co = torch.tensor(np.asarray(pts, np.float32)[:, None], device=self.dev)            # (n,1,3)
            lb = torch.ones(n, 1, dtype=torch.long, device=self.dev)
            low_t = decode(self.turbo, None, emb.expand(n, -1, -1, -1, -1), co, lb, None, None)
            bv = torch.tensor(box_variants(box), dtype=torch.float32, device=self.dev)        # (5,2,3)
            low_b = decode(self.s2b, self.box, emb.expand(5, -1, -1, -1, -1), None, None, bv, None)
            p1 = torch.tensor(np.asarray([dp], np.float32)[:, None], device=self.dev)
            l1 = torch.ones(1, 1, dtype=torch.long, device=self.dev)
            low_p = decode(self.s2b, self.box, emb, p1, l1, None, None)
            low_bp = decode(self.s2b, self.box, emb, p1, l1, bv[:1], None)
            mt = upsample(low_t.float())[:, 0] > 0
            ms = upsample(torch.cat([low_b, low_p, low_bp]).float())[:, 0] > 0
        c = torch.from_numpy(np.ascontiguousarray(comp)).to(self.dev)
        out = {f"turbo_{k}": v for k, v in mask_stats(mt, c).items()}
        out.update({f"s2b_{k}": v for k, v in mask_stats(ms, c).items()})
        out.update(deep_point=list(dp), n_turbo_points=len(set(pts)))
        return out


def load_prompter(sam_ckpt, turbo_ckpt, random_weights):
    import torch
    import s3a_seg2sam as s3
    if random_weights:
        torch.manual_seed(0)
    turbo = s3.load_sam_med3d(None if random_weights else turbo_ckpt, s3.DEV)
    s2b = copy.deepcopy(turbo)                                          # identical encoder by construction
    s3.add_lora(s2b.mask_decoder, ("q_proj", "v_proj"), 8, 16.0)
    box = s3.BoxCorners(s2b.prompt_encoder)
    if not random_weights:
        st = torch.load(sam_ckpt, map_location="cpu", weights_only=False)
        s2b.mask_decoder.load_state_dict(st["mask_decoder"]); box.load_state_dict(st["box"])
    for m in (turbo, s2b, box):
        m.to(s3.DEV).eval()
    return Prompter(turbo, s2b, box, s3.DEV, s3.DEV == "cuda")


# --------------------------------------------------------------------------------------------- steps
def step_precheck(a, out):
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    import s3a_seg2sam as s3
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any SAM call
    tab = pd.read_csv(Path(a.s3k_out).expanduser() / "train_oof_table.csv")
    tab = tab[tab.k >= 0]
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    paths = dict(require_cases(pd.read_csv(a.splits), root, "train"))
    cdir = Path(a.train_cases).expanduser()
    cdir = cdir if any(cdir.glob("*.npz")) else next(cdir.rglob("*.npz")).parent
    pats = sorted(tab.cache_id.unique())
    if a.limit:
        pats = pats[:a.limit]
    part = out / "candidates_stability.partial.csv"
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {r["cache_id"] for r in rows}
    todo = [c for c in pats if c not in done]
    print(f"S3s pre-check | {len(pats)} patients ({len(done)} already done) | "
          f"{int(tab.cache_id.isin(pats).sum())} candidates", flush=True)
    pr = load_prompter(a.sam_ckpt, a.turbo_ckpt, a.random_weights)
    t0 = time.time()
    for i, cid in enumerate(todo):
        img, _ = load_case(paths[cid])
        lab = np.load(cdir / f"{cid}.npz")["lab"]
        g = tab[tab.cache_id == cid]
        ks = g.k.astype(int).tolist()
        cc = s3.candidate_crops(img, lab, [(k, float((lab == k).sum())) for k in ks])
        for j, (k, on) in enumerate(zip(ks, g.on_tumour)):
            t1 = time.time()
            r = pr(cc["crops"][j][0], cc["boxes"][j], cc["comp"][j], f"{cid}_{k}")
            rows.append(dict(cache_id=cid, k=k, on_tumour=int(on), seconds=time.time() - t1, **r))
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"  {i + 1}/{len(todo)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "candidates_stability.csv", index=False)
    if part.exists():
        part.unlink()
    print(f"PRECHECK DONE: {len(df)} candidates | {df.seconds.mean():.2f} s per candidate -> run --step gate")


def top_hits(c, col):
    o = c.sort_values(["cache_id", col, "size_vox"], ascending=[True, False, False])
    return o.groupby("cache_id").on.first()


def step_gate(a, out):
    from sklearn.metrics import roc_auc_score
    src = Path(a.s3k_out).expanduser()
    st = pd.read_csv(out / "candidates_stability.csv")
    tab = pd.read_csv(src / "train_oof_table.csv")
    tab = tab[tab.k >= 0][["cache_id", "k", "size_vox", "seg_prob_mean", "seg_prob_max", "agree"]]
    c = st.merge(tab, on=["cache_id", "k"], how="left", validate="one_to_one")
    if a.memory_csv:
        mem = pd.read_csv(Path(a.memory_csv).expanduser())[["cache_id", "k", "mem_knn"]]
        c = c.merge(mem, on=["cache_id", "k"], how="left", validate="one_to_one")
    c["on"] = c.on_tumour == 1
    pp = pd.read_csv(src / "per_patient.csv")
    pp = pp[pp.split == "train_oof"].set_index("cache_id")
    g = c.groupby("cache_id")
    df = pd.DataFrame(dict(has=g.on.any(), chance=g.on.mean())).join(pp[["collection", "B0_rad_dice"]])
    sets = {"wrong": df.has & (df.B0_rad_dice < 0.1), "ok": df.has & (df.B0_rad_dice >= 0.1)}
    names = dict(seg_prob_mean="finder anchor (seg_prob_mean)", seg_prob_max="peak SegResNet prob",
                 size_vox="largest", agree="SAM box agree (S3l)", mem_knn="SAM memory (S3r)",
                 turbo_stab="TURBO STABILITY (primary)", s2b_stab="S2b stability (secondary)",
                 turbo_comp_dice="turbo click-vs-blob Dice", s2b_comp_dice="S2b prompt-vs-blob Dice")
    rows = []
    for col, nm in names.items():
        if col not in c or c[col].isna().all():
            continue
        h = top_hits(c, col)
        rows.append(dict(score=nm, column=col, wrong_hits=int(h[sets["wrong"]].sum()),
                         ok_elsewhere=int((~h[sets["ok"]]).sum()), auc=float(roc_auc_score(c.on, c[col].fillna(0)))))
    res = pd.DataFrame(rows)
    rows.append(dict(score="chance", column="", wrong_hits=round(float(df.chance[sets["wrong"]].sum()), 1),
                     ok_elsewhere=round(float((1 - df.chance)[sets["ok"]].sum()), 1), auc=0.5))
    n_wrong = int(sets["wrong"].sum())
    best_non = int(res[res.column.isin(RULE["non_sam"])].wrong_hits.max())
    need = max(math.ceil(0.5 * n_wrong), best_non + 5)

    def verdict(col):
        r = res[res.column == col].iloc[0]
        return dict(wrong_hits=int(r.wrong_hits), needed=need, ok_elsewhere=int(r.ok_elsewhere),
                    harm_max=RULE["harm_max"], passes=bool(r.wrong_hits >= need and r.ok_elsewhere <= RULE["harm_max"]))
    gate = dict(n_wrong=n_wrong, n_ok=int(sets["ok"].sum()), best_non_sam_hits=best_non,
                primary_turbo=verdict("turbo_stab"), secondary_s2b_reported=verdict("s2b_stab"))
    gate["PASS"] = gate["primary_turbo"]["passes"]
    pd.DataFrame(rows).to_csv(out / "gate_table.csv", index=False)
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    by = c.groupby("on")[[f"{s}_{f}" for s in ("turbo", "s2b") for f in FEATS]].median().T
    by.columns = ["false candidates (median)", "tumour candidates (median)"]
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n" + by.round(3).to_string())
    print("\nGATE:", json.dumps(gate, indent=1))
    print(f"\nVERDICT: {'PASS -> full matched S3s' if gate['PASS'] else 'FAIL -> stop (SAM prompt stability)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["precheck", "gate"], required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--s3k_out", required=True)
    ap.add_argument("--train_cases", default="", help="S3g cases/ (OOF component maps, key lab)")
    ap.add_argument("--sam_ckpt", default="")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--memory_csv", default="", help="S3r train_oof_memory_features.csv (reference, optional)")
    ap.add_argument("--random_weights", action="store_true", help="tests only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for f in (a.train_cases, a.memory_csv):
        if "test" in Path(f).name.lower():
            raise SystemExit("S3s is a development study: test files are refused")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.step == "precheck":
        if not a.train_cases:
            raise SystemExit("--train_cases is required")
        if not a.random_weights and not (a.sam_ckpt and a.turbo_ckpt):
            raise SystemExit("--sam_ckpt (S2b best.pt) and --turbo_ckpt are required")
        step_precheck(a, out)
    else:
        if not (out / "candidates_stability.csv").exists():
            raise SystemExit("run --step precheck first")
        step_gate(a, out)


if __name__ == "__main__":
    main()

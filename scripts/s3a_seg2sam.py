"""S3a: fully automatic SegResNet -> SAM pipeline (first council step, no new training).

Per case (val 120 for choosing the rule, test 306 once):
  1. SegResNet (image-only best.pt, Exp 1) -> tumour probability at 1 mm (sliding window, same as S1 test)
  2. candidate regions = connected components of the SegResNet mask (26-conn., >= 0.1 ml), top-K by size
  3. for every candidate: 128^3 SAM crop at 1.5 mm centred on the candidate, box = candidate bounding box
     -> frozen SAM-Med3D-turbo encoder -> fine-tuned decoder (S2a LoRA best.pt) with the box prompt
  4. jury: agreement = Dice(SAM mask, SegResNet candidate) inside the crop
Rules compared (all fully automatic; chosen by validation Dice, applied once to test):
  seg_rel20        SegResNet + rel20 post-processing (the S1 baseline, 0.655 on test)
  sam_rel20        SAM refinement of the rel20 components
  avg_rel20        mean of SegResNet and SAM probabilities on the rel20 components
  jury_t (t=0.3/0.5/0.7)  SAM on candidates whose agreement >= t (fallback: best-agreeing candidate)
  segjury_t        the same jury, but keep the SegResNet mask of the accepted candidates
  avgjury_t        the same jury, mean of SegResNet and SAM probabilities on the accepted candidates
No ground truth is used for prompting anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.data.cache import SAM_SIZE, crop_pad, find_cache_root, load_case, resample, znorm  # noqa: E402
from cids.data.volumes import VolumeDataset, unpad  # noqa: E402
from cids.eval.metrics import all_metrics, dice  # noqa: E402
from cids.models.sam_lora import BoxCorners, add_lora, decode, upsample  # noqa: E402
from cids.models.sam_med3d import download_turbo, load_sam_med3d  # noqa: E402
from cids.models.segresnet_text import TextSegResNet  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STRUCT = np.ones((3, 3, 3))
JURY_T = (0.3, 0.5, 0.7)
REJECT_ML = 0.1                 # SAM output below this (in its crop) = "no tumour here" (S2b can say this)
RULES = (["seg_rel20", "sam_rel20", "avg_rel20"] + [f"jury_{t}" for t in JURY_T]
         + [f"segjury_{t}" for t in JURY_T] + [f"avgjury_{t}" for t in JURY_T]
         + ["segkeep", "samkeep"])


def find_file(name: str, explicit: str = "", root: str = "/kaggle/input") -> Path:
    if explicit:
        return Path(explicit)
    hits = sorted(Path(root).rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"need exactly one {name} under {root} (found {hits}); pass the path explicitly")
    return hits[0]


# ---------------------------------------------------------------------------- models
def load_models(seg_ckpt, sam_lora_ckpt, turbo_ckpt):
    seg = TextSegResNet(text=False).to(DEV).eval()
    seg.load_state_dict(torch.load(seg_ckpt, map_location=DEV, weights_only=False))
    sam = load_sam_med3d(turbo_ckpt, DEV)
    add_lora(sam.mask_decoder, ("q_proj", "v_proj"), 8, 16.0)
    box = BoxCorners(sam.prompt_encoder)
    st = torch.load(sam_lora_ckpt, map_location="cpu", weights_only=False)
    sam.mask_decoder.load_state_dict(st["mask_decoder"])
    box.load_state_dict(st["box"])
    sam.to(DEV).eval(); box.to(DEV).eval()
    return seg, sam, box


@torch.no_grad()
def seg_prob(seg, image, amp, overlap=0.5):
    from monai.inferers import sliding_window_inference
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        logits = sliding_window_inference(image[None].to(DEV), (96,) * 3, sw_batch_size=4, predictor=seg,
                                          overlap=overlap, mode="gaussian")
    return torch.sigmoid(logits.float())[0, 0].cpu().numpy()


# ---------------------------------------------------------------------------- candidates
def components(mask, min_vox=100, top_k=5):
    lab, n = ndimage.label(mask, structure=STRUCT)
    if n == 0:
        return lab, []
    size = ndimage.sum(mask, lab, range(1, n + 1))
    order = [int(i) + 1 for i in np.argsort(-size) if size[i] >= min_vox][:top_k]
    if not order:                                   # keep the largest even if tiny
        order = [int(np.argmax(size)) + 1]
    return lab, [(k, float(size[k - 1])) for k in order]


def candidate_crops(img, lab, cands, lbl=None):
    """SAM input for every candidate: 128^3 post1 crop at 1.5 mm centred on the candidate + its box.
    Shared by S3 (pipeline) and S2c (hard-negative mining), so both see exactly the same crops.
    Returns dict(crops (K,1,128^3) f32, boxes (K,2,3), origins (K,3), comp (K,128^3) bool,
    gt (K,128^3) bool or None, shape15)."""
    img15, lbl15 = resample(img, np.zeros(lab.shape, np.uint8) if lbl is None else lbl)
    shape15 = img15.shape[1:]
    lab15 = F.interpolate(torch.from_numpy(lab.astype(np.float32))[None, None], size=shape15,
                          mode="nearest")[0, 0].numpy().astype(np.int32)
    post, fg = img15[1], (img15[1] > 0).astype(np.uint8)
    crops, boxes, origins, comp, gts = [], [], [], [], []
    for k, _ in cands:
        idx = np.argwhere(lab15 == k)
        if len(idx) == 0:                           # component vanished at 1.5 mm: use 1 mm centre / 1.5
            idx = (np.argwhere(lab == k) / 1.5).round().astype(int)
        lo, hi = idx.min(0), idx.max(0)
        c = (lo + hi) // 2
        vc, origin = crop_pad(post, c)
        fc, _ = crop_pad(fg, c)
        crops.append(znorm(vc, fc)[None])
        boxes.append(np.stack([np.clip(lo - origin, 0, SAM_SIZE - 1), np.clip(hi - origin, 0, SAM_SIZE - 1)]))
        origins.append(origin)
        comp.append(crop_pad((lab15 == k).astype(np.uint8), c)[0] > 0)
        if lbl is not None:
            gts.append(crop_pad(lbl15, c)[0] > 0)
    return dict(crops=np.stack(crops), boxes=np.stack(boxes), origins=np.stack(origins), comp=np.stack(comp),
                gt=np.stack(gts) if lbl is not None else None, shape15=shape15)


@torch.no_grad()
def sam_refine(sam, box_mod, img, lab, cands, amp, adapter=None):
    """Returns per candidate: SAM probability pasted into a 1 mm volume, and agreement with SegResNet.
    adapter (optional, S4 SamFreqAdapter): residual added to SAM's image embedding from the same crop."""
    shape1 = lab.shape
    cc = candidate_crops(img, lab, cands)
    shape15, boxes, origins, comp_crops = cc["shape15"], cc["boxes"], cc["origins"], cc["comp"]
    x = torch.from_numpy(cc["crops"]).to(DEV)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        emb = sam.image_encoder(x)
        if adapter is not None:
            emb = emb + adapter(x)
        low = decode(sam, box_mod, emb.float(), None, None, torch.from_numpy(np.stack(boxes)).float().to(DEV), None)
    prob = torch.sigmoid(upsample(low.float()))[:, 0].cpu()           # (K,128,128,128)
    out = []
    for j, (k, _) in enumerate(cands):
        p = prob[j]
        agree = dice((p > 0.5).numpy(), comp_crops[j])
        sam_ml = float((p > 0.5).sum()) * 1.5 ** 3 / 1000
        vol = torch.zeros(tuple(int(s) for s in shape15))
        o = origins[j]; lo_ = np.maximum(o, 0); hi_ = np.minimum(o + SAM_SIZE, shape15)
        a, b = lo_ - o, hi_ - o
        vol[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]] = p[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
        up = F.interpolate(vol[None, None], size=shape1, mode="trilinear", align_corners=False)[0, 0].numpy()
        out.append(dict(k=k, prob=up, agree=float(agree), sam_ml=sam_ml))
    return out


def predict_rules(sprob, refined, lab, cands):
    """All rules from one SegResNet probability map and the SAM refinements."""
    smask = sprob > 0.5
    preds = {}
    if not cands:
        for r in RULES:
            preds[r] = np.zeros_like(smask)
        return preds
    size = {k: s for k, s in cands}
    keep20 = [k for k, s in cands if s >= 0.2 * max(size.values())]
    by_k = {d["k"]: d for d in refined}
    seg20 = np.isin(lab, keep20)
    preds["seg_rel20"] = seg20
    sam20 = np.max([by_k[k]["prob"] for k in keep20], 0)
    preds["sam_rel20"] = sam20 > 0.5
    preds["avg_rel20"] = (0.5 * sprob * seg20 + 0.5 * sam20) > 0.5
    for t in JURY_T:
        ks = [d["k"] for d in refined if d["agree"] >= t] or [max(refined, key=lambda d: d["agree"])["k"]]
        samk = np.max([by_k[k]["prob"] for k in ks], 0)
        segk = np.isin(lab, ks)
        preds[f"jury_{t}"] = samk > 0.5                                   # SAM masks of the kept candidates
        preds[f"segjury_{t}"] = segk                                       # jury only filters SegResNet regions
        preds[f"avgjury_{t}"] = (0.5 * sprob * segk + 0.5 * samk) > 0.5    # fuse both on kept candidates
    # keep the candidates SAM did not reject (needs a checker trained with negative boxes, S2b)
    ks = [d["k"] for d in refined if d["sam_ml"] >= REJECT_ML] or [max(refined, key=lambda d: d["agree"])["k"]]
    preds["segkeep"] = np.isin(lab, ks)
    preds["samkeep"] = np.max([by_k[k]["prob"] for k in ks], 0) > 0.5
    return preds


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--seg_ckpt", default="")
    ap.add_argument("--sam_lora_ckpt", default="")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--splits", default=str(Path(__file__).resolve().parents[1] / "configs" / "splits.csv"))
    ap.add_argument("--out", default="/kaggle/working/s3a_seg2sam")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--skip_val", action="store_true", help="use --rule instead of choosing on val")
    ap.add_argument("--rule", default="")
    a = ap.parse_args()
    amp = DEV == "cuda"
    t0 = time.time()
    out = Path(a.out); (out / "test_pred").mkdir(parents=True, exist_ok=True)
    root = find_cache_root(a.cache_root)
    seg_ckpt = find_file("segresnet_img_best.pt", a.seg_ckpt)
    sam_ckpt = find_file("sam_lora_best.pt", a.sam_lora_ckpt)
    turbo = None if a.random_weights else (a.turbo_ckpt or download_turbo("/tmp/sam_ckpt"))
    print(f"cache {root}\nSegResNet {seg_ckpt}\nSAM LoRA {sam_ckpt}\ndevice {DEV}")
    seg, sam, box = load_models(seg_ckpt, sam_ckpt, turbo)

    sp = pd.read_csv(a.splits)
    def items(split):
        r = sp[sp.split == split]
        lst = [(c, root / ("train" if o == "train" else "val") / f"{c}.npz") for c, o in zip(r.cache_id, r.official)]
        lst = [x for x in lst if x[1].exists()]
        return lst[:a.limit_eval] if a.limit_eval else lst

    def run(cases, desc, full, save_rule=None):
        """full: set of rules that get HD95/NSD as well (Dice for every rule)."""
        rows, case_rows, cand_rows = [], [], []
        ds = VolumeDataset(cases, 96)
        for i in tqdm(range(len(ds)), desc=desc, unit="case", file=sys.stdout, dynamic_ncols=True, mininterval=10):
            item = ds[i]; cid = item["cid"]
            img, lbl = load_case(cases[i][1]); gt = lbl > 0
            sprob = unpad(seg_prob(seg, item["image"], amp), item["pads"])
            lab, cands = components(sprob > 0.5, top_k=a.top_k)
            refined = sam_refine(sam, box, img, lab, cands, amp) if cands else []
            preds = predict_rules(sprob, refined, lab, cands)
            hit = [bool((lab == k)[gt].any()) for k, _ in cands]
            case_rows.append(dict(cache_id=cid, n_cand=len(cands), any_cand_on_tumour=any(hit),
                                  agree=";".join(f"{d['agree']:.2f}" for d in refined),
                                  sam_ml=";".join(f"{d['sam_ml']:.2f}" for d in refined),
                                  on_tumour=";".join(str(int(h)) for h in hit)))
            largest = max((s_ for _, s_ in cands), default=1)
            for rank, ((k, size), d, h) in enumerate(zip(cands, refined, hit)):   # features for a learned council
                comp = lab == k
                ctr = np.argwhere(comp).mean(0) / np.array(comp.shape)
                cand_rows.append(dict(cache_id=cid, rank=rank, seg_ml=size / 1000, rel_size=size / largest,
                                      seg_prob_mean=float(sprob[comp].mean()), seg_prob_max=float(sprob[comp].max()),
                                      agree=d["agree"], sam_ml=d["sam_ml"], cx=ctr[0], cy=ctr[1], cz=ctr[2],
                                      n_cand=len(cands), on_tumour=int(h),
                                      overlap_gt=float((comp & gt).sum() / max(comp.sum(), 1))))
            for r, p in preds.items():
                m = all_metrics(p, gt) if (full and r in full) else dict(dice=dice(p, gt))
                rows.append(dict(cache_id=cid, collection=cid.split("_")[0].upper(), rule=r, **m))
            if save_rule:
                np.savez_compressed(out / "test_pred" / f"{cid}.npz", mask=preds[save_rule].astype(np.uint8))
        return pd.DataFrame(rows), pd.DataFrame(case_rows), pd.DataFrame(cand_rows)

    if a.skip_val:
        best, top = a.rule or "jury_0.7", []
    else:
        val, vc, vcand = run(items("val"), "val", None)
        val.to_csv(out / "val_metrics.csv", index=False); vc.to_csv(out / "val_cases.csv", index=False)
        vcand.to_csv(out / "val_candidates.csv", index=False)
        vs = val.groupby("rule").dice.agg(["mean", "median"]).loc[RULES]
        best = vs["mean"].idxmax()
        top = list(vs["mean"].sort_values(ascending=False).index[:3])
        print("\nVALIDATION (120) Dice:"); print(vs.round(4).to_string()); print("-> chosen rule:", best)
    full = {"seg_rel20", "sam_rel20", best, *top}

    te, tc, tcand = run(items("test"), "test", full, save_rule=best)
    te.to_csv(out / "test_metrics.csv", index=False); tc.to_csv(out / "test_cases.csv", index=False)
    tcand.to_csv(out / "test_candidates.csv", index=False)
    te_full = te[te.rule.isin(full)]
    ts = te.groupby("rule").dice.agg(["mean", "std", "median"]).loc[RULES].rename(
        columns={"mean": "dice", "std": "dice_sd", "median": "dice_median"})
    ts = ts.join(te_full.groupby("rule").agg(hd95_median=("hd95", "median"), nsd=("nsd", "mean")))
    ts["missed"] = te.groupby("rule").dice.apply(lambda x: int((x < 0.1).sum()))
    print(f"\nTEST (306), all rules (chosen on val: {best}):"); print(ts.round(4).to_string())
    print("\nper collection Dice:"); print(te.pivot_table(index="collection", columns="rule", values="dice")[RULES].round(3).to_string())
    print("\nper collection HD95 median (rules with full metrics):")
    print(te_full.pivot_table(index="collection", columns="rule", values="hd95", aggfunc="median").round(1).to_string())
    print(f"\ncandidates per case: mean {tc.n_cand.mean():.2f}; a candidate lies on the tumour in "
          f"{tc.any_cand_on_tumour.mean():.1%} of test cases")
    if len(tcand):
        rej = tcand.sam_ml < REJECT_ML
        print(f"SAM rejected {rej[tcand.on_tumour == 0].mean():.1%} of false candidates and "
              f"{rej[tcand.on_tumour == 1].mean():.1%} of true candidates (test)")
    rej_stats = {}
    if len(tcand):
        rej = tcand.sam_ml < REJECT_ML
        rej_stats = dict(false_candidates_rejected=float(rej[tcand.on_tumour == 0].mean()),
                         true_candidates_rejected=float(rej[tcand.on_tumour == 1].mean()))
    json.dump({"chosen_rule_on_val": best, "seg_ckpt": str(seg_ckpt), "sam_ckpt": str(sam_ckpt),
               "sam_rejection": rej_stats, "test": ts.round(4).to_dict(orient="index"),
               "candidate_on_tumour_rate": float(tc.any_cand_on_tumour.mean()),
               "minutes": (time.time() - t0) / 60}, open(out / "summary.json", "w"), indent=2, default=float)
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

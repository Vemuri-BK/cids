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
RULES = ["seg_rel20", "sam_rel20", "avg_rel20"] + [f"jury_{t}" for t in JURY_T]


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


@torch.no_grad()
def sam_refine(sam, box_mod, img, lab, cands, amp):
    """Returns per candidate: SAM probability pasted into a 1 mm volume, and agreement with SegResNet."""
    shape1 = lab.shape
    img15, _ = resample(img, np.zeros(shape1, np.uint8))
    shape15 = img15.shape[1:]
    lab15 = F.interpolate(torch.from_numpy(lab.astype(np.float32))[None, None], size=shape15,
                          mode="nearest")[0, 0].numpy().astype(np.int32)
    post, fg = img15[1], (img15[1] > 0).astype(np.uint8)
    crops, boxes, origins, comp_crops = [], [], [], []
    for k, _ in cands:
        idx = np.argwhere(lab15 == k)
        if len(idx) == 0:                           # component vanished at 1.5 mm: use 1 mm centre / 1.5
            idx = (np.argwhere(lab == k) / 1.5).round().astype(int)
        lo, hi = idx.min(0), idx.max(0)
        c = (lo + hi) // 2
        vc, origin = crop_pad(post, c)
        fc, _ = crop_pad(fg, c)
        crops.append(znorm(vc, fc)[None])
        b = np.stack([np.clip(lo - origin, 0, SAM_SIZE - 1), np.clip(hi - origin, 0, SAM_SIZE - 1)])
        boxes.append(b)
        origins.append(origin)
        comp_crops.append(crop_pad((lab15 == k).astype(np.uint8), c)[0] > 0)
    x = torch.from_numpy(np.stack(crops)).to(DEV)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        emb = sam.image_encoder(x)
        low = decode(sam, box_mod, emb.float(), None, None, torch.from_numpy(np.stack(boxes)).float().to(DEV), None)
    prob = torch.sigmoid(upsample(low.float()))[:, 0].cpu()           # (K,128,128,128)
    out = []
    for j, (k, _) in enumerate(cands):
        p = prob[j]
        agree = dice((p > 0.5).numpy(), comp_crops[j])
        vol = torch.zeros(tuple(int(s) for s in shape15))
        o = origins[j]; lo_ = np.maximum(o, 0); hi_ = np.minimum(o + SAM_SIZE, shape15)
        a, b = lo_ - o, hi_ - o
        vol[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]] = p[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
        up = F.interpolate(vol[None, None], size=shape1, mode="trilinear", align_corners=False)[0, 0].numpy()
        out.append(dict(k=k, prob=up, agree=float(agree)))
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
        preds[f"jury_{t}"] = np.max([by_k[k]["prob"] for k in ks], 0) > 0.5
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
        rows, case_rows = [], []
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
                                  on_tumour=";".join(str(int(h)) for h in hit)))
            for r, p in preds.items():
                m = all_metrics(p, gt) if full else dict(dice=dice(p, gt))
                rows.append(dict(cache_id=cid, collection=cid.split("_")[0].upper(), rule=r, **m))
            if save_rule:
                np.savez_compressed(out / "test_pred" / f"{cid}.npz", mask=preds[save_rule].astype(np.uint8))
        return pd.DataFrame(rows), pd.DataFrame(case_rows)

    if a.skip_val:
        best = a.rule or "jury_0.5"
    else:
        val, vc = run(items("val"), "val", False)
        val.to_csv(out / "val_metrics.csv", index=False); vc.to_csv(out / "val_cases.csv", index=False)
        vs = val.groupby("rule").dice.agg(["mean", "median"]).loc[RULES]
        best = vs["mean"].idxmax()
        print("\nVALIDATION (120) Dice:"); print(vs.round(4).to_string()); print("-> chosen rule:", best)

    te, tc = run(items("test"), "test", True, save_rule=best)
    te.to_csv(out / "test_metrics.csv", index=False); tc.to_csv(out / "test_cases.csv", index=False)
    agg = dict(dice=("dice", "mean"), dice_sd=("dice", "std"), dice_median=("dice", "median"),
               hd95_median=("hd95", "median"), nsd=("nsd", "mean"), missed=("dice", lambda x: int((x < 0.1).sum())))
    ts = te.groupby("rule").agg(**agg).loc[RULES]
    print(f"\nTEST (306), all rules (chosen on val: {best}):"); print(ts.round(4).to_string())
    print("\nper collection Dice:"); print(te.pivot_table(index="collection", columns="rule", values="dice")[RULES].round(3).to_string())
    print("\nper collection HD95 median:")
    print(te.pivot_table(index="collection", columns="rule", values="hd95", aggfunc="median")[RULES].round(1).to_string())
    print(f"\ncandidates per case: mean {tc.n_cand.mean():.2f}; a candidate lies on the tumour in "
          f"{tc.any_cand_on_tumour.mean():.1%} of test cases")
    json.dump({"chosen_rule_on_val": best, "test": ts.round(4).to_dict(orient="index"),
               "candidate_on_tumour_rate": float(tc.any_cand_on_tumour.mean()),
               "minutes": (time.time() - t0) / 60}, open(out / "summary.json", "w"), indent=2, default=float)
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

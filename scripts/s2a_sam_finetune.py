"""S2a / S2b: SAM-Med3D-turbo fine-tuned on MAMA-MIA images only.

S2a (--box_aug clean, default): box prompts are the GT box +-5 voxels.
S2b (--box_aug realistic): box prompts look like SegResNet candidates -- clean (GT +- jitter), messy
  (scaled 0.7-1.8x, shifted up to 40% of tumour size, may cover part of it) or negative (on tumour-free
  tissue, target = EMPTY mask, so SAM learns to answer "no tumour here"). Validation score =
  mean(clean-box Dice, messy-box Dice, negative-box rejection rate); rejection = predicted volume < 0.1 ml.

S2c (--cand_root, on top of S2b, usually --init_ckpt = S2b best.pt): adds SegResNet's REAL candidates
  (scripts/s2c_mine_candidates.py). A share --p_hard of the batches are hard negatives: crop + box of a
  candidate that does NOT touch the tumour, target = empty mask. A share --p_realpos are real true candidates
  (target = tumour), so SAM does not start rejecting real tumours. Validation adds, on val's real candidates
  (pipeline setting), hard_reject (false candidates rejected) and real_keep (true candidates kept);
  score = mean(box1, messy1, hard_reject, real_keep). Test reports both for every model (S2b = 'compare').

Frozen image encoder -> cached post1 embeddings from S0 (`cids-sam-emb`); only the mask decoder
is adapted (LoRA r=8 on q/v, ~70k params; or --tune decoder_full) + 2 learnable box-corner tokens.
No text, no frequency prompts.

Training (train 1080 of configs/splits.csv, 4 cached crops per patient, one random crop per epoch):
  each batch uses either a jittered GT box (p=--p_box) or a random first click, followed by up to
  --max_clicks-1 corrective clicks (SAM-Med3D style, previous low-res mask fed back as dense prompt);
  loss = BCE + soft Dice on the 128^3 upsampled logits, averaged over the steps.
Validation (120, tumour-centred crop): mean of Dice@1 click, Dice@5 clicks, Dice@box -> best.pt.
Test (306, once): the zero-shot model and the fine-tuned model with the same seeded clicks;
  crop-level Dice (1.5 mm, comparable with S0) for clicks 1..5 and box(+clicks);
  full-resolution Dice / HD95 / NSD@2mm at 1 mm (crop pasted back; tumour outside the crop = missed)
  if the 2-channel cache is available.

Example
  python scripts/s2a_sam_finetune.py --out /kaggle/working/s2a_lora
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.eval.metrics import all_metrics  # noqa: E402
from cids.models.sam_lora import (BoxCorners, add_lora, decode, gt_boxes, interactive,  # noqa: E402
                                  negative_box, messy_box, realistic_boxes, sample_clicks, upsample)
from cids.models.sam_med3d import download_turbo, load_sam_med3d  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VOX = 128 ** 3
FULL_RES_PROMPTS = [("click", 1), ("click", 5), ("box", 1), ("messy", 1)]
REJECT_VOX = 30                     # < 30 voxels at 1.5 mm (~0.1 ml, the candidate minimum) = "no tumour"


# ---------------------------------------------------------------------------- data
def find_emb_root(explicit: str = "", search_root: str = "/kaggle/input") -> Path:
    if explicit:
        return Path(explicit)
    hits = sorted({p.parent for p in Path(search_root).rglob("index.csv")
                   if (p.parent / "train").is_dir() and (p.parent / "test").is_dir()
                   and next((p.parent / "test").glob("*.npz"), None)}, key=lambda h: (len(h.parts), str(h)))
    if not hits:
        raise FileNotFoundError(f"no embedding folder (index.csv + train/ + test/) under {search_root}; set --emb_root")
    if len(hits) > 1:
        print(f"[note] several embedding folders found, using {hits[0]}; others: {hits[1:]}")
    return hits[0]


class EmbFiles(Dataset):
    def __init__(self, items, crops="all"):
        self.items, self.crops = items, crops

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, path = self.items[i]
        d = np.load(path)
        emb, lab = d["emb_post1"], d["label"]
        if self.crops == "first":
            emb, lab = emb[:1], lab[:1]
        bits = np.packbits(lab.reshape(len(lab), -1) > 0, axis=1)
        out = dict(cid=cid, emb=emb.astype(np.float16), bits=bits)
        if self.crops == "first":
            out.update(origin=d["origin"][0], shape15=d["shape15"])
        return out


def _identity(x):
    return x


def _worker_init(_):
    torch.set_num_threads(1)


def preload(items, crops, workers, desc):
    dl = DataLoader(EmbFiles(items, crops), batch_size=None, collate_fn=_identity, num_workers=workers,
                    worker_init_fn=_worker_init)
    return [x for x in tqdm(dl, desc=desc, unit="case", file=sys.stdout, dynamic_ncols=True, mininterval=10)]


def load_cands(root, split, keep_ids=None):
    """Real SegResNet candidates of one split (s2c_mine_candidates.py) -> flat arrays (N candidates)."""
    files = sorted(Path(root, split).glob("*.npz"))
    if keep_ids is not None:
        files = [f for f in files if f.stem in keep_ids]
    emb, box, true, bits, cid = [], [], [], [], []
    for f in tqdm(files, desc=f"load {split} candidates", unit="case", file=sys.stdout, mininterval=10):
        d = np.load(f)
        if len(d["is_true"]) == 0:
            continue
        emb.append(d["emb"]); box.append(d["box"].astype(np.float32)); true.append(d["is_true"])
        bits.extend(list(d["gt_bits"])); cid.extend([f.stem] * len(d["is_true"]))
    if not emb:
        return None
    c = dict(emb=np.concatenate(emb), box=np.concatenate(box), is_true=np.concatenate(true), bits=bits, cid=cid)
    c["false_idx"], c["true_idx"] = np.flatnonzero(~c["is_true"]), np.flatnonzero(c["is_true"])
    print(f"{split}: {len(c['cid'])} real candidates = {len(c['true_idx'])} true + {len(c['false_idx'])} false")
    return c


def cand_batch(cands, idx):
    emb = torch.from_numpy(cands["emb"][idx]).to(DEV).float()
    boxes = torch.from_numpy(cands["box"][idx]).to(DEV)
    gt = unpack([cands["bits"][i] for i in idx]).to(DEV)
    gt &= torch.from_numpy(cands["is_true"][idx]).to(DEV)[:, None, None, None]     # false candidate -> empty target
    return emb, boxes, gt


def unpack(bits_list) -> torch.Tensor:
    lab = np.unpackbits(np.stack(bits_list), axis=1, count=VOX).reshape(-1, 128, 128, 128)
    return torch.from_numpy(lab.astype(bool))


# ---------------------------------------------------------------------------- model
def build(args, ckpt):
    model = load_sam_med3d(ckpt, DEV)
    model.image_encoder = torch.nn.Identity()          # embeddings are cached; free ~370 MB
    for p in model.parameters():
        p.requires_grad_(False)
    box = BoxCorners(model.prompt_encoder).to(DEV)
    if args.tune == "lora":
        names = add_lora(model.mask_decoder, ("q_proj", "v_proj"), args.r, args.alpha)
        model.to(DEV)
        print(f"LoRA r={args.r} on {len(names)} decoder projections")
    elif args.tune == "decoder_full":
        for p in model.mask_decoder.parameters():
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad] + list(box.parameters())
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    return model, box, params


def trainable_state(model, box):
    return dict(mask_decoder=model.mask_decoder.state_dict(), box=box.state_dict())


def seg_loss(logits, gt):
    """logits (B,1,D,H,W), gt (B,D,H,W) bool -> BCE + soft Dice (per sample, averaged)."""
    x, g = logits[:, 0].float(), gt.float()
    bce = F.binary_cross_entropy_with_logits(x, g)
    p = torch.sigmoid(x).flatten(1)
    g = g.flatten(1)
    d = 1 - (2 * (p * g).sum(1) + 1) / (p.sum(1) + g.sum(1) + 1)
    return bce + d.mean()


def lr_at(step, total, warm, base):
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


# ---------------------------------------------------------------------------- train
def train_epoch(model, box, opt, scaler, data, args, epoch, step0, total_steps, gen, cands=None):
    model.train()
    rng = random.Random(args.seed * 1000 + epoch)
    order = list(range(len(data)))
    rng.shuffle(order)
    nb = len(order) // args.batch
    losses, kinds = [], {"gt": 0, "hard": 0, "realpos": 0}
    bar = tqdm(range(nb), desc=f"ep {epoch}", unit="it", file=sys.stdout, dynamic_ncols=True, mininterval=10)
    for b in bar:
        idx = order[b * args.batch:(b + 1) * args.batch]
        ks = [rng.randrange(len(data[i]["emb"])) for i in idx]
        emb = torch.from_numpy(np.stack([data[i]["emb"][k] for i, k in zip(idx, ks)])).to(DEV).float()
        gt = unpack([data[i]["bits"][k] for i, k in zip(idx, ks)]).to(DEV)
        use_box = rng.random() < args.p_box
        n_steps = rng.randint(1, args.max_clicks)
        for g in opt.param_groups:
            g["lr"] = lr_at(step0 + b, total_steps, args.warmup_steps, args.lr)
        kind, r = "gt", rng.random()
        if cands is not None:
            if r < args.p_hard and len(cands["false_idx"]):
                kind, pool = "hard", cands["false_idx"]
            elif r < args.p_hard + args.p_realpos and len(cands["true_idx"]):
                kind, pool = "realpos", cands["true_idx"]
        if kind != "gt":                       # SegResNet's real box, one decoder pass, as in the pipeline
            cidx = np.array([pool[rng.randrange(len(pool))] for _ in idx])
            cemb, cbox, cgt = cand_batch(cands, cidx)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                loss = seg_loss(upsample(decode(model, box, cemb, None, None, cbox, None)), cgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item()); kinds[kind] += 1
            continue
        kinds["gt"] += 1
        coords = torch.zeros(len(idx), 0, 3, device=DEV)
        labels = torch.zeros(len(idx), 0, dtype=torch.long, device=DEV)
        boxes, tgt = None, gt
        if use_box and args.box_aug == "realistic":
            boxes, neg = realistic_boxes(gt, rng, args.p_messy, args.p_neg, args.box_jitter)
            tgt = gt & ~neg[:, None, None, None]                  # negative box -> empty target
        elif use_box:
            boxes = gt_boxes(gt, args.box_jitter, gen)
        prev, low, loss = torch.zeros_like(gt), None, 0.0
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            for s in range(n_steps):
                if not (use_box and s == 0):
                    c, l = sample_clicks(prev, tgt, gen)
                    coords, labels = torch.cat([coords, c], 1), torch.cat([labels, l], 1)
                low = decode(model, box, emb, coords, labels, boxes, None if low is None else low.detach())
                logits = upsample(low)
                loss = loss + seg_loss(logits, tgt) / n_steps
                prev = (logits.detach()[:, 0] > 0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if b % 20 == 0:
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}", lr=f"{opt.param_groups[0]['lr']:.1e}")
    bar.close()
    if cands is not None:
        print(f"  batches: {kinds}")
    return float(np.mean(losses)) if losses else float("nan"), nb


def batch_dice(prob, gt):
    p, g = prob > 0.5, gt
    inter = (p & g).flatten(1).sum(1).float()
    s = p.flatten(1).sum(1).float() + g.flatten(1).sum(1).float()
    return torch.where(s > 0, 2 * inter / s.clamp(min=1), torch.ones_like(s)).cpu().numpy()


def eval_boxes(gt, seed):
    """Fixed messy and negative boxes for evaluation (same seed -> same boxes for every model).
    Returns messy (B,2,3), neg (B,2,3) and has_neg (B,) bool (False if no tumour-free place was found)."""
    rng = random.Random(seed)
    messy, neg, ok = [], [], []
    for g in gt:
        m = messy_box(g, rng)
        n = negative_box(g, rng)
        messy.append(m if m is not None else torch.stack([torch.zeros(3, device=g.device),
                                                          torch.full((3,), 127., device=g.device)]))
        ok.append(n is not None)
        neg.append(n if n is not None else torch.zeros(2, 3, device=g.device))
    return torch.stack(messy), torch.stack(neg), torch.tensor(ok)


@torch.no_grad()
def run_prompts(model, box, emb, gt, args, with_box=True, seed=233):
    """Returns {('click'|'box'|'messy'|'neg', step): prob (B,128^3)} with seeded clicks/boxes
    (same for every model). 'messy'/'neg' are single box prompts without clicks."""
    model.eval()
    out = {}
    with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
        g = torch.Generator(device=DEV).manual_seed(seed)
        for k, p in enumerate(interactive(model, box, emb, gt, args.test_clicks, False, g), 1):
            out[("click", k)] = p.float()
        if with_box:
            g = torch.Generator(device=DEV).manual_seed(seed)
            for k, p in enumerate(interactive(model, box, emb, gt, args.test_clicks, True, g), 1):
                out[("box", k)] = p.float()
            messy, neg, _ = eval_boxes(gt, seed)
            for name, bx in (("messy", messy), ("neg", neg)):
                low = decode(model, box, emb, None, None, bx, None)
                out[(name, 1)] = torch.sigmoid(upsample(low.float()))[:, 0]
    return out


def neg_stats(prob, gt, seed):
    """Rejection (predicted < REJECT_VOX voxels) and predicted ml on the negative boxes; NaN where no
    tumour-free box exists."""
    _, _, ok = eval_boxes(gt, seed)
    vox = (prob > 0.5).flatten(1).sum(1).float().cpu().numpy()
    rej = np.where(ok.numpy(), (vox < REJECT_VOX).astype(float), np.nan)
    ml = np.where(ok.numpy(), vox * 1.5 ** 3 / 1000, np.nan)
    return rej, ml


@torch.no_grad()
def cand_eval(model, box, cands, args):
    """SAM on real SegResNet candidates (box only, as in the pipeline).
    hard_reject = share of FALSE candidates rejected (< REJECT_VOX = 0.1 ml); real_keep = share of TRUE
    candidates kept; real_dice = Dice on true candidates (crop, 1.5 mm)."""
    model.eval()
    vox, dsc = [], []
    for s in range(0, len(cands["cid"]), args.batch):
        idx = np.arange(s, min(s + args.batch, len(cands["cid"])))
        emb, bx, gt = cand_batch(cands, idx)
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            prob = torch.sigmoid(upsample(decode(model, box, emb, None, None, bx, None).float()))[:, 0]
        vox.extend((prob > 0.5).flatten(1).sum(1).cpu().numpy()); dsc.extend(batch_dice(prob, gt))
    vox, dsc, t = np.array(vox), np.array(dsc), cands["is_true"]
    return dict(hard_reject=float((vox[~t] < REJECT_VOX).mean()) if (~t).any() else float("nan"),
                real_keep=float((vox[t] >= REJECT_VOX).mean()) if t.any() else float("nan"),
                real_dice=float(dsc[t].mean()) if t.any() else float("nan"),
                hard_pred_ml=float(np.median(vox[~t]) * 1.5 ** 3 / 1000) if (~t).any() else float("nan"))


def validate(model, box, data, args, vcands=None):
    res = {("click", 1): [], ("click", 5): [], ("box", 1): [], ("messy", 1): []}
    rej = []
    for s in range(0, len(data), args.batch):
        chunk = data[s:s + args.batch]
        emb = torch.from_numpy(np.stack([c["emb"][0] for c in chunk])).to(DEV).float()
        gt = unpack([c["bits"][0] for c in chunk]).to(DEV)
        out = run_prompts(model, box, emb, gt, args, seed=233 + s)
        for key in res:
            k = (key[0], min(key[1], args.test_clicks)) if key[0] == "click" else key
            res[key].extend(batch_dice(out[k], gt))
        rej.extend(neg_stats(out[("neg", 1)], gt, 233 + s)[0])
    m = {f"{a}{b}": float(np.mean(v)) for (a, b), v in res.items()}
    m["neg_reject"] = float(np.nanmean(rej))
    if vcands is not None:
        m.update(cand_eval(model, box, vcands, args))
        m["score"] = float(np.nanmean([m["box1"], m["messy1"], m["hard_reject"], m["real_keep"]]))
    elif args.box_aug == "realistic":
        m["score"] = float(np.mean([m["box1"], m["messy1"], m["neg_reject"]]))
    else:                                                       # S2a criterion, unchanged
        m["score"] = float(np.mean([m["click1"], m["click5"], m["box1"]]))
    return m


# ---------------------------------------------------------------------------- test
def full_res_metrics(prob_crop, origin, shape15, cache_path):
    """Paste the 128^3 crop probability (1.5 mm) into the case volume, resample to the 1 mm cache grid."""
    _, lbl = load_case(cache_path)
    vol = torch.zeros(tuple(int(s) for s in shape15))
    lo = np.maximum(origin, 0)
    hi = np.minimum(origin + 128, shape15)
    if np.all(hi > lo):
        a, b = lo - origin, hi - origin
        vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = prob_crop[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    up = F.interpolate(vol[None, None], size=lbl.shape, mode="trilinear", align_corners=False)[0, 0]
    return all_metrics((up > 0.5).numpy(), lbl > 0)


def test(models, test_items, coll, args, cache_root, out, tcands=None):
    crop_rows, full_rows = [], []
    real = {}
    if tcands is not None:
        for name, (model, box, with_box) in models.items():
            if with_box:
                real[name] = {k: round(v, 4) for k, v in cand_eval(model, box, tcands, args).items()}
        print("\nTest, SegResNet's REAL candidates (pipeline setting): hard_reject = false candidates rejected, "
              "real_keep = true candidates kept")
        print(pd.DataFrame(real).T.to_string())
    for s in tqdm(range(0, len(test_items), args.batch), desc="test", unit="batch", file=sys.stdout,
                  dynamic_ncols=True, mininterval=10):
        chunk = test_items[s:s + args.batch]
        emb = torch.from_numpy(np.stack([c["emb"][0] for c in chunk])).to(DEV).float()
        gt = unpack([c["bits"][0] for c in chunk]).to(DEV)
        for name, (model, box, with_box) in models.items():
            outs = run_prompts(model, box, emb, gt, args, with_box=with_box, seed=233 + s)
            for (ptype, k), prob in outs.items():
                if ptype == "neg":
                    rej, ml = neg_stats(prob, gt, 233 + s)
                    for c, r_, v in zip(chunk, rej, ml):
                        crop_rows.append(dict(cache_id=c["cid"], collection=coll[c["cid"]], model=name,
                                              prompt="neg", step=k, reject=r_, neg_pred_ml=v))
                    continue
                for c, d in zip(chunk, batch_dice(prob, gt)):
                    crop_rows.append(dict(cache_id=c["cid"], collection=coll[c["cid"]], model=name,
                                          prompt=ptype, step=k, dice=float(d)))
            if cache_root is not None:
                if name == next(iter(models)):         # resampling ceiling: GT crop pasted back to 1 mm
                    for j, c in enumerate(chunk):
                        m = full_res_metrics(gt[j].float().cpu(), c["origin"], c["shape15"],
                                             cache_root / "val" / f"{c['cid']}.npz")
                        full_rows.append(dict(cache_id=c["cid"], collection=coll[c["cid"]],
                                              model="gt_crop_ceiling", prompt="-", **m))
                for j, c in enumerate(chunk):
                    path = cache_root / "val" / f"{c['cid']}.npz"
                    for key in FULL_RES_PROMPTS:
                        if key not in outs:
                            continue
                        m = full_res_metrics(outs[key][j].cpu(), c["origin"], c["shape15"], path)
                        full_rows.append(dict(cache_id=c["cid"], collection=coll[c["cid"]], model=name,
                                              prompt=f"{key[0]}{key[1]}", **m))
    cr = pd.DataFrame(crop_rows)
    cr.to_csv(out / "test_crop_metrics.csv", index=False)
    summary = {"crop_dice_1.5mm": {}, "negative_boxes": {}}
    for name, g in cr[cr.prompt == "neg"].groupby("model"):
        summary["negative_boxes"][name] = dict(reject_rate=round(float(g.reject.mean()), 4),
                                               mean_pred_ml=round(float(g.neg_pred_ml.mean()), 3),
                                               n=int(g.reject.notna().sum()))
    print("\nNegative boxes (no tumour inside): rejection rate = output < 0.1 ml")
    print(json.dumps(summary["negative_boxes"], indent=2))
    cr_d = cr[cr.prompt != "neg"]
    for (name, ptype), g in cr_d.groupby(["model", "prompt"]):
        summary["crop_dice_1.5mm"][f"{name}/{ptype}"] = {int(k): round(float(v), 4)
                                                         for k, v in g.groupby("step").dice.mean().items()}
    print("\nTest, crop-level Dice (1.5 mm, same seeded clicks for every model):")
    print(cr_d.pivot_table(index="step", columns=["model", "prompt"], values="dice").round(4).to_string())
    print("\n1 click, per collection:")
    print(cr[(cr.prompt == "click") & (cr.step == 1)].pivot_table(index="collection", columns="model",
                                                                  values="dice").round(4).to_string())
    if full_rows:
        fr = pd.DataFrame(full_rows)
        fr.to_csv(out / "test_full_metrics.csv", index=False)
        agg = fr.groupby(["model", "prompt"]).agg(dice=("dice", "mean"), hd95=("hd95", "median"),
                                                  nsd=("nsd", "mean"), n_hd95_nan=("hd95", lambda x: int(x.isna().sum())))
        print("\nTest, full resolution (1 mm; Dice/NSD mean, HD95 median mm):")
        print(agg.round(4).to_string())
        summary["full_res_1mm"] = {f"{m}/{p}": {k: (round(float(v), 4) if pd.notna(v) else None)
                                                for k, v in r.items()} for (m, p), r in agg.iterrows()}
        per_coll = fr.pivot_table(index="collection", columns=["model", "prompt"], values="dice").round(4)
        print("\nfull-res Dice per collection:")
        print(per_coll.to_string())
    if real:
        summary["real_candidates"] = real
    json.dump(summary, open(out / "test_summary.json", "w"), indent=2)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_root", default="")
    ap.add_argument("--cache_root", default="", help="2-channel cache for full-resolution metrics ('none' to skip)")
    ap.add_argument("--splits", default=str(Path(__file__).resolve().parents[1] / "configs" / "splits.csv"))
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--out", default="/kaggle/working/s2a_lora")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--tune", choices=["lora", "decoder_full"], default="lora")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0, help="default 5e-4 (lora) / 1e-4 (decoder_full)")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2, help="epochs")
    ap.add_argument("--max_clicks", type=int, default=5)
    ap.add_argument("--p_box", type=float, default=0.5)
    ap.add_argument("--box_jitter", type=int, default=5, help="voxels at 1.5 mm")
    ap.add_argument("--box_aug", choices=["clean", "realistic"], default="clean", help="S2a = clean, S2b = realistic")
    ap.add_argument("--p_messy", type=float, default=0.4, help="share of box prompts that are messy (S2b)")
    ap.add_argument("--p_neg", type=float, default=0.2, help="share of box prompts that are negative (S2b)")
    ap.add_argument("--cand_root", default="", help="S2c: output of s2c_mine_candidates.py (train/ val/ test/)")
    ap.add_argument("--p_hard", type=float, default=0.25, help="S2c: share of batches = real false candidates")
    ap.add_argument("--p_realpos", type=float, default=0.10, help="S2c: share of batches = real true candidates")
    ap.add_argument("--init_ckpt", default="", help="start from this best.pt (S2c: the S2b checkpoint)")
    ap.add_argument("--compare_ckpt", default="", help="best.pt of another run (same --tune/--r), tested as 'compare'")
    ap.add_argument("--test_clicks", type=int, default=5)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--time_budget_h", type=float, default=10.0)
    ap.add_argument("--resume_from", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--no_zero_shot", action="store_true", help="skip the zero-shot rows at test")
    args = ap.parse_args()
    args.amp = DEV == "cuda"
    args.lr = args.lr or (5e-4 if args.tune == "lora" else 1e-4)
    t_start = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    emb_root = find_emb_root(args.emb_root)
    cache_root = None
    if args.cache_root != "none":
        try:
            cache_root = find_cache_root(args.cache_root)
        except FileNotFoundError as e:
            print("[warn] no 2-channel cache -> crop-level metrics only.", e)
    sp = pd.read_csv(args.splits)
    coll = dict(zip(sp.cache_id, sp.collection))

    def items(split):
        rows = sp[sp[args.split_col] == split]
        sub = lambda off: "train" if off == "train" else "test"          # S0 folder names
        lst = [(c, emb_root / sub(o) / f"{c}.npz") for c, o in zip(rows.cache_id, rows.official)]
        return [x for x in lst if x[1].exists()]

    tr, va, te = items("train"), items("val"), items("test")
    if args.limit_train:
        tr = tr[:args.limit_train]
    if args.limit_eval:
        va, te = va[:args.limit_eval], te[:args.limit_eval]
    print(f"embeddings: {emb_root} | cache: {cache_root}\ntrain {len(tr)} | val {len(va)} | test {len(te)} | device {DEV}")

    ckpt = None if args.random_weights else (args.ckpt or download_turbo("/tmp/sam_ckpt"))
    model, box, params = build(args, ckpt)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if args.init_ckpt:
        st = torch.load(args.init_ckpt, map_location=DEV, weights_only=False)
        model.mask_decoder.load_state_dict(st["mask_decoder"]); box.load_state_dict(st["box"])
        print("initialised from", args.init_ckpt)
    start_epoch, best, log = 1, -1.0, []
    resume = Path(args.resume_from) if args.resume_from else out / "last.pt"
    if resume.exists():
        st = torch.load(resume, map_location=DEV, weights_only=False)
        model.mask_decoder.load_state_dict(st["state"]["mask_decoder"]); box.load_state_dict(st["state"]["box"])
        opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        start_epoch, best, log = st["epoch"] + 1, st["best"], st["log"]
        if resume.parent != out:
            for f in ("best.pt",):
                if (resume.parent / f).exists() and not (out / f).exists():
                    shutil.copy(resume.parent / f, out / f)
        print(f"resumed from {resume} at epoch {start_epoch}, best {best:.4f}")

    if start_epoch <= args.epochs:
        train_data = preload(tr, "all", args.workers, "load train")
        val_data = preload(va, "first", args.workers, "load val")
        tcand = vcand = None
        if args.cand_root:
            tcand = load_cands(args.cand_root, "train", {c for c, _ in tr})
            vcand = load_cands(args.cand_root, "val", {c for c, _ in va})
            if tcand is None or vcand is None or not len(tcand["false_idx"]):
                raise SystemExit(f"--cand_root {args.cand_root}: no mined train/val candidates (or no false ones) "
                                 f"-> run s2c_mine_candidates.py first (did it fail?)")
            if start_epoch == 1:                                    # epoch 0 = the starting model (S2b)
                m = validate(model, box, val_data, args, vcand)
                log.append(dict(epoch=0, **{f"val_{k}": v for k, v in m.items()}))
                print("epoch 0 (start):", {k: round(v, 4) for k, v in m.items()})
        nb = len(train_data) // args.batch
        total = nb * args.epochs
        args.warmup_steps = nb * args.warmup
        ep_times = []
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            loss, _ = train_epoch(model, box, opt, scaler, train_data, args, epoch, (epoch - 1) * nb, total, gen,
                                  tcand)
            row = dict(epoch=epoch, loss=loss, lr=opt.param_groups[0]["lr"])
            if epoch % args.val_every == 0 or epoch == args.epochs:
                m = validate(model, box, val_data, args, vcand)
                row.update({f"val_{k}": v for k, v in m.items()})
                if m["score"] > best:
                    best = m["score"]
                    torch.save(trainable_state(model, box), out / "best.pt")
                print(f"epoch {epoch}: loss {loss:.4f} | val click1 {m['click1']:.4f} click5 {m['click5']:.4f} "
                      f"box {m['box1']:.4f} messy {m['messy1']:.4f} neg_reject {m['neg_reject']:.3f} "
                      + (f"| REAL hard_reject {m['hard_reject']:.3f} real_keep {m['real_keep']:.3f} "
                         if 'hard_reject' in m else "")
                      + f"| score {m['score']:.4f} (best {best:.4f})")
            else:
                print(f"epoch {epoch}: loss {loss:.4f}")
            ep_times.append(time.time() - t0)
            row["epoch_min"] = ep_times[-1] / 60
            log.append(row)
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
            torch.save(dict(state=trainable_state(model, box), opt=opt.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch, best=best, log=log), out / "last.pt")
            used = (time.time() - t_start) / 3600
            if used + 2 * np.mean(ep_times) / 3600 + 0.5 > args.time_budget_h and epoch < args.epochs:
                print(f"time budget reached after epoch {epoch} ({used:.2f} h); stopping training")
                break
        del train_data, tcand

    if args.skip_test:
        return
    st = torch.load(out / "best.pt", map_location=DEV, weights_only=False)
    model.mask_decoder.load_state_dict(st["mask_decoder"]); box.load_state_dict(st["box"])
    models = {"finetuned": (model, box, True)}
    if args.compare_ckpt:
        cm, cb, _ = build(args, ckpt)
        cs = torch.load(args.compare_ckpt, map_location=DEV, weights_only=False)
        cm.mask_decoder.load_state_dict(cs["mask_decoder"]); cb.load_state_dict(cs["box"])
        models["compare"] = (cm, cb, True)
        print("comparison model:", args.compare_ckpt)
    if not args.no_zero_shot:
        zs = load_sam_med3d(ckpt, DEV); zs.image_encoder = torch.nn.Identity()
        models = {"zero_shot": (zs, None, False), **models}            # turbo has no 3D box prompt
    test_data = preload(te, "first", args.workers, "load test")
    tc = None
    if args.cand_root and Path(args.cand_root, "test").is_dir():
        tc = load_cands(args.cand_root, "test", {c for c, _ in te})
    test(models, test_data, coll, args, cache_root, out, tc)
    print(f"total time {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()

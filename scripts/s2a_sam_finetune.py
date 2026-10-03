"""S2a: SAM-Med3D-turbo fine-tuned on MAMA-MIA images only (Exp 3 baseline, variant A).

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
                                  sample_clicks, upsample)
from cids.models.sam_med3d import download_turbo, load_sam_med3d  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VOX = 128 ** 3
FULL_RES_PROMPTS = [("click", 1), ("click", 5), ("box", 1)]


# ---------------------------------------------------------------------------- data
def find_emb_root(explicit: str = "", search_root: str = "/kaggle/input") -> Path:
    if explicit:
        return Path(explicit)
    hits = sorted({p.parent for p in Path(search_root).rglob("index.csv")
                   if (p.parent / "train").is_dir() and (p.parent / "test").is_dir()})
    if len(hits) != 1:
        raise FileNotFoundError(f"Set --emb_root explicitly; candidates: {hits}")
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
def train_epoch(model, box, opt, scaler, data, args, epoch, step0, total_steps, gen):
    model.train()
    rng = random.Random(args.seed * 1000 + epoch)
    order = list(range(len(data)))
    rng.shuffle(order)
    nb = len(order) // args.batch
    losses = []
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
        coords = torch.zeros(len(idx), 0, 3, device=DEV)
        labels = torch.zeros(len(idx), 0, dtype=torch.long, device=DEV)
        boxes = gt_boxes(gt, args.box_jitter, gen) if use_box else None
        prev, low, loss = torch.zeros_like(gt), None, 0.0
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            for s in range(n_steps):
                if not (use_box and s == 0):
                    c, l = sample_clicks(prev, gt, gen)
                    coords, labels = torch.cat([coords, c], 1), torch.cat([labels, l], 1)
                low = decode(model, box, emb, coords, labels, boxes, None if low is None else low.detach())
                logits = upsample(low)
                loss = loss + seg_loss(logits, gt) / n_steps
                prev = (logits.detach()[:, 0] > 0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if b % 20 == 0:
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}", lr=f"{opt.param_groups[0]['lr']:.1e}")
    bar.close()
    return float(np.mean(losses)) if losses else float("nan"), nb


def batch_dice(prob, gt):
    p, g = prob > 0.5, gt
    inter = (p & g).flatten(1).sum(1).float()
    s = p.flatten(1).sum(1).float() + g.flatten(1).sum(1).float()
    return torch.where(s > 0, 2 * inter / s.clamp(min=1), torch.ones_like(s)).cpu().numpy()


@torch.no_grad()
def run_prompts(model, box, emb, gt, args, with_box=True, seed=233):
    """Returns {('click'|'box', step): prob (B,128^3)} with seeded clicks (same for every model)."""
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
    return out


def validate(model, box, data, args):
    res = {("click", 1): [], ("click", 5): [], ("box", 1): []}
    for s in range(0, len(data), args.batch):
        chunk = data[s:s + args.batch]
        emb = torch.from_numpy(np.stack([c["emb"][0] for c in chunk])).to(DEV).float()
        gt = unpack([c["bits"][0] for c in chunk]).to(DEV)
        out = run_prompts(model, box, emb, gt, args, seed=233 + s)
        for key in res:
            k = (key[0], min(key[1], args.test_clicks))
            res[key].extend(batch_dice(out[k], gt))
    m = {f"{a}{b}": float(np.mean(v)) for (a, b), v in res.items()}
    m["score"] = float(np.mean(list(m.values())))
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


def test(models, test_items, coll, args, cache_root, out):
    crop_rows, full_rows = [], []
    for s in tqdm(range(0, len(test_items), args.batch), desc="test", unit="batch", file=sys.stdout,
                  dynamic_ncols=True, mininterval=10):
        chunk = test_items[s:s + args.batch]
        emb = torch.from_numpy(np.stack([c["emb"][0] for c in chunk])).to(DEV).float()
        gt = unpack([c["bits"][0] for c in chunk]).to(DEV)
        for name, (model, box, with_box) in models.items():
            outs = run_prompts(model, box, emb, gt, args, with_box=with_box, seed=233 + s)
            for (ptype, k), prob in outs.items():
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
    summary = {"crop_dice_1.5mm": {}}
    for (name, ptype), g in cr.groupby(["model", "prompt"]):
        summary["crop_dice_1.5mm"][f"{name}/{ptype}"] = {int(k): round(float(v), 4)
                                                         for k, v in g.groupby("step").dice.mean().items()}
    print("\nTest, crop-level Dice (1.5 mm, same seeded clicks for every model):")
    print(cr.pivot_table(index="step", columns=["model", "prompt"], values="dice").round(4).to_string())
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
        nb = len(train_data) // args.batch
        total = nb * args.epochs
        args.warmup_steps = nb * args.warmup
        ep_times = []
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            loss, _ = train_epoch(model, box, opt, scaler, train_data, args, epoch, (epoch - 1) * nb, total, gen)
            row = dict(epoch=epoch, loss=loss, lr=opt.param_groups[0]["lr"])
            if epoch % args.val_every == 0 or epoch == args.epochs:
                m = validate(model, box, val_data, args)
                row.update({f"val_{k}": v for k, v in m.items()})
                if m["score"] > best:
                    best = m["score"]
                    torch.save(trainable_state(model, box), out / "best.pt")
                print(f"epoch {epoch}: loss {loss:.4f} | val click1 {m['click1']:.4f} click5 {m['click5']:.4f} "
                      f"box {m['box1']:.4f} | score {m['score']:.4f} (best {best:.4f})")
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
        del train_data

    if args.skip_test:
        return
    st = torch.load(out / "best.pt", map_location=DEV, weights_only=False)
    model.mask_decoder.load_state_dict(st["mask_decoder"]); box.load_state_dict(st["box"])
    models = {"finetuned": (model, box, True)}
    if not args.no_zero_shot:
        zs = load_sam_med3d(ckpt, DEV); zs.image_encoder = torch.nn.Identity()
        models = {"zero_shot": (zs, None, False), **models}            # turbo has no 3D box prompt
    test_data = preload(te, "first", args.workers, "load test")
    test(models, test_data, coll, args, cache_root, out)
    print(f"total time {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()

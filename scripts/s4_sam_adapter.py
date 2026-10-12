"""S4: frequency adapter for SAM-S2b (⑤ in the CIDS diagram, no metadata yet).

SAM's own input crop -> 3D Haar (8 bands) -> learned band weights -> small CNN -> residual added to SAM's
cached image embedding. Starts from the S2b checkpoint (decoder LoRA + box tokens); by default ONLY the
adapter trains (--train_decoder also trains LoRA + box tokens), so any change is the adapter's.
Same prompts as S2b: clean / messy / negative boxes (+ clicks). Validation score as S2b:
mean(clean-box Dice, messy-box Dice, negative rejection).
Test: S2b alone vs S2b + adapter on identical boxes/clicks: crop Dice, negative rejection, full-resolution
Dice/HD95/NSD (box, messy). Also writes the learned band weights (band_weights.json).

Data: crop 0 (tumour-centred) of every case; its embedding from S0 (`cids-sam-emb`) and its image crop
rebuilt from the 1 mm cache exactly as S0 made it (cids/fcouncil/sam_crops.py), kept in RAM (~5 GB).

Example
  python scripts/s4_sam_adapter.py --emb_root <cids-sam-emb> --cache_root <cache> \\
      --sam_ckpt <cids-checkpoints>/sam_s2b_best.pt --out /kaggle/working/s4_adapter
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s2a_sam_finetune as s2a  # noqa: E402
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.fcouncil.adapter import SamFreqAdapter  # noqa: E402
from cids.fcouncil.sam_crops import sam_crop_from_case  # noqa: E402
from cids.models.sam_lora import (BoxCorners, add_lora, decode, realistic_boxes, sample_clicks,  # noqa: E402
                                  upsample)
from cids.models.sam_med3d import download_turbo, load_sam_med3d  # noqa: E402

DEV = s2a.DEV


class CropData(Dataset):
    """(cid, emb npz, cache npz) -> embedding of crop 0, its label bits, its SAM input crop, origin, shape15."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, ef, cf = self.items[i]
        d = np.load(ef)
        img, lbl = load_case(cf)
        crop = sam_crop_from_case(img, lbl, d["origin"][0])
        lab = d["label"][0]
        return dict(cid=cid, emb=d["emb_post1"][0].astype(np.float16), crop=crop.astype(np.float16),
                    bits=np.packbits(lab.reshape(-1) > 0), origin=d["origin"][0], shape15=d["shape15"])


def _identity(x):
    return x


def _worker_init(_):
    torch.set_num_threads(1)


def preload(items, workers, desc):
    dl = DataLoader(CropData(items), batch_size=None, num_workers=workers, collate_fn=_identity,
                    worker_init_fn=_worker_init)
    return [x for x in tqdm(dl, desc=desc, unit="case", file=sys.stdout, mininterval=10)]


def unpack(bits):
    return torch.from_numpy(np.unpackbits(np.stack(bits), axis=1, count=128 ** 3).reshape(-1, 128, 128, 128).astype(bool))


def batch(data, idx):
    emb = torch.from_numpy(np.stack([data[i]["emb"] for i in idx])).to(DEV).float()
    crop = torch.from_numpy(np.stack([data[i]["crop"] for i in idx]))[:, None].to(DEV).float()
    gt = unpack([data[i]["bits"] for i in idx]).to(DEV)
    return emb, crop, gt


def build(a, turbo):
    sam = load_sam_med3d(turbo, DEV)
    sam.image_encoder = torch.nn.Identity()
    for p in sam.parameters():
        p.requires_grad_(False)
    add_lora(sam.mask_decoder, ("q_proj", "v_proj"), 8, 16)
    box = BoxCorners(sam.prompt_encoder)
    st = torch.load(a.sam_ckpt, map_location=DEV, weights_only=False)
    sam.mask_decoder.load_state_dict(st["mask_decoder"]); box.load_state_dict(st["box"])
    sam.to(DEV); box.to(DEV)
    for p in list(sam.parameters()) + list(box.parameters()):
        p.requires_grad_(False)
    if a.train_decoder:
        for n, p in sam.mask_decoder.named_parameters():
            p.requires_grad_(n.endswith((".A", ".B")))
        for p in box.parameters():
            p.requires_grad_(True)
    return sam, box


def train_epoch(sam, box, adapter, opt, scaler, data, a, epoch, step0, total, gen):
    sam.train(); adapter.train()
    rng = random.Random(a.seed * 1000 + epoch)
    order = list(range(len(data))); rng.shuffle(order)
    nb = len(order) // a.batch
    losses = []
    for b in tqdm(range(nb), desc=f"ep {epoch}", unit="it", file=sys.stdout, mininterval=10):
        idx = order[b * a.batch:(b + 1) * a.batch]
        emb, crop, gt = batch(data, idx)
        for g in opt.param_groups:
            g["lr"] = s2a.lr_at(step0 + b, total, a.warmup_steps, g["base_lr"])
        use_box = rng.random() < a.p_box
        n_steps = rng.randint(1, a.max_clicks)
        boxes, tgt = None, gt
        if use_box:
            boxes, neg = realistic_boxes(gt, rng, a.p_messy, a.p_neg, a.box_jitter)
            tgt = gt & ~neg[:, None, None, None]
        coords = torch.zeros(len(idx), 0, 3, device=DEV); labels = torch.zeros(len(idx), 0, dtype=torch.long, device=DEV)
        prev, low, loss = torch.zeros_like(gt), None, 0.0
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
            e = emb + adapter(crop)
            for s in range(n_steps):
                if not (use_box and s == 0):
                    c, l = sample_clicks(prev, tgt, gen)
                    coords, labels = torch.cat([coords, c], 1), torch.cat([labels, l], 1)
                low = decode(sam, box, e, coords, labels, boxes, None if low is None else low.detach())
                logits = upsample(low)
                loss = loss + s2a.seg_loss(logits, tgt) / n_steps
                prev = logits.detach()[:, 0] > 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses)), nb


@torch.no_grad()
def adapted(emb, crop, adapter, amp):
    if adapter is None:
        return emb
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        return (emb + adapter(crop)).float()


def validate(sam, box, adapter, data, a):
    sam.eval(); adapter.eval()
    res = {k: [] for k in ("click1", "click5", "box1", "messy1")}
    rej = []
    for s in range(0, len(data), a.batch):
        idx = list(range(s, min(s + a.batch, len(data))))
        emb, crop, gt = batch(data, idx)
        out = s2a.run_prompts(sam, box, adapted(emb, crop, adapter, a.amp), gt, a, seed=233 + s)
        res["click1"].extend(s2a.batch_dice(out[("click", 1)], gt))
        res["click5"].extend(s2a.batch_dice(out[("click", a.test_clicks)], gt))
        res["box1"].extend(s2a.batch_dice(out[("box", 1)], gt))
        res["messy1"].extend(s2a.batch_dice(out[("messy", 1)], gt))
        rej.extend(s2a.neg_stats(out[("neg", 1)], gt, 233 + s)[0])
    m = {k: float(np.mean(v)) for k, v in res.items()}
    m["neg_reject"] = float(np.nanmean(rej))
    m["score"] = float(np.mean([m["box1"], m["messy1"], m["neg_reject"]]))
    return m


def test(models, adapter, data, coll, a, cache_root, out):
    """models: [(name, sam, box, adapter_or_None)] -- all get identical boxes and clicks."""
    rows, full = [], []
    for s in tqdm(range(0, len(data), a.batch), desc="test", unit="batch", file=sys.stdout, mininterval=10):
        idx = list(range(s, min(s + a.batch, len(data))))
        emb, crop, gt = batch(data, idx)
        for name, sam, box, ad in models:
            o = s2a.run_prompts(sam, box, adapted(emb, crop, ad, a.amp), gt, a, seed=233 + s)
            rej, ml = s2a.neg_stats(o[("neg", 1)], gt, 233 + s)
            for j, i in enumerate(idx):
                c = data[i]["cid"]
                r = dict(cache_id=c, collection=coll[c], model=name, neg_reject=rej[j], neg_pred_ml=ml[j])
                for (pt, k), p in o.items():
                    if pt != "neg":
                        r[f"{pt}{k}"] = float(s2a.batch_dice(p[j:j + 1], gt[j:j + 1])[0])
                rows.append(r)
                if cache_root is not None:
                    path = cache_root / ("train" if a.official[c] == "train" else "val") / f"{c}.npz"
                    for key in (("box", 1), ("messy", 1)):
                        m = s2a.full_res_metrics(o[key][j].cpu(), data[i]["origin"], data[i]["shape15"], path)
                        full.append(dict(cache_id=c, collection=coll[c], model=name, prompt=f"{key[0]}{key[1]}", **m))
    cr = pd.DataFrame(rows); cr.to_csv(out / "test_crop_metrics.csv", index=False)
    keys = ["box1", "messy1", "click1", f"click{a.test_clicks}", "neg_reject"]
    summ = {"crop": cr.groupby("model")[keys].mean().round(4).to_dict(orient="index")}
    from scipy.stats import wilcoxon
    p = cr.pivot_table(index="cache_id", columns="model", values="messy1")
    summ["messy1_adapter_minus_s2b"] = dict(mean=float((p.s2b_adapter - p.s2b).mean()),
                                           wilcoxon_p=float(wilcoxon(p.s2b_adapter, p.s2b).pvalue))
    if full:
        fr = pd.DataFrame(full); fr.to_csv(out / "test_full_metrics.csv", index=False)
        agg = fr.groupby(["model", "prompt"]).agg(dice=("dice", "mean"), hd95=("hd95", "median"), nsd=("nsd", "mean"))
        summ["full_res_1mm"] = {f"{m}/{q}": r.round(4).to_dict() for (m, q), r in agg.iterrows()}
        print("\nfull resolution (1 mm):"); print(agg.round(4).to_string())
    print("\ncrop level (1.5 mm):"); print(cr.groupby("model")[keys].mean().round(4).to_string())
    print("\nmessy-box Dice per collection:"); print(cr.pivot_table(index="collection", columns="model", values="messy1").round(4).to_string())
    summ["band_weights"] = adapter.readout()
    json.dump(summ, open(out / "test_summary.json", "w"), indent=2, default=float)
    print(json.dumps(summ, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_root", required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--sam_ckpt", required=True, help="S2b best.pt (sam_s2b_best.pt)")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", default="/kaggle/working/s4_adapter")
    ap.add_argument("--train_decoder", action="store_true", help="also train LoRA + box tokens (lr x 0.2)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max_clicks", type=int, default=5)
    ap.add_argument("--p_box", type=float, default=0.5)
    ap.add_argument("--box_jitter", type=int, default=5)
    ap.add_argument("--p_messy", type=float, default=0.4)
    ap.add_argument("--p_neg", type=float, default=0.2)
    ap.add_argument("--test_clicks", type=int, default=5)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--skip_test", action="store_true")
    a = ap.parse_args()
    a.amp = DEV == "cuda"
    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    gen = torch.Generator(device=DEV).manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=2)
    emb_root = s2a.find_emb_root(a.emb_root)
    cache_root = find_cache_root(a.cache_root)
    sp = pd.read_csv(a.splits)
    coll = dict(zip(sp.cache_id, sp.collection)); a.official = dict(zip(sp.cache_id, sp.official))

    def items(split):
        r = sp[sp.split == split]
        lst = [(c, emb_root / ("train" if o == "train" else "test") / f"{c}.npz",
                cache_root / ("train" if o == "train" else "val") / f"{c}.npz") for c, o in zip(r.cache_id, r.official)]
        return [x for x in lst if x[1].exists() and x[2].exists()]

    tr, va, te = items("train"), items("val"), items("test")
    if a.limit_train:
        tr = tr[:a.limit_train]
    if a.limit_eval:
        va, te = va[:a.limit_eval], te[:a.limit_eval]
    print(f"embeddings {emb_root} | cache {cache_root}\ntrain {len(tr)} | val {len(va)} | test {len(te)} | {DEV}")

    turbo = None if a.random_weights else (a.turbo_ckpt or download_turbo("/tmp/sam_ckpt"))
    sam, box = build(a, turbo)
    adapter = SamFreqAdapter().to(DEV)
    groups = [dict(params=list(adapter.parameters()), lr=a.lr, base_lr=a.lr)]
    dec = [p for p in list(sam.parameters()) + list(box.parameters()) if p.requires_grad]
    if dec:
        groups.append(dict(params=dec, lr=a.lr * 0.2, base_lr=a.lr * 0.2))
    print(f"trainable: adapter {sum(p.numel() for p in adapter.parameters()):,}"
          + (f" + decoder LoRA/box {sum(p.numel() for p in dec):,}" if dec else " (SAM-S2b decoder frozen)"))
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)

    best, log = -1.0, []
    train_data = preload(tr, a.workers, "load train")
    val_data = preload(va, a.workers, "load val")
    m0 = validate(sam, box, adapter, val_data, a)                      # = S2b (adapter is zero at start)
    print(f"epoch 0 (= S2b): box {m0['box1']:.4f} messy {m0['messy1']:.4f} neg_reject {m0['neg_reject']:.3f} score {m0['score']:.4f}")
    log.append(dict(epoch=0, **{f"val_{k}": v for k, v in m0.items()}))
    nb = len(train_data) // a.batch
    a.warmup_steps = nb * a.warmup
    for epoch in range(1, a.epochs + 1):
        te0 = time.time()
        loss, _ = train_epoch(sam, box, adapter, opt, scaler, train_data, a, epoch, (epoch - 1) * nb, nb * a.epochs, gen)
        row = dict(epoch=epoch, loss=loss, epoch_min=(time.time() - te0) / 60)
        msg = f"epoch {epoch}: loss {loss:.4f}"
        if epoch % a.val_every == 0 or epoch == a.epochs:
            m = validate(sam, box, adapter, val_data, a)
            row.update({f"val_{k}": v for k, v in m.items()})
            if m["score"] > best:
                best = m["score"]
                torch.save(dict(adapter=adapter.state_dict(), mask_decoder=sam.mask_decoder.state_dict(),
                                box=box.state_dict()), out / "best.pt")
            bw = " ".join(f"{k}={v:.2f}" for k, v in adapter.readout().items())
            msg += (f" | val box {m['box1']:.4f} messy {m['messy1']:.4f} neg_reject {m['neg_reject']:.3f} "
                    f"click1 {m['click1']:.4f} | score {m['score']:.4f} (best {best:.4f}) | bands {bw}")
        print(msg)
        log.append(row)
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    del train_data
    if a.skip_test:
        return
    st = torch.load(out / "best.pt", map_location=DEV, weights_only=False)
    adapter.load_state_dict(st["adapter"]); sam.mask_decoder.load_state_dict(st["mask_decoder"]); box.load_state_dict(st["box"])
    json.dump(adapter.readout(), open(out / "band_weights.json", "w"), indent=2)
    adapter.eval()
    ref = argparse.Namespace(**{**vars(a), "train_decoder": False})
    sam0, box0 = build(ref, turbo)                                     # the original S2b, untouched
    models = [("s2b", sam0.eval(), box0.eval(), None), ("s2b_adapter", sam.eval(), box.eval(), adapter)]
    test_data = preload(te, a.workers, "load test")
    test(models, adapter, test_data, coll, a, cache_root, out)
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

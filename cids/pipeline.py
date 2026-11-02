"""CIDS-S3d as ONE file: SegResNet-100 (finder) + SAM-Med3D-turbo/S2b (checker) + learned jury -> tumour mask.

Test Dice 0.735 on MAMA-MIA (306 official test cases). Fully automatic, no prompts, no ground truth.

    from cids.pipeline import CIDSPipeline
    pipe = CIDSPipeline.load("cids_s3d_pipeline.pt")          # cuda if available
    out = pipe.predict(img)                                    # img: (2, X, Y, Z) [pre, post1], 1 mm RAS,
    mask = out["mask"]                                         #      each channel scaled to [0, 1] (1-99 pct)
    out["candidates"]                                          # per candidate: clues + P(tumour) + kept

Bundle contents (torch.save dict): SegResNet state_dict, the FULL SAM state_dict (turbo encoder + LoRA
decoder + box tokens, so no download is needed), the fitted jury (scikit-learn; load with the same major
scikit-learn version), and the settings (threshold 0.1 ml candidates, top 5, jury cut-off).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import s3a_seg2sam as s3  # noqa: E402
import s3d_learned_jury as s3d  # noqa: E402

from cids.data.volumes import pad_to, three_channel, unpad  # noqa: E402
from cids.models.sam_lora import BoxCorners, add_lora  # noqa: E402
from cids.models.sam_med3d import load_sam_med3d  # noqa: E402
from cids.models.segresnet_text import TextSegResNet  # noqa: E402

VERSION = "cids-s3d-1"


class CIDSPipeline:
    def __init__(self, seg, sam, box, jury, cfg, device):
        self.seg, self.sam, self.box, self.jury, self.cfg, self.device = seg, sam, box, jury, cfg, device

    # ------------------------------------------------------------------ build / save / load
    @classmethod
    def build(cls, seg_ckpt, sam_ckpt, turbo_ckpt, jury, threshold, top_k=5, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        s3.DEV = device
        seg, sam, box = s3.load_models(seg_ckpt, sam_ckpt, turbo_ckpt)
        cfg = dict(version=VERSION, threshold=float(threshold), top_k=int(top_k), min_vox=100,
                   features=list(s3d.FEATURES), seg_prob_threshold=0.5, lora_r=8, lora_alpha=16.0)
        return cls(seg, sam, box, jury, cfg, device)

    def save(self, path):
        torch.save(dict(cfg=self.cfg, segresnet=self.seg.state_dict(), sam=self.sam.state_dict(),
                        box=self.box.state_dict(), jury=self.jury), path)

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        s3.DEV = device
        b = torch.load(path, map_location="cpu", weights_only=False)
        cfg = b["cfg"]
        seg = TextSegResNet(text=False); seg.load_state_dict(b["segresnet"]); seg.to(device).eval()
        sam = load_sam_med3d(None, "cpu")                       # architecture only; weights from the bundle
        add_lora(sam.mask_decoder, ("q_proj", "v_proj"), cfg["lora_r"], cfg["lora_alpha"])
        sam.load_state_dict(b["sam"]); sam.to(device).eval()
        box = BoxCorners(sam.prompt_encoder); box.load_state_dict(b["box"]); box.to(device).eval()
        return cls(seg, sam, box, b["jury"], cfg, device)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, img: np.ndarray) -> dict:
        """img (2, X, Y, Z) float [pre, post1] at 1 mm -> dict(mask (X,Y,Z) bool, seg_prob, candidates)."""
        img = np.asarray(img, np.float32)
        amp = self.device == "cuda"
        x, pads = pad_to(three_channel(img), (96, 96, 96))
        sprob = unpad(s3.seg_prob(self.seg, torch.from_numpy(x), amp), pads)
        lab, cands = s3.components(sprob > self.cfg["seg_prob_threshold"], min_vox=self.cfg["min_vox"],
                                   top_k=self.cfg["top_k"])
        if not cands:
            return dict(mask=np.zeros(sprob.shape, bool), seg_prob=sprob, candidates=pd.DataFrame())
        refined = s3.sam_refine(self.sam, self.box, img, lab, cands, amp)
        feats = pd.DataFrame(s3d.candidate_rows("case", sprob, lab, cands, refined, np.zeros(sprob.shape, bool)))
        p = self.jury.predict_proba(feats[self.cfg["features"]].to_numpy(float))[:, 1]
        keep = p >= self.cfg["threshold"]
        if not keep.any():                                     # every case has a tumour: keep the most likely
            keep[int(np.argmax(p))] = True
        feats = feats.drop(columns=["size_vox", "inter_vox", "gt_vox", "on_tumour", "cache_id"])
        feats["p_tumour"], feats["kept"] = p, keep
        return dict(mask=np.isin(lab, feats.k[keep].to_numpy()), seg_prob=sprob, candidates=feats)

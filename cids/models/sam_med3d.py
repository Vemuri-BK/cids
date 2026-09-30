"""SAM-Med3D-turbo loading and click-prompt inference (zero-shot baseline).

Model definition comes from the official `medim` package (uni-medical); weights from
huggingface.co/blueyo0/SAM-Med3D (sam_med3d_turbo.pth, key 'model_state_dict').
We load the state dict ourselves with a strict key check, because medim loads with
strict=False and would silently ignore a mismatched checkpoint.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

HF_REPO = "blueyo0/SAM-Med3D"
HF_FILE = "sam_med3d_turbo.pth"


def download_turbo(local_dir: str = "/kaggle/working/ckpt") -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, local_dir=local_dir)


def load_sam_med3d(ckpt_path: Optional[str], device: str = "cuda"):
    import medim
    model = medim.create_model("SAM-Med3D", pretrained=False)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"checkpoint missing {len(missing)} keys, e.g. {missing[:5]}")
        if unexpected:
            print(f"[warn] {len(unexpected)} unexpected keys ignored, e.g. {unexpected[:3]}")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
def sample_click(prev: torch.Tensor, gt: torch.Tensor, rng: np.random.Generator
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """SAM-Med3D 'random' click: a voxel from the error region (FN -> positive, FP -> negative);
    on the first click (empty prev) a random tumour voxel. prev/gt: (D,H,W) bool."""
    fn = gt & ~prev
    fp = ~gt & prev
    err = fn | fp
    if not err.any():
        pts = torch.argwhere(gt)
        p = pts[rng.integers(len(pts))]
        pos = True
    else:
        pts = torch.argwhere(err)
        p = pts[rng.integers(len(pts))]
        pos = bool(fn[p[0], p[1], p[2]])
    return p.reshape(1, 1, 3).float(), torch.tensor([[int(pos)]], dtype=torch.long)


@torch.no_grad()
def click_inference(model, emb: torch.Tensor, gt: torch.Tensor, n_clicks: int, seed: int
                    ) -> list:
    """Iterative click prompting on one crop. emb: (1,384,8,8,8) on device; gt: (128,128,128) bool (cpu).
    Returns the binary 128^3 prediction after each click (list of cpu bool tensors)."""
    dev = emb.device
    rng = np.random.default_rng(seed)
    coords = torch.zeros(1, 0, 3, device=dev)
    labels = torch.zeros(1, 0, dtype=torch.long, device=dev)
    low = torch.zeros(1, 1, 32, 32, 32, device=dev)
    prev = torch.zeros_like(gt, dtype=torch.bool)
    preds = []
    for _ in range(n_clicks):
        c, l = sample_click(prev, gt, rng)
        coords = torch.cat([coords, c.to(dev)], 1)
        labels = torch.cat([labels, l.to(dev)], 1)
        sp, de = model.prompt_encoder(points=[coords, labels], boxes=None, masks=low)
        low, _ = model.mask_decoder(image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                                    multimask_output=False)
        hi = F.interpolate(low, size=gt.shape, mode="trilinear", align_corners=False)
        prev = (torch.sigmoid(hi)[0, 0] > 0.5).cpu()
        preds.append(prev)
    return preds


def dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p, g = pred.bool(), gt.bool()
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2 * (p & g).sum() / s)

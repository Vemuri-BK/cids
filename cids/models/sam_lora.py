"""SAM-Med3D-turbo adaptation on cached embeddings (image only, no text / frequency prompts).

- LoRA on the mask decoder's attention q/v projections (14 Linear layers, ~70k params),
  or full fine-tuning of the mask decoder (--tune decoder_full).
- Box prompts: SAM-Med3D-turbo was trained with points only (its prompt encoder has 2 point
  embeddings and a 2D-only box path), so we add two learnable 3D box-corner embeddings,
  initialised from the positive-point embedding. A box is encoded as 2 corner tokens.
- Batched click simulation on GPU, same rule as SAM-Med3D's 'random' click: first click a random
  tumour voxel; later clicks a random voxel of the error region (FN -> positive, FP -> negative).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero -> identical to base at init
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def add_lora(module: nn.Module, targets=("q_proj", "v_proj"), r: int = 8, alpha: float = 16.0) -> List[str]:
    """Replace every nn.Linear whose attribute name is in `targets` by a LoRALinear (in place)."""
    done = []
    for name, sub in list(module.named_modules()):
        for t in targets:
            child = getattr(sub, t, None)
            if isinstance(child, nn.Linear):
                setattr(sub, t, LoRALinear(child, r, alpha))
                done.append(f"{name}.{t}" if name else t)
    return done


# ---------------------------------------------------------------------------- box prompt
class BoxCorners(nn.Module):
    def __init__(self, prompt_encoder):
        super().__init__()
        w = prompt_encoder.point_embeddings[1].weight.detach().clone()     # (1, C) positive point
        self.corner = nn.Parameter(torch.cat([w, w], 0))                   # (2, C)

    def forward(self, prompt_encoder, boxes: torch.Tensor) -> torch.Tensor:
        """boxes: (B, 2, 3) voxel coords [min corner, max corner] -> (B, 2, C)."""
        pe = prompt_encoder.pe_layer.forward_with_coords(boxes.float() + 0.5, prompt_encoder.input_image_size)
        return pe + self.corner[None]


def gt_boxes(gt: torch.Tensor, jitter: int = 0, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """gt: (B, D, H, W) bool -> (B, 2, 3) [min, max] voxel coords of the tumour (whole crop if empty),
    each side randomly moved by up to +-jitter voxels."""
    B, *S = gt.shape
    out = torch.zeros(B, 2, 3, device=gt.device)
    for b in range(B):
        idx = torch.nonzero(gt[b])
        if len(idx):
            out[b, 0], out[b, 1] = idx.min(0).values.float(), idx.max(0).values.float()
        else:
            out[b, 1] = torch.tensor(S, device=gt.device, dtype=torch.float) - 1
    if jitter:
        out = out + torch.randint(-jitter, jitter + 1, out.shape, generator=gen, device=gt.device)
        hi = torch.tensor(S, device=gt.device, dtype=torch.float) - 1
        out = torch.minimum(torch.clamp(out, min=0), hi)
        out[:, 1] = torch.maximum(out[:, 1], out[:, 0])
    return out


# ---------------------------------------------------------------------------- clicks
def _random_voxel(mask: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    """mask (B, N) bool, every row non-empty -> (B,) flat index of a uniformly random True voxel."""
    r = torch.rand(mask.shape, generator=gen, device=mask.device)   # gen must live on mask.device
    return (r * mask).argmax(1)


def sample_clicks(prev: torch.Tensor, gt: torch.Tensor, gen: Optional[torch.Generator] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """prev, gt: (B, D, H, W) bool. Returns coords (B, 1, 3) float and labels (B, 1) long."""
    B, D, H, W = gt.shape
    g, p = gt.reshape(B, -1), prev.reshape(B, -1)
    fn, fp = g & ~p, ~g & p
    err = fn | fp
    has_err, has_gt = err.any(1), g.any(1)
    pool = torch.where(has_err[:, None], err, torch.where(has_gt[:, None], g, torch.ones_like(g)))
    i = _random_voxel(pool, gen)
    pos = torch.where(has_err, fn.gather(1, i[:, None])[:, 0], has_gt)
    coords = torch.stack([i // (H * W), (i // W) % H, i % W], 1).float()[:, None]
    return coords, pos.long()[:, None]


# ---------------------------------------------------------------------------- decoding
def decode(model, box_mod: Optional[BoxCorners], emb: torch.Tensor,
           coords: Optional[torch.Tensor], labels: Optional[torch.Tensor],
           boxes: Optional[torch.Tensor], low_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """One pass of prompt encoder + mask decoder. emb (B,384,8,8,8). Returns low-res logits (B,1,32,32,32)."""
    pe_ = model.prompt_encoder
    B = emb.shape[0]
    parts = []
    if coords is not None and coords.shape[1] > 0:
        parts.append(pe_._embed_points(coords, labels, pad=boxes is None))
    if boxes is not None:
        parts.append(box_mod(pe_, boxes))
    sparse = torch.cat(parts, 1) if parts else torch.empty(B, 0, pe_.embed_dim, device=emb.device)
    if low_mask is not None:
        dense = pe_._embed_masks(low_mask)
    else:
        dense = pe_.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(B, -1, *pe_.image_embedding_size)
    low, _ = model.mask_decoder(image_embeddings=emb, image_pe=pe_.get_dense_pe(),
                                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                multimask_output=False)
    return low


def upsample(low: torch.Tensor, size: int = 128) -> torch.Tensor:
    return F.interpolate(low, size=(size,) * 3, mode="trilinear", align_corners=False)


@torch.no_grad()
def interactive(model, box_mod, emb, gt, n_clicks: int, use_box: bool, gen=None) -> List[torch.Tensor]:
    """Evaluation protocol (batched). gt (B,128,128,128) bool on device.
    use_box=False: n_clicks iterative clicks (first = random tumour voxel).
    use_box=True : box first (no click), then n_clicks-1 corrective clicks.
    Returns the probability map (B,128,128,128) after every step."""
    coords = torch.zeros(gt.shape[0], 0, 3, device=gt.device)
    labels = torch.zeros(gt.shape[0], 0, dtype=torch.long, device=gt.device)
    boxes = gt_boxes(gt) if use_box else None
    prev = torch.zeros_like(gt)
    low, outs = None, []
    steps = n_clicks if not use_box else max(n_clicks, 1)
    for k in range(steps):
        if not (use_box and k == 0):
            c, l = sample_clicks(prev, gt, gen)
            coords, labels = torch.cat([coords, c], 1), torch.cat([labels, l], 1)
        low = decode(model, box_mod, emb, coords, labels, boxes, low)
        prob = torch.sigmoid(upsample(low.float()))[:, 0]
        prev = prob > 0.5
        outs.append(prob)
    return outs

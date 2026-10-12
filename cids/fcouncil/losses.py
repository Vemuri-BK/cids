"""Dual-supervision loss for the frequency council.

  L = DiceCE(final, GT)
    + λ_exp · mean_k DiceCE(expert_k, GT)                each independent expert must segment on its own
    + λ_kd  · agreement-gated distillation               the two teachers: finder (SegResNet) and checker (SAM)
    + λ_bal · KL(mean gate weight || uniform)             keep all 4 experts in use

Agreement-gated distillation: where the finder and the checker agree (|p_f - p_c| < tau) the final mask
is pulled towards their mean soft probability; where they disagree only the ground truth decides.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def dice_ce(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """logits, target (B,1,D,H,W); squared soft Dice + BCE (as MONAI DiceCELoss(sigmoid, squared_pred))."""
    p = torch.sigmoid(logits.float())
    t = target.float()
    dims = tuple(range(2, p.dim()))
    dice = 1 - (2 * (p * t).sum(dims) + smooth) / ((p * p).sum(dims) + (t * t).sum(dims) + smooth)
    return dice.mean() + F.binary_cross_entropy_with_logits(logits.float(), t)


def council_loss(out: Dict[str, torch.Tensor], gt: torch.Tensor, p_finder: torch.Tensor, p_checker: torch.Tensor,
                 lam_exp: float = 0.5, lam_kd: float = 0.3, lam_bal: float = 0.01, tau: float = 0.5,
                 use_teachers: bool = True) -> Dict[str, torch.Tensor]:
    main = dice_ce(out["final"], gt)
    exp = torch.stack([dice_ce(out["experts"][:, k:k + 1], gt) for k in range(out["experts"].shape[1])]).mean()
    kd = torch.zeros((), device=gt.device)
    if use_teachers and lam_kd > 0:
        agree = (p_finder - p_checker).abs() < tau
        if agree.any():
            target = 0.5 * (p_finder + p_checker)
            bce = F.binary_cross_entropy_with_logits(out["final"].float(), target.float(), reduction="none")
            kd = bce[agree].mean()
    w = out["weights"].float().mean(dim=(0, 2, 3, 4))
    bal = (w * torch.log(w * w.numel() + 1e-8)).sum()
    total = main + lam_exp * exp + lam_kd * kd + lam_bal * bal
    return dict(total=total, main=main.detach(), experts=exp.detach(), kd=kd.detach(), bal=bal.detach())

"""Council member B: MONAI SegResNet, optionally conditioned on the clinical instruction.

Text conditioning = FiLM at the bottleneck and after every decoder stage:
    x <- x * (1 + gamma(t)) + beta(t)
gamma/beta come from a small MLP on the (frozen) BioClinicalBERT sentence vector t.
The last FiLM layer is zero-initialised, so at step 0 the text model is exactly the
image-only model; any difference that appears is learned from the data.
With text=False the FiLM modules are not created (identical image-only network).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class FiLM(nn.Module):
    def __init__(self, text_dim: int, channels: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(text_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        g, b = self.net(t).chunk(2, dim=-1)
        shape = (x.shape[0], x.shape[1]) + (1,) * (x.dim() - 2)
        return x * (1 + g.view(shape)) + b.view(shape)


class TextSegResNet(SegResNet):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_filters: int = 16,
                 text: bool = False, text_dim: int = 768, dropout_prob: Optional[float] = None):
        super().__init__(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                         init_filters=init_filters, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                         dropout_prob=dropout_prob)
        self.use_text = text
        if text:
            n_down = len(self.down_layers)
            bott = init_filters * 2 ** (n_down - 1)
            ups = [init_filters * 2 ** (n_down - 2 - i) for i in range(len(self.up_layers))]
            self.film_bott = FiLM(text_dim, bott)
            self.film_up = nn.ModuleList([FiLM(text_dim, c) for c in ups])

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()
        if self.use_text:
            if t is None:
                raise ValueError("text model needs the text vector t")
            x = self.film_bott(x, t)
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            if self.use_text:
                x = self.film_up[i](x, t)
        if self.use_conv_final:
            x = self.conv_final(x)
        return x

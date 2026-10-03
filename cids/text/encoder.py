"""Frozen BioClinicalBERT sentence encoder (mean-pooled last hidden state), with a cache."""
from __future__ import annotations

from typing import Dict, List

import torch

DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


class TextEncoder:
    def __init__(self, name: str = DEFAULT_MODEL, device: str = "cuda", max_len: int = 160):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device, self.max_len = device, max_len
        self.dim = self.model.config.hidden_size
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Returns (N, dim) float32 on self.device. Identical strings are computed once."""
        todo = [s for s in dict.fromkeys(texts) if s not in self._cache]
        for i in range(0, len(todo), 32):
            chunk = todo[i:i + 32]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_len,
                           return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            for s, row in zip(chunk, v):
                self._cache[s] = row.float()
        return torch.stack([self._cache[s] for s in texts])

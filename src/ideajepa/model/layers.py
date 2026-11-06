"""Shared transformer building blocks (pre-LN, SDPA)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d, mult=4, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(d, mult * d)
        self.proj = nn.Linear(mult * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc(x)
        if not getattr(self, "linear", False):   # linear = True: identity activation (P linear start)
            h = F.gelu(h)
        return self.drop(self.proj(h))


class Attention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dh = d // n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.dropout = dropout

    def split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)

    def kv_heads(self, x):
        k, v = self.kv(x).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def attend(self, q, k, v, mask=None, is_causal=False):
        """q: [B, Lq, d] (unsplit), k/v already split [B, H, Lk, dh]. mask: bool, True = keep."""
        qh = self.split(self.q(q))
        out = F.scaled_dot_product_attention(qh, k, v, attn_mask=mask, is_causal=is_causal,
                                             dropout_p=self.dropout if self.training else 0.0)
        B, H, L, dh = out.shape
        return self.o(out.transpose(1, 2).reshape(B, L, H * dh))

    def forward(self, x, mask=None, is_causal=False, ctx=None):
        k, v = self.kv_heads(x if ctx is None else ctx)
        return self.attend(x, k, v, mask, is_causal)


class Block(nn.Module):
    def __init__(self, d, n_heads, mlp_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d, mlp_mult, dropout)

    def forward(self, x, mask=None, is_causal=False):
        x = x + self.attn(self.ln1(x), mask=mask, is_causal=is_causal)
        return x + self.mlp(self.ln2(x))


def causal_mask(L, device):
    return torch.ones(L, L, dtype=torch.bool, device=device).tril()


def init_weights(module: nn.Module, std=0.02):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)

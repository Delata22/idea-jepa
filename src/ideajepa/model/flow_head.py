"""Generative idea head by convergence (rectified flow), standalone module (not wired into IdeaJEPA).

The target idea z1 is the r slots of E_ema in the JEPA normalisation: [k ; v/|v|] per slot (keys are already
unit-norm). A velocity field F(z_tau, tau, c) is trained on straight paths from noise to the target:
    z_tau = (1 - tau) z0 + tau z1,   loss = || F(z_tau, tau, c) - (z1 - z0) ||^2,   tau ~ logit-normal,
with z0 ~ N(0, I) scaled by a running per-dimension std of the targets. Sampling integrates dz/dtau = F with
Euler steps from tau = 0 to 1. The context c is P's per-slot output (r tokens), used twice: added slot to
slot, and as the memory of a cross-attention in every block.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, Attention, init_weights


def to_target(k, v):
    """JEPA normalisation of idea slots: [k ; v/|v|] -> [N, r, 2*ds]."""
    return torch.cat([F.normalize(k, dim=-1), F.normalize(v, dim=-1)], -1)


def from_target(z):
    """Back to (k, v) with unit-norm keys and unit-norm values (the decoder rescales values itself when
    z_norm='idea'; otherwise a norm must be supplied by the caller)."""
    k, v = z.chunk(2, -1)
    return F.normalize(k, dim=-1), F.normalize(v, dim=-1)


def tau_embedding(tau, dim, max_period=1000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=tau.device, dtype=torch.float32) / half)
    ang = tau.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class FlowBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), Attention(d, n_heads)
        self.ln2, self.cross = nn.LayerNorm(d), Attention(d, n_heads)
        self.ln3, self.mlp = nn.LayerNorm(d), MLP(d)

    def forward(self, x, ctx):
        x = x + self.attn(self.ln1(x))
        x = x + self.cross(self.ln2(x), ctx=ctx)
        return x + self.mlp(self.ln3(x))


class FlowHead(nn.Module):
    """cfg: d (256), n_heads, r (slots), d_slot (defaults to d), flow_layers (2), d_ctx (dimension of the
    context tokens, defaults to 2*d_slot = P's [k_hat ; v_hat]), std_momentum (0.99)."""

    def __init__(self, cfg):
        super().__init__()
        d, h, r = cfg["d"], cfg["n_heads"], cfg["r"]
        ds = cfg.get("d_slot", d)
        self.r, self.dz = r, 2 * ds
        dc = cfg.get("d_ctx", 2 * ds)
        self.inp = nn.Linear(self.dz, d)
        self.ctx_in = nn.Linear(dc, d)
        self.slot_emb = nn.Embedding(r, d)
        self.tau_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([FlowBlock(d, h) for _ in range(cfg.get("flow_layers", 2))])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.dz)
        self.d = d
        self.momentum = cfg.get("std_momentum", 0.99)
        self.register_buffer("z_std", torch.ones(self.dz))
        init_weights(self)
        nn.init.zeros_(self.out.weight)  # start from zero velocity
        nn.init.zeros_(self.out.bias)

    def velocity(self, z, tau, c):
        """z: [N, r, dz] ; tau: [N] in [0, 1] ; c: [N, r, d_ctx] -> velocity [N, r, dz]."""
        ctx = self.ctx_in(c)
        x = self.inp(z) + ctx + self.slot_emb.weight[None] + self.tau_mlp(tau_embedding(tau, self.d))[:, None]
        for b in self.blocks:
            x = b(x, ctx)
        return self.out(self.ln(x))

    def noise(self, shape, device, temperature=1.0, generator=None):
        return torch.randn(shape, device=device, generator=generator) * self.z_std * temperature

    def loss(self, z1, c, generator=None):
        """Rectified-flow loss for targets z1 [N, r, dz] (use to_target(k_ema, v_ema)) and context c."""
        if self.training:
            with torch.no_grad():
                std = z1.float().reshape(-1, self.dz).std(0).clamp_min(1e-3)
                self.z_std.mul_(self.momentum).add_(std, alpha=1 - self.momentum)
        N = z1.shape[0]
        z0 = self.noise(z1.shape, z1.device, generator=generator)
        tau = torch.sigmoid(torch.randn(N, device=z1.device, generator=generator))  # logit-normal
        zt = (1 - tau[:, None, None]) * z0 + tau[:, None, None] * z1
        return (self.velocity(zt, tau, c) - (z1 - z0)).float().pow(2).sum(-1).mean()

    @torch.no_grad()
    def sample(self, c, n_steps=16, temperature=1.0, generator=None, return_path=False):
        """Euler integration from noise (tau = 0) to an idea (tau = 1). c: [N, r, d_ctx]."""
        N = c.shape[0]
        z = self.noise((N, self.r, self.dz), c.device, temperature, generator)
        path = [z]
        for i in range(n_steps):
            tau = torch.full((N,), i / n_steps, device=c.device)
            z = z + self.velocity(z, tau, c) / n_steps
            path.append(z)
        return (z, path) if return_path else z

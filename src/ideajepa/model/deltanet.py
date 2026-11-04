"""DeltaNet memory (Schlag et al. 2021; Yang et al. 2024), chunked parallel form in pure PyTorch.

State S_t (d_v x d_k):  S_t = S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T,   o_t = S_t q_t.
Chunked (UT transform): T = (I + tril(diag(b) K K^T, -1))^{-1} diag(b); W = T K; U = T V;
per chunk: U' = U - W S^T;  O = Q S^T + (Q K^T * causal) U';  S <- S + U'^T K.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def delta_rule_naive(q, k, v, beta, log_alpha=None):
    """q, k: [B, H, T, dk]; v: [B, H, T, dv]; beta: [B, H, T]; log_alpha: [B, H, T] (log of the forget gate,
    None = no forgetting). Reference recurrence S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T."""
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, H, dv, dk)
    out = []
    for t in range(T):
        kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t, None, None]
        if log_alpha is not None:
            S = S * log_alpha[:, :, t, None, None].exp()
        S = S - bt * (S @ kt[..., None]) @ kt[..., None, :] + bt * vt[..., None] @ kt[..., None, :]
        out.append((S @ q[:, :, t, :, None])[..., 0])
    return torch.stack(out, 2)


def delta_rule_chunked(q, k, v, beta, chunk=64, log_alpha=None):
    """Chunked (UT transform) delta rule, optionally gated (Gated DeltaNet). With g_t = cumsum of log a within
    the chunk and G_tj = exp(g_t - g_j):  A_tj = b_t G_tj k_t.k_j (j < t),  T = (I + A)^-1 diag(b),
    W = T diag(e^g) K,  U = T V - W S0^T,  O = diag(e^g) Q S0^T + (Q K^T * G * causal) U,
    S <- e^{g_C} S0 + U^T diag(e^{g_C - g}) K.  log_alpha = None or 0 gives the plain delta rule."""
    B, H, T, dk = k.shape
    dv = v.shape[-1]
    pad = (-T) % chunk
    if log_alpha is None:
        log_alpha = torch.zeros_like(beta)
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta = F.pad(beta, (0, pad))
        log_alpha = F.pad(log_alpha, (0, pad))
    n = (T + pad) // chunk
    q, k, v = (x.reshape(B, H, n, chunk, -1) for x in (q, k, v))
    beta = beta.reshape(B, H, n, chunk)
    g = log_alpha.reshape(B, H, n, chunk).cumsum(-1)                    # log gamma_t within each chunk
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=k.device).tril()
    Gm = (g[..., :, None] - g[..., None, :]).masked_fill(~causal, float("-inf")).exp()   # gamma_t / gamma_j, j <= t
    kb = k * beta[..., None]
    A = torch.tril((kb @ k.transpose(-1, -2)) * Gm, -1)                 # [B,H,n,C,C]
    I = torch.eye(chunk, dtype=k.dtype, device=k.device)
    Tm = torch.linalg.solve_triangular(I + A, torch.diag_embed(beta), upper=False, unitriangular=True)
    eg = g.exp()[..., None]                                             # gamma_t
    W, U = Tm @ (k * eg), Tm @ v
    S = q.new_zeros(B, H, dv, dk)
    outs = []
    for i in range(n):
        Ui = U[:, :, i] - W[:, :, i] @ S.transpose(-1, -2)
        qi, ki = q[:, :, i], k[:, :, i]
        att = (qi @ ki.transpose(-1, -2)) * Gm[:, :, i]
        outs.append((qi * eg[:, :, i]) @ S.transpose(-1, -2) + att @ Ui)
        gC = g[:, :, i, -1]                                             # log gamma at the chunk end
        S = S * gC.exp()[..., None, None] + Ui.transpose(-1, -2) @ (ki * (gC[..., None] - g[:, :, i]).exp()[..., None])
    o = torch.stack(outs, 2).reshape(B, H, n * chunk, dv)
    return o[:, :, :T]


def entnet_scan(q, k, v, beta, W, scale=4.0, block=1):
    """notes §34.3, EntNet-style write (Henaff et al. 2017): m slots per head with FIXED learned keys W [H, m, dh]; a token
    writes every slot j through a content gate g_j = sigmoid(k.h^_j + k.w_j) (h^_j = normalised slot content) and a delta
    update h_j <- h_j + g_j beta (v - h_j) (gates of the tokens of one idea rescaled to sum <= 1 per slot); a token reads
    softmax_j(scale q.(w_j + h^_j)) weighted slot contents.
    block = r: one write step per IDEA (its r slot tokens write in parallel, from the state before the idea, as EntNet writes
    one sentence per step), then the idea's tokens read the updated state; causal across ideas.
    q, k: [B, H, T, dh] (normalised); v: [B, H, T, dv]; beta: [B, H, T]."""
    B, H, T, dh = q.shape
    Wn = F.normalize(W.float(), dim=-1)[None, :, None]                          # [1, H, 1, m, dh]
    h = q.new_zeros(B, H, W.shape[1], v.shape[-1])
    outs = []
    for s in range(0, T, block):
        kt, vt = k[:, :, s:s + block, None], v[:, :, s:s + block, None]         # [B, H, b, 1, d]
        bt, qt = beta[:, :, s:s + block, None, None], q[:, :, s:s + block, None]
        hn = F.normalize(h, dim=-1)[:, :, None]                                 # [B, H, 1, m, dv]
        g = torch.sigmoid((hn * kt).sum(-1) + (Wn * kt).sum(-1))[..., None]     # [B, H, b, m, 1]
        # normalised gates: the r tokens of an idea write the same slots in parallel; their weights a_t = g_t beta_t
        # are rescaled so that sum_t a_t <= 1 per slot, so h <- (1 - A) h + sum_t a_t v_t is a convex combination
        # (no overshoot of the target, maths session 29/09)
        aw = g * bt
        aw = aw / aw.sum(2, keepdim=True).clamp_min(1.0)
        entnet_scan.last_gate_sum = aw.sum(2).detach()
        if getattr(entnet_scan, "record", None) is not None:   # diagnostics: write strength per step and slot
            entnet_scan.record.append(aw.sum(2)[..., 0].detach().float().cpu())   # [B, H, m]
        h = h + (aw * (vt - h[:, :, None])).sum(2)
        p = torch.softmax(scale * (qt * (Wn + F.normalize(h, dim=-1)[:, :, None])).sum(-1), -1)[..., None]
        outs.append((p * h[:, :, None]).sum(3))                                 # [B, H, b, dv]
    return torch.cat(outs, 2)


class DeltaMemory(nn.Module):
    """Per-token written associative memory, one per head; output gated and projected."""

    def __init__(self, d, n_heads, mode="delta", center=False, center_v=False, decay=None, decay_init=0.95,
                 decay_every=1, n_slots=16):
        super().__init__()
        if mode == "entnet":   # m slots per head with fixed learned keys (entnet_scan)
            self.slot_keys = nn.Parameter(torch.randn(n_heads, n_slots, d // n_heads) / math.sqrt(d // n_heads))
        # decay_every = k: the forget gate acts only on positions t % k == 0 (k = r slot tokens per idea -> alpha is a
        # per-IDEA forgetting factor instead of alpha^r per idea)
        self.decay_every = int(decay_every)
        # decay (forget gate, Gated DeltaNet): None = no forgetting; a float = fixed alpha for every step and head;
        # "learned" = one alpha per head, alpha = sigmoid(w_h), initialised at decay_init
        self.decay = decay
        if decay == "learned":
            self.decay_logit = nn.Parameter(torch.full((n_heads,), math.log(decay_init / (1 - decay_init))))
        self.mode = mode  # "delta": delta rule (overwrite per key) | "sum": plain causal linear attention
        self.h, self.dh = n_heads, d // n_heads
        # center=True: subtract a running mean (EMA over training batches, fixed at eval, hence causal) from the raw
        # queries and keys before normalisation: removes the offset shared by all keys (diag_key_interference, 28/09)
        self.center = center
        if center:
            self.register_buffer("q_mu", torch.zeros(n_heads, d // n_heads))
            self.register_buffer("k_mu", torch.zeros(n_heads, d // n_heads))
            self.register_buffer("mu_init", torch.zeros(()))
        # center_v=True: the written values are also centred by a running mean (shared value component, 28/09)
        self.center_v = center_v
        if center_v:
            self.register_buffer("v_mu", torch.zeros(n_heads, d // n_heads))
            self.register_buffer("vmu_init", torch.zeros(()))
        self.qkv = nn.Linear(d, 3 * d)
        self.b = nn.Linear(d, n_heads)
        self.g = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln = nn.LayerNorm(self.dh)

    def forward(self, x, kcode=None):
        """kcode [B, H, T, c] (IdeaPredictor p_bow_keys with p_bow_where: mem, notes §35.15): ADDED to the first c dimensions
        of each head's raw write key (before normalisation; the learned key is kept); queries unchanged."""
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = q.float(), k.float()
        if kcode is not None:
            k = torch.cat([k[..., :kcode.shape[-1]] + kcode.float(), k[..., kcode.shape[-1]:]], -1)
        if self.center:
            if self.training:
                with torch.no_grad():
                    qm, km = q.mean(dim=(0, 2)), k.mean(dim=(0, 2))       # [H, dh]
                    if float(self.mu_init) == 0:
                        self.q_mu.copy_(qm); self.k_mu.copy_(km); self.mu_init.fill_(1)
                    else:
                        self.q_mu.mul_(0.99).add_(0.01 * qm); self.k_mu.mul_(0.99).add_(0.01 * km)
            q = q - self.q_mu[None, :, None]; k = k - self.k_mu[None, :, None]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        if self.center_v:
            v = v.float()
            if self.training:
                with torch.no_grad():
                    vm = v.mean(dim=(0, 2))
                    if float(self.vmu_init) == 0:
                        self.v_mu.copy_(vm); self.vmu_init.fill_(1)
                    else:
                        self.v_mu.mul_(0.99).add_(0.01 * vm)
            v = v - self.v_mu[None, :, None]
        beta = torch.sigmoid(self.b(x).float()).transpose(1, 2)
        if getattr(self, "capture", False):  # diagnostics: keep the normalised queries/keys and the values
            self.last_qk = (q.detach(), k.detach())
            self.last_v = v.detach().float()
            self.last_beta = beta.detach().float()
        ov = getattr(self, "q_override", None)
        if ov:   # inference oracle of the query (chantier 3 bis §5): q[b, :, qs:qs+n] := k[b, :, ks:ks+n]
            q = q.clone()   # (mix < 1: normalise((1 - mix) q + mix k)); q_override_cos = cos(normal query, oracle key)
            cos = []
            for b_, qs, ks, n_, mix in ov:
                q0, k0 = q[b_, :, qs:qs + n_], k[b_, :, ks:ks + n_]
                cos.append(float((q0 * k0).sum(-1).mean()))
                q[b_, :, qs:qs + n_] = F.normalize((1 - mix) * q0 + mix * k0, dim=-1)
            self.q_override_cos = sum(cos) / len(cos)
        if getattr(self, "keep_q", False):   # training term on the query (loss.w_qoracle), with grad: [B, H, T, dh]
            self.kept_qk = (q, k)
        if getattr(self, "keep", False):     # training term on the written keys / values (loss.w_memsep), with grad
            self.kept = (k, v.float())
        log_alpha = None
        if self.decay == "learned":
            log_alpha = F.logsigmoid(self.decay_logit.float())[None, :, None].expand(B, -1, T)
        elif self.decay is not None:
            log_alpha = torch.full((B, self.h, T), math.log(float(self.decay)), device=x.device)
        if log_alpha is not None and self.decay_every > 1:
            keep = (torch.arange(T, device=x.device) % self.decay_every == 0).float()
            log_alpha = log_alpha * keep
        with torch.autocast("cuda", enabled=False):
            if self.mode == "entnet":
                o = entnet_scan(q, k, v.float(), beta, self.slot_keys, block=self.decay_every)
            elif self.mode == "sum":
                att = (q @ k.transpose(-1, -2)) * beta[..., None, :]                 # [B, H, T, T] weights beta_s
                causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                if log_alpha is not None:                                           # decayed linear attention
                    gc = log_alpha.cumsum(-1)
                    att = att * (gc[..., :, None] - gc[..., None, :]).masked_fill(~causal, float("-inf")).exp()
                att = att.masked_fill(~causal, 0.0)
                o = att @ v.float()
            else:
                o = delta_rule_chunked(q, k, v.float(), beta, log_alpha=log_alpha)
        ro = getattr(self, "read_override", None)
        if ro:   # inference (notes §32.18): the tokens [qs, qs+n) read a GIVEN state S [H, dv, dk] (another story's, or 0)
            o = o.clone()
            for b_, qs, n_, S_ in ro:
                o[b_, :, qs:qs + n_] = torch.einsum("hvk,htk->htv", S_.to(o.dtype), q[b_, :, qs:qs + n_].to(o.dtype))
        if ov:   # q_override_readcos: cos(raw read at the query tokens, value written at the addressed tokens)
            self.q_override_readcos = sum(float(F.cosine_similarity(o[b_, :, qs:qs + n_].reshape(-1).float(),
                                                                    v[b_, :, ks:ks + n_].reshape(-1).float(), dim=0))
                                          for b_, qs, ks, n_, _ in ov) / len(ov)
        o = self.ln(o).transpose(1, 2).reshape(B, T, d)
        return self.o(o.to(x.dtype) * torch.sigmoid(self.g(x)))

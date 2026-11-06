"""Baselines. B0: full-context causal transformer on documents.
B1: same per-idea local window as the Idea-JEPA decoder (idea + m previous tokens), no idea matrix.
B3: documents, sliding-window attention (w tokens) + per-token DeltaNet matrix memory in every block
    (local + compressive matrix memory, but no segmentation, no ideas, no JEPA)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deltanet import DeltaMemory
from .layers import MLP, Attention, Block, init_weights

PAD = 0


class MemBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, n_heads)
        self.mem = DeltaMemory(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d)

    def forward(self, x, mask=None, is_causal=False):
        h = self.ln1(x)
        x = x + self.attn(h, mask=mask) + self.mem(h)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d"]
        self.cfg = cfg
        self.kind = cfg["model"]  # "b0" | "b1"
        self.emb = nn.Embedding(cfg["vocab_size"], d)
        self.pos = nn.Embedding(cfg["max_seq"] if self.kind in ("b0", "b3") else cfg["max_window"], d)
        # shared = T > 0 ("UT-mots"): ONE block applied T times with a learned pass embedding, then one distinct
        # output block (the words-level twin of IdeaPredictor p_shared); n_layers is ignored
        self.shared = int(cfg.get("shared", 0))
        mk = (lambda: MemBlock(d, cfg["n_heads"])) if self.kind == "b3" else \
             (lambda: Block(d, cfg["n_heads"], dropout=cfg.get("dropout", 0.0)))
        if self.shared:
            one = mk()
            self.blocks = nn.ModuleList([one] * self.shared + [mk()])
            self.pass_emb = nn.Embedding(self.shared, d)
        else:
            self.blocks = nn.ModuleList([mk() for _ in range(cfg["n_layers"])])
            self.pass_emb = None
        self.window = cfg.get("window", 16)
        self.ln_f = nn.LayerNorm(d)
        init_weights(self)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_parameters(self):
        return list(self.parameters())

    def compile_parts(self, mode="blocks"):
        for mod in self.blocks:
            mod.compile(dynamic=True)

    def run(self, inp):
        N, T = inp.shape
        x = self.emb(inp) + self.pos(torch.arange(T, device=inp.device))
        kvalid = inp != PAD
        eye = torch.eye(T, dtype=torch.bool, device=inp.device)
        causal = torch.ones(T, T, dtype=torch.bool, device=inp.device).tril()
        if self.kind == "b3":
            causal = causal & ~torch.ones(T, T, dtype=torch.bool, device=inp.device).tril(-self.window)
        mask = ((causal[None] & kvalid[:, None, :]) | eye[None])[:, None]
        for li, b in enumerate(self.blocks):
            if self.pass_emb is not None and li < self.shared:
                x = x + self.pass_emb.weight[li].to(x.dtype)
            x = b(x, mask=mask)
        return F.linear(self.ln_f(x), self.emb.weight)

    def token_logits(self, batch):
        """Returns (logits [*, V], targets [*]) over exactly the set of real tokens."""
        if self.kind in ("b0", "b3"):
            seq = batch["seq"]
            logits = self.run(seq[:, :-1])
            tgt = seq[:, 1:]
            m = tgt != PAD
            if self.cfg.get("loss_ded_only", False) and "seq_ded" in batch:  # Path-Star protocol: loss on the path only
                m = m & batch["seq_ded"][:, 1:]
            return logits[m], tgt[m]
        win, M, length = batch["win"], batch["M"], batch["idea_len"]
        logits = self.run(win[:, :-1])[:, M - 1:]
        tgt = win[:, M:]
        m = torch.arange(tgt.shape[1], device=tgt.device)[None] < length[:, None]
        return logits[m], tgt[m]

    def losses(self, batch, **_):
        logits, tgt = self.token_logits(batch)
        ce = F.cross_entropy(logits.float(), tgt)
        return {"loss": ce, "L_tok": ce.detach(), "ce_gen": ce.detach()}, {}

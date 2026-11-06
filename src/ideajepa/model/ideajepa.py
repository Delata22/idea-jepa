"""Idea-JEPA: role->content idea slots, JEPA idea predictor, erasing idea->words decoder.

Shapes: N = number of ideas in the batch, r = slots, d = model dim,
L = max idea length, M = prefix length (m tokens of the previous idea, or the
whole previous document for ablation A11), W = M + L (window length).
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, Attention, Block, init_weights
from .deltanet import DeltaMemory

PAD = 0
UNK = 3


def safe_norm(x, dim=-1):
    """L2 norm with a finite gradient at 0 (Z is exactly 0 for the first idea of B1', or for L_contrib)."""
    return (x.pow(2).sum(dim) + 1e-12).sqrt()


def slot_norm(V):
    """sqrt(sum_i ||v_i||^2) -- equals ||Z||_F when keys are orthonormal (see REPORT, ambiguity 1)."""
    return (V.float().pow(2).sum(dim=(-1, -2)) + 1e-12).sqrt()


def frob_norm(K, V):
    """True ||Z||_F for Z = sum_i v_i k_i^T: ||Z||_F^2 = sum_ij (v_i.v_j)(k_i.k_j)."""
    K, V = K.float(), V.float()
    g = (V @ V.transpose(-1, -2)) * (K @ K.transpose(-1, -2))
    return g.sum(dim=(-1, -2)).clamp_min(0).sqrt()


def erase_step(Vres, a, g, e, mode="local"):
    """Projective erasure. Vres [N, r, d], a [N, r], g [N] (beta_t = g_t), e [N, d] unit
    (or [N, h, d] orthonormal rows for rank-h erasure: projection onto their span).
    local:  v_i <- v_i - g a_i P_e v_i       global (A5): v_i <- v_i - g P_e v_i"""
    coef = g[:, None] * (a if mode == "local" else torch.ones_like(a))
    if e.dim() == 2:
        proj = torch.einsum("nrd,nd->nr", Vres, e)
        return Vres - (coef * proj)[..., None] * e[:, None, :]
    proj = torch.einsum("nrd,nhd->nrh", Vres, e)
    return Vres - coef[..., None] * torch.einsum("nrh,nhd->nrd", proj, e)



N_REL, N_OFF = 3, 32   # step 2: relation classes (none / justified_by / sequence), pointer offsets

class IdeaEncoder(nn.Module):
    """Bidirectional transformer over a segment + Perceiver-style cross-attention with r learned queries."""

    def __init__(self, cfg, emb: nn.Embedding):
        super().__init__()
        d, h = cfg["d"], cfg["n_heads"]
        self.emb = emb
        self.pos = nn.Embedding(cfg["max_idea_len"], d)
        self.blocks = nn.ModuleList([Block(d, h, dropout=cfg.get("enc_dropout", cfg.get("dropout", 0.0)))
                                     for _ in range(cfg["enc_layers"])])
        self.ln = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(cfg["r"], d) * 0.02)
        self.ln_q = nn.LayerNorm(d)
        self.cross = Attention(d, h)
        self.ln_m = nn.LayerNorm(d)
        self.mlp = MLP(d)
        self.ln_out = nn.LayerNorm(d)
        ds = cfg.get("d_slot", d)
        self.out_k = nn.Linear(d, ds)
        self.out_v = nn.Linear(d, ds)

    def forward(self, tok, length):
        N, L = tok.shape
        self.last_tok = tok        # read by IdeaPredictor p_id_keys (entity words of the ideas P is about to read)
        x = self.emb(tok) + self.pos(torch.arange(L, device=tok.device))
        valid = torch.arange(L, device=tok.device)[None] < length[:, None]
        mask = valid[:, None, None, :] | torch.eye(L, dtype=torch.bool, device=tok.device)[None, None]
        for b in self.blocks:
            x = b(x, mask=mask)
        x = self.ln(x)
        self.last_words, self.last_valid = x, valid   # word states (IdeaPredictor p_word_kv)
        q = self.queries[None].expand(N, -1, -1)
        s = q + self.cross(self.ln_q(q), mask=valid[:, None, None, :], ctx=x)
        s = s + self.mlp(self.ln_m(s))
        s = self.ln_out(s)
        k = F.normalize(self.out_k(s).float(), dim=-1)
        v = self.out_v(s).float()
        return k, v


class IdeaPredictor(nn.Module):
    """Causal transformer at the idea level: slots of ideas < n  ->  slots of idea n."""

    def __init__(self, cfg):
        super().__init__()
        d, h, r = cfg["d"], cfg["n_heads"], cfg["r"]
        self.r = r
        self.D = d
        self.ds = ds = cfg.get("d_slot", d)
        self.inp = nn.Linear(2 * ds, d)
        self.start = nn.Parameter(torch.randn(r, d) * 0.02)
        self.idea_pos = nn.Embedding(cfg["max_ideas"], d)
        self.slot_emb = nn.Embedding(r, d)
        # p_role_emb: learned premise / deduction embedding added to every input idea (known in generation: all that
        # follows the connector is a deduction)
        self.role_emb = nn.Embedding(2, d) if cfg.get("p_role_emb", False) else None
        # rel_cond (chantier 3 bis step 2): relation of each input idea (none / justified_by / sequence) + pointer to the
        # justifying idea as a relative offset (0 = no pointer), added to its slot tokens
        self.rel_emb = nn.Embedding(N_REL, d) if cfg.get("rel_cond", False) else None
        self.rel_off_emb = nn.Embedding(N_OFF, d) if cfg.get("rel_cond", False) else None
        # plan_cond (plan variable): the input slots of a past idea o are added, with a "plan" type
        # embedding, to the block that predicts the next idea
        self.plan_emb = nn.Parameter(torch.randn(r, d) * 0.02) if cfg.get("plan_cond", False) else None
        self.plan_inject = cfg.get("plan_inject", "input")        # "input" (every block, memory included) | "output"
        self.plan_out = MLP(d) if cfg.get("plan_cond", False) and self.plan_inject == "output" else None
        # p_shared = T > 0 (Universal-Transformer-style depth): ONE block (and its memory) applied T times, with a
        # learned pass embedding added before each pass, then one distinct output block; pred_layers is ignored.
        # self.blocks / self.mems list the T passes (the same module repeated) + the output block.
        self.p_shared = int(cfg.get("p_shared", 0))
        n_blk = self.p_shared + 1 if self.p_shared else cfg["pred_layers"]
        mk_block = lambda: Block(d, h, dropout=cfg.get("p_dropout", cfg.get("dropout", 0.0)))   # p_dropout: P only
        if self.p_shared:
            shared = mk_block()
            self.blocks = nn.ModuleList([shared] * self.p_shared + [mk_block()])
            self.pass_emb = nn.Embedding(self.p_shared, d)
        else:
            self.blocks = nn.ModuleList([mk_block() for _ in range(n_blk)])
            self.pass_emb = None
        # p_pos: "abs" = learned absolute idea-position embedding (default) | "alibi" = no position embedding, a recency
        # bias -slope_h * (distance in ideas) on the attention logits (ALiBi slopes 2^(-8(h+1)/H))
        self.p_pos = cfg.get("p_pos", "abs")
        if self.p_pos == "alibi":
            self.register_buffer("alibi_slopes", torch.tensor([2.0 ** (-8.0 * (i + 1) / h) for i in range(h)]),
                                 persistent=False)
        # p_id_keys (weak supervision: word classes): in every attention block of P, the first quarter of each
        # head's KEY dimensions is replaced by entity codes of the idea at that position: sub-block A (1/8) = the first
        # proper name of the sentence, sub-block B (1/8) = its object noun (zeros if absent); codes = learned per-head
        # projections (shared by the layers) of the model's word embedding. Queries stay fully learned. Word ids come from
        # model_config (p_id_name_ids / p_id_obj_ids, resolved from the word lists of the config).
        # p_bow_keys (the RECIPE candidate; no word list): the same quarter of key dimensions receives a learned
        # per-head projection of the MEAN word embedding of the sentence (all its words; the question too).
        # p_id_keys is a DIAGNOSTIC (hard-coded word classes: ceiling "what if the reading channel had identity").
        self.bow_keys = bool(cfg.get("p_bow_keys", False))
        assert not (self.bow_keys and cfg.get("p_id_keys")), "p_bow_keys and p_id_keys are exclusive"
        self.id_keys = bool(cfg.get("p_id_keys")) or self.bow_keys
        # p_bow_where: "attn" (default: keys of P's attention) | "mem" (write keys of the delta memory only)
        self.bow_where = cfg.get("p_bow_where", "attn")
        assert self.bow_where in ("attn", "mem")
        assert not (self.bow_where == "mem" and cfg.get("idea_memory", "none") == "none"), "p_bow_where mem needs a memory"
        if self.bow_keys:
            dh = d // h
            assert dh % 8 == 0, "p_bow_keys: head dimension must be a multiple of 8"
            self.id_sub = dh // 8
            self.bow = nn.Linear(d, h * 2 * self.id_sub, bias=False)
        elif self.id_keys:
            dh = d // h
            assert dh % 8 == 0, "p_id_keys: head dimension must be a multiple of 8"
            self.id_sub = dh // 8
            self.id_a = nn.Linear(d, h * self.id_sub, bias=False)
            self.id_b = nn.Linear(d, h * self.id_sub, bias=False)
            self.register_buffer("id_name_ids", torch.tensor(cfg["p_id_name_ids"], dtype=torch.long), persistent=False)
            self.register_buffer("id_obj_ids", torch.tensor(cfg["p_id_obj_ids"], dtype=torch.long), persistent=False)
        # p_word_kv (variant (c)): the keys / values of P's attention also contain E's WORD states of past
        # ideas (contextualised in their sentence, after E's blocks), with a "word" type embedding and the idea position;
        # queries stay the slots; causal per idea (position n sees the words of the ideas at input positions <= n).
        # "all": every past idea exposes its words; "support": ORACLE, only the bAbI supporting facts do (labels).
        self.word_kv = cfg.get("p_word_kv", "none")
        if self.word_kv != "none":
            assert self.word_kv in ("all", "support")
            self.word_type = nn.Parameter(torch.randn(d) * 0.02)
        # "B3 at the idea level": P attends only to the last p_window ideas (0 = all), and an associative
        # memory over all past slot tokens (learned k/v projections, "sum" or "delta" rule) in every block.
        self.p_window = int(cfg.get("p_window", 0))
        self.idea_memory = cfg.get("idea_memory", "none")
        if self.idea_memory != "none":
            mk_mem = lambda: DeltaMemory(d, h, mode=self.idea_memory, center=cfg.get("mem_center", False),
                                         center_v=cfg.get("mem_center_v", False),
                                         decay=cfg.get("mem_decay"), decay_init=cfg.get("mem_decay_init", 0.95),
                                         decay_every=r if cfg.get("mem_decay_per", "idea") == "idea" else 1,
                                         n_slots=cfg.get("mem_slots", 16))
            if self.p_shared:
                shared_m = mk_mem()
                self.mems = nn.ModuleList([shared_m] * self.p_shared + [mk_mem()])
            else:
                self.mems = nn.ModuleList([mk_mem() for _ in range(n_blk)])
        self.ln = nn.LayerNorm(d)
        # k_hyp > 1 (phase 2 control against the averaging problem): K hypotheses per next idea
        # + a score per hypothesis (multiple-choice learning). forward() returns the best-scored one and
        # keeps all of them in self.last_hyp = (pk [B, Nm, K, r, ds], pv [...], scores [B, Nm, K]).
        # p_head_dropout (small models): dropout on P's states before its regression head
        self.head_drop = nn.Dropout(cfg.get("p_head_dropout", 0.0))
        self.K = int(cfg.get("k_hyp", 1))
        self.head = nn.Linear(d, 2 * ds * self.K)
        if self.K > 1:
            self.score = nn.Linear(d, self.K)
        self.last_hyp = None

    def _entity_codes(self, tokens):
        """tokens [NI, L] -> key codes [NI, H, 2 * id_sub]: p_id_keys (diagnostic) A = first proper name, B = first object
        noun, zeros if none; p_bow_keys = projection of the sentence's mean word embedding."""
        H = self.blocks[0].attn.h
        if self.bow_keys:
            m = (tokens != 0).to(self.word_emb.weight.dtype)                   # PAD = 0
            mean = (self.word_emb(tokens) * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
            return self.bow(mean).view(-1, H, 2 * self.id_sub)
        out = []
        for ids, proj in ((self.id_name_ids, self.id_a), (self.id_obj_ids, self.id_b)):
            hit = (tokens[..., None] == ids).any(-1)                            # [NI, L]
            has = hit.any(1)
            first = hit.float().argmax(1)
            word = tokens.gather(1, first[:, None])[:, 0]
            c = proj(self.word_emb.weight[word]) * has[:, None].to(self.word_emb.weight.dtype)
            out.append(c.view(-1, H, self.id_sub))
        return torch.cat(out, -1)

    def forward(self, k, v, doc_index, role=None, rel=None, plan=None, tokens=None, words=None):
        """k, v: [NI, r, d]; doc_index: [B, Nmax] index into NI (-1 = pad); role: [NI] 0 premise / 1 deduction
        (p_role_emb only). Returns [B, Nmax, r, d] x2."""
        B, Nm = doc_index.shape
        r, d = self.r, self.D
        tok = self.inp(torch.cat([k, v], -1).to(self.inp.weight.dtype))
        if self.role_emb is not None:
            assert role is not None, "p_role_emb: the batch needs 'idea_role' (or 'is_deduction')"
            tok = tok + self.role_emb(role.long())[:, None].to(tok.dtype)
        if self.rel_emb is not None:
            assert rel is not None, "rel_cond: the batch needs 'rel_id' and 'rel_off'"
            tok = tok + (self.rel_emb(rel[0].long()) + self.rel_off_emb(rel[1].long().clamp(0, N_OFF - 1)))[:, None].to(tok.dtype)
        self.last_tok = tok                                   # [NI, r, d] input slots (plan pointer keys)
        valid = doc_index >= 0
        slots = tok[doc_index.clamp_min(0)] * valid[..., None, None]
        x = torch.cat([self.start[None, None].expand(B, 1, r, d).to(slots.dtype), slots[:, :-1]], 1)
        if self.p_pos == "alibi":
            x = x + self.slot_emb.weight[None, None]
        else:
            x = x + self.idea_pos(torch.arange(Nm, device=k.device))[None, :, None] + self.slot_emb.weight[None, None]
        plan_out = None
        if plan is not None:   # plan [B, Nm]: story index o of the plan idea for the block predicting idea n (-1 = none)
            has = (plan >= 0) & valid
            ptok = slots.gather(1, plan.clamp_min(0)[..., None, None].expand(-1, -1, r, d))
            padd = (ptok + self.plan_emb.to(ptok.dtype)[None, None]) * has[..., None, None].to(ptok.dtype)
            if self.plan_inject == "output":   # after the blocks: enters neither the memory writes nor the attention
                plan_out = padd.reshape(B, Nm * r, d)
            else:
                x = x + padd
        x = x.reshape(B, Nm * r, d)
        ii = torch.arange(Nm, device=k.device).repeat_interleave(r)
        causal = ii[None, :] <= ii[:, None]
        if self.p_window:
            causal = causal & (ii[:, None] - ii[None, :] < self.p_window)
        mask = causal[None, None]
        if self.p_pos == "alibi":   # additive float mask [1, H, T, T]: -slope * (i - j) in ideas, -inf outside
            dist = (ii[:, None] - ii[None, :]).to(torch.float32)
            bias = -self.alibi_slopes[:, None, None] * dist[None]
            mask = bias.masked_fill(~causal[None], float("-inf"))[None].to(x.dtype)
        idc = None
        if self.id_keys:   # codes per input position (position 0 = start: zeros; position j = idea j - 1), per slot token
            assert tokens is not None and tokens.shape[0] == k.shape[0], "p_id_keys: P needs the tokens of its input ideas"
            c = self._entity_codes(tokens)                                     # [NI, H, 2s]
            cpos = c[doc_index.clamp_min(0)] * valid[..., None, None]           # [B, Nm, H, 2s]
            cpos = torch.cat([torch.zeros_like(cpos[:, :1]), cpos[:, :-1]], 1)
            idc = cpos.repeat_interleave(r, 1).permute(0, 2, 1, 3)              # [B, H, Nm * r, 2s]
        wseq = None
        if self.word_kv != "none":   # words [NI, L, d], valid [NI, L], sel [NI] -> key/value sequence [B, Nm * L, d]
            assert words is not None, "p_word_kv: P needs E's word states of its input ideas"
            ws, wv, sel = words
            assert ws.shape[0] == k.shape[0]
            Lw = ws.shape[1]
            W = ws[doc_index.clamp_min(0)]                                       # [B, Nm, Lw, d]
            Wm = wv[doc_index.clamp_min(0)] & sel[doc_index.clamp_min(0)][..., None] & valid[..., None]
            W = torch.cat([torch.zeros_like(W[:, :1]), W[:, :-1]], 1)            # input position j + 1 <- idea j
            Wm = torch.cat([torch.zeros_like(Wm[:, :1]), Wm[:, :-1]], 1)
            pw = torch.arange(Nm, device=k.device)
            W = W + self.word_type.to(W.dtype) + (0 if self.p_pos == "alibi" else self.idea_pos(pw)[None, :, None].to(W.dtype))
            wseq = W.reshape(B, Nm * Lw, d).to(x.dtype)
            wpos = pw.repeat_interleave(Lw)
            wmask = (wpos[None, :] <= ii[:, None])[None] & Wm.reshape(B, 1, Nm * Lw)   # [B, Tq, Tw]
            assert mask.dtype == torch.bool, "p_word_kv: not with p_pos alibi"
            mask = torch.cat([mask.expand(B, 1, -1, -1), wmask[:, None]], -1)   # [B, 1, Tq, Tq + Tw]
        for li, b in enumerate(self.blocks):
            if self.pass_emb is not None and li < self.p_shared:
                x = x + self.pass_emb.weight[li].to(x.dtype)
            if wseq is not None:
                assert idc is None, "p_word_kv: not with key codes"
                h_ = b.ln1(x)
                kk, vv = b.attn.kv_heads(torch.cat([h_, b.ln1(wseq)], 1))
                x = x + b.attn.attend(h_, kk, vv, mask) + (self.mems[li](h_) if self.idea_memory != "none" else 0)
                x = x + b.mlp(b.ln2(x))
            elif idc is not None and self.bow_where == "mem":
                h_ = b.ln1(x)
                x = x + b.attn(h_, mask=mask) + self.mems[li](h_, kcode=idc)
                x = x + b.mlp(b.ln2(x))
            elif idc is not None:
                h_ = b.ln1(x)
                kk, vv = b.attn.kv_heads(h_)
                kk = torch.cat([idc.to(kk.dtype), kk[..., idc.shape[-1]:]], -1)
                x = x + b.attn.attend(h_, kk, vv, mask) + (self.mems[li](h_) if self.idea_memory != "none" else 0)
                x = x + b.mlp(b.ln2(x))
            elif self.idea_memory != "none":
                h_ = b.ln1(x)
                x = x + b.attn(h_, mask=mask) + self.mems[li](h_)
                x = x + b.mlp(b.ln2(x))
            else:
                x = b(x, mask=mask)
        if getattr(self, "keep_mem", False) and self.idea_memory != "none":
            self.mem_rows = self._written_rows(doc_index, k.shape[0])
        if plan_out is not None:
            x = x + self.plan_out(plan_out)
        hx = self.ln(x)
        self.last_hidden = hx.view(B, Nm, r, -1)   # slot states before the regression head (flow context)
        if self.K > 1:
            out = self.head(hx).view(B, Nm, r, self.K, 2 * self.ds).permute(0, 1, 3, 2, 4).float()
            pk_all, pv_all = out.chunk(2, -1)                                  # [B, Nm, K, r, ds]
            pk_all = F.normalize(pk_all, dim=-1)
            scores = self.score(hx.view(B, Nm, r, -1).mean(2)).float()         # [B, Nm, K]
            self.last_hyp = (pk_all, pv_all, scores)
            best = scores.argmax(-1)[..., None, None, None].expand(-1, -1, 1, r, self.ds)
            return pk_all.gather(2, best)[:, :, 0], pv_all.gather(2, best)[:, :, 0]
        out = self.head(self.head_drop(hx)).view(B, Nm, r, 2 * self.ds).float()
        pk, pv = out.chunk(2, -1)
        return F.normalize(pk, dim=-1), pv


def _written_rows_impl(self, doc_index, NI):
    """Keys and values each memory layer WRITES for every idea row (the input at position p + 1 is idea p):
    {"k": [NI, L, H*r*dh], "v": [NI, L, H*r*dh], "valid": [NI]} (the last idea of a document is never written)."""
    B, Nm = doc_index.shape
    r = self.r
    ks, vs = [], []
    for m in self.mems:
        k, v = m.kept                                                       # [B, H, T, dh], T = Nm * r
        H, dh = k.shape[1], k.shape[-1]
        f = lambda t: t.reshape(B, H, Nm, r, dh)[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(B, Nm - 1, H * r * dh)
        ks.append(f(k)); vs.append(f(v))
    K, Vv = torch.stack(ks, 2), torch.stack(vs, 2)                          # [B, Nm-1, L, D]
    idx = doc_index[:, :-1]
    sel = idx >= 0
    out_k = K.new_zeros(NI, K.shape[2], K.shape[3]).index_put((idx[sel],), K[sel])
    out_v = Vv.new_zeros(NI, Vv.shape[2], Vv.shape[3]).index_put((idx[sel],), Vv[sel])
    valid = torch.zeros(NI, dtype=torch.bool, device=doc_index.device)
    valid[idx[sel]] = True
    return {"k": out_k, "v": out_v, "valid": valid}


IdeaPredictor._written_rows = _written_rows_impl


class IdeaMaskPredictor(IdeaPredictor):
    """Bidirectional predictor at the idea level: the slots of masked ideas are replaced by a learned mask
    embedding; every idea attends to every other idea of the document; the output at a masked position
    predicts that idea's slots (target: E_ema). Used for stage-A training and as an energy for candidates."""

    def __init__(self, cfg):
        super().__init__({**cfg, "k_hyp": 1})
        self.mask_emb = nn.Parameter(torch.randn(cfg["r"], cfg["d"]) * 0.02)

    def forward(self, k, v, doc_index, masked):
        """masked: [NI] bool over idea rows. Returns predicted (k, v) for every idea row [NI, r, d]."""
        B, Nm = doc_index.shape
        r, d = self.r, self.inp.weight.shape[0]
        tok = self.inp(torch.cat([k, v], -1).to(self.inp.weight.dtype))       # [NI, r, d]
        tok = torch.where(masked[:, None, None], self.mask_emb[None].to(tok.dtype), tok)
        valid = doc_index >= 0
        x = tok[doc_index.clamp_min(0)] * valid[..., None, None]              # [B, Nm, r, d]
        x = x + self.idea_pos(torch.arange(Nm, device=k.device))[None, :, None] + self.slot_emb.weight[None, None]
        x = x.reshape(B, Nm * r, d)
        kv_valid = valid.repeat_interleave(r, 1)                              # [B, Nm*r]
        mask = kv_valid[:, None, None, :]
        for b in self.blocks:
            x = b(x, mask=mask)
        out = self.head(self.ln(x)).view(B, Nm, r, 2 * self.ds).float()  # head outputs slot dims (d_slot)
        pk, pv = out.chunk(2, -1)
        pk, pv = F.normalize(pk, dim=-1), pv
        # gather back to idea rows
        rows_k = torch.zeros_like(k, dtype=pk.dtype); rows_v = torch.zeros_like(v, dtype=pv.dtype)
        bi, ji = valid.nonzero(as_tuple=True)
        rows_k[doc_index[bi, ji]] = pk[bi, ji]; rows_v[doc_index[bi, ji]] = pv[bi, ji]
        return rows_k, rows_v


class OutputEncoder(nn.Module):
    """e_t: unit vector in value space describing what the last w emitted tokens expressed."""

    def __init__(self, cfg, emb: nn.Embedding):
        super().__init__()
        d = cfg["d"]
        self.w = cfg["out_window"]
        self.mode = cfg["e_mode"]
        self.emb = emb
        self.h = cfg.get("erase_rank", 1)
        ds = cfg.get("d_slot", d)
        if self.mode == "context":
            self.net = nn.Sequential(nn.Linear(self.w * d, 2 * d), nn.GELU(), nn.Linear(2 * d, self.h * ds))
        elif self.mode == "frozen":  # A6b: fixed random unit direction per token (O not learned)
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("fixed", F.normalize(torch.randn(cfg["vocab_size"], self.h * ds, generator=g), dim=-1))
        else:  # "static": A6
            self.net = nn.Linear(d, self.h * ds)

    def forward(self, win, M):
        """win: [N, W] window tokens; returns e for every idea position: [N, W - M, d]."""
        if self.mode == "frozen":
            e = self.fixed[win[:, M:]]
            if self.h == 1:
                return e
            N, L = e.shape[:2]
            q, _ = torch.linalg.qr(e.view(N, L, self.h, -1).transpose(-1, -2))
            return q.transpose(-1, -2)
        x = self.emb(win)
        if self.mode == "context":
            N, W, d = x.shape
            xp = F.pad(x, (0, 0, self.w - 1, 0))
            ctx = xp.unfold(1, self.w, 1)[:, M:]          # [N, L, d, w]  (tokens t-w+1..t)
            ctx = ctx.permute(0, 1, 3, 2).reshape(N, W - M, self.w * d)
            e = self.net(ctx)
        else:
            e = self.net(x[:, M:])
        if self.h == 1:
            return F.normalize(e.float(), dim=-1)
        N, L = e.shape[:2]
        q, _ = torch.linalg.qr(e.float().view(N, L, self.h, -1).transpose(-1, -2))  # [N, L, d, h]
        return q.transpose(-1, -2)  # [N, L, h, d] orthonormal rows


class ReadBlock(Block):
    """Transformer block + read from residual idea slots (after attention, before MLP)."""

    def __init__(self, cfg):
        d = cfg["d"]
        super().__init__(d, cfg["n_heads"], dropout=cfg.get("dropout", 0.0))
        self.rho_proj = nn.Linear(1, d)
        ds = cfg.get("d_slot", d)
        self.ln_r = nn.LayerNorm(d)
        self.wq = nn.Linear(d, ds)
        self.wo = nn.Linear(ds, d)
        self.wg = nn.Linear(d + 1, 1)
        self.wb = nn.Linear(d, 1)  # only used by erase="free" (A4)


class DeepDecoder(nn.Module):
    """arch='deep': reads injected inside the last `read_layers` transformer blocks, rho added to the
    residual stream. Later blocks' attention sees read-augmented states -> sequential over positions."""

    def __init__(self, cfg, emb: nn.Embedding):
        super().__init__()
        d, h = cfg["d"], cfg["n_heads"]
        self.cfg = cfg
        self.emb = emb
        self.pos = nn.Embedding(cfg["max_window"], d)
        nr = cfg["read_layers"]
        self.base = nn.ModuleList([Block(d, h, dropout=cfg.get("dropout", 0.0))
                                   for _ in range(cfg["dec_layers"] - nr)])
        self.reads = nn.ModuleList([ReadBlock(cfg) for _ in range(nr)])
        self.ln_f = nn.LayerNorm(d)
        self.gate = cfg["gate"]
        self.erase = cfg["erase"]
        self.read_mode = cfg["read_mode"]
        self.scale = 1.0 / math.sqrt(d)
        self.log_scale = (nn.Parameter(torch.tensor(math.log(cfg.get("read_scale_init", 8.0))))
                          if cfg.get("read_scale", "sqrt_d") == "learned" else None)

    def logits(self, x):
        return F.linear(self.ln_f(x), self.emb.weight)

    def _read(self, blk, xs, K, Vres, rho):
        """xs [N, d]; K, Vres [N, r, d] (fp32). Returns new xs, a [N, r], g [N], hq."""
        xs = xs + blk.rho_proj(rho[:, None].to(xs.dtype))
        hq = blk.ln_r(xs)
        with torch.autocast("cuda", enabled=False):
            q = blk.wq(hq.float())
            s = torch.einsum("nrd,nd->nr", K, q)
            if self.read_mode == "softmax":
                a = torch.softmax(s * (self.log_scale.exp() if self.log_scale is not None else self.scale), -1)
            else:  # "linear": A12
                a = s.clamp(0.0, 1.0)
            rvec = torch.einsum("nr,nrd->nd", a, Vres)
            if self.gate:
                g = torch.sigmoid(blk.wg(torch.cat([hq.float(), rho[:, None]], -1)))[:, 0]
            else:
                g = torch.ones_like(rho)
            upd = g[:, None] * blk.wo(rvec)
        return xs + upd.to(xs.dtype), a, g, hq

    def _erase(self, Vres, a, g, e, hq, blk):
        with torch.autocast("cuda", enabled=False):
            if self.erase == "none":
                return Vres
            if self.erase == "free":
                beta = torch.sigmoid(blk.wb(hq.float()))[:, 0]
                return Vres * (1 - beta[:, None] * a)[..., None]
            return erase_step(Vres, a, g, e, self.erase)

    def forward(self, win, length, M, K, V0, e, return_trace=True):
        """win: [N, M+L] window tokens (prefix + idea); length: [N] idea lengths.
        K, V0: [N, r, d] idea slots; e: [N, L, d] output-encoder vectors for idea tokens.
        Returns logits [N, L, V] (position s predicts idea token s) and a trace dict."""
        N, W = win.shape
        L = W - M
        dev = win.device
        inp = win[:, :-1]
        Wi = W - 1
        x = self.emb(inp) + self.pos(torch.arange(Wi, device=dev))
        kvalid = inp != PAD
        eye = torch.eye(Wi, dtype=torch.bool, device=dev)
        mask = (torch.ones(Wi, Wi, dtype=torch.bool, device=dev).tril()[None] & kvalid[:, None, :]) | eye[None]
        mask = mask[:, None]
        for b in self.base:
            x = b(x, mask=mask)

        blk0 = self.reads[0]
        x = x + blk0.attn(blk0.ln1(x), mask=mask)       # first read block attention is parallel
        P = M - 1                                        # number of prefix positions without a read
        caches = [None]
        if len(self.reads) > 1:
            pre = x[:, :P]
            pre = pre + blk0.mlp(blk0.ln2(pre))
            pmask = mask[:, :, :P, :P]
            for blk in self.reads[1:]:
                hp = blk.ln1(pre)
                kp, vp = blk.attn.kv_heads(hp)
                caches.append(([kp], [vp]))
                pre = pre + blk.attn.attend(hp, kp, vp, mask=pmask)
                pre = pre + blk.mlp(blk.ln2(pre))
        pvalid = kvalid[:, :P]

        Vres = V0.float()
        n0 = slot_norm(Vres)
        inv0 = 1.0 / n0.clamp_min(1e-6)
        rho = slot_norm(Vres) * inv0
        outs, rhos, gs, As, gall = [], [rho], [], [], []
        for s in range(L):
            p = P + s
            xs = x[:, p]
            gl = []
            for j, blk in enumerate(self.reads):
                if j > 0:
                    h1 = blk.ln1(xs)[:, None]
                    kn, vn = blk.attn.kv_heads(h1)
                    ck, cv = caches[j]
                    ck.append(kn)
                    cv.append(vn)
                    Kc, Vc = torch.cat(ck, 2), torch.cat(cv, 2)
                    km = torch.cat([pvalid, torch.ones(N, s + 1, dtype=torch.bool, device=dev)], 1)
                    xs = xs + blk.attn.attend(h1, Kc, Vc, mask=km[:, None, None, :])[:, 0]
                xs, a, g, hq = self._read(blk, xs, K, Vres, rho)
                gl.append(g)
                xs = xs + blk.mlp(blk.ln2(xs))
            outs.append(xs)
            active = (s < length)[:, None, None]
            Vnew = self._erase(Vres, a, g, e[:, s].float(), hq, self.reads[-1])
            Vres = torch.where(active, Vnew, Vres)
            rho = slot_norm(Vres) * inv0
            rhos.append(rho)
            gs.append(g)
            As.append(a)
            gall.append(torch.stack(gl, -1))
        logits = self.logits(torch.stack(outs, 1))
        trace = {"rho": torch.stack(rhos, 1), "g": torch.stack(gs, 1), "a": torch.stack(As, 1),
                 "g_all": torch.stack(gall, 1), "V_end": Vres, "n0": n0}
        return logits, trace


class ParallelDecoder(nn.Module):
    """arch='parallel' (default). h_local = local causal transformer (fully parallel); optional read of
    the *unerased* Z_0 before the last block; then a light sequential loop (read / gate / erase / rho,
    O(r*d) per step) and a final per-position layer that sees [rho, per-slot residual fractions]."""

    def __init__(self, cfg, emb: nn.Embedding):
        super().__init__()
        d, h, r = cfg["d"], cfg["n_heads"], cfg["r"]
        self.cfg = cfg
        self.emb = emb
        self.pos = nn.Embedding(cfg["max_window"], d)
        self.blocks = nn.ModuleList([Block(d, h, dropout=cfg.get("dropout", 0.0))
                                     for _ in range(cfg["dec_layers"])])
        self.pre_read = cfg.get("pre_read", True)
        ds = cfg.get("d_slot", d)
        # deep read: after each of the last `z_read_layers` non-final blocks, a gated softmax read of the
        # UNERASED Z_0 (parallel over positions); the final block keeps the residual read.
        self.z_read_layers = int(cfg.get("z_read_layers", 0))
        assert 0 <= self.z_read_layers <= cfg["dec_layers"] - 1
        if self.z_read_layers:
            n_zr = self.z_read_layers
            self.zr_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_zr)])
            self.zr_wq = nn.ModuleList([nn.Linear(d, ds) for _ in range(n_zr)])
            self.zr_wo = nn.ModuleList([nn.Linear(ds, d) for _ in range(n_zr)])
            self.zr_wg = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_zr)])
        if self.pre_read:
            self.ln_r0 = nn.LayerNorm(d)
            self.wq0 = nn.Linear(d, ds)
            self.wo0 = nn.Linear(ds, d)
            self.wg0 = nn.Linear(d, 1)
        self.ln_r = nn.LayerNorm(d)
        self.wq = nn.Linear(d, ds)
        self.wo = nn.Linear(ds, d)
        self.wg_h = nn.Linear(d, 1)
        self.wg_rho = nn.Linear(r + 1, 1, bias=False)
        self.wb = nn.Linear(d, 1)  # erase="free" (A4)
        self.feat = nn.Linear(r + 1, d)
        self.ln_fin = nn.LayerNorm(d)
        self.fin = MLP(d)
        self.ln_f = nn.LayerNorm(d)
        self.stop = nn.Linear(d, 1) if cfg.get("stop_head", False) else None
        self.word_dropout = float(cfg.get("word_dropout", 0.0))
        self.form_emb = nn.Embedding(cfg.get("n_forms", 10), d) if cfg.get("form_cond", False) else None
        self.rel_emb = nn.Embedding(N_REL, d) if cfg.get("rel_cond", False) else None   # step 2: r given with Z
        self.local_window = cfg.get("local_window", 0)
        self.gate = cfg["gate"]
        self.erase = cfg["erase"]
        self.read_mode = cfg["read_mode"]
        self.scale = 1.0 / math.sqrt(d)
        # keys are unit-norm, so q.k/sqrt(d) is ~flat at init; read_scale="learned" uses a learnable
        # temperature initialised at 1 (i.e. logits = q.k), shared by the pre-read and the residual read
        self.log_scale = (nn.Parameter(torch.tensor(math.log(cfg.get("read_scale_init", 8.0))))
                          if cfg.get("read_scale", "sqrt_d") == "learned" else None)
        # interventions (diagnostics only): force the gate to a constant, or cut h_local
        self.force_gate = None
        self.cut_local = False
        # beta_t = g_t, but with beta_detach the gate only gets gradient through its read role
        # (otherwise L_exh pushes g -> 1 everywhere, since more erasure always lowers L_exh)
        self.beta_detach = cfg.get("beta_detach", False)
        # erase_norm="max": erasure coefficient g * a_i / max_j a_j (the slot being read is erased fully
        # along e_t; a softmax never reaches 1, so otherwise every content direction keeps (1 - g a_i))
        self.erase_norm = cfg.get("erase_norm", "none")

    def _attn_weights(self, K, q):
        s = torch.einsum("nrd,nld->nlr", K, q.float())
        if self.read_mode == "softmax":
            scale = self.log_scale.exp() if self.log_scale is not None else self.scale
            return torch.softmax(s * scale, -1)
        return s.clamp(0.0, 1.0)

    def forward(self, win, length, M, K, V0, e, return_trace=True, form=None, rel=None):
        N, W = win.shape
        L = W - M
        dev = win.device
        inp = win[:, :-1]
        if self.training and self.word_dropout > 0:
            # stage A: replace visible INPUT tokens (previous idea's m tokens + tokens already emitted, forced
            # connector included) by <unk>; targets are read from `win` by the caller and are never dropped
            drop = (torch.rand(inp.shape, device=inp.device) < self.word_dropout) & (inp != PAD)
            inp = torch.where(drop, torch.full_like(inp, UNK), inp)
        Wi = W - 1
        x = self.emb(inp) + self.pos(torch.arange(Wi, device=dev))
        if self.form_emb is not None and form is not None:  # "separate form": the template is given, not in Z
            x = x + self.form_emb(form)[:, None]
        if self.rel_emb is not None:
            assert rel is not None, "rel_cond: the decoder needs the relation of the idea it writes"
            x = x + self.rel_emb(rel.long())[:, None].to(x.dtype)
        kvalid = inp != PAD
        eye = torch.eye(Wi, dtype=torch.bool, device=dev)
        causal = torch.ones(Wi, Wi, dtype=torch.bool, device=dev).tril()
        if self.local_window:  # A15: sliding window of the last `local_window` tokens only
            causal = causal & ~torch.ones(Wi, Wi, dtype=torch.bool, device=dev).tril(-self.local_window)
        mask = ((causal[None] & kvalid[:, None, :]) | eye[None])[:, None]
        K = K.float()
        V0 = V0.float()
        nb = len(self.blocks)
        for i, b in enumerate(self.blocks[:-1]):
            x = b(x, mask=mask)
            j = i - (nb - 1 - self.z_read_layers)          # index into the deep-read modules (>= 0 if active)
            if self.z_read_layers and j >= 0:
                hz = self.zr_ln[j](x)
                with torch.autocast("cuda", enabled=False):
                    az = self._attn_weights(K, self.zr_wq[j](hz.float()))
                    rz = torch.einsum("nlr,nrd->nld", az, V0)
                    gz = torch.sigmoid(self.zr_wg[j](hz.float())) if self.gate else torch.ones_like(rz[..., :1])
                    if self.force_gate is not None:
                        gz = torch.full_like(gz, self.force_gate)
                    x = x + (gz * self.zr_wo[j](rz)).to(x.dtype)
        if self.pre_read:
            h0 = self.ln_r0(x)
            with torch.autocast("cuda", enabled=False):
                a0 = self._attn_weights(K, self.wq0(h0.float()))
                r0 = torch.einsum("nlr,nrd->nld", a0, V0)
                g0 = torch.sigmoid(self.wg0(h0.float())) if self.gate else torch.ones_like(r0[..., :1])
                if self.force_gate is not None:
                    g0 = torch.full_like(g0, self.force_gate)
                x = x + (g0 * self.wo0(r0)).to(x.dtype)
        x = self.blocks[-1](x, mask=mask)
        P = M - 1
        hl = x[:, P:P + L]                                  # h_local at read positions [N, L, d]
        hq = self.ln_r(hl)
        with torch.autocast("cuda", enabled=False):
            hqf = hq.float()
            a = self._attn_weights(K, self.wq(hqf))         # [N, L, r]  (keys never erased -> parallel)
            gh = self.wg_h(hqf)[..., 0]                      # [N, L]
            beta_free = torch.sigmoid(self.wb(hqf))[..., 0] if self.erase == "free" else None
            n0_i = safe_norm(V0)                           # [N, r]
            n0 = n0_i.pow(2).sum(-1).add(1e-12).sqrt()
            inv0 = 1.0 / n0.clamp_min(1e-6)
            inv0_i = 1.0 / n0_i.clamp_min(1e-6)
            Vres = V0
            reads, gs, feats, rhos, projs = [], [], [], [], []
            for s in range(L):
                ni = safe_norm(Vres)
                rho = ni.pow(2).sum(-1).add(1e-12).sqrt() * inv0
                f = torch.cat([rho[:, None], ni * inv0_i], -1)   # [N, r+1]
                g = torch.sigmoid(gh[:, s] + self.wg_rho(f)[:, 0]) if self.gate else torch.ones_like(rho)
                if self.force_gate is not None:
                    g = torch.full_like(rho, self.force_gate)
                a_s = a[:, s]
                if return_trace:
                    es_ = e[:, s].float()
                    pr_ = (torch.einsum("nrd,nd->nr", Vres, es_).abs() if es_.dim() == 2
                           else torch.einsum("nrd,nhd->nrh", Vres, es_).norm(dim=-1))
                    projs.append((pr_ * a_s).sum(-1) * inv0)
                reads.append(torch.einsum("nr,nrd->nd", a_s, Vres))
                gs.append(g)
                feats.append(f)
                rhos.append(rho)
                if self.erase == "none":
                    continue
                if self.erase == "free":
                    Vnew = Vres * (1 - beta_free[:, s, None] * a_s)[..., None]
                else:
                    beta = g.detach() if self.beta_detach else g
                    a_e = a_s / a_s.max(-1, keepdim=True).values.clamp_min(1e-6) if self.erase_norm == "max" else a_s
                    Vnew = erase_step(Vres, a_e, beta, e[:, s].float(), self.erase)
                Vres = torch.where((s < length)[:, None, None], Vnew, Vres)
            rhos.append(safe_norm(Vres).pow(2).sum(-1).add(1e-12).sqrt() * inv0)
            R = torch.stack(reads, 1)
            G = torch.stack(gs, 1)
            F_ = torch.stack(feats, 1)
            upd = G[..., None] * self.wo(R) + self.feat(F_)
        h = (torch.zeros_like(hl) if self.cut_local else hl) + upd.to(hl.dtype)
        h = h + self.fin(self.ln_fin(h))
        hf = self.ln_f(h)
        logits = F.linear(hf, self.emb.weight)
        trace = {"rho": torch.stack(rhos, 1), "g": G, "a": a, "V_end": Vres, "n0": n0}
        if self.stop is not None:
            # stop logit at step s = "close the idea now, before producing token s". The input at step s is
            # token s-1, so step n (after the last token) is computed from real tokens: target 1 there.
            trace["stop_logit"] = self.stop(hf).float()[..., 0]
        if projs:
            trace["ev"] = torch.stack(projs, 1)  # sum_i a_i |e_t . v_i^res| / ||Z_0||
        return logits, trace


def Decoder(cfg, emb):
    return (DeepDecoder if cfg.get("arch", "parallel") == "deep" else ParallelDecoder)(cfg, emb)


class IdeaJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["d"]
        self.emb = nn.Embedding(cfg["vocab_size"], d)
        self.E = IdeaEncoder(cfg, self.emb)
        self.P = IdeaPredictor(cfg)
        if self.P.id_keys:   # plain attribute (not a submodule: the embedding stays owned by the LM)
            object.__setattr__(self.P, "word_emb", self.emb)
        self.PM = IdeaMaskPredictor(cfg) if cfg.get("mask_predictor", False) else None
        self.register_buffer("e_slot_norm", torch.zeros(cfg["r"]), persistent=True)  # running mean |v_i| of E_ema
        # phase 2: generative idea head by convergence (rectified flow), conditioned on P's slots
        self.Fh = None
        if cfg.get("flow_head", False):
            from .flow_head import FlowHead
            ds = cfg.get("d_slot", d)
            # flow context: "reg" = P's regressed slots [k_hat ; v_hat] (a mean, poor for choosing a mode),
            # "hidden" = P's slot states before its regression head, "hidden+reg" = both concatenated
            self.flow_ctx = cfg.get("flow_ctx", "reg")
            d_ctx = {"reg": 2 * ds, "hidden": d, "hidden+reg": d + 2 * ds}[self.flow_ctx]
            self.Fh = FlowHead({"d": d, "n_heads": cfg["n_heads"], "r": cfg["r"], "d_slot": ds,
                                "flow_layers": cfg.get("flow_layers", 2), "d_ctx": d_ctx})
        self.O = OutputEncoder(cfg, self.emb)
        self.dec = Decoder(cfg, self.emb)
        # step 2: P predicts r_{t+1} (relation class + pointer offset) from its slot states for idea t+1
        self.rel_head = nn.Linear(d, N_REL) if cfg.get("rel_cond", False) else None
        self.ptr_head = nn.Linear(d, N_OFF) if cfg.get("rel_cond", False) else None
        # plan pointer head: query from P's slot states WITHOUT the plan, keys from the past ideas' input slots
        self.plan_q = nn.Linear(d, d) if cfg.get("plan_cond", False) else None
        self.plan_k = nn.Linear(d, d) if cfg.get("plan_cond", False) else None
        self.plan_mode = "sample"            # inference: "sample" (T = 1) | "argmax"; batch["plan_o"] overrides
        self.plan_gen = None                 # torch.Generator for sampling
        init_weights(self, std=cfg.get("init_std", 0.02))   # model.init_std (0.01 = initial weights x 0.5)
        self.E_ema = copy.deepcopy(self.E)
        for p in self.E_ema.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        """E_ema (the JEPA target) always runs without dropout (enc_dropout: SimCSE positives)."""
        super().train(mode)
        self.E_ema.eval()
        return self

    def compile_parts(self, mode="blocks"):
        """torch.compile the transformer blocks (dynamic shapes). Fuses LayerNorm/GELU/residuals, which
        cuts both memory (fp32 LayerNorm outputs) and kernel launches; the intra-idea loop stays eager."""
        for mod in list(self.E.blocks) + list(self.P.blocks) + list(getattr(self.dec, "blocks", [])) + \
                list(getattr(self.dec, "base", [])):
            mod.compile(dynamic=True)
        for mod in (self.E.mlp, self.dec.fin) if hasattr(self.dec, "fin") else (self.E.mlp,):
            mod.compile(dynamic=True)

    def trainable_parameters(self):
        ema = {id(p) for p in self.E_ema.parameters()}
        return [p for p in self.parameters() if id(p) not in ema]

    def n_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    @torch.no_grad()
    def ema_update(self, tau):
        for pe, p in zip(self.E_ema.parameters(), self.E.parameters()):
            pe.mul_(tau).add_(p.detach(), alpha=1 - tau)

    def _p_tokens(self, k):
        """Tokens of the ideas P reads (p_id_keys): those of the last E call, which produced k."""
        if not self.P.id_keys:
            return None
        tok = getattr(self.E, "last_tok", None)
        assert tok is not None and tok.shape[0] == k.shape[0], "p_id_keys: the last E call did not encode P's inputs"
        return tok

    def _p_words(self, k, batch):
        """(word states, valid, selected ideas) of the last E call (p_word_kv); "support" (ORACLE) needs
        batch["idea_support"]."""
        if self.P.word_kv == "none":
            return None
        ws, wv = getattr(self.E, "last_words", None), getattr(self.E, "last_valid", None)
        assert ws is not None and ws.shape[0] == k.shape[0], "p_word_kv: the last E call did not encode P's inputs"
        if self.P.word_kv == "support":
            sel = batch["idea_support"].to(ws.device)
        else:
            sel = torch.ones(ws.shape[0], dtype=torch.bool, device=ws.device)
        return ws, wv, sel

    def encode(self, batch):
        return self.E(batch["idea_tok"], batch["idea_len"])

    def predict(self, k, v, batch):
        if self.cfg.get("predictor", "transformer") == "mean":
            return self._mean_predict(k, v, batch)
        if self.cfg.get("detach_P_inputs", False):  # E not trained through P's inputs
            k, v = k.detach(), v.detach()
        role = batch.get("idea_role", batch.get("is_deduction"))
        rel = (batch["rel_id"], batch["rel_off"]) if self.cfg.get("rel_cond", False) else None
        if self.cfg.get("plan_cond", False):
            pk, pv = self._predict_plan(k, v, batch, role, rel)
        else:
            pk, pv = self.P(k, v, batch["doc_index"], role=role, rel=rel, tokens=self._p_tokens(k), words=self._p_words(k, batch))
        pos = (batch["idea_doc"], batch["idea_pos"])
        self._p_hidden = self.P.last_hidden[pos]    # [NI, r, d], read by flow_context()
        return pk[pos], pv[pos]

    def predict_hyp(self, k, v, batch):
        """All K hypotheses for every idea row: (pk [NI, K, r, ds], pv [NI, K, r, ds], scores [NI, K])."""
        self.predict(k, v, batch)
        pk_all, pv_all, scores = self.P.last_hyp
        pos = (batch["idea_doc"], batch["idea_pos"])
        return pk_all[pos], pv_all[pos], scores[pos]

    def flow_context(self, pk, pv):
        """Context of the flow head for the ideas of the last predict() call."""
        from .flow_head import to_target
        mode = getattr(self, "flow_ctx", "reg")
        if mode == "reg":
            return to_target(pk, pv)
        h = self._p_hidden.to(pk.dtype)
        return h if mode == "hidden" else torch.cat([h, to_target(pk, pv)], -1)

    def flow_target(self, k, v):
        """Idea slots -> the flow's space. flow_space="direction" (default): [k ; v/|v|] (norms supplied at
        sampling time); "scaled": [k ; v / e_slot_norm_i] (the flow also models each slot's relative norm)."""
        from .flow_head import to_target
        if self.cfg.get("flow_space", "direction") == "scaled":
            s = self.e_slot_norm.clamp_min(1e-6).to(v.dtype)[None, :, None]
            return torch.cat([F.normalize(k, dim=-1), v / s], -1)
        return to_target(k, v)

    def flow_decode(self, z, pv=None):
        """Inverse of flow_target for sampled z. Direction space: value norms from E (flow_norms="E") or P."""
        from .flow_head import from_target
        if self.cfg.get("flow_space", "direction") == "scaled":
            zk, zv = z.chunk(2, -1)
            return F.normalize(zk, dim=-1), zv * self.e_slot_norm.clamp_min(1e-6).to(zv.dtype)[None, :, None]
        kf, vf = from_target(z)
        if self.cfg.get("flow_norms", "P") == "E" and hasattr(self, "e_slot_norm") and float(self.e_slot_norm.sum()) > 0:
            return kf, vf * self.e_slot_norm.to(vf.dtype)[None, :, None]
        return kf, vf * pv.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def predict_flow(self, k, v, batch, n_steps=16, temperature=1.0, generator=None):
        """Next-idea slots sampled by the flow head (see flow_target / flow_decode for the value norms)."""
        from .flow_head import from_target
        pk, pv = self.predict(k, v, batch)
        z = self.Fh.sample(self.flow_context(pk, pv), n_steps=n_steps, temperature=temperature, generator=generator)
        if self.cfg.get("flow_space", "direction") == "scaled":
            return self.flow_decode(z)
        kf, vf = from_target(z)
        if self.cfg.get("flow_norms", "P") == "E" and hasattr(self, "e_slot_norm") and float(self.e_slot_norm.sum()) > 0:
            # value norms at the E_ema scale (running mean per slot, filled during flow training or by
            # scripts/calibrate_flow_norms.py): P's norms are ~13x larger than E's and break the decoder when
            # combined with a non-P direction (diag_flow_decode, 26/09)
            return kf, vf * self.e_slot_norm.to(vf.dtype)[None, :, None]
        return kf, vf * pv.norm(dim=-1, keepdim=True)

    def _mean_predict(self, k, v, batch):
        """B1': Z for idea n = mean of the encoder slots of ideas < n (zeros for the first idea)."""
        di = batch["doc_index"]
        valid = (di >= 0).float()[..., None, None]
        vk = k[di.clamp_min(0)] * valid
        vv = v[di.clamp_min(0)] * valid
        cnt = valid.cumsum(1) - valid                                 # number of previous ideas
        mk = (vk.cumsum(1) - vk) / cnt.clamp_min(1)
        mv = (vv.cumsum(1) - vv) / cnt.clamp_min(1)
        mk = F.normalize(mk + (cnt == 0).float() * 1e-3, dim=-1)
        pos = (batch["idea_doc"], batch["idea_pos"])
        return mk[pos], mv[pos]

    def _predict_plan(self, k, v, batch, role, rel):
        """Plan variable. Pass 1 WITHOUT plan -> pointer logits over the past ideas (self.plan_logits,
        [B, Nm, Nm], -inf outside 0 <= j < n); o = batch["plan_o"] (per idea row, story index, -1 = none) if given,
        else drawn from the head (plan_mode). Pass 2 WITH o -> prediction."""
        di = batch["doc_index"]
        B, Nm = di.shape
        self.P(k, v, di, role=role, rel=rel, tokens=self._p_tokens(k), words=self._p_words(k, batch))
        h = self.P.last_hidden.mean(2)                                         # [B, Nm, d] block n predicts idea n
        kin = self.P.last_tok.mean(1)
        if self.cfg.get("plan_head_detach", False):                            # no gradient from the pointer head into P / E
            h, kin = h.detach(), kin.detach()
        keys = self.plan_k(kin)[di.clamp_min(0)]                               # [B, Nm, d] idea j
        lg = torch.einsum("bnd,bjd->bnj", self.plan_q(h), keys).float() / math.sqrt(h.shape[-1])
        n_idx = torch.arange(Nm, device=di.device)
        ok = (n_idx[None, :] < n_idx[:, None])[None] & (di >= 0)[:, None, :]
        lg = lg.masked_fill(~ok, float("-inf"))
        self.plan_logits = lg
        if batch.get("plan_o") is not None:        # training: every block gets its target
            plan_o = batch["plan_o"]
        else:                                        # inference: past blocks get the same label-free target computed
            from .objective import plan_targets      # on the ideas already there (online E); the predicted block
            plan_o = plan_targets(v, di)             # gets batch["plan_next"] or a draw from the head
        plan = torch.full((B, Nm), -1, dtype=torch.long, device=di.device)
        plan[di >= 0] = plan_o[di[di >= 0]].long()
        if batch.get("plan_o") is None:
            pb, pn = batch["idea_doc"], batch["idea_pos"]
            if batch.get("plan_next") is not None:
                nxt = batch["plan_next"].long()
            else:
                safe = lg[pb, pn]
                has = torch.isfinite(safe).any(-1)
                safe = safe.masked_fill(~torch.isfinite(safe), -1e9)
                nxt = safe.argmax(-1) if self.plan_mode == "argmax" else \
                    torch.multinomial(torch.softmax(safe, -1), 1, generator=self.plan_gen)[:, 0]
                nxt = nxt.masked_fill(~has, -1)
            plan[pb, pn] = nxt
        self.plan_choice = plan
        return self.P(k, v, di, role=role, rel=rel, plan=plan, tokens=self._p_tokens(k), words=self._p_words(k, batch))

    def rel_logits(self):
        """Step 2: (relation logits [NI, 3], pointer-offset logits [NI, 32]) for the idea each row predicts, from the
        slot states of the last predict() call."""
        h = self._p_hidden.mean(1)
        if self.cfg.get("rel_head_detach", False):   # no gradient from the relation heads into P
            h = h.detach()
        return self.rel_head(h).float(), self.ptr_head(h).float()

    def dec_rel(self, rel=None, **kw):
        """kwargs for self.dec: the relation of the written idea when rel_cond (step 2), nothing otherwise."""
        return dict(kw, rel=rel) if self.cfg.get("rel_cond", False) else kw

    def decode(self, batch, K, V, idx=None):
        win, length = batch["win"], batch["idea_len"]
        rel = batch.get("rel_id")
        if idx is not None:
            win, length = win[idx], length[idx]
            rel = rel[idx] if rel is not None else None
        M = batch["M"]
        e = self.O(win, M)
        if self.cfg.get("rel_cond", False):
            return self.dec(win, length, M, K, V, e, rel=rel)
        if self.cfg.get("form_cond", False):
            form = batch.get("form_id")
            if form is not None and idx is not None:
                form = form[idx]
            return self.dec(win, length, M, K, V, e, form=form)
        return self.dec(win, length, M, K, V, e)

"""Greedy / sampled generation for Idea-JEPA (idea-level) and the baselines."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model.baseline import GPT
from ..model.ideajepa import IdeaJEPA

PAD, BOS, EOS = 0, 1, 2


def _pick(logits, temperature, top_k, gen):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        v, _ = logits.topk(top_k, -1)
        logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
    return torch.multinomial(F.softmax(logits.float(), -1), 1, generator=gen)[:, 0]


@torch.no_grad()
def generate_idea(model: IdeaJEPA, prefix, K, V, max_len, is_end, close_rule="punct", tau=0.1,
                  extra_after_rho=3, temperature=0.0, top_k=None, gen=None, banned=None, form=None):
    """Decode one idea per row from slots (K, V). prefix: [N, M] tokens.
    is_end: bool tensor [vocab] marking terminal-punctuation tokens.
    close_rule: "punct" | "rho" | "rho_punct". Returns list of token lists, rho traces, close positions."""
    N, M = prefix.shape
    dev = prefix.device
    toks = torch.full((N, max_len + 1), PAD, dtype=torch.long, device=dev)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    length = torch.zeros(N, dtype=torch.long, device=dev)
    since_rho = torch.full((N,), -1, dtype=torch.long, device=dev)
    rho_hist = []
    for s in range(max_len):
        win = torch.cat([prefix, toks[:, :s + 1]], 1)
        n = torch.full((N,), s + 1, dtype=torch.long, device=dev)
        e = model.O(win, M)
        logits, tr = (model.dec(win, n, M, K, V, e, form=form) if form is not None else model.dec(win, n, M, K, V, e))
        rho_s = tr["rho"][:, s]  # residual before producing token s (after erasing tokens < s)
        rho_hist.append(rho_s)
        if s > 0:
            last = toks[:, s - 1]
            if close_rule == "punct":
                close = is_end[last]
            elif close_rule == "rho":
                close = rho_s < tau
            elif close_rule == "stop":
                close = tr["stop_logit"][:, s] > 0
            else:
                crossed = (rho_s < tau) & (since_rho < 0)
                since_rho = torch.where(crossed, torch.zeros_like(since_rho), since_rho)
                close = (since_rho >= 0) & (is_end[last] | (since_rho >= extra_after_rho))
                since_rho = torch.where(since_rho >= 0, since_rho + 1, since_rho)
            newly = close & ~done
            length = torch.where(newly, torch.full_like(length, s), length)
            done |= close
        if done.all():
            break
        lg = logits[:, s].float()
        if banned is not None:
            lg[:, banned] = float("-inf")
        nxt = _pick(lg, temperature, top_k, gen)
        toks[:, s] = torch.where(done, torch.full_like(nxt, PAD), nxt)
    length = torch.where(done, length, torch.full_like(length, s + 1 if not done.all() else s))
    out = [toks[i, :length[i]].tolist() for i in range(N)]
    return out, torch.stack(rho_hist, 1), length


@torch.no_grad()
def generate_story_idea(model: IdeaJEPA, n_ideas, is_end, max_idea_len=48, m_prefix=3, n_stories=1,
                        close_rule="rho_punct", tau=0.1, temperature=0.0, top_k=None, seed=0,
                        max_tokens=None, device="cuda"):
    """Full generation: E encodes generated ideas, P predicts the next idea's slots, decoder expresses it."""
    gen = torch.Generator(device=device).manual_seed(seed)
    stories = [[] for _ in range(n_stories)]  # list of idea token lists
    for j in range(n_ideas):
        # build batch of past ideas
        B = n_stories
        rows, doc_index = [], torch.full((B, j + 1), -1, dtype=torch.long)
        for b in range(B):
            for i, idea in enumerate(stories[b]):
                doc_index[b, i] = len(rows)
                rows.append(idea if idea else [PAD])
        r = model.cfg["r"]
        d = model.cfg.get("d_slot", model.cfg["d"])
        if rows:
            L = max(len(t) for t in rows)
            tok = torch.full((len(rows), L), PAD, dtype=torch.long)
            for i, t in enumerate(rows):
                tok[i, :len(t)] = torch.tensor(t)
            k, v = model.E(tok.to(device), torch.tensor([max(1, len(t)) for t in rows], device=device))
        else:
            k = torch.zeros(1, r, d, device=device)
            v = torch.zeros(1, r, d, device=device)
        # predictions for idea j in each story: P with doc_index where idea j is a placeholder
        di = doc_index.clone()
        di[:, j] = 0 if rows else -1
        batch = {"doc_index": di.to(device), "idea_doc": torch.arange(B, device=device),
                 "idea_pos": torch.full((B,), j, device=device)}
        if model.cfg.get("predictor", "transformer") == "mean":
            # mean predictor ignores slot j itself (strictly previous ideas)
            pass
        pk, pv = model.predict(k, v, batch)
        prefix = torch.full((B, m_prefix), PAD, dtype=torch.long)
        for b in range(B):
            hist = [BOS] + [t for idea in stories[b] for t in idea]
            p = hist[-m_prefix:]
            prefix[b, m_prefix - len(p):] = torch.tensor(p)
        ideas, _, _ = generate_idea(model, prefix.to(device), pk, pv, max_idea_len, is_end, close_rule, tau,
                                    temperature=temperature, top_k=top_k, gen=gen, banned=[PAD, BOS, EOS])
        for b in range(B):
            stories[b].append(ideas[b])
        if max_tokens and all(sum(len(i) for i in s) >= max_tokens for s in stories):
            break
    return stories


@torch.no_grad()
def generate_gpt(model: GPT, n_tokens, n_stories=1, temperature=0.0, top_k=None, seed=0, device="cuda",
                 is_end=None, m_prefix=3, max_idea_len=48):
    """B0: plain autoregressive. B1: window = current segment (S1 rule on generated tokens) + m prefix tokens."""
    gen = torch.Generator(device=device).manual_seed(seed)
    seq = torch.full((n_stories, 1), BOS, dtype=torch.long, device=device)
    seg_start = torch.ones(n_stories, dtype=torch.long, device=device)  # index of current idea's first token
    banned = [PAD, BOS, EOS]
    for t in range(n_tokens):
        if model.kind in ("b0", "b3"):
            ctx = seq[:, -model.pos.num_embeddings:]
            logits = model.run(ctx)[:, -1]
        else:
            T = seq.shape[1]
            W = m_prefix + max_idea_len
            win = torch.full((n_stories, W), PAD, dtype=torch.long, device=device)
            lens = []
            for b in range(n_stories):
                ss = int(seg_start[b])
                pre = seq[b, max(0, ss - m_prefix):ss]
                win[b, m_prefix - len(pre):m_prefix] = pre      # left-padded prefix, as in training
                cur = seq[b, ss:T]
                win[b, m_prefix:m_prefix + len(cur)] = cur
                lens.append(m_prefix + len(cur))
            out = model.run(win)
            logits = out[torch.arange(n_stories), torch.tensor(lens, device=device) - 1]
        logits = logits.float()
        logits[:, banned] = float("-inf")
        nxt = _pick(logits, temperature, top_k, gen)
        seq = torch.cat([seq, nxt[:, None]], 1)
        if is_end is not None:
            T = seq.shape[1]
            seg_start = torch.where(is_end[nxt] | (T - seg_start >= max_idea_len),
                                    torch.full_like(seg_start, T), seg_start)
    return seq[:, 1:].tolist()

"""Where the next idea comes from, for generation and phase-2 diagnostics.

    idea_candidates(model, k, v, batch, sampler, n=1, flow_steps=16, flow_temp=1.0)
        -> (K [NI, C, r, ds], V [NI, C, r, ds])
sampler:
  "P"     the predictor's regression (C = 1)
  "flow"  n samples of the flow head (C = n; direction from the flow, value norms from P)
  "khyp"  the K hypotheses of a k_hyp predictor, sorted by decreasing score (C = K; n is ignored)
Candidate 0 is always the default choice (P's output, the first flow sample, the best-scored hypothesis).
"""
from __future__ import annotations

import torch

SAMPLERS = ("P", "flow", "khyp")


@torch.no_grad()
def idea_candidates(model, k, v, batch, sampler="P", n=1, flow_steps=16, flow_temp=1.0, generator=None):
    if sampler == "P":
        pk, pv = model.predict(k, v, batch)
        return pk[:, None], pv[:, None]
    if sampler == "flow":
        if getattr(model, "Fh", None) is None:
            raise ValueError("sampler=flow needs a checkpoint trained with model.flow_head: true")
        ks, vs = zip(*[model.predict_flow(k, v, batch, n_steps=flow_steps, temperature=flow_temp,
                                          generator=generator) for _ in range(n)])
        return torch.stack(ks, 1), torch.stack(vs, 1)
    if sampler == "khyp":
        if getattr(model.P, "K", 1) < 2:
            raise ValueError("sampler=khyp needs a checkpoint trained with model.k_hyp >= 2")
        pk_all, pv_all, scores = model.predict_hyp(k, v, batch)
        order = scores.argsort(-1, descending=True)
        idx = order[..., None, None].expand(-1, -1, *pk_all.shape[2:])
        return pk_all.gather(1, idx), pv_all.gather(1, idx)
    raise ValueError(f"unknown sampler {sampler}")

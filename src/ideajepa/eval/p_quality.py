"""Question (a): does the idea predictor P beat trivial predictors of the next idea's slots?

python -m ideajepa.eval.p_quality runs/stage1/v_full_s0 [...]
Targets = E_ema slots of the true idea (the L_jepa targets). Predictors compared:
  P        : the trained idea predictor;
  mean     : constant = mean target slots over the fit split (per slot index);
  copy     : slots of the previous idea (E_ema) (for the first idea: the mean);
Metrics (per slot, weighted by the target slot norm, so empty slots do not count):
  cos      : cosine(prediction, target);
  jepa     : ||k_hat - k||^2 + ||v_hat/|v_hat| - v/|v|||^2 (the L_jepa distance);
  ce_gen   : token CE of the decoder with Z_0 = each predictor (the quantity that matters).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.loaders import make_loader
from ..data.dataset import to_device
from ..model.objective import per_idea_ce
from ..train import load_model

DEV = "cuda"


@torch.no_grad()
def collect_mean(model, loader, n_batches=30):
    sk = sv = None
    n = 0
    for i, b in enumerate(loader.batches()):
        if i >= n_batches:
            break
        b = to_device(b, DEV)
        k, v = model.E_ema(b["idea_tok"], b["idea_len"])
        sk = k.sum(0) if sk is None else sk + k.sum(0)
        sv = v.sum(0) if sv is None else sv + v.sum(0)
        n += k.shape[0]
    return F.normalize(sk / n, dim=-1), sv / n


@torch.no_grad()
def evaluate(run_dir: Path, n_batches=30):
    model, cfg, _ = load_model(run_dir / "ckpt.pt")
    if cfg["model"].get("predictor", "transformer") == "mean":
        return None
    mk, mv = collect_mean(model, make_loader(cfg, "val", shuffle=False), n_batches)
    agg = {}
    tot_tok = 0
    for i, b in enumerate(make_loader(cfg, "test_iid", shuffle=False).batches()):
        if i >= n_batches:
            break
        b = to_device(b, DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            k, v = model.encode(b)
            kt, vt = model.E_ema(b["idea_tok"], b["idea_len"])
            pk, pv = model.predict(k, v, b)
        kt, vt, pk, pv = kt.float(), vt.float(), pk.float(), pv.float()
        NI = kt.shape[0]
        # copy predictor: previous idea in the same document
        prev = torch.full((NI,), -1, dtype=torch.long, device=DEV)
        di = b["doc_index"]
        prev_idx = di[b["idea_doc"], (b["idea_pos"] - 1).clamp_min(0)]
        prev = torch.where(b["idea_pos"] > 0, prev_idx, prev)
        ck = torch.where((prev >= 0)[:, None, None], kt[prev.clamp_min(0)], mk[None].expand(NI, -1, -1))
        cv = torch.where((prev >= 0)[:, None, None], vt[prev.clamp_min(0)], mv[None].expand(NI, -1, -1))
        preds = {"P": (pk, pv), "mean": (mk[None].expand(NI, -1, -1), mv[None].expand(NI, -1, -1)),
                 "copy": (ck, cv)}
        w = vt.norm(dim=-1)
        ntok = int(b["idea_len"].sum())
        tot_tok += ntok
        for name, (hk, hv) in preds.items():
            cos = ((F.cosine_similarity(hv, vt, dim=-1) * w).sum(-1) / w.sum(-1).clamp_min(1e-8)).mean()
            jd = ((hk - kt).pow(2).sum(-1) + (F.normalize(hv, dim=-1) - F.normalize(vt, dim=-1)).pow(2).sum(-1))
            jd = ((jd * w).sum(-1) / w.sum(-1).clamp_min(1e-8)).mean()
            V0 = hv
            if model.cfg.get("z_norm", "none") == "idea":
                V0 = V0 / V0.pow(2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()[:, None, None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model.decode(b, hk, V0)
            ce, m = per_idea_ce(lg, b["win"], b["M"], b["idea_len"])
            ce = (ce * m).sum() / m.sum()
            for key, val in (("cos", cos), ("jepa", jd), ("ce_gen", ce)):
                agg[f"{name}_{key}"] = agg.get(f"{name}_{key}", 0.0) + val.item() * ntok
    return {k: v / tot_tok for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for r in a.runs:
        res = evaluate(Path(r))
        if res is None:
            continue
        (Path(r) / "p_quality.json").write_text(json.dumps(res, indent=1))
        print(Path(r).name, " ".join(f"{k}={v:.3f}" for k, v in res.items()))


if __name__ == "__main__":
    main()

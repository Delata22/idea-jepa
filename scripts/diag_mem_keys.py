"""Cos of the memory KEYS written by P between DIFFERENT facts of the SAME template (predicate, role) in the
same story, per layer (raw and centred by the batch mean), val batches of the run's own data view, teacher forcing
(plan models: the training plan targets). Compares plan (the plan enters the written keys) vs prop.
usage: uv run python scripts/diag_mem_keys.py <run> [<run> ...] [--batches 8] [--summary out.md]
"""
import argparse, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.data.dataset import to_device
from ideajepa.data.loaders import make_loader
from ideajepa.model.objective import plan_targets
from ideajepa.train import load_model

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--batches", type=int, default=8)
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
res = {}
for run in a.runs:
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV); model.eval()
    loader = make_loader(cfg, "val", False)
    acc = {}
    with torch.no_grad():
        for bi, b in enumerate(loader.batches()):
            if bi >= a.batches:
                break
            b = to_device(b, DEV)
            if "tmpl_key" not in b:
                continue
            k, v = model.encode(b)
            if model.cfg.get("plan_cond", False):
                b["plan_o"] = plan_targets(model.E_ema(b["idea_tok"], b["idea_len"])[1], b["doc_index"])
            model.P.keep_mem = True
            for m in model.P.mems:
                m.keep = True
            model.predict(k, v, b)
            rows = model.P.mem_rows
            valid, tk, fk, doc = rows["valid"], b["tmpl_key"], b["fact_key"], b["idea_doc"]
            pair = ((doc[:, None] == doc[None, :]) & (tk[:, None] == tk[None, :]) & (tk[:, None] >= 0)
                    & (fk[:, None] != fk[None, :]) & valid[:, None] & valid[None, :])
            pair = torch.triu(pair, 1)
            x = rows["k"].float()
            for li in range(x.shape[1]):
                xr = F.normalize(x[:, li], dim=-1)
                xc = F.normalize(x[:, li] - x[valid, li].mean(0, keepdim=True), dim=-1)
                for nm, z in (("raw", xr), ("centred", xc)):
                    c = (z @ z.T)[pair]
                    s, n = acc.get((li, nm), (0.0, 0))
                    acc[(li, nm)] = (s + float(c.sum()), n + int(c.numel()))
    res[run] = {k_: s / max(n, 1) for k_, (s, n) in acc.items()}
    print(run + ": " + "  ".join(f"L{li} {nm} {v_:.3f}" for (li, nm), v_ in sorted(res[run].items())), flush=True)
    del model
if a.summary:
    L = ["| run | couche | cos clés, faits différents même gabarit (brut) | centré |", "|---|---|---|---|"]
    for run, r in res.items():
        for li in sorted({k_[0] for k_ in r}):
            L.append(f"| {run.split('/')[-1]} | {li} | {r[(li, 'raw')]:.3f} | {r[(li, 'centred')]:.3f} |")
    open(a.summary, "w").write("\n".join(L) + "\n"); print("\n".join(L))

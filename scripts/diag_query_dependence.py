"""Mémo §34 (goal arm, diagnostic): does the memory READ query of the upper layers depend on what the lower layers read?
For each memory layer l >= 1, reference context (val stories, all families), P's query at every token: normal pass vs a pass
where the memory OUTPUT of every layer < l is zeroed. Mean cos(q_l normal, q_l ablated) over tokens (normalised queries,
as used by the read), and the same for the queries of the deduction-predicting blocks only; plus the change of layer l's
read output ||Δo|| / ||o||.  cos ≈ 1: the upper queries do not depend on the lower reads.
usage: uv run python scripts/diag_query_dependence.py <run> [<run> ...] [--batches 8] [--summary out.md]
"""
import argparse, collections, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.data.dataset import to_device
from ideajepa.data.loaders import make_loader
from ideajepa.train import load_model

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--batches", type=int, default=8)
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
res = {}
for run in a.runs:
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV); model.eval()
    mems = list(model.P.mems); L = len(mems)
    loader = make_loader(cfg, "val", False)
    acc = collections.defaultdict(list)
    with torch.no_grad():
        for bi, b in enumerate(loader.batches()):
            if bi >= a.batches:
                break
            b = to_device(b, DEV)
            k, v = model.encode(b)

            def pass_(zero_below):
                orig = [m.forward for m in mems]
                for li in range(zero_below):
                    mems[li].forward = (lambda x, q_bias=None, _m=mems[li]: torch.zeros_like(x))
                for m in mems:
                    m.capture = True
                try:
                    model.P(k, v, b["doc_index"], role=b.get("is_deduction"))
                    return [(m.last_qk[0].clone(), m.last_v.clone()) if hasattr(m, "last_qk") else None for m in mems]
                finally:
                    for m, f in zip(mems, orig):
                        m.forward = f; m.capture = False
            normal = pass_(0)
            for l in range(1, L):
                abl = pass_(l)
                qn, qa = normal[l][0], abl[l][0]                         # [B, H, T, dh], normalised
                acc[(l, "cos_q")].append(float(F.cosine_similarity(qn, qa, dim=-1).mean()))
    res[run] = {k_: sum(v_) / len(v_) for k_, v_ in acc.items()}
    print(run + ": " + "  ".join(f"layer {l}: cos(q, q | lower reads zeroed) {res[run][(l, 'cos_q')]:.3f}" for l in range(1, L)), flush=True)
    del model
if a.summary:
    Ls = sorted({k_[0] for r in res.values() for k_ in r})
    T = ["| run | " + " | ".join(f"couche {l} : cos(q, q sans lectures des couches < {l})" for l in Ls) + " |", "|---|" + "---|" * len(Ls)]
    for run, r in res.items():
        T.append(f"| {run.split('/')[-1]} | " + " | ".join(f"{r[(l, 'cos_q')]:.3f}" for l in Ls) + " |")
    open(a.summary, "w").write("\n".join(T) + "\n"); print("\n".join(T))

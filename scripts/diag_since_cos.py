"""Where does E put a restated sentence "since A , B ."? cos(E(since A , B), E(A)) vs cos(E(since A , B), E(B)), A and B
in their "so" form 0 (and A in its premise form), values (per-slot normalised, flattened) and keys, over the restated
sentences of the first N val docs. Appends nothing; prints a markdown table (mean ± std over the seeds of each group).
usage: uv run python scripts/diag_since_cos.py <run> [<run> ...] [--docs 600]
"""
import argparse, collections, os, statistics as st, sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.data import reasoning as R, verifier as V
from ideajepa.train import load_model

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--docs", type=int, default=600)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
pairs = []
for d in data["docs"][:a.docs]:
    for idea in d["ideas"]:
        if idea.get("restated"):
            f = tuple(tuple(x) if isinstance(x, list) else x for x in idea["fact"])
            A = V.parse_sentence(" ".join(vocab[t] for t in idea["tokens"]))[1]
            if A is not None:
                pairs.append((list(idea["tokens"]), A, f))
KEYS = ["A_so0", "A_premise", "B_so0"]
res = collections.defaultdict(lambda: collections.defaultdict(list))
for run in a.runs:
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    name = Path(run).name; g = name.rsplit("_s", 1)[0]
    def enc(sents):
        L = max(map(len, sents)); tok = torch.zeros(len(sents), L, dtype=torch.long)
        for i, t in enumerate(sents): tok[i, :len(t)] = torch.tensor(t)
        with torch.no_grad():
            k, v = model.E(tok.to(DEV), torch.tensor(list(map(len, sents)), device=DEV))
        return {"values": F.normalize(F.normalize(v.float(), dim=-1).flatten(1), dim=-1),
                "keys": F.normalize(k.float().flatten(1), dim=-1)}
    so0 = lambda f: [w2i[x] for x in R.render_with_template(f, 2).w]
    prem = lambda f: [w2i[x] for x in R.premise_forms(f)[0]]
    zs, za, zap, zb = (enc([p[0] for p in pairs]), enc([so0(p[1]) for p in pairs]),
                       enc([prem(p[1]) for p in pairs]), enc([so0(p[2]) for p in pairs]))
    for sp in ("values", "keys"):
        run_vals = {"A_so0": (zs[sp] * za[sp]).sum(-1), "A_premise": (zs[sp] * zap[sp]).sum(-1), "B_so0": (zs[sp] * zb[sp]).sum(-1)}
        for kk, x in run_vals.items():
            res[g][(sp, kk)].append(float(x.mean()))
        res[g][(sp, "closer_to_A")].append(float((run_vals["A_so0"] > run_vals["B_so0"]).float().mean()))
    print(f"{name}: " + " ".join(f"{sp}:{kk}={res[g][(sp, kk)][-1]:.3f}" for sp in ("values", "keys") for kk in KEYS + ["closer_to_A"]), flush=True)
fmt = lambda xs: f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"
print(f"\n## f. Where E puts \"since A , B\" ({len(pairs)} restated sentences, E online)\n")
print("| measure | " + " | ".join(f"{g} (n={len(res[g][('values', 'B_so0')])})" for g in res) + " |")
print("|---|" + "---|" * len(res))
for sp in ("values", "keys"):
    for kk, lab in (("A_so0", "cos(E(since A,B), E(A so form 0))"), ("A_premise", "cos(E(since A,B), E(A premise form))"),
                    ("B_so0", "cos(E(since A,B), E(B so form 0))"), ("closer_to_A", "share closer to A than to B")):
        print(f"| {sp}: {lab} | " + " | ".join(fmt(res[g][(sp, kk)]) for g in res) + " |")

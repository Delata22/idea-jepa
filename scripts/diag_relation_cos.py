"""F2: does E (and P's prediction) confuse the relation of a deduction B with the relation of its antecedent edge A?
For each F2 deduction B = (rel, X, r, Z) of the first N val stories, A = its antecedent (written one for a restated
sentence, else the generator's rule) = (rel, Y, r2, Z) with r2 != r:
  B'  = (rel, X, r2, Z)   same people, relation of A
  B'' = (rel, X, r3, Z)   same people, a third relation (neither r nor r2)
E: cos(E(so B), E(so B')) vs cos(E(so B), E(so B'')), values (per-slot normalised, flattened) and keys; "so" form 0.
P: at rank 1 of each F2 story (reference context = the premises, story surfaces), nearest of {B, B', B''} to P's
prediction (E_ema "so" form 0 values).
usage: uv run python scripts/diag_relation_cos.py <run> [<run> ...] [--docs 600]
"""
import argparse, collections, os, random, statistics as st, sys
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
RELS = ["parent", "grandparent", "sibling", "pibling"]
rng = random.Random(0)
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)
triples, rank1 = [], []
for di, d in enumerate(data["docs"][:a.docs]):
    if d["family"] != "F2":
        continue
    facts = [as_fact(i["fact"]) for i in d["ideas"]]
    first = True
    for ii, idea in enumerate(d["ideas"]):
        if idea["step"] != "deduction":
            continue
        B = facts[ii]
        A = (V.parse_sentence(" ".join(vocab[t] for t in idea["tokens"]))[1] if idea.get("restated")
             else R.antecedent("F2", B, facts[:ii]))
        if A is None or A[2] == B[2]:
            first = False
            continue
        r3 = rng.choice([x for x in RELS if x not in (B[2], A[2])])
        t = (B, ("rel", B[1], A[2], B[3]), ("rel", B[1], r3, B[3]))
        triples.append(t)
        if first:
            prem = [i["tokens"] for i in d["ideas"] if i["step"] == "premise"]
            rank1.append((prem, t))
        first = False
so0 = lambda f: [w2i[x] for x in R.render_with_template(f, 2).w]
res = collections.defaultdict(lambda: collections.defaultdict(list))
for run in a.runs:
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    g = Path(run).name.rsplit("_s", 1)[0]

    def enc(sents, ema=False):
        L = max(map(len, sents)); tok = torch.zeros(len(sents), L, dtype=torch.long)
        for i, t in enumerate(sents): tok[i, :len(t)] = torch.tensor(t)
        with torch.no_grad():
            k, v = (model.E_ema if ema else model.E)(tok.to(DEV), torch.tensor(list(map(len, sents)), device=DEV))
        return F.normalize(F.normalize(v.float(), dim=-1).flatten(1), dim=-1), F.normalize(k.float().flatten(1), dim=-1)
    zb, zp, zo = (enc([so0(t[j]) for t in triples]) for j in range(3))
    for sp, j in (("values", 0), ("keys", 1)):
        res[g][f"{sp}: cos(B, B' relation of A)"].append(float((zb[j] * zp[j]).sum(-1).mean()))
        res[g][f"{sp}: cos(B, B'' third relation)"].append(float((zb[j] * zo[j]).sum(-1).mean()))
    hits = collections.Counter()
    for prem, t in rank1:
        story = [list(p) for p in prem]
        L = max(map(len, story)); tok = torch.zeros(len(story), L, dtype=torch.long)
        for i, s in enumerate(story): tok[i, :len(s)] = torch.tensor(s)
        n = len(story)
        with torch.no_grad():
            k, v = model.E(tok.to(DEV), torch.tensor([len(s) for s in story], device=DEV))
            di_ = torch.arange(n + 1, device=DEV)[None].clone(); di_[0, n] = 0
            pk, pv = model.predict(k, v, {"doc_index": di_, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                                          "idea_pos": torch.tensor([n], device=DEV),
                                          "idea_role": torch.zeros(n, dtype=torch.long, device=DEV)})
        p = F.normalize(F.normalize(pv[0].float(), dim=-1).flatten(), dim=0)
        zc = enc([so0(f) for f in t], ema=True)[0]
        hits[["right relation", "relation of A", "third relation"][int((zc @ p).argmax())]] += 1
    for c in ("right relation", "relation of A", "third relation"):
        res[g][f"P rank 1 nearest: {c}"].append(hits[c] / max(1, len(rank1)))
    print(run, {k: round(v[-1], 3) for k, v in res[g].items()}, flush=True)
fmt = lambda xs: f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"
print(f"\n{len(triples)} F2 deductions with an antecedent of another relation; {len(rank1)} rank-1 steps\n")
print("| measure | " + " | ".join(f"{g} (n={len(next(iter(res[g].values())))})" for g in res) + " |")
print("|---|" + "---|" * len(res))
for key in next(iter(res.values())):
    print(f"| {key} | " + " | ".join(fmt(res[g][key]) for g in res) + " |")

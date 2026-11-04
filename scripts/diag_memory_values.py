"""Stage B target measure: are same-template facts about DIFFERENT entities distinct in memory?

For N val stories (the model's own collation: canonical stream under surfaces=True), E encodes the ideas, P runs
with the idea memories' capture on; the value written for idea i = the memory tokens at P's input position i + 1.
Centred cosine (the batch mean of each layer/head/slot is subtracted, then cos over slots, mean over layers and
heads) for pairs of ideas of the same story:
  same_template_diff_entities   same predicate and same role (premise / deduction), different arguments
                                (e.g. "the book is in the attic" vs "the kite is in the school")  -> target ~0.2
  same_fact                     same fact tuple, different role or surface (premise <-> deduction)   -> high
  diff_template                 different predicates                                               -> reference
Also reported for E's values (per-slot normalised, centred), the input of the memory.
usage: uv run python scripts/diag_memory_values.py <run> [--stories 200]
"""
import argparse
import collections
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ideajepa.data.dataset import Collator
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--stories", type=int, default=200)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
assert getattr(model.P, "mems", None), "needs a checkpoint with an idea memory"
for m in model.P.mems:
    m.capture = True
data = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
docs = [d for d in data["docs"] if len(d["ideas"]) < cfg["data"]["max_ideas"]][:a.stories]
coll = Collator(m_prefix=cfg["model"]["m_prefix"], surfaces=cfg["data"].get("surfaces", False), w2i=w2i)
r = model.cfg["r"]


def as_fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f)


def kind(f1, s1, f2, s2):
    if f1 == f2:
        return "same_fact"
    if f1[0] == f2[0] and s1 == s2:
        return "same_template_diff_entities"
    return "diff_template"


mem_vals, e_vals, metas = [], [], []
with torch.no_grad():
    for start in range(0, len(docs), a.bs):
        chunk = docs[start:start + a.bs]
        b = {k_: (v_.to(DEV) if torch.is_tensor(v_) else v_) for k_, v_ in coll(chunk).items()}
        k, v = model.encode(b)
        di = b["doc_index"]
        # one extra (placeholder) idea per document so that the last idea's value is written
        Bn, Nm = di.shape
        di2 = torch.cat([di, torch.full((Bn, 1), -1, device=DEV, dtype=di.dtype)], 1)
        for bi in range(Bn):
            n = int((di[bi] >= 0).sum())
            di2[bi, n] = di[bi, n - 1]
        model.P(k, v, di2)
        mv = torch.stack([m_.last_v for m_ in model.P.mems])              # [L, B, H, T, dh]
        Lr, _, H, T, dh = mv.shape
        mv = mv.view(Lr, Bn, H, T // r, r, dh).permute(1, 3, 0, 2, 4, 5)  # [B, pos, L, H, r, dh]
        for bi, d in enumerate(chunk):
            for j, idea in enumerate(d["ideas"]):
                row = int(di[bi, j])
                mem_vals.append(mv[bi, j + 1].cpu())                     # value written for idea j
                e_vals.append(F.normalize(v[row], dim=-1).cpu())
                metas.append((start + bi, as_fact(idea["fact"]), idea["step"], d["family"]))

MV = torch.stack(mem_vals); EV = torch.stack(e_vals)                     # [N, L, H, r, dh], [N, r, ds]
MVc = MV - MV.mean(0, keepdim=True); EVc = EV - EV.mean(0, keepdim=True)
mflat = F.normalize(MVc.flatten(3), dim=-1)    # [N, L, H, r*dh]
eflat = F.normalize(EVc.flatten(1), dim=-1)                              # [N, r*ds]
acc = collections.defaultdict(list)
by_story = collections.defaultdict(list)
for i, m_ in enumerate(metas):
    by_story[m_[0]].append(i)
for idx in by_story.values():
    for x in range(len(idx)):
        for y in range(x + 1, len(idx)):
            i, j = idx[x], idx[y]
            kd = kind(metas[i][1], metas[i][2], metas[j][1], metas[j][2])
            fam = metas[i][3]
            cm = float((mflat[i] * mflat[j]).sum(-1).mean())
            ce = float(eflat[i] @ eflat[j])
            for key in (fam, "all"):
                acc[(key, kd, "memory")].append(cm); acc[(key, kd, "E")].append(ce)
mean = lambda x: sum(x) / len(x) if x else float("nan")
kinds = ["same_template_diff_entities", "same_fact", "diff_template"]
print(f"{a.run.rstrip('/').split('/')[-1]}: centred cosine between ideas of the same story ({len(metas)} ideas)")
res = {}
for src in ("memory", "E"):
    print(f"[{src} values]  " + "  ".join(f"{k_:>28s}" for k_ in kinds))
    for fam in ("F1", "F2", "F3", "all"):
        cells = []
        for k_ in kinds:
            vals = acc[(fam, k_, src)]
            res.setdefault(src, {}).setdefault(fam, {})[k_] = mean(vals)
            cells.append(f"{mean(vals):28.3f}" if vals else f"{'–':>28s}")
        print(f"  {fam:4s} " + "  ".join(cells))
json.dump({"run": a.run, "results": res}, open(f"{a.run}/memory_values.json", "w"), indent=1)

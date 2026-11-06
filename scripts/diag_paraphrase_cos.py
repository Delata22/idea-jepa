"""Stage A measure: is the idea space separated from the form?

For the deductions of N val stories, every surface of the fact is encoded by E (or E_ema with --ema):
  same_deduction   cos between two different deduction templates of the same fact (3 connectors x 2 forms)
  premise_vs_ded   cos between the premise form and a deduction template of the same tuple (F1 objin, F2 rel)
  diff_same_story  cos between surfaces of two DIFFERENT deductions of the same story (hard negatives)
  diff_other       cos between surfaces of deductions of different stories
  pos_vs_goes      F1: cos("X goes to L", "X is in L") vs control "X is in L'" (another place)
  pos_vs_takes     F1: cos("X takes O", "the O is in L_X") vs control "the O is in L'"
reported for keys, values (per-slot normalised) and the flow space [k ; v/|v|], per family. Targets:
same_deduction 0.64 -> > 0.85 while diff_* stay low (~0.31).
usage: uv run python scripts/diag_paraphrase_cos.py <run> [--stories 200] [--ema]
"""
import argparse
import collections
import itertools
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--stories", type=int, default=200)
ap.add_argument("--ema", action="store_true", help="encode with E_ema instead of E")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
enc = model.E_ema if a.ema else model.E
data = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
rng = random.Random(a.seed)


def as_fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f)


@torch.no_grad()
def encode(sents):
    L = max(map(len, sents)); tok = torch.zeros(len(sents), L, dtype=torch.long)
    for i, t in enumerate(sents): tok[i, :len(t)] = torch.tensor(t)
    k, v = enc(tok.to(DEV), torch.tensor(list(map(len, sents)), device=DEV))
    kf = F.normalize(k.flatten(1), dim=-1)
    vn = F.normalize(v, dim=-1)
    vf = F.normalize(vn.flatten(1), dim=-1)
    zf = F.normalize(torch.cat([k, vn], -1).flatten(1), dim=-1)
    return {"keys": kf, "values": vf, "flow": zf}


stats = collections.defaultdict(lambda: collections.defaultdict(list))   # (family, kind) -> space -> cos list
story_reps = []
for d in data["docs"][:a.stories]:
    fam = d["family"]
    facts = list(dict.fromkeys(as_fact(i["fact"]) for i in d["ideas"] if i["step"] == "deduction"))
    reps = {}
    for f in facts:
        ded = [[w2i[x] for x in R.render_with_template(f, t).w] for t in R.DEDUCTION_TEMPLATES]
        prem = [[w2i[x] for x in w] for w in R.premise_forms(f)] if f[0] in R.PREMISE_PREDS else []
        e = encode(ded + prem)
        reps[f] = {sp: e[sp][:len(ded)] for sp in e}
        for sp, z in e.items():
            zd = z[:len(ded)]
            sim = zd @ zd.T
            iu = torch.triu_indices(len(ded), len(ded), 1)
            stats[(fam, "same_deduction")][sp] += sim[iu[0], iu[1]].tolist()
            if prem:
                stats[(fam, "premise_vs_ded")][sp] += (z[len(ded):] @ zd.T).flatten().tolist()
    for f1, f2 in itertools.combinations(facts, 2):
        for sp in reps[f1]:
            j1, j2 = rng.randrange(6), rng.randrange(6)
            stats[(fam, "diff_same_story")][sp].append(float(reps[f1][sp][j1] @ reps[f2][sp][j2]))
    story_reps.append((fam, reps))
for (fam1, r1), (fam2, r2) in zip(story_reps, story_reps[1:] + story_reps[:1]):
    if not r1 or not r2:
        continue
    f1, f2 = rng.choice(list(r1)), rng.choice(list(r2))
    for sp in r1[f1]:
        stats[(fam1, "diff_other")][sp].append(float(r1[f1][sp][rng.randrange(6)] @ r2[f2][sp][rng.randrange(6)]))

# position <-> movement (F1): does E encode the position a movement implies?
#   goes(X, L)  vs  "X is in L"       (same person, same place)       control: "X is in L'" (another place)
#   takes(X, O) vs  "the O is in L_X" (L_X = X's location at that time) control: "the O is in L'"
for d in data["docs"][:a.stories]:
    if d["family"] != "F1":
        continue
    prem = [as_fact(i["fact"]) for i in d["ideas"] if i["step"] == "premise"]
    locs = sorted({f[2] for f in prem if f[0] in ("at", "goes")} | {f[2] for f in prem if f[0] == "objin"})
    pos = {}
    for f in prem:
        if f[0] == "at":
            pos[f[1]] = f[2]
        pairs = []
        if f[0] == "goes":
            pos[f[1]] = f[2]
            other = [l for l in locs if l != f[2]]
            if other:
                pairs.append(("pos_vs_goes", f, ("at", f[1], f[2]), ("at", f[1], rng.choice(other))))
        elif f[0] == "takes" and f[1] in pos:
            other = [l for l in locs if l != pos[f[1]]]
            if other:
                pairs.append(("pos_vs_takes", f, ("objin", f[2], pos[f[1]]), ("objin", f[2], rng.choice(other))))
        for kind, move, same, ctrl in pairs:
            sents = [[w2i[x] for x in w] for w in (R.premise_forms(move)[0], R.premise_forms(same)[0], R.premise_forms(ctrl)[0])]
            e = encode(sents)
            for sp, z in e.items():
                stats[("F1", kind)][sp].append(float(z[0] @ z[1]))
                stats[("F1", kind + "_ctrl")][sp].append(float(z[0] @ z[2]))

kinds = ["same_deduction", "premise_vs_ded", "diff_same_story", "diff_other"]
move_kinds = ["pos_vs_goes", "pos_vs_goes_ctrl", "pos_vs_takes", "pos_vs_takes_ctrl"]
res = {}
print(f"{a.run.rstrip('/').split('/')[-1]}  encoder={'E_ema' if a.ema else 'E'}  ({a.stories} stories)")
for sp in ("flow", "values", "keys"):
    print(f"[{sp}]  " + "  ".join(f"{k:>16s}" for k in kinds))
    for fam in ("F1", "F2", "F3", "all"):
        cells = []
        for kind in kinds:
            vals = [x for (f_, k_), d_ in stats.items() if k_ == kind and (fam == "all" or f_ == fam) for x in d_[sp]]
            m = sum(vals) / len(vals) if vals else float("nan")
            res.setdefault(sp, {}).setdefault(fam, {})[kind] = m
            cells.append(f"{m:16.3f}" if vals else f"{'–':>16s}")
        print(f"  {fam:4s} " + "  ".join(cells))
print("[position <-> movement, F1]  (ctrl = same movement vs a position at ANOTHER place)")
for sp in ("flow", "values", "keys"):
    cells = []
    for kind in move_kinds:
        vals = stats[("F1", kind)][sp]
        m = sum(vals) / len(vals) if vals else float("nan")
        res.setdefault(sp, {}).setdefault("F1_move", {})[kind] = m
        cells.append(f"{kind} {m:.3f}")
    print(f"  {sp:6s} " + "  ".join(cells) + f"  (n={len(stats[('F1', 'pos_vs_goes')][sp])}/{len(stats[('F1', 'pos_vs_takes')][sp])})")
json.dump({"run": a.run, "encoder": "E_ema" if a.ema else "E", "results": res},
          open(f"{a.run}/paraphrase_cos{'_ema' if a.ema else ''}.json", "w"), indent=1)

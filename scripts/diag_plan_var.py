"""A "plan variable" in the DATA? Encoder only (E_ema of a trained run), no predictor, no decoder.
For every reference deduction n of a val story: o = argmax over the past ideas j < n (story surfaces) of
cos(Ē(idea n), Ē(idea j)), Ē = E_ema values of the sentence, per-slot normalised, flattened, L2; the deduction itself is
encoded in its "so" form 0 (the same form as the candidates below).
Per family: share of o = the useful premise (F1: last premise that fixed the object's location, R.f1_useful_premises;
F2 / F3: the generator's antecedent R.antecedent), o = another premise, o = a deduction; number of valid candidates of the
step (V.valid_next given the reference deductions said) that have the SAME o (1 = only the reference, 2, 3+);
mean cos to the 1st and 2nd nearest past idea; 5 examples.
usage: uv run python scripts/diag_plan_var.py <run> [<run> ...] [--n 200] [--summary out.md]
"""
import argparse
import collections
import json
import os
import statistics as st
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V
from ideajepa.train import load_model

PAD = 0
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--n", type=int, default=200, help="stories per family")
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
w = lambda t: " ".join(vocab[x] for x in t)
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()

    def enc(tok_lists):
        L = max(len(t) for t in tok_lists)
        tok = torch.full((len(tok_lists), L), PAD, dtype=torch.long)
        for i, t in enumerate(tok_lists):
            tok[i, :len(t)] = torch.tensor(t)
        _, v = model.E_ema(tok.to(DEV), torch.tensor([len(t) for t in tok_lists], device=DEV))
        return F.normalize(F.normalize(v.float(), dim=-1).flatten(1), dim=-1)

    so0 = lambda f: [w2i[x] for x in R.render_with_template(f, 2).w]
    cnt = collections.defaultdict(collections.Counter)
    cosl = collections.defaultdict(lambda: collections.defaultdict(list))
    ex = collections.defaultdict(list)
    per = collections.Counter()
    with torch.no_grad():
        for doc in docs:
            fam = doc.get("family")
            if per[fam] >= a.n:
                continue
            ideas = doc["ideas"]
            if sum(i["step"] == "deduction" for i in ideas) < 1:
                continue
            per[fam] += 1
            facts = [as_fact(i["fact"]) for i in ideas]
            prem_idx = [j for j, i in enumerate(ideas) if i["step"] == "premise"]
            pf = [facts[j] for j in prem_idx]
            useful_f1 = R.f1_useful_premises(pf) if fam == "F1" else {}
            Zpast = enc([i["tokens"] for i in ideas])
            said = []
            for n, idea in enumerate(ideas):
                if idea["step"] != "deduction":
                    continue
                f = facts[n]
                cands = sorted(V.valid_next(fam, pf, said) - set(said), key=str)
                zc = enc([so0(c) for c in cands]) if cands else None
                zn = enc([so0(f)])[0]
                sims = Zpast[:n] @ zn
                top = sims.topk(min(2, n))
                o = int(top.indices[0])
                if fam == "F1":
                    u = prem_idx[useful_f1[f[1]]] if f[0] == "objin" and f[1] in useful_f1 else None
                else:
                    ante = R.antecedent(fam, f, facts[:n])
                    u = max((j for j in range(n) if facts[j] == ante), default=None) if ante is not None else None
                cat = "useful" if o == u else ("other_premise" if ideas[o]["step"] == "premise" else "deduction")
                c_ = cnt[fam]
                c_["n"] += 1; c_[cat] += 1; c_["has_useful"] += u is not None
                if zc is not None and f in cands:
                    os_ = (zc @ Zpast[:n].T).argmax(1).tolist()
                    same = sum(x == o for x in os_)
                    c_["same_o_" + ("1" if same <= 1 else ("2" if same == 2 else "3+"))] += 1
                    c_["n_cands_known"] += 1
                    cosl[fam]["n_cands"].append(len(cands))
                cosl[fam]["cos1"].append(float(top.values[0]))
                if len(top.values) > 1:
                    cosl[fam]["cos2"].append(float(top.values[1]))
                if len(ex[fam]) < 5 and cat != "useful":
                    ex[fam].append({"deduction": w(idea["tokens"]), "nearest": w(ideas[o]["tokens"]),
                                    "useful": w(ideas[u]["tokens"]) if u is not None else None, "cos": round(float(top.values[0]), 3)})
                said.append(f)
    res = {fam: {**dict(c_), **{k: st.mean(v) for k, v in cosl[fam].items()}} for fam, c_ in cnt.items()}
    json.dump({"run": run, "results": res, "examples": ex}, open(f"{run}/plan_var.json", "w"), indent=1)
    print(f"{run}: " + "  ".join(f"{fam} useful {c_['useful'] / c_['n']:.3f}" for fam, c_ in sorted(cnt.items())), flush=True)
    del model
    return res, ex


allr = {r: run_one(r) for r in a.runs}
if a.summary:
    L = ["| run | famille | n | o = prémisse utile | o = autre prémisse | o = déduction | candidates au même o : 1 / 2 / 3+ | cos 1er | cos 2e | candidates valides (moy.) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r, (res, _) in allr.items():
        for fam in ("F1", "F2", "F3"):
            c = res.get(fam)
            if not c:
                continue
            n, nk = c["n"], max(1, c.get("n_cands_known", 0))
            f = lambda k, d=n: f"{c.get(k, 0) / d:.3f}"
            L.append(f"| {r.split('/')[-1]} | {fam} | {n} | {f('useful')} | {f('other_premise')} | {f('deduction')} | "
                     f"{f('same_o_1', nk)} / {f('same_o_2', nk)} / {f('same_o_3+', nk)} | {c.get('cos1', 0):.3f} | {c.get('cos2', 0):.3f} | "
                     f"{c.get('n_cands', 0):.2f} |")
    L += ["", "Exemples (o ≠ prémisse utile) :"]
    for r, (_, ex) in list(allr.items())[:1]:
        for fam, es in ex.items():
            for e in es[:2]:
                L.append(f"- {fam} : déduction « {e['deduction']} » → plus proche « {e['nearest']} » (cos {e['cos']}) ; prémisse utile « {e['useful']} »")
    open(a.summary, "w").write("\n".join(L) + "\n")
    print("\n".join(L))

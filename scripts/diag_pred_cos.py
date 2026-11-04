"""Why does stage A lose free-generation accuracy? Where does P's prediction land among the story's facts?

Phase-1 diagnostic (diag_averaging.py) redone with ONE surface for every fact so that models trained with and without
surface pooling are compared on equal terms: each fact is encoded by E_ema from its "so" + clause form 0 sentence
(render_with_template(f, 2)), premises included; same space as phase 1 (per-slot-normalised values, flattened, L2).
Teacher forcing on val, at every deduction step with >= 1 valid candidate, p = P's predicted idea:
  cos_best / cos_centroid      max over the valid candidates / centroid of the valid candidates
  cos_invalid_max / _mean      invalid facts of the story: the true fact with ONE argument replaced by a value seen in
                               the same argument slot of the same predicate in the story (invalid_facts pools),
                               kept only if not entailed; all of them, not capped
  cos_premise_max / _mean      the story's premises, "so" form   (…_nat: in their own premise form, for reference)
  cos_said_max                 deductions already written (true, but repeats)
  margin                       cos_best - cos_invalid_max
  nn_valid / nn_invalid / nn_premise / nn_said   nearest neighbour of p in valid ∪ invalid ∪ premises ∪ said
Split by family and by number of valid candidates (1, 2-3, 4+).
usage: uv run python scripts/diag_pred_cos.py <run> [<run> ...] [--n 90] [--summary runs/v3/pred_cos_summary.md]
"""
import argparse
import collections
import json
import os
import random
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V
from ideajepa.data.dataset import Collator
from ideajepa.eval.samplers import idea_candidates
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--n", type=int, default=90, help="val stories (n/3 per family)")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--summary", default=None, help="markdown table (mean ± std over the runs of each group)")
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
PAD = 0
KEYS = ["cos_best", "cos_centroid", "cos_invalid_max", "cos_invalid_mean", "cos_premise_max", "cos_premise_mean",
        "cos_premise_nat_max", "cos_said_max", "margin", "nn_valid", "nn_invalid", "nn_premise", "nn_said"]


def as_fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f)


def nbin(n):
    return "1" if n == 1 else ("2-3" if n <= 3 else "4+")


def invalid_pool(fact, prem, truth):
    """diag_averaging.invalid_facts pools, uncapped: one argument of the true fact replaced by a value seen in the same
    slot of the same predicate in the story (premises or entailed facts); kept if not entailed."""
    pool = collections.defaultdict(set)
    for f in list(prem) + list(truth):
        for pos in range(1, len(f)):
            pool[(f[0], pos)].add(f[pos])
    alts = []
    for pos in range(1, len(fact)):
        for e in sorted(pool.get((fact[0], pos), ()), key=str):
            if e == fact[pos]:
                continue
            g = list(fact); g[pos] = e; g = tuple(g)
            if g not in truth:
                alts.append(g)
    return list(dict.fromkeys(alts))


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    data = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
    vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
    per_fam = collections.Counter(); docs = []
    for d in data["docs"]:
        if len(d["ideas"]) < cfg["data"]["max_ideas"] and per_fam[d["family"]] < a.n // 3:
            docs.append(d); per_fam[d["family"]] += 1
    coll = Collator(m_prefix=cfg["model"]["m_prefix"], surfaces=cfg["data"].get("surfaces", False), w2i=w2i, seed=a.seed)
    cache = {}

    @torch.no_grad()
    def enc(sents):
        """E_ema of token lists -> per-slot-normalised values, flattened, L2 [n, r*ds]."""
        todo = [s for s in sents if s not in cache]
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]; L = max(map(len, chunk))
            tok = torch.full((len(chunk), L), PAD, dtype=torch.long)
            for j, t in enumerate(chunk):
                tok[j, :len(t)] = torch.tensor(t)
            _, v = model.E_ema(tok.to(DEV), torch.tensor([len(t) for t in chunk], device=DEV))
            z = F.normalize(F.normalize(v.float(), dim=-1).flatten(1), dim=-1)
            for t, zz in zip(chunk, z):
                cache[t] = zz
        return torch.stack([cache[s] for s in sents]) if sents else None

    so = lambda f: tuple(w2i[x] for x in R.render_with_template(f, 2).w)
    nat = lambda f: tuple(w2i[x] for x in R.premise_forms(f)[0])
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    with torch.no_grad():
        for s in range(0, len(docs), a.bs):
            chunk = docs[s:s + a.bs]
            b = {k_: (v_.to(DEV) if torch.is_tensor(v_) else v_) for k_, v_ in coll(chunk).items()}
            k, v = model.E(b["idea_tok"], b["idea_len"])
            pk, pv = idea_candidates(model, k, v, b, "P", 1)
            P = F.normalize(F.normalize(pv[:, 0].float(), dim=-1).flatten(1), dim=-1)
            di = b["doc_index"]
            for bi, d in enumerate(chunk):
                prem = V.doc_premises(d)
                truth = V.derivable(d["family"], prem)
                zprem, zprem_nat = enc([so(f) for f in prem]), enc([nat(f) for f in prem])
                said = []
                for j, idea in enumerate(d["ideas"]):
                    fact = as_fact(idea["fact"])
                    cands = [as_fact(c) for c in (idea.get("candidates") or [])]
                    if idea["step"] != "deduction" or not cands:
                        if idea["step"] == "deduction":
                            said.append(fact)
                        continue
                    p = P[int(di[bi, j])]
                    zc = enc([so(c) for c in cands])
                    inv = [g for g in invalid_pool(fact, prem, truth) if g not in cands]
                    zi = enc([so(g) for g in inv])
                    zs = enc([so(f) for f in said])
                    cv = zc @ p
                    row = {"cos_best": float(cv.max()), "cos_centroid": float(F.normalize(zc.mean(0), dim=0) @ p),
                           "cos_premise_max": float((zprem @ p).max()), "cos_premise_mean": float((zprem @ p).mean()),
                           "cos_premise_nat_max": float((zprem_nat @ p).max())}
                    best = {"valid": float(cv.max()), "premise": row["cos_premise_max"]}
                    if zi is not None:
                        ci = zi @ p
                        row.update(cos_invalid_max=float(ci.max()), cos_invalid_mean=float(ci.mean()),
                                   margin=float(cv.max() - ci.max()))
                        best["invalid"] = float(ci.max())
                    if zs is not None:
                        row["cos_said_max"] = float((zs @ p).max()); best["said"] = row["cos_said_max"]
                    nn = max(best, key=best.get)
                    for c in ("valid", "invalid", "premise", "said"):
                        row[f"nn_{c}"] = float(nn == c)
                    for fk in (d["family"], "all"):
                        for nk in (nbin(len(cands)), "all"):
                            for key, val in row.items():
                                acc[f"{fk}|{nk}"][key].append(val)
                            acc[f"{fk}|{nk}"]["N"].append(1)
                    said.append(fact)
    out = {g: {key: (sum(vs) / len(vs) if key != "N" else len(vs)) for key, vs in d_.items()} for g, d_ in acc.items()}
    json.dump({"run": run, "n_stories": len(docs), "results": out}, open(f"{run}/pred_cos.json", "w"), indent=1)
    r = out["all|all"]
    print(f"{run}: N={r['N']}  best {r['cos_best']:.3f}  inval_max {r['cos_invalid_max']:.3f}  prem_max "
          f"{r['cos_premise_max']:.3f}  margin {r['margin']:+.3f}  nn valid/inval/prem/said {r['nn_valid']:.2f}/"
          f"{r['nn_invalid']:.2f}/{r['nn_premise']:.2f}/{r['nn_said']:.2f}", flush=True)
    del model; torch.cuda.empty_cache()
    return out


results = {}
for run in a.runs:
    p = Path(run) / "pred_cos.json"
    results[run] = run_one(run)

if a.summary:
    groups = collections.defaultdict(list)                     # strip the seed suffix: xxx_s0/_s1/_s2 -> xxx
    for run, res in results.items():
        name = Path(run).name
        groups[name.rsplit("_s", 1)[0] if name[-1].isdigit() and "_s" in name else name].append(res)
    fmt = lambda xs: (f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) >= 2 else f"{xs[0]:.3f}") if xs else "–"
    lines = [f"# P's prediction vs the story's facts (teacher forcing, {a.n} val stories, E_ema values, \"so\" form 0)", "",
             "mean ± std over seeds; margin = cos_best − cos_invalid_max; nn_* = nearest neighbour of the prediction "
             "among valid ∪ invalid ∪ premises ∪ already said.", ""]
    strata = [f"{f}|all" for f in ("all", "F1", "F2", "F3")] + [f"all|{n}" for n in ("1", "2-3", "4+")]
    for stratum in strata:
        lines += [f"## {stratum.replace('|all', '').replace('all|', 'n valid = ')}", "",
                  "| measure | " + " | ".join(f"{g} (n={len(rs)})" for g, rs in groups.items()) + " |",
                  "|---|" + "---|" * len(groups)]
        for key in ["N"] + KEYS:
            lines.append(f"| {key} | " + " | ".join(fmt([r[stratum][key] for r in rs if stratum in r and key in r[stratum]])
                                                  for rs in groups.values()) + " |")
        lines.append("")
    Path(a.summary).write_text("\n".join(lines))
    print(f"-> {a.summary}")

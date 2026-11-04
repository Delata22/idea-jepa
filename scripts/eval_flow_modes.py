"""Phase 2: does the generative idea head land IN a mode (one valid deduction) rather than between them?

Teacher forcing on v2 val, deduction steps with n >= 2 valid candidates. Each candidate c is represented by
the mean of E_ema over its 6 paraphrases (3 connectors x 2 forms), in the flow space [k ; v/|v|] flattened
and L2-normalised; the centroid is the mean of the candidates. For every sample s of the head
(16 flow samples, or the K hypotheses of a k_hyp predictor) and for P's regression:
  hit          cos(s, nearest valid) > cos(s, centroid)  and  cos(s, nearest valid) > --thr
  near_invalid cos(s, nearest invalid) > cos(s, nearest valid) and > --thr   (invalid = the story's
               deductions that are not valid here: already said, or not derivable yet)
  coverage     distinct candidates hit at this step / n
--thr defaults to 0.64, the cosine between paraphrases of one fact measured in phase 1 (RESEARCH_NOTES §28.2).
usage: uv run python scripts/eval_flow_modes.py <run> [--sampler flow|khyp] [--n_samples 16] [--stories 300]
"""
import argparse
import collections
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data.dataset import Collator
from ideajepa.eval.samplers import idea_candidates
from ideajepa.model.flow_head import to_target
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--sampler", default="flow", choices=["flow", "khyp", "truth"],
                help="truth = sanity reference: the 'sample' is E_ema of the true next sentence (any checkpoint)")
ap.add_argument("--n_samples", type=int, default=16)
ap.add_argument("--flow_steps", type=int, default=16)
ap.add_argument("--flow_temp", type=float, default=1.0)
ap.add_argument("--thr", type=float, default=0.64)
ap.add_argument("--stories", type=int, default=300)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--family", default=None, choices=[None, "F1", "F2", "F3"], help="restrict to one family")
ap.add_argument("--misses", action="store_true", help="break down the head's missed samples")
ap.add_argument("--between_margin", type=float, default=0.05)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
data = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
docs = [d for d in data["docs"] if len(d["ideas"]) <= cfg["data"]["max_ideas"] and (a.family is None or d["family"] == a.family)][:a.stories]
MISS = collections.defaultdict(collections.Counter)      # family -> miss category counts (head samples)
MISS_COS = collections.defaultdict(lambda: collections.defaultdict(float))
gen = torch.Generator(device=DEV).manual_seed(a.seed)
_rep_cache = {}


def flat(k_, v_):
    return F.normalize(to_target(k_, v_).flatten(1), dim=-1)


@torch.no_grad()
def fact_rep(fact):
    """Mean over the 6 paraphrases of a deduction, encoded by E_ema, in the flow space."""
    if fact not in _rep_cache:
        sents = []
        for conn in R.CONNECT:
            for form in (0, 1):
                s = R.S().add(conn)
                s.ext(R.render_clause(fact, None, form)[0]).add(".")
                sents.append([w2i[x] for x in s.w])
        L = max(map(len, sents))
        tok = torch.zeros(len(sents), L, dtype=torch.long)
        for i, t in enumerate(sents): tok[i, :len(t)] = torch.tensor(t)
        k_, v_ = model.E_ema(tok.to(DEV), torch.tensor(list(map(len, sents)), device=DEV))
        _rep_cache[fact] = F.normalize(flat(k_, v_).mean(0), dim=0)
    return _rep_cache[fact]


def as_fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f) if isinstance(f, (list, tuple)) else f


acc = collections.defaultdict(lambda: collections.defaultdict(float))


def add(key, name, v):
    acc[key][name] += v


coll = Collator(m_prefix=cfg["model"]["m_prefix"])
with torch.no_grad():
    for start in range(0, len(docs), a.bs):
        chunk = docs[start:start + a.bs]
        b = {k_: (v_.to(DEV) if torch.is_tensor(v_) else v_) for k_, v_ in coll(chunk).items()}
        k, v = model.E(b["idea_tok"], b["idea_len"])
        PK, PV = idea_candidates(model, k, v, b, "P")
        n_s = a.n_samples if a.sampler == "flow" else 1
        if a.sampler == "truth":
            tk, tv = model.E_ema(b["idea_tok"], b["idea_len"])
            SK, SV = tk[:, None], tv[:, None]
        else:
            SK, SV = idea_candidates(model, k, v, b, a.sampler, n_s, a.flow_steps, a.flow_temp, gen)
        r = 0
        for d in chunk:
            story_ded = [as_fact(i["fact"]) for i in d["ideas"] if i["step"] == "deduction"]
            for i in d["ideas"]:
                if i["step"] == "deduction" and len(i.get("candidates") or []) >= 2:
                    cands = [as_fact(c) for c in i["candidates"]]
                    invalid = [f for f in story_ded if f not in cands]
                    C = torch.stack([fact_rep(c) for c in cands])                    # [n, D]
                    cent = F.normalize(C.mean(0), dim=0)
                    I = torch.stack([fact_rep(f) for f in invalid]) if invalid else None
                    n = len(cands); nb = "4+" if n >= 4 else str(n)
                    for name, (KK, VV) in (("P", (PK[r], PV[r])), (a.sampler, (SK[r], SV[r]))):
                        z = flat(KK, VV)                                                 # [S, D]
                        cv = z @ C.T                                                     # [S, n]
                        best, which = cv.max(1)
                        hit = (best > z @ cent) & (best > a.thr)
                        near_inv = torch.zeros_like(hit)
                        if I is not None:
                            ci = (z @ I.T).max(1).values
                            near_inv = (ci > best) & (ci > a.thr)
                        if a.misses and name == a.sampler:
                            fam = d["family"]
                            ref = as_fact(i["fact"])
                            inter = (C @ C.T)[~torch.eye(n, dtype=torch.bool, device=C.device)].mean()
                            MISS_COS[fam]["inter_candidate_cos"] += float(inter); MISS_COS[fam]["steps"] += 1
                            top2 = cv.topk(2, dim=1).values
                            cc = z @ cent
                            for j in range(len(z)):
                                if hit[j]:
                                    MISS[fam]["hit"] += 1; continue
                                if near_inv[j]:
                                    cat = "near_invalid"
                                elif top2[j, 1] > a.thr and top2[j, 0] - top2[j, 1] < a.between_margin:
                                    cat = "between_two_candidates"
                                elif top2[j, 0] > cc[j]:
                                    cat = "under_threshold"          # right direction, weak (paraphrase of a fact?)
                                else:
                                    cat = "centroid"                 # averaging
                                MISS[fam][cat] += 1
                                MISS[fam]["nearest_is_reference|" + cat] += int(cands[int(which[j])] == ref)
                                for nm, val in (("best", top2[j, 0]), ("second", top2[j, 1]), ("centroid", cc[j])):
                                    MISS_COS[fam][f"{cat}|{nm}"] += float(val)
                        for key in (f"n={nb}", d["family"], "all"):
                            add(key, f"{name}_samples", len(z))
                            add(key, f"{name}_hit", float(hit.sum()))
                            add(key, f"{name}_near_invalid", float(near_inv.sum()))
                            add(key, f"{name}_best_cos", float(best.sum()))
                            add(key, f"{name}_centroid_cos", float((z @ cent).sum()))
                            add(key, f"{name}_coverage", len(set(which[hit].tolist())) / n)
                            add(key, f"{name}_steps", 1)
                r += 1

res = {}
print(f"{a.run.rstrip('/').split('/')[-1]}  sampler={a.sampler} (x{a.n_samples if a.sampler == 'flow' else ('K' if a.sampler == 'khyp' else 1)})  thr={a.thr}")
print(f"{'':6s} {'steps':>6s} | {'hit':>11s} | {'coverage':>11s} | {'near-invalid':>12s} | {'cos best/centroid (P)':>22s} | {'cos best/centroid (head)':>24s}")
for key in ["n=2", "n=3", "n=4+", "F1", "F2", "F3", "all"]:
    q = acc.get(key)
    if not q:
        continue
    row = {}
    for name in ("P", a.sampler):
        S, T = q[f"{name}_samples"], q[f"{name}_steps"]
        row[name] = {"hit": q[f"{name}_hit"] / S, "near_invalid": q[f"{name}_near_invalid"] / S,
                     "coverage": q[f"{name}_coverage"] / T, "best_cos": q[f"{name}_best_cos"] / S,
                     "centroid_cos": q[f"{name}_centroid_cos"] / S}
    res[key] = row
    P, H = row["P"], row[a.sampler]
    print(f"{key:6s} {int(q['P_steps']):6d} | {P['hit']:.2f} / {H['hit']:.2f} | {P['coverage']:.2f} / {H['coverage']:.2f} | "
          f"{P['near_invalid']:.2f} / {H['near_invalid']:.2f} | {P['best_cos']:.2f} / {P['centroid_cos']:.2f}{'':12s} | "
          f"{H['best_cos']:.2f} / {H['centroid_cos']:.2f}")
print("(each cell: P / head)")
json.dump({"sampler": a.sampler, "thr": a.thr, "results": res}, open(f"{a.run}/flow_modes_{a.sampler}.json", "w"), indent=1)

if a.misses:
    print("\nMissed samples of the head, by family (share of all head samples; mean cos best / 2nd / centroid):")
    for fam in sorted(MISS):
        c = MISS[fam]; tot = sum(v for k, v in c.items() if "|" not in k)
        q = MISS_COS[fam]
        print(f"  {fam}: {tot} samples, hit {c['hit'] / tot:.2f}; mean cos BETWEEN candidates of a step = "
              f"{q['inter_candidate_cos'] / max(1, q['steps']):.3f}")
        for cat in ("between_two_candidates", "under_threshold", "centroid", "near_invalid"):
            k = c[cat]
            if not k: continue
            print(f"     {cat:24s} {k / tot:.2f}   cos {q[cat + '|best'] / k:.3f} / {q[cat + '|second'] / k:.3f} / "
                  f"{q[cat + '|centroid'] / k:.3f}   nearest = reference fact: {c['nearest_is_reference|' + cat] / k:.2f}")

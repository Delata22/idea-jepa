"""A flow sample counts as a hit if it DECODES into a valid deduction (verifier), not on a cosine threshold.

Teacher forcing on v2 val deduction steps (n = 1 and n >= 2). For every step: S samples of the flow head
(model.predict_flow: flow direction + the checkpoint's value norms, E if calibrated) and P's regression; every
idea is decoded greedily (batched) and judged by the verifier from the TRUE state (previous true deductions).
Per step:
  decoded_hit    share of the S samples that decode into a valid deduction
  coverage       distinct valid facts obtained / n
  any_valid@K    at least one of the first K samples is valid (teacher-forcing oracle ceiling of best-of-K)
  cos_hit        the old measure (closer to a valid candidate than to the centroid, and cos > --thr)
  verdicts       valid / premature / repeat / premise / false / unparsed
usage: uv run python scripts/eval_decoded_hits.py <flow_run> [--stories 300] [--n_samples 16] [--flow_steps 32]
       [--flow_temp 0.5]      (IDEAJEPA_DEVICE=cpu to run without GPU)
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
from ideajepa.data import verifier as V
from ideajepa.data.dataset import Collator
from ideajepa.eval.generate import generate_idea
from ideajepa.model.flow_head import to_target
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--stories", type=int, default=300)
ap.add_argument("--n_samples", type=int, default=16)
ap.add_argument("--flow_steps", type=int, default=32)
ap.add_argument("--flow_temp", type=float, default=0.5)
ap.add_argument("--thr", type=float, default=0.64)
ap.add_argument("--bs", type=int, default=16, help="stories per batch")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
torch.manual_seed(a.seed)
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
assert getattr(model, "Fh", None) is not None, "needs a flow checkpoint"
data = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
w = lambda toks: " ".join(vocab[t] for t in toks)
is_end = torch.zeros(len(vocab), dtype=torch.bool, device=DEV); is_end[w2i["."]] = True
docs = [d for d in data["docs"] if len(d["ideas"]) <= cfg["data"]["max_ideas"]][:a.stories]
gen = torch.Generator(device=DEV).manual_seed(a.seed)
KS = [k for k in (1, 4, 8, 16) if k <= a.n_samples]
_rep = {}


def as_fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f) if isinstance(f, (list, tuple)) else f


def flat(k_, v_):
    return F.normalize(to_target(k_, v_).flatten(1), dim=-1)


@torch.no_grad()
def fact_rep(f):
    if f not in _rep:
        sents = [[w2i[x] for x in R.render_with_template(f, t).w] for t in R.DEDUCTION_TEMPLATES]
        L = max(map(len, sents)); tok = torch.zeros(len(sents), L, dtype=torch.long)
        for i, t in enumerate(sents): tok[i, :len(t)] = torch.tensor(t)
        k_, v_ = model.E_ema(tok.to(DEV), torch.tensor(list(map(len, sents)), device=DEV))
        _rep[f] = F.normalize(flat(k_, v_).mean(0), dim=0)
    return _rep[f]


def verdict(fam, prem, said, toks):
    f, _ = V.parse_sentence(w(toks))
    if f is None: return "unparsed", None
    if f in said: return "repeat", f
    if f in V.valid_next(fam, prem, list(said)): return "valid", f
    if f in set(prem): return "premise", f
    return ("premature" if f in V.derivable(fam, prem) else "false"), f


acc = collections.defaultdict(lambda: collections.defaultdict(float))
coll = Collator(m_prefix=cfg["model"]["m_prefix"])
with torch.no_grad():
    for start in range(0, len(docs), a.bs):
        chunk = docs[start:start + a.bs]
        b = {k_: (v_.to(DEV) if torch.is_tensor(v_) else v_) for k_, v_ in coll(chunk).items()}
        k, v = model.E(b["idea_tok"], b["idea_len"])
        pk, pv = model.predict(k, v, b)
        samples = [model.predict_flow(k, v, b, n_steps=a.flow_steps, temperature=a.flow_temp, generator=gen)
                   for _ in range(a.n_samples)]
        # step rows = deduction ideas; build the decode batch: P + S samples per step
        rows, meta = [], []
        r = 0
        for d in chunk:
            prem = V.doc_premises(d); said = []
            for i in d["ideas"]:
                if i["step"] == "deduction" and i.get("candidates"):
                    rows.append(r); meta.append((d["family"], prem, list(said), [as_fact(c) for c in i["candidates"]]))
                if i["step"] == "deduction":
                    said.append(as_fact(i["fact"]))
                r += 1
        if not rows:
            continue
        rows_t = torch.tensor(rows, device=DEV)
        S = a.n_samples
        Ks = torch.cat([pk[rows_t][:, None]] + [s[0][rows_t][:, None] for s in samples], 1)     # [N, 1+S, r, ds]
        Vs = torch.cat([pv[rows_t][:, None]] + [s[1][rows_t][:, None] for s in samples], 1)
        N = len(rows)
        prefix = b["win"][rows_t, :b["M"]].repeat_interleave(1 + S, 0)
        out, _, _ = generate_idea(model, prefix, Ks.flatten(0, 1), Vs.flatten(0, 1), 25, is_end, "punct")
        for j, (fam, prem, said, cands) in enumerate(meta):
            n = len(cands); nb = "n=1" if n == 1 else ("n=2" if n == 2 else ("n=3" if n == 3 else "n=4+"))
            C = torch.stack([fact_rep(c) for c in cands]); cent = F.normalize(C.mean(0), dim=0)
            res = [verdict(fam, prem, said, out[j * (1 + S) + c]) for c in range(1 + S)]
            zf = flat(Ks[j, 1:], Vs[j, 1:]); cv = zf @ C.T; best = cv.max(1).values
            cos_hit = ((best > zf @ cent) & (best > a.thr)).float().mean().item() if n > 1 else (best > a.thr).float().mean().item()
            flow_res = res[1:]
            valid = [vd == "valid" for vd, _ in flow_res]
            cover = len({f for (vd, f) in flow_res if vd == "valid"}) / n
            for key in [nb, fam, "all"] + (["n>=2"] if n > 1 else []):
                q = acc[key]
                q["steps"] += 1
                q["P_valid"] += res[0][0] == "valid"
                q["P_coverage"] += (res[0][0] == "valid") / n
                q["flow_decoded_hit"] += sum(valid) / S
                q["flow_coverage"] += cover
                q["flow_cos_hit"] += cos_hit
                for K_ in KS:
                    q[f"any_valid@{K_}"] += any(valid[:K_])
                for vd, _ in flow_res:
                    q[f"flowverdict_{vd}"] += 1 / S
                q[f"Pverdict_{res[0][0]}"] += 1

order = ["n=1", "n=2", "n=3", "n=4+", "n>=2", "F1", "F2", "F3", "all"]
print(f"{a.run.rstrip('/').split('/')[-1]}  S={a.n_samples}  flow_steps={a.flow_steps}  T={a.flow_temp}  "
      f"norms={cfg['model'].get('flow_norms', 'P')}  ({int(acc['all']['steps'])} steps)")
print(f"{'':6s} {'steps':>5s} | {'P valid':>7s} | {'flow: decoded hit':>17s} {'cos hit':>7s} {'coverage':>8s} | "
      + " ".join(f"{'any@' + str(k):>7s}" for k in KS) + " | flow false / premature / unparsed")
res_json = {}
for key in order:
    q = acc.get(key)
    if not q or not q["steps"]:
        continue
    s_ = q["steps"]
    row = {m: q[m] / s_ for m in q if m != "steps"}; row["steps"] = s_
    res_json[key] = row
    print(f"{key:6s} {int(s_):5d} | {row['P_valid']:7.2f} | {row['flow_decoded_hit']:17.2f} {row['flow_cos_hit']:7.2f} "
          f"{row['flow_coverage']:8.2f} | " + " ".join(f"{row[f'any_valid@{k}']:7.2f}" for k in KS)
          + f" | {row.get('flowverdict_false', 0):.2f} / {row.get('flowverdict_premature', 0):.2f} / {row.get('flowverdict_unparsed', 0):.2f}")
json.dump({"run": a.run, "args": vars(a), "results": res_json}, open(f"{a.run}/decoded_hits.json", "w"), indent=1)

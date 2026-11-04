"""Anti-repetition test on REAL generated repetitions: does P's idea memory see "already said"?

Reads gen_eval --jsonl outputs of a model with an idea memory (P.mems), rebuilds each generated story (true
premises + the model's own sentences), runs E then P with the memories' query/key capture on, and measures:
  key_cos_repeat_same     cos(key written for the repeated idea, key written for its first occurrence), same surface
  key_cos_repeat_para     same, different surface (paraphrase of the same fact)
  key_cos_diff_facts      baseline: keys of two different deductions of the same story
  retrieval               at the position where P predicts idea j, rank of each earlier idea by its best
                          query.key score (max over slots, mean over heads and layers):
                            repeat steps: is the first occurrence ranked 1st?
                            other steps: share where the top-ranked idea is an already-said deduction
Key of idea i = the memory tokens at P's input position i + 1 (P reads the slots of idea i there); query for idea
j = the tokens at position j. A placeholder idea is appended so that the last idea's key exists.
usage: uv run python scripts/diag_repeat_keys.py <run> <gen_eval.jsonl> [--model NAME]
"""
import argparse
import collections
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ideajepa.data import verifier as V
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("jsonl")
ap.add_argument("--model", default=None, help="rows of this model name only (default: the run's name)")
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
assert getattr(model.P, "mems", None), "needs a checkpoint with an idea memory (idea_memory: delta/sum)"
for m in model.P.mems:
    m.capture = True
data = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
name = a.model or a.run.rstrip("/").split("/")[-1]
r = model.cfg["r"]


def ideas_keys_queries(ideas):
    """-> keys [n, L, H, r, dh] (key block written for idea i), queries [n, L, H, r, dh] (query block at idea i)."""
    ideas = ideas + [ideas[-1]]                                   # placeholder so the last idea's key exists
    L_ = max(map(len, ideas)); tok = torch.zeros(len(ideas), L_, dtype=torch.long)
    for i, t in enumerate(ideas): tok[i, :len(t)] = torch.tensor(t)
    k, v = model.E(tok.to(DEV), torch.tensor(list(map(len, ideas)), device=DEV))
    model.P(k, v, torch.arange(len(ideas), device=DEV)[None])
    qs = torch.stack([m.last_qk[0][0] for m in model.P.mems])     # [L, H, T, dh]
    ks = torch.stack([m.last_qk[1][0] for m in model.P.mems])
    n = len(ideas) - 1
    Lr, H, T, dh = ks.shape
    ks = ks.view(Lr, H, n + 1, r, dh).permute(2, 0, 1, 3, 4)       # [pos, L, H, r, dh]
    qs = qs.view(Lr, H, n + 1, r, dh).permute(2, 0, 1, 3, 4)
    return ks[1:n + 1], qs[:n]                                    # key of idea i at pos i+1 ; query for idea j at pos j


def key_cos(ka, kb):
    return float(F.cosine_similarity(ka.flatten(-2), kb.flatten(-2), dim=-1).mean())   # mean over layers, heads


acc = collections.defaultdict(list)
rows = [json.loads(l) for l in open(a.jsonl)]
rows = [x for x in rows if x["model"] == name] or rows
with torch.no_grad():
    for row in rows:
        d = data["docs"][row["story"]]
        prem = [i["tokens"] for i in d["ideas"] if i["step"] == "premise"]
        gen = [[w2i[x] for x in s.split() if x in w2i] for s in row["sentences"]]
        if not gen or any(len(g_) == 0 for g_ in gen):
            continue
        ks, qs = ideas_keys_queries(prem + gen)
        P0 = len(prem)
        facts = [V.parse_sentence(s)[0] for s in row["sentences"]]
        ded_facts = [f for f in facts]
        for j, (vd, f) in enumerate(zip(row["verdicts"], facts)):
            gj = P0 + j
            # retrieval: rank earlier ideas by best query.key score (max over slots), mean over layers/heads
            sc = torch.einsum("lhsd,nlhtd->nlhst", qs[gj], ks[:gj]).amax(dim=(-1, -2)).mean(dim=(1, 2))   # [gj]
            top = int(sc.argmax()) if gj > 0 else -1
            if vd == "repeat" and f is not None:
                first = next((i for i in range(j) if facts[i] == f), None)
                if first is None:
                    continue
                gi = P0 + first
                same = row["sentences"][first] == row["sentences"][j]
                acc["key_cos_repeat_same" if same else "key_cos_repeat_para"].append(key_cos(ks[gj], ks[gi]))
                rank = int((sc > sc[gi]).sum()) + 1
                acc["repeat_first_occ_rank1"].append(rank == 1)
                acc["repeat_first_occ_rank"].append(rank)
            elif vd.startswith("valid") and j > 0:
                acc["valid_top_is_said_deduction"].append(top >= P0)
            if f is not None:
                for i in range(j):
                    if facts[i] is not None and facts[i] != f:
                        acc["key_cos_diff_facts"].append(key_cos(ks[gj], ks[P0 + i]))

mean = lambda x: sum(map(float, x)) / len(x) if x else float("nan")
print(f"{name}: {len(rows)} stories")
for k_ in ("key_cos_repeat_same", "key_cos_repeat_para", "key_cos_diff_facts", "repeat_first_occ_rank1",
           "repeat_first_occ_rank", "valid_top_is_said_deduction"):
    print(f"  {k_:30s} {mean(acc[k_]):.3f}   (n={len(acc[k_])})")
json.dump({k_: mean(v_) for k_, v_ in acc.items()} | {f"n_{k_}": len(v_) for k_, v_ in acc.items()},
          open(f"{a.run}/repeat_keys.json", "w"), indent=1)

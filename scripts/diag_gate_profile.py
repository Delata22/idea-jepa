"""§30 stage A measure: gate profile per token — where does the decoder read the idea?

Teacher forcing on the deductions of N val stories (the decoder's main rows: the "so" form under surfaces=True,
else the original sentence). The residual-read gate g_t (trace["g"], the gate used to PREDICT token t) is averaged
per token class of the predicted token and per position in the sentence:
  connector  so / then / this / means / that / since
  function   the, is, of, can, by, 's, ...
  entity     person names, colours, objects, places, items (the arguments of the fact)
  relation   predicate content words (mother, opens, gets, ends, ...)
  punct      . ,
Expected after stage A: low g on connector/function (local context suffices), high on entities.
usage: uv run python scripts/diag_gate_profile.py <run> [--stories 200] [--path recon|gen]
"""
import argparse
import collections
import json
import os
import sys

import torch

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data.dataset import Collator
from ideajepa.model.objective import idea_losses
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--stories", type=int, default=200)
ap.add_argument("--path", default="recon", choices=["recon", "gen"])
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
data = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
docs = [d for d in data["docs"] if len(d["ideas"]) <= cfg["data"]["max_ideas"]][:a.stories]
CONNECTOR = {"so", "then", "this", "means", "that", "since"}
ENTITY = set(R.NAMES) | set(R.COLORS) | set(R.OBJS) | set(R.LOCS) | set(R.ITEMS) | {"key", "box"}


def cls(word):
    if word in (".", ","): return "punct"
    if word in CONNECTOR: return "connector"
    if word in ENTITY: return "entity"
    if word in R.FUNCTION: return "function"
    return "relation"


coll = Collator(m_prefix=cfg["model"]["m_prefix"], surfaces=cfg["data"].get("surfaces", False), w2i=w2i)
by_cls = collections.defaultdict(list)
by_pos = collections.defaultdict(list)
by_fam_cls = collections.defaultdict(list)
with torch.no_grad():
    for start in range(0, len(docs), a.bs):
        chunk = docs[start:start + a.bs]
        b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in coll(chunk).items()}
        _, aux = idea_losses(model, b, cfg["loss"], force_path=a.path, train=False)
        g = aux["trace"]["g"].float().cpu()
        tok = b["idea_tok"].cpu(); L = b["idea_len"].cpu(); ded = b["is_deduction"].cpu()
        fams = [d["family"] for d in chunk for _ in d["ideas"]]
        for r in range(tok.shape[0]):
            if not ded[r]:
                continue
            for s in range(int(L[r])):
                c = cls(vocab[int(tok[r, s])])
                by_cls[c].append(float(g[r, s])); by_pos[min(s, 10)].append(float(g[r, s]))
                by_fam_cls[(fams[r], c)].append(float(g[r, s]))
mean = lambda x: sum(x) / len(x) if x else float("nan")
print(f"{a.run.rstrip('/').split('/')[-1]}  path={a.path}  gate g when PREDICTING a token of each class (deductions)")
classes = ["connector", "function", "relation", "entity", "punct"]
print("  class      " + " ".join(f"{c:>10s}" for c in classes))
for fam in ("F1", "F2", "F3"):
    print(f"  {fam:10s} " + " ".join(f"{mean(by_fam_cls[(fam, c)]):10.3f}" for c in classes))
print(f"  {'all':10s} " + " ".join(f"{mean(by_cls[c]):10.3f}" for c in classes) + "   (n = "
      + ", ".join(f"{c} {len(by_cls[c])}" for c in classes) + ")")
print("  by position: " + " ".join(f"{p}:{mean(by_pos[p]):.2f}" for p in sorted(by_pos)))
json.dump({"run": a.run, "path": a.path, "by_class": {c: mean(v) for c, v in by_cls.items()},
           "by_position": {str(p): mean(v) for p, v in by_pos.items()},
           "by_family_class": {f"{f}|{c}": mean(v) for (f, c), v in by_fam_cls.items()}},
          open(f"{a.run}/gate_profile_{a.path}.json", "w"), indent=1)

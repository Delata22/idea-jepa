"""Stage A measure: mid-sentence patching of the idea — does the content come from Z or from the local prefix?

At deduction steps with n >= 2 valid candidates (A = the reference fact, B = another valid candidate), the first k
words of A's main form ("so" form 0) are forced, then the decoder completes the sentence greedily while reading
Z_B = E(main form of B) instead of Z_A. Outcome of the completed sentence (verifier parse):
  follows_B    the fact is B              (content carried by the idea)
  follows_A    the fact is A              (the prefix / local context wins)
  composition  the argument fixed by the prefix comes from A, the rest of the fact from Z_B
  other        another parsable fact, or unparsable
Reported separately for steps where B is still COMPATIBLE with the forced words (B's form starts with them) and
where it is not: "follows_B" alone is mostly decided by that compatibility, not by the model.
Controls: Z_A with A's prefix (should give A); k = 1 forces only "so" (no content in the prefix).
usage: uv run python scripts/diag_patch_midsentence.py <run> [--stories 150] [--ks 1 3 5]
"""
import argparse
import collections
import json
import os
import random
import sys

import torch

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V
from ideajepa.data.dataset import Collator
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--stories", type=int, default=150)
ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
data = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
w = lambda toks: " ".join(vocab[t] for t in toks)
is_end = torch.zeros(len(vocab), dtype=torch.bool, device=DEV); is_end[w2i["."]] = True
docs = [d for d in data["docs"] if len(d["ideas"]) <= cfg["data"]["max_ideas"]][:a.stories]
rng = random.Random(a.seed)
M = cfg["model"]["m_prefix"]
coll = Collator(m_prefix=M, surfaces=cfg["data"].get("surfaces", False), w2i=w2i)


def as_fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f)


def main_form(f):
    return [w2i[x] for x in R.render_with_template(f, 2).w]      # "so" + clause form 0


@torch.no_grad()
def encode(toks):
    t = torch.tensor([toks], device=DEV)
    return model.E(t, torch.tensor([len(toks)], device=DEV))


@torch.no_grad()
def complete(prefix, forced, K, Vv, max_len=25):
    toks = list(forced)
    for s_ in range(len(forced), max_len):
        win = torch.cat([prefix, torch.tensor([toks + [PAD]], device=DEV)], 1)
        e = model.O(win, M)
        logits, tr = model.dec(win, torch.tensor([s_ + 1], device=DEV), M, K, Vv, e)
        if toks and is_end[toks[-1]]:
            break
        lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf"); toks.append(int(lg.argmax()))
    return toks


res = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
for d in docs:
    b = coll([d])
    for r, idea in enumerate(d["ideas"]):
        cands = [as_fact(c) for c in (idea.get("candidates") or [])]
        if idea["step"] != "deduction" or len(cands) < 2:
            continue
        A = as_fact(idea["fact"]); B = rng.choice([c for c in cands if c != A])
        prefix = b["win"][r:r + 1, :M].to(DEV)
        fa, fb = main_form(A), main_form(B)
        kA, vA = encode(fa); kB, vB = encode(fb)
        for k in a.ks:
            if k >= len(fa) - 1:
                continue
            compat = fb[:k] == fa[:k]          # B's main form starts with the forced words: B is still possible
            for name, (K, Vv) in (("Z_B", (kB, vB)), ("control_Z_A", (kA, vA))):
                out = complete(prefix, fa[:k], K, Vv)
                f, _ = V.parse_sentence(w(out))
                oc = "follows_B" if f == B else ("follows_A" if f == A else "other")
                if oc == "other" and f is not None and len(f) == len(A) and f[0] == A[0] == B[0]:
                    # composition: the argument fixed by the prefix comes from A, the rest from B
                    if any(f == tuple(A[:i]) + tuple(B[i:]) for i in range(2, len(A))):
                        oc = "composition"
                for key in ((name, k, d["family"]), (name, k, "all")):
                    res[key][oc] += 1
                    res[key + ("compat" if compat else "incompat",)][oc] += 1
                if name == "Z_B" and len(examples[k]) < 3:
                    examples[k].append((w(fa[:k]), w(out), w(fa), w(fb)))

print(f"{a.run.rstrip('/').split('/')[-1]}  (first k words of A forced, then the decoder reads Z_B)")
out_json = {}
OCS = ("follows_B", "follows_A", "composition", "other")
for name in ("Z_B", "control_Z_A"):
    for k in a.ks:
        for fam in ("F1", "F2", "F3", "all"):
            for sub in (None, "compat", "incompat"):
                key = (name, k, fam) + ((sub,) if sub else ())
                c = res.get(key)
                if not c:
                    continue
                tot = sum(c.values())
                row = {oc: c[oc] / tot for oc in OCS}
                out_json[f"{name}|k={k}|{fam}" + (f"|{sub}" if sub else "")] = {**row, "n": tot}
                if name == "Z_B" and (fam == "all" or sub is None) or (name != "Z_B" and fam == "all" and sub is None):
                    print(f"  {name:12s} k={k} {fam:3s} {sub or 'all':8s} n={tot:4d}  " + "  ".join(f"{oc} {row[oc]:.2f}" for oc in OCS))
print("  (compat = B's main form starts with the forced words, so B is still reachable; incompat = it is not:\n"
      "   'composition' = the argument fixed by the prefix comes from A and the rest of the fact from Z_B)")
for k, ex in examples.items():
    for forced, out, fa, fb in ex[:2]:
        print(f"  e.g. k={k}: forced «{forced}» → «{out}»   (A: «{fa}», B: «{fb}»)")
json.dump({"run": a.run, "results": out_json}, open(f"{a.run}/patch_midsentence.json", "w"), indent=1)

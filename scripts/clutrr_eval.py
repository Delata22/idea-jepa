"""CLUTRR evaluation: free generation of as many deductions as the target has (1 for the "answer" variant,
the proof length for "chain"), after the story + question (goal_cond models: the question is the goal G). Measures per k:
answer accuracy (the LAST generated sentence parses to the query fact (A, target, B): "so B is the target of A ."),
relation-only accuracy (right relation word, any entities), out-of-format rate; chain variant: precision of the intermediate
generated facts (fraction in the proof's derived facts). 95 % bootstrap intervals (1000 resamples) per k and per band
(k 2-3, 4-6, 7-10).
usage: python scripts/clutrr_eval.py <run> [<run> ...] [--data data/clutrr/train23_chain] [--ks 2 3 ... 10] [--json out.json]
"""
import argparse
import collections
import json
import os
import random
import re
import sys

import torch

sys.path.insert(0, "src")
from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--data", default=None)
ap.add_argument("--ks", type=int, nargs="+", default=list(range(2, 11)))
ap.add_argument("--split", default=None, help="val instead of the test files")
ap.add_argument("--json", default=None)
ap.add_argument("--show", type=int, default=4)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
BANDS = {"k2-3": (2, 3), "k4-6": (4, 6), "k7-10": (7, 10)}


def parse(words):
    s = " ".join(words)
    m = re.fullmatch(r"so (ent_\d+) is the ([a-z\-]+) of (ent_\d+) \.", s)
    return (m.group(3), m.group(2), m.group(1)) if m else None      # (X, r, Y): "Y is the r of X"


def boot(xs, n=1000, seed=0):
    rng = random.Random(seed)
    if not xs:
        return (float("nan"), float("nan"))
    ms = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    path = a.data or cfg["data"]["path"]
    is_im = isinstance(model, IdeaJEPA)
    goal_cond = is_im and model.cfg.get("goal_cond", False)
    M = cfg["model"].get("m_prefix", 3)
    files = [("val", "val")] if a.split == "val" else [(f"k{k}", f"test_k{k}") for k in a.ks]
    per = collections.defaultdict(list)
    counters = collections.defaultdict(collections.Counter)
    ex = []
    for tag, fname in files:
        data = torch.load(f"{path}/{fname}.pt", weights_only=False)
        docs, vocab = data["docs"], data["vocab"]
        w2i = {x: i for i, x in enumerate(vocab)}
        FORCE, DOT = w2i["so"], w2i["."]
        w = lambda t: [vocab[x] for x in t]

        def enc_tokens(ideas):
            L = max(len(t) for t in ideas)
            tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
            for i, t in enumerate(ideas):
                tok[i, :len(t)] = torch.tensor(t)
            return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)

        def gen_im(ctx, goal, n_ded, max_len=24):
            story = [list(t) for t in ctx]
            n0 = len(story)
            for _ in range(n_ded):
                tok, lens = enc_tokens(story)
                k, v = model.E(tok, lens)
                n = len(story)
                di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
                batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                         "idea_pos": torch.tensor([n], device=DEV),
                         "idea_role": (torch.arange(n, device=DEV) >= n0).long()}
                if goal is not None:
                    batch["goal_tok"] = torch.tensor([goal], device=DEV); batch["goal_len"] = torch.tensor([len(goal)], device=DEV)
                    batch["goal_present"] = torch.ones(1, device=DEV)
                pk, pv = model.predict(k, v, batch)
                hist = [BOS] + [t for idea in story for t in idea]; p = hist[-M:]
                pre = torch.full((1, M), PAD, dtype=torch.long); pre[0, M - len(p):] = torch.tensor(p); pre = pre.to(DEV)
                toks = [FORCE]
                for s_ in range(1, max_len):
                    win = torch.cat([pre, torch.tensor([toks + [PAD]], device=DEV)], 1)
                    logits, tr = model.dec(win, torch.tensor([s_ + 1], device=DEV), M, pk, pv, model.O(win, M))
                    if toks[-1] == DOT or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)):
                        break
                    lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf")
                    toks.append(int(lg.argmax()))
                story.append(toks)
            return story[n0:]

        def gen_gpt(ctx, n_ded, max_len=24):
            seq = [BOS] + [t for p in ctx for t in p]
            outs = []
            for _ in range(n_ded):
                cur = [FORCE]; seq = seq + [FORCE]
                for _ in range(max_len):
                    x = torch.tensor([seq[-model.pos.num_embeddings:]], device=DEV)
                    lg = model.run(x)[:, -1].float(); lg[:, [PAD, BOS, EOS]] = float("-inf")
                    nxt = int(lg.argmax()); seq.append(nxt); cur.append(nxt)
                    if nxt == DOT:
                        break
                outs.append(cur)
            return outs

        with torch.no_grad():
            for d in docs:
                ctx = [i["tokens"] for i in d["ideas"] if i["step"] == "premise"]
                q = [i["tokens"] for i in d["ideas"] if i["step"] == "question"][0]
                n_ded = sum(i["step"] == "deduction" for i in d["ideas"])
                if is_im:
                    outs = gen_im(ctx, q, n_ded) if goal_cond else gen_im(ctx + [q], None, n_ded)
                else:
                    outs = gen_gpt(ctx + [q], n_ded)
                facts = [parse(w(o)) for o in outs]
                A, B = d["query"]
                final = facts[-1]
                ok = final == (A, d["answer"], B)
                rel_ok = final is not None and final[1] == d["answer"]
                proof = set(map(tuple, d["proof_facts"]))
                inter = facts[:-1]
                c = counters[tag]
                c["n"] += 1; c["correct"] += ok; c["rel_correct"] += rel_ok; c["oof"] += final is None
                c["inter"] += len(inter); c["inter_valid"] += sum(f in proof for f in inter if f is not None)
                per[tag].append(ok)
                if len(ex) < a.show and not ok:
                    ex.append({"k": tag, "question": " ".join(w(q)), "target": f"so {B} is the {d['answer']} of {A} .",
                               "generated": " | ".join(" ".join(w(o)) for o in outs)})
    res = {}
    for tag, c in counters.items():
        lo, hi = boot(per[tag])
        res[tag] = {**dict(c), "acc": c["correct"] / c["n"], "ci": (lo, hi), "rel_acc": c["rel_correct"] / c["n"],
                    "oof": c["oof"] / c["n"], "chain_precision": c["inter_valid"] / c["inter"] if c["inter"] else None}
    for band, (lo_k, hi_k) in BANDS.items():
        xs = [x for tag, v in per.items() if tag.startswith("k") and lo_k <= int(tag[1:]) <= hi_k for x in v]
        if xs:
            res[band] = {"n": len(xs), "acc": sum(xs) / len(xs), "ci": boot(xs)}
    print(f"{run} [{path}]: " + "  ".join(f"{t} {r['acc']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] (n={r['n']})"
                                         for t, r in sorted(res.items(), key=lambda x: (len(x[0]), x[0]))), flush=True)
    for e in ex:
        print(f"    ✗ {e['k']} {e['question']} → « {e['generated']} » (attendu « {e['target']} »)")
    del model
    return {"run": run, "data": path, "results": res, "examples": ex}


out = [run_one(r) for r in a.runs]
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)

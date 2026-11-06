"""Does the stage-C loop settle or oscillate? Agreement between the decoded sentence D(Ẑ) and the nearest
neighbour of Ẑ among the encodings of the candidate facts, in the reference context. No training.

Reference context (premises + reference deductions < j), P's prediction Ẑ, greedy decoding "so" forced (as gen_eval).
Candidates = the valid next deductions at step j (V.valid_next). Encoding of a fact: E_ema of its "so" form 0 sentence,
values per-slot normalised, flattened, L2 (diag_pred_cos space; --space flow = [k ; v/|v|]). NN = argmax cos(Ẑ, fact).
agree = the decoded sentence parses to the NN fact. Reported overall, per family, and on the steps with >= 2 candidates
(with a single candidate, agreement = "decoded valid"); also decoded_valid and nn_is_ref (NN = reference fact).
Target: >= 90 % the loop settles, <= 70 % it oscillates.
usage: uv run python scripts/diag_nn_agree.py <run> [<run> ...] [--n 30] [--space values|flow] [--summary out.md]
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
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--n", type=int, default=30, help="stories per family")
ap.add_argument("--space", default="values", choices=["values", "flow"])
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
FORCE = w2i["so"]
w = lambda toks: " ".join(vocab[t] for t in toks)
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)


def stories():
    per = collections.Counter()
    for doc in docs:
        fam = doc.get("family")
        if per[fam] >= a.n:
            continue
        prem = [i["tokens"] for i in doc["ideas"] if i["step"] == "premise"]
        ded = [i for i in doc["ideas"] if i["step"] == "deduction"]
        if len(ded) < 2:
            continue
        per[fam] += 1
        yield fam, prem, ded, V.doc_premises(doc)


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    M = cfg["model"]["m_prefix"]
    endm = torch.zeros(len(vocab), dtype=torch.bool, device=DEV); endm[w2i["."]] = True

    def enc_tokens(ideas):
        L = max(len(t) for t in ideas)
        tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
        for i, t in enumerate(ideas):
            tok[i, :len(t)] = torch.tensor(t)
        return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)

    def flat(k, v):
        vn = F.normalize(v.float(), dim=-1)
        x = vn if a.space == "values" else torch.cat([k.float(), vn], -1)
        return F.normalize(x.flatten(1), dim=-1)

    zc = {}

    def zfacts(facts):
        todo = [f for f in facts if f not in zc]
        if todo:
            t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(f, 2).w] for f in todo])
            for f, z in zip(todo, flat(*model.E_ema(t_, l_))):
                zc[f] = z
        return torch.stack([zc[f] for f in facts])

    def predict(story, n_prem):
        tok, lens = enc_tokens(story)
        k, v = model.E(tok, lens)
        n = len(story)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                 "idea_pos": torch.tensor([n], device=DEV), "idea_role": (torch.arange(n, device=DEV) >= n_prem).long()}
        return model.predict(k, v, batch)

    def decode(prefix, pk, pv, max_len=48):
        toks = [FORCE]
        for s_ in range(1, max_len):
            win = torch.cat([prefix, torch.tensor([toks + [PAD]], device=DEV)], 1)
            e = model.O(win, M)
            logits, tr = model.dec(win, torch.tensor([s_ + 1], device=DEV), M, pk, pv, e)
            if endm[toks[-1]] or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)):
                break
            lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf")
            toks.append(int(lg.argmax()))
        return toks

    cnt = collections.defaultdict(collections.Counter)
    ex = []
    with torch.no_grad():
        for fam, prem, ded, pf in stories():
            ref = [as_fact(i["fact"]) for i in ded]
            for j in range(len(ded)):
                ctx = [list(p) for p in prem] + [list(i["tokens"]) for i in ded[:j]]
                cands = sorted(V.valid_next(fam, pf, ref[:j]) - set(ref[:j]), key=str)
                if not cands:
                    continue
                hist = [BOS] + [t for idea in ctx for t in idea]; p = hist[-M:]
                pre = torch.full((1, M), PAD, dtype=torch.long); pre[0, M - len(p):] = torch.tensor(p)
                pk, pv = predict(ctx, len(prem))
                out = decode(pre.to(DEV), pk, pv)
                f_out = V.parse_sentence(w(out))[0]
                zhat = flat(pk, pv)[0]
                sims = zfacts(cands) @ zhat
                nn = cands[int(sims.argmax())]
                agree = f_out == nn
                for key in ((fam, "all"), ("all", "all")) + (((fam, "multi"), ("all", "multi")) if len(cands) > 1 else ()):
                    c = cnt[key]
                    c["n"] += 1; c["agree"] += agree; c["decoded_valid"] += f_out in cands; c["nn_is_ref"] += nn == ref[j]
                    c["decoded_is_ref"] += f_out == ref[j]
                if not agree and len(cands) > 1 and len(ex) < 6:
                    ex.append({"fam": fam, "decoded": w(out), "nn": " ".join(R.render_with_template(nn, 2).w),
                               "n_cands": len(cands), "decoded_valid": f_out in cands})
    res = {f"{k[0]}|{k[1]}": dict(c) for k, c in cnt.items()}
    json.dump({"run": run, "space": a.space, "results": res, "examples": ex},
              open(f"{run}/nn_agree_{a.space}.json", "w"), indent=1)
    c = res["all|all"]; cm = res.get("all|multi", {"n": 1})
    print(f"{run}: agree {c['agree'] / c['n']:.3f} (steps with >= 2 candidates: {cm.get('agree', 0) / cm['n']:.3f}, "
          f"n={cm['n']})", flush=True)
    del model
    return res


allr = {r: run_one(r) for r in a.runs}
if a.summary:
    keys = ["all|all", "all|multi", "F1|all", "F1|multi", "F2|all", "F2|multi", "F3|all", "F3|multi"]
    lines = ["| run | group | n | agree D(Ẑ) = NN | decoded valid | NN = reference | decoded = reference |", "|---|---|---|---|---|---|---|"]
    for r, res in allr.items():
        for k in keys:
            c = res.get(k)
            if c:
                f = lambda x: f"{c.get(x, 0) / c['n']:.3f}"
                lines.append(f"| {r.split('/')[-1]} | {k} | {c['n']} | {f('agree')} | {f('decoded_valid')} | {f('nn_is_ref')} | {f('decoded_is_ref')} |")
    open(a.summary, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

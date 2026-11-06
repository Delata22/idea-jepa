"""ORACLE OF THE QUERY AT INFERENCE, no training. Does addressing drive content?

F1 only. "Useful premise" of an object o (the generator has no antecedent in F1): the LAST premise that fixed o's
location, i.e. its initial "the o is in L", or the last "X goes to L" of the person X carrying o at that time (taking
or dropping o does not move it). Its location is o's final location.
At a deduction step, P's memory query for the predicted idea (its r query tokens, block n) is replaced by the r keys
written for the useful premise p (block p + 1) of a target object, in every memory layer (variant a, default) or in
one layer (--layer, variant b). Conditions, on the same steps:
  normal    no intervention
  oracle    target = the object of the reference deduction (reference context) / a valid unsaid object (free gen.)
  control   target = ANOTHER object whose deduction is still valid and unsaid at that step
Contexts: (b) reference context (premises + reference deductions < j), (c) free generation (the model's own
deductions; under oracle / control the chosen sentence enters the context, as in generation).
Per step: verdict (valid / repeat / premature / false ...); obj_target = the decoded fact is about the target object;
loc_ok = its location is the true final location of the object it names; chimera = wrong fact whose object and
location both come from valid candidates of the step but from different ones.
usage: uv run python scripts/diag_query_oracle.py <run> [<run> ...] [--n 30] [--layer L] [--summary out.md]
"""
import argparse
import collections
import json
import os
import random
import statistics as st
import sys

import torch

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--n", type=int, default=30, help="F1 stories")
ap.add_argument("--layer", type=int, default=None, help="variant b: intervene in this memory layer only")
ap.add_argument("--mix", type=float, default=1.0, help="soft variant: query = normalise((1 - mix) q + mix k_oracle)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
FORCE = w2i["so"]
w = lambda toks: " ".join(vocab[t] for t in toks)
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)
is_end = torch.zeros(len(vocab), dtype=torch.bool); is_end[w2i["."]] = True
SUFFIX = (f"_L{a.layer}" if a.layer is not None else "") + (f"_mix{a.mix:g}" if a.mix != 1.0 else "")


useful_premises = R.f1_useful_premises


def stories():
    n = 0
    for doc in docs:
        if doc.get("family") != "F1" or n >= a.n:
            continue
        prem = [i for i in doc["ideas"] if i["step"] == "premise"]
        ded = [i for i in doc["ideas"] if i["step"] == "deduction"]
        if len(ded) < 2:
            continue
        n += 1
        pf = [as_fact(i["fact"]) for i in prem]
        up = useful_premises(pf)
        final = R.final_state_f1(pf)
        assert all(pf[up[o]][-1] == final[o] for o in final), "useful premise must carry the final location"
        yield [i["tokens"] for i in prem], ded, pf, up, final


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    M, r = cfg["model"]["m_prefix"], model.P.r
    mems = list(model.P.mems)
    layers = [a.layer] if a.layer is not None else list(range(len(mems)))
    endm = is_end.to(DEV)
    rng = random.Random(a.seed)

    def enc_tokens(ideas):
        L = max(len(t) for t in ideas)
        tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
        for i, t in enumerate(ideas):
            tok[i, :len(t)] = torch.tensor(t)
        return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)

    def prefix_of(story):
        hist = [BOS] + [t for idea in story for t in idea]; p = hist[-M:]
        pre = torch.full((1, M), PAD, dtype=torch.long); pre[0, M - len(p):] = torch.tensor(p)
        return pre.to(DEV)

    def predict(story, n_prem, target_premise=None, cond=None, probe=False):
        """probe: no intervention (mix 0), only the read measure against target_premise's value."""
        tok, lens = enc_tokens(story)
        k, v = model.E(tok, lens)
        n = len(story)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                 "idea_pos": torch.tensor([n], device=DEV), "idea_role": (torch.arange(n, device=DEV) >= n_prem).long()}
        if target_premise is not None:   # query block n := key block (premise index + 1)
            for li in layers:
                mems[li].q_override = [(0, n * r, (target_premise + 1) * r, r, 0.0 if probe else a.mix)]
        try:
            out = model.predict(k, v, batch)
            if target_premise is not None:
                if not probe:
                    COS.append(sum(mems[li].q_override_cos for li in layers) / len(layers))
                READ[cond].append(sum(mems[li].q_override_readcos for li in layers) / len(layers))
            return out
        finally:
            for m in mems:
                m.q_override = None

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

    COS = []                                                 # cos(normal query, oracle key), intervened layers
    READ = collections.defaultdict(list)                     # condition -> cos(read, value of the target premise)
    cnt = collections.defaultdict(collections.Counter)      # (context, condition) -> counts
    ex = collections.defaultdict(list)

    def score(key, out, said_txt, pf, final, target, cands, story_ref):
        verdict = V.judge("F1", pf, said_txt + [w(out)])[-1]["verdict"]
        verdict = "valid" if verdict.startswith("valid") else verdict
        f, _ = V.parse_sentence(w(out))
        c = cnt[key]
        c["n"] += 1; c[verdict] += 1
        if f is not None and f[0] == "objin":
            c["obj_target"] += f[1] == target
            c["loc_ok"] += final.get(f[1]) == f[2]
            c["obj_target_and_loc_ok"] += f[1] == target and final.get(f[1]) == f[2]
            others = {final[o] for o in cands if o != f[1]}
            c["chimera"] += verdict != "valid" and f[1] in cands and f[2] in others and final.get(f[1]) != f[2]
        if len(ex[key]) < 3 and (verdict != "valid" or key[1] != "normal"):
            ex[key].append({"context_said": said_txt[-2:], "target": target, "out": w(out), "verdict": verdict})

    with torch.no_grad():
        for prem, ded, pf, up, final in stories():
            ref_facts = [as_fact(i["fact"]) for i in ded]
            ref_txt = [w(i["tokens"]) for i in ded]
            # (b) reference context
            for j in range(len(ded)):
                ctx = [list(p) for p in prem] + [list(i["tokens"]) for i in ded[:j]]
                said = ref_facts[:j]
                cands = sorted(o for o, l in final.items() if ("objin", o, l) not in said)
                tgt = ref_facts[j][1]
                others = [o for o in cands if o != tgt]
                pre = prefix_of(ctx)
                score(("ref", "normal"), decode(pre, *predict(ctx, len(prem), up[tgt], "ref|normal", probe=True)),
                      ref_txt[:j], pf, final, tgt, cands, ctx)
                score(("ref", "oracle"), decode(pre, *predict(ctx, len(prem), up[tgt], "ref|oracle")), ref_txt[:j], pf, final, tgt, cands, ctx)
                if others:
                    o2 = rng.choice(others)
                    score(("ref", "control"), decode(pre, *predict(ctx, len(prem), up[o2], "ref|control")), ref_txt[:j], pf, final, o2, cands, ctx)
            # (c) free generation, one trajectory per condition
            for cond in ("normal", "oracle", "control"):
                own = []
                for j in range(len(ded)):
                    story = [list(p) for p in prem] + own
                    said = [f for f in (V.parse_sentence(w(x))[0] for x in own) if f is not None]
                    cands = sorted(o for o, l in final.items() if ("objin", o, l) not in said)
                    if not cands:
                        break
                    valid_obj = sorted(f[1] for f in V.valid_next("F1", pf, said))
                    tgt = rng.choice(valid_obj) if valid_obj else rng.choice(cands)
                    if cond == "control":
                        others = [o for o in cands if o != tgt]
                        if not others:
                            break
                        tgt = rng.choice(others)
                    out = decode(prefix_of(story), *predict(story, len(prem), up[tgt], "free|" + cond, probe=cond == "normal"))
                    score(("free", cond), out, [w(x) for x in own], pf, final, tgt, cands, story)
                    own.append(out)
    res = {f"{k[0]}|{k[1]}": dict(c) for k, c in cnt.items()}
    json.dump({"run": run, "n": a.n, "layers": layers, "mix": a.mix, "cos_query_oracle_key": st.mean(COS) if COS else None,
               "cos_read_target_value": {k: st.mean(v) for k, v in READ.items() if v},
               "results": res,
               "examples": {f"{k[0]}|{k[1]}": v for k, v in ex.items()}},
              open(f"{run}/query_oracle{SUFFIX}.json", "w"), indent=1)
    line = "  ".join(f"{k}: valid {c.get('valid', 0) / c['n']:.3f} obj {c.get('obj_target', 0) / c['n']:.3f}"
                     for k, c in sorted(res.items()))
    print(f"{run}: cos(q, k_oracle) {st.mean(COS) if COS else float('nan'):.3f}  read→target value "
          + " ".join(f"{k} {st.mean(v):.3f}" for k, v in sorted(READ.items()) if v) + f"  {line}", flush=True)
    for k, v in READ.items():
        if v and f"{k}" in res:
            res[k]["cos_read_target_value"] = st.mean(v)
    del model
    return res


all_res = {r: run_one(r) for r in a.runs}
if a.summary:
    keys = sorted({k for res in all_res.values() for k in res})
    lines = ["| run | context/condition | n | valid | obj = target | loc ok | obj = target & loc ok | chimera | repeat | false | cos(read, target v) |",
             "|" + "---|" * 11]
    for r, res in all_res.items():
        for k in keys:
            c = res.get(k)
            if not c:
                continue
            f = lambda x: f"{c.get(x, 0) / c['n']:.3f}"
            lines.append(f"| {r.split('/')[-1]} | {k} | {c['n']} | {f('valid')} | {f('obj_target')} | {f('loc_ok')} | "
                         f"{f('obj_target_and_loc_ok')} | "
                         f"{f('chimera')} | {f('repeat')} | {f('false')} | {c.get('cos_read_target_value', float('nan')):.3f} |")
    open(a.summary, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

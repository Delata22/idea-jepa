"""Provoked repetition: does the anti-repetition of P depend on the surface of what was already said?

gen_eval stories and decoding (greedy, "so" forced). Reference context = premises + reference deductions < k (k >= 1).
One already-said deduction s (random among those < k) is re-inserted at the end of the context:
  (i)   same     the same sentence as in the story
  (ii)  other    another deduction template of the same fact (different sentence)
  (iii) none     no insertion (control)
At the next step: P's prediction is decoded and judged against the reference deductions said (+ the insertion).
  rep_inserted   the decoded fact is s                    repeat   verifier verdict "repeat" (any said fact)
  valid          verifier verdict valid
  nn_s / nn_said nearest neighbour of P's prediction (diag_pred_cos space, "so" form 0 encodings) among
                 valid-now ∪ said is s / is any said fact;     cos_s  cos(prediction, s)
Memory keys (normalised DeltaMemory keys, per layer, mean over heads and slots): cos between the key written for s
at its first occurrence and at the re-insertion ((i) vs (ii)); reference: cos between the keys of s and of another
said deduction of a different fact (control context).
usage: uv run python scripts/diag_provoked_repeat.py <run> [<run> ...] [--n 30] [--summary runs/v2/provoked_repeat_summary.md]
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
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--n", type=int, default=30, help="stories per family (gen_eval --n)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--summary", default=None)
ap.add_argument("--summary_only", action="store_true")
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
w = lambda toks: " ".join(vocab[t] for t in toks)
FORCE = w2i["so"]
CONDS = ["same", "other", "none"]
MEAS = ["rep_inserted", "repeat", "valid", "nn_s", "nn_said", "cos_s", "cos_valid_now"]


def as_fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f)


def rank(k):
    return "1" if k == 1 else ("2" if k == 2 else "3+")


def stories():
    per_fam = collections.Counter()
    for doc in docs:
        fam = doc.get("family")
        if per_fam[fam] >= a.n:
            continue
        prem = [i["tokens"] for i in doc["ideas"] if i.get("depth", -1) < 0]
        ded = [i for i in doc["ideas"] if i.get("depth", -1) >= 0]
        if len(ded) < 2:
            continue
        per_fam[fam] += 1
        yield fam, prem, ded, V.doc_premises(doc)


def run_one(run):
    rng = random.Random(a.seed)
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    M, r = cfg["model"]["m_prefix"], model.cfg["r"]
    mems = getattr(model.P, "mems", None) or []
    for m_ in mems:
        m_.capture = True
    is_end = torch.zeros(len(vocab), dtype=torch.bool, device=DEV); is_end[w2i["."]] = True

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

    def decode(prefix, pk, pv, max_len=48):
        toks = [FORCE]
        for s_ in range(1, max_len):
            win = torch.cat([prefix, torch.tensor([toks + [PAD]], device=DEV)], 1)
            e = model.O(win, M)
            logits, tr = model.dec(win, torch.tensor([s_ + 1], device=DEV), M, pk, pv, e)
            if is_end[toks[-1]] or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)):
                break
            lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf")
            toks.append(int(lg.argmax()))
        return toks

    def predict(story, n_prem):
        tok, lens = enc_tokens(story)
        k, v = model.E(tok, lens)
        n = len(story)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                 "idea_pos": torch.tensor([n], device=DEV), "idea_role": (torch.arange(n, device=DEV) >= n_prem).long()}
        pk, pv = model.predict(k, v, batch)
        keys = torch.stack([m_.last_qk[1][0] for m_ in mems]) if mems else None   # [L, H, T, dh]
        return pk, pv, keys

    def key_cos(keys, i, j):
        """per-layer cos between the keys written for ideas i and j (write position = idea index + 1)."""
        ki = keys[:, :, (i + 1) * r:(i + 2) * r]; kj = keys[:, :, (j + 1) * r:(j + 2) * r]
        return (F.normalize(ki, dim=-1) * F.normalize(kj, dim=-1)).sum(-1).mean((1, 2)).tolist()

    zcache = {}

    def zf(f):
        if f not in zcache:
            t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(f, 2).w]])
            _, v_ = model.E_ema(t_, l_)
            zcache[f] = F.normalize(F.normalize(v_[0].float(), dim=-1).flatten(), dim=0)
        return zcache[f]

    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    with torch.no_grad():
        for fam, prem, ded, prem_facts in stories():
            ref_txt = [w(i["tokens"]) for i in ded]
            facts = [as_fact(i["fact"]) for i in ded]
            for k in range(1, len(ded)):
                si = rng.randrange(k); s = facts[si]
                others = [t for t in R.DEDUCTION_TEMPLATES if R.render_with_template(s, t).w != ref_txt[si].split()]
                if not others:
                    continue
                other_tok = [w2i[x] for x in R.render_with_template(s, rng.choice(others)).w]
                base_ctx = [list(p) for p in prem] + [list(i["tokens"]) for i in ded[:k]]
                said = facts[:k]
                now = [f for f in sorted(V.valid_next(fam, prem_facts, said), key=str) if f not in said]
                keys_ = [(fam, rank(k)), (fam, "all"), ("all", rank(k)), ("all", "all")]
                for cond in CONDS:
                    ins = {"same": list(ded[si]["tokens"]), "other": other_tok, "none": None}[cond]
                    ctx = base_ctx + ([ins] if ins is not None else [])
                    pk, pv, keys = predict(ctx, len(prem))
                    out = decode(prefix_of(ctx), pk, pv)
                    txt_said = ref_txt[:k] + ([w(ins)] if ins is not None else [])
                    verdict = V.judge(fam, prem_facts, txt_said + [w(out)])[-1]["verdict"]
                    f_out, _ = V.parse_sentence(w(out))
                    p = F.normalize(F.normalize(pv[0].float(), dim=-1).flatten(), dim=0)
                    cs = {f: float(zf(f) @ p) for f in list(dict.fromkeys(said + now))}
                    nn = max(cs, key=cs.get)
                    row = {"rep_inserted": float(f_out == s), "repeat": float(verdict == "repeat"),
                           "valid": float(verdict.startswith("valid")), "nn_s": float(nn == s),
                           "nn_said": float(nn in said), "cos_s": cs[s]}
                    if now:
                        row["cos_valid_now"] = max(cs[f] for f in now)
                    if keys is not None:
                        n_prem = len(prem)
                        if cond != "none":
                            row["key_cos_reinsert"] = key_cos(keys, n_prem + si, len(ctx) - 1)
                        else:
                            diff = [j for j in range(k) if facts[j] != s]
                            if diff:
                                row["key_cos_other_fact"] = key_cos(keys, n_prem + si, n_prem + rng.choice(diff))
                    for kk in keys_:
                        acc[(cond,) + kk]["N"].append(1)
                        for m_, v_ in row.items():
                            acc[(cond,) + kk][m_].append(v_)

    def mean(xs):
        if isinstance(xs[0], list):
            return [sum(c) / len(c) for c in zip(*xs)]
        return sum(xs) / len(xs)
    res = {"|".join(key): {m_: (mean(v_) if m_ != "N" else len(v_)) for m_, v_ in d_.items()} for key, d_ in acc.items()}
    json.dump({"run": run, "n_per_family": a.n, "results": res}, open(f"{run}/provoked_repeat.json", "w"), indent=1)
    g = lambda c, m_: res[f"{c}|all|all"].get(m_, float("nan"))
    print(f"{run}: rep_inserted same/other/none {g('same', 'rep_inserted'):.3f}/{g('other', 'rep_inserted'):.3f}/"
          f"{g('none', 'rep_inserted'):.3f}  valid {g('same', 'valid'):.3f}/{g('other', 'valid'):.3f}/{g('none', 'valid'):.3f}",
          flush=True)
    return res


results = {}
for run in a.runs:
    results[run] = (json.load(open(f"{run}/provoked_repeat.json"))["results"] if a.summary_only else run_one(run))

if a.summary:
    groups = collections.defaultdict(list)
    for run, res in results.items():
        name = Path(run).name
        groups[name.rsplit("_s", 1)[0] if "_s" in name and name[-1].isdigit() else name].append(res)

    def fmt(xs, pct):
        if not xs:
            return "–"
        f = (lambda x: f"{100 * x:.0f}") if pct else (lambda x: f"{x:.3f}")
        return f"{f(st.mean(xs))} ± {f(st.stdev(xs))}" if len(xs) >= 2 else f(xs[0])

    L = [f"# Provoked repetition ({a.n} stories per family, reference context, greedy \"so\")", "",
         "A deduction s already said is re-inserted at the end of the context: (same) same sentence, (other) another "
         "template of the same fact, (none) no insertion. Measured at the next step. Rates in %, mean ± std over seeds.", ""]
    for g, rs in groups.items():
        L += [f"## {g} (n={len(rs)})", ""]
        for fam in ("all", "F1", "F2", "F3"):
            for rk in ("all", "1", "2", "3+"):
                key = lambda c: f"{c}|{fam}|{rk}"
                if not any(key("same") in r_ for r_ in rs):
                    continue
                L += [f"### {fam}, rank {rk}", "", "| measure | same | other | none |", "|---|---|---|---|"]
                for m_ in ["N"] + MEAS:
                    pct = m_ not in ("cos_s", "cos_valid_now", "N")
                    cells = []
                    for c in CONDS:
                        xs = [r_[key(c)][m_] for r_ in rs if key(c) in r_ and m_ in r_[key(c)]]
                        cells.append(fmt(xs, pct) if m_ != "N" else (str(int(st.mean(xs))) if xs else "–"))
                    L.append(f"| {m_} | " + " | ".join(cells) + " |")
                if fam == "all":
                    nl = max((len(r_[key("same")].get("key_cos_reinsert", [])) for r_ in rs if key("same") in r_), default=0)
                    for li in range(nl):
                        cells = []
                        for c, m_ in (("same", "key_cos_reinsert"), ("other", "key_cos_reinsert"), ("none", "key_cos_other_fact")):
                            xs = [r_[key(c)][m_][li] for r_ in rs if key(c) in r_ and m_ in r_[key(c)]]
                            cells.append(fmt(xs, False))
                        L.append(f"| memory key cos, layer {li} (reinsert vs 1st write; none: other fact) | " + " | ".join(cells) + " |")
                L.append("")
    Path(a.summary).write_text("\n".join(L) + "\n")
    print(f"-> {a.summary}")

"""Where are free-generation deductions lost: the decoder, P's prediction, or the context drift of free generation?

Same stories and decoding as gen_eval.py (first n stories per family with >= 2 deductions, premises then deductions,
m-token prefix from the history, greedy, "so" forced, stop at "." / stop head). Three conditions per deduction step j:
  (a) decoder alone   context = premises + reference deductions < j; idea = E_ema(reference fact j, "so" form 0);
                      success = the decoded sentence parses to that same fact
  (b) P, teacher forcing   same reference context; idea = P's prediction; verdict of the verifier given the
                      reference deductions already said (valid / premature / repeat / premise / false / unparsed)
  (c) free generation context = premises + the model's OWN previous deductions (= gen_eval); same verdicts
Idea side (diag_pred_cos space: E_ema values of the "so" form 0 sentence, per-slot normalised, flattened, L2), for P's
prediction BEFORE decoding, (b) in the reference context and (d) in the free-generation context of (c). Pools given
the facts S already said in that context (reference deductions for (b), the model's own parsed sentences for (d)):
  valid_now   V.valid_next(premises, S) minus S          premature  entailed, not valid now, not said, not a premise
  said        S                                          premise    the story's premises
  invalid     not entailed: one argument of a valid-now fact replaced by a value seen in the same slot of the same
              predicate in the story (diag_pred_cos pools, built around every valid-now fact)
  margin = cos best valid_now - cos max invalid; nn_* = nearest neighbour among the five pools
Split by family and by rank of the deduction in the chain (1st, 2nd, 3rd+).
usage: uv run python scripts/diag_decode_verify.py <run> [<run> ...] [--n 30] [--summary runs/v3/decode_verify_summary.md]
       (--n = stories per family; --summary_only rebuilds the table from the <run>/decode_verify.json files)
"""
import argparse
import collections
import json
import os
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
ap.add_argument("--summary", default=None)
ap.add_argument("--summary_only", action="store_true")
ap.add_argument("--fit_split", type=int, default=None, help="keep only the stories whose split version has at most "
                "this many ideas (same subset for as-is / --strip / --split); suffix _fitN")
ap.add_argument("--split", action="store_true", help="evaluation option: a restated reference sentence \"since A , B .\" "
                "is given to the model as TWO ideas, \"so A .\" (form 0) then \"so B .\"; outputs to <run>/<name>_split.json")
ap.add_argument("--gold_rel", action="store_true", help="rel_cond models, (b): the context ideas get their TRAINING "
                "relation labels (split_compound, the run's data.rel_source) instead of the head's predictions; suffix _goldrel")
ap.add_argument("--strip", action="store_true", help="evaluation option: a restated reference sentence \"since A , B .\" "
                "is given to the model as \"so B .\" (B in the clause form written); outputs to <run>/<name>_strip.json")
ap.add_argument("--canon", action="store_true", help="context in the canonical surfaces of the stage-A training stream "
                "(premises: premise form 0, reference deductions: \"so\" + form 0) instead of the story's own surfaces; "
                "outputs to <run>/decode_verify_canon.json")
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
w = lambda toks: " ".join(vocab[t] for t in toks)
FORCE = w2i["so"]
SUFFIX = ("_goldrel" if a.gold_rel else "") + ("_canon" if a.canon else "") + ("_strip" if a.strip else "") + ("_split" if a.split else "") + (f"_fit{a.fit_split}" if a.fit_split else "")
VERDICTS = ["valid", "repeat", "premature", "premise", "false", "unparsed"]


LAST_REL = [(0, 0)]   # step 2 (rel_cond models): head's prediction at the last predict() call

def as_fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f)


def invalid_pool(facts, prem, truth):
    """diag_pred_cos pools around each fact of `facts`: one argument replaced by a value seen in the same slot of the
    same predicate in the story (premises or entailed facts); kept if not entailed."""
    pool = collections.defaultdict(set)
    for f in list(prem) + list(truth):
        for pos in range(1, len(f)):
            pool[(f[0], pos)].add(f[pos])
    alts = []
    for fact in facts:
        for pos in range(1, len(fact)):
            for e in sorted(pool.get((fact[0], pos), ()), key=str):
                if e == fact[pos]:
                    continue
                g = list(fact); g[pos] = e; g = tuple(g)
                if g not in truth:
                    alts.append(g)
    return list(dict.fromkeys(alts))


IDEA_KEYS = ["cos_valid_now", "cos_premature", "cos_invalid_max", "cos_said_max", "cos_premise_max", "margin",
             "nn_valid_now", "nn_premature", "nn_invalid", "nn_said", "nn_premise"]


def rank(j):
    return "1" if j == 0 else ("2" if j == 1 else "3+")


def kind(verdict):
    return "valid" if verdict.startswith("valid") else verdict


EXCLUDED = collections.Counter()   # --fit_split: stories left out per family


def stories():
    per_fam = collections.Counter()
    for di, doc in enumerate(docs):
        fam = doc.get("family")
        if per_fam[fam] >= a.n:
            continue
        prem = [i["tokens"] for i in doc["ideas"] if i.get("depth", -1) < 0]
        ded = [i for i in doc["ideas"] if i.get("depth", -1) >= 0]
        # antecedent A of a restated "since A , B" (reinsertion measure), before any --strip / --split
        ded = [{**i, "_A": V.parse_sentence(" ".join(vocab[t] for t in i["tokens"]))[1]} if i.get("restated") else i
               for i in ded]
        if a.fit_split and len(doc["ideas"]) + sum(bool(i.get("restated")) for i in ded) > a.fit_split:
            EXCLUDED[fam] += 1
            continue
        if a.split:   # "since A , B" -> two context ideas "so A ." then "so B ."; scored as one step (B)
            out_ = []
            for i in ded:
                if i.get("restated"):
                    f = as_fact(i["fact"])
                    A = V.parse_sentence(" ".join(vocab[t] for t in i["tokens"]))[1]
                    tb = [w2i[x] for x in R.render_with_template(f, 2 + int(i.get("form") or 0)).w]
                    ta = [w2i[x] for x in R.render_with_template(A, 2).w]
                    out_.append({**i, "tokens": tb, "ctx_tokens": [ta, tb]})
                else:
                    out_.append(i)
            ded = out_
        if a.strip:
            ded = [{**i, "tokens": [w2i[x] for x in R.render_with_template(as_fact(i["fact"]), 2 + int(i.get("form") or 0)).w]}
                   if i.get("restated") else i for i in ded]
        if a.canon:
            prem = [[w2i[x] for x in R.premise_forms(as_fact(i["fact"]))[0]] for i in doc["ideas"] if i.get("depth", -1) < 0]
            ded = [{**i, "tokens": [w2i[x] for x in R.render_with_template(as_fact(i["fact"]), 2).w]} for i in ded]
        if len(ded) < 2:
            continue
        per_fam[fam] += 1
        yield di, fam, prem, ded, V.doc_premises(doc)


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    assert not model.cfg.get("form_cond", False), "form_cond models: use gen_eval --form"
    M = cfg["model"]["m_prefix"]
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

    def decode(prefix, pk, pv, max_len=48, rel=None):          # = gen_eval.decode_idea, greedy
        toks = [FORCE]
        for s_ in range(1, max_len):
            win = torch.cat([prefix, torch.tensor([toks + [PAD]], device=DEV)], 1)
            e = model.O(win, M)
            logits, tr = model.dec(win, torch.tensor([s_ + 1], device=DEV), M, pk, pv, e, **model.dec_rel(rel))
            if is_end[toks[-1]] or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)):
                break
            lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf")
            toks.append(int(lg.argmax()))
        return toks

    def predict(story, n_prem, rels=None):                       # = gen_eval.gen_ideajepa, sampler P
        tok, lens = enc_tokens(story)
        k, v = model.E(tok, lens)
        n = len(story)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                 "idea_pos": torch.tensor([n], device=DEV), "idea_role": (torch.arange(n, device=DEV) >= n_prem).long()}
        if model.cfg.get("rel_cond", False):
            # step 2: relation labels of the context ideas = what the head predicted when each was the next idea
            # (premises / unknown: none); the head's prediction for the next idea goes to LAST_REL
            rels = list(rels or []) + [(0, 0)] * (n - len(rels or []))
            batch["rel_id"] = torch.tensor([x[0] for x in rels[:n]], device=DEV)
            batch["rel_off"] = torch.tensor([x[1] for x in rels[:n]], device=DEV)
            out = model.predict(k, v, batch)
            lr, lp = model.rel_logits()
            LAST_REL[0] = (int(lr[0].argmax()), int(lp[0].argmax()))
            return out
        return model.predict(k, v, batch)

    zcache = {}

    def zfacts(facts):
        """E_ema of the "so" form 0 sentences -> [n, r*ds] (diag_pred_cos space); None if empty."""
        facts = list(facts)
        todo = [f for f in dict.fromkeys(facts) if f not in zcache]
        if todo:
            t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(f, 2).w] for f in todo])
            _, v_ = model.E_ema(t_, l_)
            for f, z in zip(todo, F.normalize(F.normalize(v_.float(), dim=-1).flatten(1), dim=-1)):
                zcache[f] = z
        return torch.stack([zcache[f] for f in facts]) if facts else None

    def idea_row(pv, fam, prem_facts, truth, said):
        """Where P's prediction lands among the pools, given the facts already said in that context."""
        p = F.normalize(F.normalize(pv[0].float(), dim=-1).flatten(), dim=0)
        said = list(dict.fromkeys(said))
        now = [f for f in sorted(V.valid_next(fam, prem_facts, said), key=str) if f not in said]
        if not now:
            return None
        pools = {"valid_now": now, "said": said, "premise": list(prem_facts),
                 "premature": [f for f in sorted(truth, key=str) if f not in now and f not in said and f not in prem_facts],
                 "invalid": [g for g in invalid_pool(now, prem_facts, truth) if g not in said]}
        best = {c: float((zfacts(fs) @ p).max()) for c, fs in pools.items() if fs}
        row = {f"cos_{c}" + ("_max" if c in ("invalid", "said", "premise") else ""): v_ for c, v_ in best.items()}
        if "invalid" in best:
            row["margin"] = best["valid_now"] - best["invalid"]
        nn = max(best, key=best.get)
        for c in pools:
            row[f"nn_{c}"] = float(nn == c)
        return row

    idea_acc = collections.defaultdict(lambda: collections.defaultdict(list))   # (cond, fam, rank) -> measure -> values
    cnt = collections.defaultdict(collections.Counter)   # (cond, fam, rank) -> outcome counts
    examples = collections.defaultdict(list)
    reins = collections.defaultdict(collections.Counter)   # (fam, "after" | "other") -> counts, condition (b)
    reins_ex = []

    def ante_of(fam, f, stated):
        try:
            return R.antecedent(fam, f, stated) if f is not None else None
        except Exception:
            return None
    with torch.no_grad():
        for sid, fam, prem, ded, prem_facts in stories():
            ref_txt = [w(i["tokens"]) for i in ded]
            own, rels_b, rels_c = [], [(0, 0)] * len(prem), [(0, 0)] * len(prem)
            if a.gold_rel:
                gdoc = R.split_compound(docs[sid], "conclusion", vocab, cfg["data"].get("rel_source", "connector"),
                                        cfg["data"].get("rel_f1", "useful"))["ideas"]
                RID = {None: 0, "none": 0, "justified_by": 1, "sequence": 2}
                gold = [(RID[i.get("rel")], n_ - i["rel_ptr"] if i.get("rel_ptr", -1) >= 0 else 0) for n_, i in enumerate(gdoc)]
                assert len(gdoc) == len(prem) + len(ded), "gold_rel: prop view only"
            truth = V.derivable(fam, prem_facts)
            for j, idea in enumerate(ded):
                keys = [(fam, rank(j)), (fam, "all"), ("all", rank(j)), ("all", "all")]
                ctx = [list(p) for p in prem] + [list(t) for i in ded[:j] for t in i.get("ctx_tokens", [i["tokens"]])]
                pre = prefix_of(ctx)
                pk_b, pv_b = predict(ctx, len(prem), rels_b)          # (b) first: its relation serves (a) and (b)
                rel_b = LAST_REL[0]
                rels_b += [(0, 0)] * (len(idea.get("ctx_tokens", [0])) - 1) + [gold[len(prem) + j] if a.gold_rel else rel_b]
                # (a) decoder alone on E_ema(true fact, "so" form 0)
                fact = as_fact(idea["fact"])
                t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(fact, 2).w]])
                ka, va = model.E_ema(t_, l_)
                out_a = decode(pre, ka, va, rel=torch.tensor([rel_b[0]], device=DEV))
                f_a, _ = V.parse_sentence(w(out_a))
                oa = "same_fact" if f_a == fact else ("unparsed" if f_a is None else "other_fact")
                # (b) P with the reference context
                pk, pv = pk_b, pv_b
                rb = idea_row(pv, fam, prem_facts, truth, [as_fact(i["fact"]) for i in ded[:j]])
                out_b = decode(pre, pk, pv, rel=torch.tensor([rel_b[0]], device=DEV))
                ob = kind(V.judge(fam, prem_facts, ref_txt[:j] + [w(out_b)])[-1]["verdict"])
                # (c) free generation
                story_c = [list(p) for p in prem] + own
                pk, pv = predict(story_c, len(prem), rels_c)
                rels_c.append(LAST_REL[0])
                own_facts = [f for f in (V.parse_sentence(w(x))[0] for x in own) if f is not None]
                rd = idea_row(pv, fam, prem_facts, truth, own_facts)
                out_c = decode(prefix_of(story_c), pk, pv, rel=torch.tensor([LAST_REL[0][0]], device=DEV))
                own.append(out_c)
                oc = kind(V.judge(fam, prem_facts, [w(x) for x in own])[-1]["verdict"])
                for key in keys:
                    cnt[("a",) + key][oa] += 1; cnt[("b",) + key][ob] += 1; cnt[("c",) + key][oc] += 1
                    for cond, row in (("b", rb), ("d", rd)):
                        idea_acc[(cond,) + key]["N"].append(1)
                        for m_, v_ in (row or {}).items():
                            idea_acc[(cond,) + key][m_].append(v_)
                # reinsertion measure, condition (b), steps j >= 1 of F2/F3: "after" = right after a
                # restated reference sentence "since A , B", "other" = the other steps (control). A wrong output
                # "continues the previous step" if the generator's rule gives it the same licensing premise as the
                # previous reference fact (= A after a reinsertion: "as if A were the last of the chain"); B said again
                # is counted apart (repeat_last, its premise is A by construction)
                if j >= 1 and fam in ("F2", "F3"):
                    prev = ded[j - 1]
                    grp = "after" if prev.get("restated") else "other"
                    c_ = reins[(fam, grp)]
                    c_["n"] += 1; c_["valid"] += ob == "valid"
                    if ob != "valid":
                        f_b = V.parse_sentence(w(out_b))[0]
                        last = as_fact(prev["fact"])
                        facts_ = [as_fact(i["fact"]) for i in ded]
                        an = ante_of(fam, f_b, list(prem_facts) + facts_[:j])
                        an_prev = prev["_A"] if prev.get("restated") else ante_of(fam, last, list(prem_facts) + facts_[:j - 1])
                        cont = f_b != last and an is not None and an == an_prev
                        c_["wrong"] += 1; c_["repeat_last"] += f_b == last; c_["cont_prev"] += cont
                        c_["wrong_" + ob] += 1
                        if grp == "after" and len(reins_ex) < 8 and f_b != last:
                            reins_ex.append({"fam": fam, "reinserted": ref_txt[j - 1], "out": w(out_b), "verdict": ob,
                                             "cont_A": cont, "ref_next": ref_txt[j]})
                if ob != "valid" and len(examples[(fam, ob)]) < 2:
                    examples[(fam, ob)].append({"ref_said": ref_txt[:j], "b": w(out_b), "ref_next": ref_txt[j]})
    res = {"|".join(k): {**dict(c), "n": sum(c.values())} for k, c in cnt.items()}
    ires = {"|".join(k): {m_: (sum(v_) / len(v_) if m_ != "N" else len(v_)) for m_, v_ in d_.items()}
            for k, d_ in idea_acc.items()}
    json.dump({"run": run, "n_per_family": a.n, "excluded_fit_split": dict(EXCLUDED), "results": res, "idea": ires,
               "examples_b": {f"{k[0]}|{k[1]}": v for k, v in examples.items()},
               "reinsertion": {f"{k[0]}|{k[1]}": dict(c_) for k, c_ in reins.items()}, "reinsertion_examples": reins_ex},
              open(f"{run}/decode_verify{SUFFIX}.json", "w"), indent=1)
    r = lambda c, key: res[f"{c}|all|all"].get(key, 0) / res[f"{c}|all|all"]["n"]
    print(f"{run}: (a) same fact {r('a', 'same_fact'):.3f}  (b) valid {r('b', 'valid'):.3f}  "
          f"(c) valid {r('c', 'valid'):.3f}", flush=True)
    del model
    return {**res, **{"idea|" + k: v_ for k, v_ in ires.items()}}


results = {}
for run in a.runs:
    if a.summary_only:
        js = json.load(open(f"{run}/decode_verify{SUFFIX}.json"))
        results[run] = {**js["results"], **{"idea|" + k: v_ for k, v_ in js.get("idea", {}).items()}}
    else:
        results[run] = run_one(run)

if a.summary:
    groups = collections.defaultdict(list)
    for run, res in results.items():
        name = Path(run).name
        groups[name.rsplit("_s", 1)[0] if "_s" in name and name[-1].isdigit() else name].append(res)

    def cell(rs, key, outcome, count=False):
        xs = [r[key].get(outcome, 0) if count else r[key].get(outcome, 0) / r[key]["n"] for r in rs if key in r]
        if not xs:
            return "–"
        f = (lambda x: f"{x:.1f}") if count else (lambda x: f"{100 * x:.0f}")
        return f"{f(st.mean(xs))} ± {f(st.stdev(xs))}" if len(xs) >= 2 else f(xs[0])

    G = list(groups)
    head = "| | " + " | ".join(f"{g} (n={len(groups[g])})" for g in G) + " |"
    sep = "|---|" + "---|" * len(G)
    L = [f"# Decode-then-verify (gen_eval stories: {a.n} per family; greedy, \"so\" forced)", "",
         "(a) decoder alone: E_ema(true fact, \"so\" form 0) with the reference context → % decoding back to the same fact.",
         "(b) P's prediction with the reference context (teacher forcing) → % valid for the verifier.",
         "(c) free generation (own previous deductions as context, = gen_eval) → % valid.",
         "Mean ± std over seeds, in %.", "", "## Summary: % success per condition, by family and by rank", "", head, sep]
    for fam in ("all", "F1", "F2", "F3"):
        for rk in ("all", "1", "2", "3+"):
            key_ = lambda c: f"{c}|{fam}|{rk}"
            for c, oc, lab in (("a", "same_fact", "(a) decoder"), ("b", "valid", "(b) P teacher-forced"),
                               ("c", "valid", "(c) free generation")):
                L.append(f"| {fam} rank {rk} {lab} | " + " | ".join(cell(groups[g], key_(c), oc) for g in G) + " |")
    L += ["", "## Error types (counts per seed, all families, all ranks)", "", head, sep]
    for c in ("b", "c"):
        for oc in VERDICTS:
            L.append(f"| ({c}) {oc} | " + " | ".join(cell(groups[g], f"{c}|all|all", oc, count=True) for g in G) + " |")
    L += ["", "## Error types by family (counts per seed)", "", head, sep]
    for fam in ("F1", "F2", "F3"):
        for c in ("b", "c"):
            for oc in VERDICTS[1:]:
                L.append(f"| {fam} ({c}) {oc} | " + " | ".join(cell(groups[g], f"{c}|{fam}|all", oc, count=True) for g in G) + " |")
    L += ["", "## (a) decoder failures (counts per seed)", "", head, sep]
    for oc in ("other_fact", "unparsed"):
        L.append(f"| (a) {oc} | " + " | ".join(cell(groups[g], "a|all|all", oc, count=True) for g in G) + " |")
    L += ["", "## Idea side: P's prediction before decoding, (b) reference context vs (d) free-generation context", "",
          "cos / nn rates (nn in %); pools relative to the facts already said in that context (see the script docstring).", ""]
    for fam in ("all", "F1", "F2", "F3"):
        for rk in ("all", "1", "2", "3+"):
            L += [f"### {fam}, rank {rk}", "", "| | " + " | ".join(f"{g} (b) | {g} (d)" for g in G) + " |",
                  "|---|" + "---|---|" * len(G)]
            for m_ in ["N"] + IDEA_KEYS:
                cells = []
                for g in G:
                    for cond in ("b", "d"):
                        xs = [r[f"idea|{cond}|{fam}|{rk}"][m_] for r in groups[g]
                              if f"idea|{cond}|{fam}|{rk}" in r and m_ in r[f"idea|{cond}|{fam}|{rk}"]]
                        sc = 100 if m_.startswith("nn_") else 1
                        fm = (lambda x: f"{x:.0f}") if (m_.startswith("nn_") or m_ == "N") else (lambda x: f"{x:.3f}")
                        cells.append("–" if not xs else (f"{fm(sc * st.mean(xs))} ± {fm(sc * st.stdev(xs))}" if len(xs) >= 2
                                                          else fm(sc * xs[0])))
                L.append(f"| {m_} | " + " | ".join(cells) + " |")
            L.append("")
    Path(a.summary).write_text("\n".join(L) + "\n")
    print(f"-> {a.summary}")

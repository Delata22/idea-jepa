"""Why does B repeat in the REFERENCE context (decode_verify (b)) but not in free generation?

gen_eval stories and decoding (greedy, "so" forced). At every deduction step j, reference context (the story's own
surfaces: premises + reference deductions < j), P's prediction decoded and judged (= decode_verify (b)); the same
story is also generated freely (= (c)). The surface of a reference deduction = its template (R.template_id):
so / then / tmt ("this means that") / since (restated "since …, …") x clause form f0 / f1.
  a. every (b) repeat: the repeated fact, the surface under which the REFERENCE had written it; repeat rate per
     surface = repeats of a fact first written under that surface / (steps x facts already said under that surface)
  b. distance in steps between the first statement (reference) and the repeat
  c. order: rank at which the model states that fact in free generation on the same story vs its reference rank:
     later / earlier or same / never
  d. cos(E(reference sentence as written), E("so" form 0 of the same fact)), per surface, over all reference
     deductions (E online; per-slot-normalised values, flattened; keys; flow space [k ; v/|v|])
  e. idea side: nearest neighbour of P's prediction among valid-now ∪ said is a said fact; cos_said_max; and per
     surface of the said sentence: mean cos(prediction, said fact) and rate at which that said fact is the nearest
     neighbour (per exposure). Facts encoded by E_ema from their "so" form 0 (diag_pred_cos space).
usage: uv run python scripts/diag_refctx_repeats.py <run> [<run> ...] [--n 30] [--summary runs/v3/B_refctx_repeats.md]
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
ap.add_argument("--strip", action="store_true", help="evaluation option: a restated reference sentence \"since A , B .\" "
                "is given to the model as \"so B .\" (B in the clause form written); outputs to <run>/<name>_strip.json")
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
w = lambda toks: " ".join(vocab[t] for t in toks)
FORCE = w2i["so"]
SURF = ["so/f0", "so/f1", "then/f0", "then/f1", "tmt/f0", "tmt/f1", "since/f0", "since/f1"]


LAST_REL = [(0, 0)]   # step 2 (rel_cond models): head's prediction at the last predict() call

def surface(idea):
    t = R.template_id(idea)
    if t >= 8:
        return f"since/f{t - 8}"
    return f"{['so', 'then', 'tmt'][(t - 2) // 2]}/f{(t - 2) % 2}"


def as_fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f)


EXCLUDED = collections.Counter()   # --fit_split: stories left out per family


def stories():
    per_fam = collections.Counter()
    for doc in docs:
        fam = doc.get("family")
        if per_fam[fam] >= a.n:
            continue
        prem = [i["tokens"] for i in doc["ideas"] if i.get("depth", -1) < 0]
        ded = [i for i in doc["ideas"] if i.get("depth", -1) >= 0]
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
        if a.strip:   # the surface label (template_id) stays "since"; the model reads "so B"
            ded = [{**i, "tokens": [w2i[x] for x in R.render_with_template(as_fact(i["fact"]), 2 + int(i.get("form") or 0)).w]}
                   if i.get("restated") else i for i in ded]
        if len(ded) < 2:
            continue
        per_fam[fam] += 1
        yield fam, prem, ded, V.doc_premises(doc)


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
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

    def decode(prefix, pk, pv, max_len=48, rel=None):
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

    def predict(story, n_prem, rels=None):
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

    def zf(f):
        if f not in zcache:
            t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(f, 2).w]])
            _, v_ = model.E_ema(t_, l_)
            zcache[f] = F.normalize(F.normalize(v_[0].float(), dim=-1).flatten(), dim=0)
        return zcache[f]

    def spaces(toks_list):
        t_, l_ = enc_tokens(toks_list)
        k_, v_ = model.E(t_, l_)
        vn = F.normalize(v_.float(), dim=-1)
        return {"values": F.normalize(vn.flatten(1), dim=-1), "keys": F.normalize(k_.float().flatten(1), dim=-1),
                "flow": F.normalize(torch.cat([k_.float(), vn], -1).flatten(1), dim=-1)}

    C = collections.Counter()                                  # counters
    L = collections.defaultdict(list)                          # lists
    rows = []
    with torch.no_grad():
        for fam, prem, ded, prem_facts in stories():
            facts = [as_fact(i["fact"]) for i in ded]
            surfs = [surface(i) for i in ded]
            ref_txt = [w(i["tokens"]) for i in ded]
            # d. invariance of E to the reference surface
            so0 = [[w2i[x] for x in R.render_with_template(f, 2).w] for f in facts]
            zs_ref, zs_so = spaces([list(i["tokens"]) for i in ded]), spaces(so0)
            for j, s in enumerate(surfs):
                for sp in zs_ref:
                    L[f"d|{sp}|{s}"].append(float(zs_ref[sp][j] @ zs_so[sp][j]))
            # c. free generation on the same story: rank at which each fact is first stated
            own, rels_c = [], [(0, 0)] * len(prem)
            for j in range(len(ded)):
                story_c = [list(p) for p in prem] + own
                pk, pv = predict(story_c, len(prem), rels_c)
                rels_c.append(LAST_REL[0])
                own.append(decode(prefix_of(story_c), pk, pv, rel=torch.tensor([LAST_REL[0][0]], device=DEV)))
            free_rank = {}
            for r_, x in enumerate(own):
                f_, _ = V.parse_sentence(w(x))
                if f_ is not None and f_ not in free_rank:
                    free_rank[f_] = r_
            # (b) reference context
            rels_b = [(0, 0)] * len(prem)
            for j in range(len(ded)):
                ctx = [list(p) for p in prem] + [list(t) for i in ded[:j] for t in i.get("ctx_tokens", [i["tokens"]])]
                pk, pv = predict(ctx, len(prem), rels_b)
                rels_b += [(0, 0)] * (len(ded[j].get("ctx_tokens", [0])) - 1) + [LAST_REL[0]]
                out = decode(prefix_of(ctx), pk, pv, rel=torch.tensor([LAST_REL[0][0]], device=DEV))
                verdict = V.judge(fam, prem_facts, ref_txt[:j] + [w(out)])[-1]["verdict"]
                verdict = "valid" if verdict.startswith("valid") else verdict
                C[f"verdict|{verdict}"] += 1; C["steps"] += 1
                f_out, _ = V.parse_sentence(w(out))
                for i in range(j):                                       # exposures per surface
                    C[f"expo|{surfs[i]}"] += 1
                if verdict == "repeat" and f_out in facts[:j]:
                    i = facts.index(f_out)
                    C[f"rep|{surfs[i]}"] += 1; C["rep"] += 1
                    L["dist"].append(j - i)
                    fr = free_rank.get(f_out)
                    cls = "never" if fr is None else ("later" if fr > i else "earlier_or_same")
                    C[f"order|{cls}"] += 1
                    rows.append({"family": fam, "fact": str(f_out), "ref_surface": surfs[i], "ref_sentence": ref_txt[i],
                                 "said_at": i, "repeated_at": j, "decoded": w(out), "free_rank": fr})
                # e. idea side
                p = F.normalize(F.normalize(pv[0].float(), dim=-1).flatten(), dim=0)
                said = facts[:j]
                now = [f for f in sorted(V.valid_next(fam, prem_facts, said), key=str) if f not in said]
                if said:
                    cs = {f: float(zf(f) @ p) for f in dict.fromkeys(said + now)}
                    nn = max(cs, key=cs.get)
                    L["e|nn_said"].append(float(nn in said)); L["e|cos_said_max"].append(max(cs[f] for f in said))
                    if now:
                        L["e|cos_valid_now"].append(max(cs[f] for f in now))
                    for i in range(j):
                        L[f"e|cos_said|{surfs[i]}"].append(cs[facts[i]])
                        L[f"e|nn_is|{surfs[i]}"].append(float(nn == facts[i]))
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    res = {"steps": C["steps"], "repeats": C["rep"],
           "verdicts": {k.split("|")[1]: v_ for k, v_ in C.items() if k.startswith("verdict|")},
           "rep_by_surface": {s: C[f"rep|{s}"] for s in SURF}, "expo_by_surface": {s: C[f"expo|{s}"] for s in SURF},
           "rep_rate_by_surface": {s: (C[f"rep|{s}"] / C[f"expo|{s}"] if C[f"expo|{s}"] else None) for s in SURF},
           "dist_mean": mean(L["dist"]), "dist_hist": dict(collections.Counter(L["dist"])),
           "order": {c: C[f"order|{c}"] for c in ("later", "earlier_or_same", "never")},
           "invariance": {sp: {s: mean(L[f"d|{sp}|{s}"]) for s in SURF} for sp in ("values", "keys", "flow")},
           "idea": {"nn_said": mean(L["e|nn_said"]), "cos_said_max": mean(L["e|cos_said_max"]),
                    "cos_valid_now": mean(L["e|cos_valid_now"]),
                    "cos_said_by_surface": {s: mean(L[f"e|cos_said|{s}"]) for s in SURF},
                    "nn_is_said_by_surface": {s: mean(L[f"e|nn_is|{s}"]) for s in SURF}},
           "examples": rows[:40]}
    json.dump({"run": run, "n_per_family": a.n, "excluded_fit_split": dict(EXCLUDED), "results": res}, open(f"{run}/refctx_repeats{'_strip' if a.strip else ''}{'_split' if a.split else ''}{f'_fit{a.fit_split}' if a.fit_split else ''}.json", "w"), indent=1)
    print(f"{run}: repeats {C['rep']}/{C['steps']}  by surface " +
          " ".join(f"{s}:{C[f'rep|{s}']}" for s in SURF) + f"  order {res['order']}", flush=True)
    return res


results = {}
for run in a.runs:
    results[run] = json.load(open(f"{run}/refctx_repeats{'_strip' if a.strip else ''}{'_split' if a.split else ''}{f'_fit{a.fit_split}' if a.fit_split else ''}.json"))["results"] if a.summary_only else run_one(run)

if a.summary:
    groups = collections.defaultdict(list)
    for run, res in results.items():
        name = Path(run).name
        groups[name.rsplit("_s", 1)[0] if "_s" in name and name[-1].isdigit() else name].append(res)
    G = list(groups)

    def cell(xs, fmt):
        xs = [x for x in xs if x is not None]
        if not xs:
            return "–"
        return f"{fmt(st.mean(xs))} ± {fmt(st.stdev(xs))}" if len(xs) >= 2 else fmt(xs[0])
    f3 = lambda x: f"{x:.3f}"; f1 = lambda x: f"{x:.1f}"; pct = lambda x: f"{100 * x:.1f}"
    head = ["| | " + " | ".join(f"{g} (n={len(groups[g])})" for g in G) + " |", "|---|" + "---|" * len(G)]
    row = lambda lab, get, fmt: f"| {lab} | " + " | ".join(cell([get(r) for r in groups[g]], fmt) for g in G) + " |"
    out = [f"# Repeats in the reference context (decode_verify (b)), by surface of the reference sentence",
           "", f"gen_eval stories ({a.n} per family), greedy \"so\". Mean ± std over seeds. Surfaces: connector so / then / "
           "tmt (\"this means that\") / since (restated) × clause form f0 / f1; the decoder of A/B is trained on so/f0 and so/f1 only.",
           "", "## Verdicts in (b) (counts per seed)", ""] + head
    for vd in ("valid", "repeat", "premature", "premise", "false", "unparsed"):
        out.append(row(vd, lambda r, vd=vd: r["verdicts"].get(vd, 0), f1))
    out += ["", "## a. Repeat rate per surface under which the reference first wrote the fact (% per exposure; raw count)", ""] + head
    for s in SURF:
        out.append(row(f"{s} rate", lambda r, s=s: r["rep_rate_by_surface"][s], pct))
        out.append(row(f"{s} count", lambda r, s=s: r["rep_by_surface"][s], f1))
    out += ["", "## b. Distance (steps) first statement → repeat", ""] + head
    out.append(row("mean distance", lambda r: r["dist_mean"], f1))
    for dd in ("1", "2", "3", "4"):
        out.append(row(f"distance {dd}{'+' if dd == '4' else ''} (share of repeats, %)",
                       lambda r, dd=dd: ((sum(v_ for k_, v_ in r["dist_hist"].items() if int(k_) >= 4) if dd == "4"
                                          else r["dist_hist"].get(dd, r["dist_hist"].get(int(dd), 0))) / r["repeats"]
                                         if r["repeats"] else None), pct))
    out += ["", "## c. Repeated fact: rank in free generation on the same story vs its reference rank (share of repeats, %)", ""] + head
    for c in ("later", "earlier_or_same", "never"):
        out.append(row(c, lambda r, c=c: r["order"][c] / r["repeats"] if r["repeats"] else None, pct))
    for sp in ("values", "flow", "keys"):
        out += ["", f"## d. Invariance of E ({sp}): cos(E(reference sentence), E(\"so\" form 0 of the same fact))", ""] + head
        for s in SURF:
            out.append(row(s, lambda r, s=s, sp=sp: r["invariance"][sp][s], f3))
    out += ["", "## e. Idea side (P's prediction, E_ema \"so\" form 0 space)", ""] + head
    out.append(row("nearest neighbour is a said fact (%)", lambda r: r["idea"]["nn_said"], pct))
    out.append(row("cos_said_max", lambda r: r["idea"]["cos_said_max"], f3))
    out.append(row("cos_valid_now (max)", lambda r: r["idea"]["cos_valid_now"], f3))
    for s in SURF:
        out.append(row(f"cos(pred, said fact written {s})", lambda r, s=s: r["idea"]["cos_said_by_surface"][s], f3))
    for s in SURF:
        out.append(row(f"said fact written {s} is the nearest neighbour (% per exposure)",
                       lambda r, s=s: r["idea"]["nn_is_said_by_surface"][s], pct))
    Path(a.summary).write_text("\n".join(out) + "\n")
    print(f"-> {a.summary}")

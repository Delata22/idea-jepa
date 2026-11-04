"""Plan variable (notes §32.20), REFERENCE context (premises + reference deductions < j, prop view: restated sentences as
"so B"), per deduction step, two regimes kept apart:
  ref     o = the REFERENCE plan target (argmax over the past ideas of cos(Ē_ema(reference next idea), Ē_ema(idea j)),
          the training target; ceiling)
  drawn   o drawn from the pointer head (T = 1); also "argmax" (head's argmax)
Per family: validity of the greedy decoding; F1 chimeras (wrong fact whose object and location come from two different
valid candidates); cos(Ẑ, nearest candidate) (Ẑ = P's values, diag_pred_cos space); copy = cos(Ẑ, target) −
cos(Ē(o), target), target = Ē_ema(reference next idea); head: accuracy (argmax = reference o), entropy, probability mass
on the o's of the valid candidates (each candidate's own nearest past idea).
usage: uv run python scripts/diag_plan_eval.py <run> [<run> ...] [--n 30] [--summary out.md]
"""
import argparse
import collections
import json
import os
import statistics as st
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
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--summary", default=None)
ap.add_argument("--families", nargs="+", default=["F1", "F2", "F3"])
ap.add_argument("--regimes", nargs="+", default=None, help="subset of ref drawn argmax init useful random none")
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
FORCE = w2i["so"]
w = lambda t: " ".join(vocab[x] for x in t)
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)
so0 = lambda f: [w2i[x] for x in R.render_with_template(f, 2).w]
import random
RNG = random.Random(a.seed)


def stories():
    per = collections.Counter()
    for doc in docs:
        fam = doc.get("family")
        if per[fam] >= a.n or fam not in a.families:
            continue
        ded = [i for i in doc["ideas"] if i["step"] == "deduction"]
        if len(ded) < 2:
            continue
        per[fam] += 1
        ded = [{**i, "tokens": [w2i[x] for x in R.render_with_template(as_fact(i["fact"]), 2 + int(i.get("form") or 0)).w]}
               if i.get("restated") else i for i in ded]
        yield fam, [i["tokens"] for i in doc["ideas"] if i["step"] == "premise"], ded, V.doc_premises(doc)


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    assert model.cfg.get("plan_cond", False)
    model.plan_gen = torch.Generator(device=DEV).manual_seed(a.seed)
    M = cfg["model"]["m_prefix"]
    endm = torch.zeros(len(vocab), dtype=torch.bool, device=DEV); endm[w2i["."]] = True

    def enc_tokens(ideas):
        L = max(len(t) for t in ideas)
        tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
        for i, t in enumerate(ideas):
            tok[i, :len(t)] = torch.tensor(t)
        return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)

    flat = lambda v: F.normalize(F.normalize(v.float(), dim=-1).flatten(1), dim=-1)

    def ebar(tok_lists):
        t_, l_ = enc_tokens(tok_lists)
        return flat(model.E_ema(t_, l_)[1])

    def predict(story, n_prem, o=None, mode="sample"):
        tok, lens = enc_tokens(story)
        k, v = model.E(tok, lens)
        n = len(story)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                 "idea_pos": torch.tensor([n], device=DEV), "idea_role": (torch.arange(n, device=DEV) >= n_prem).long()}
        if o is not None:   # the predicted block's plan; past blocks get their label-free targets (see _predict_plan)
            batch["plan_next"] = torch.tensor([o], device=DEV)
        model.plan_mode = mode
        pk, pv = model.predict(k, v, batch)
        return pk, pv, model.plan_logits[0, n].float(), int(model.plan_choice[0, n])

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
    L = collections.defaultdict(list)
    ex = []
    with torch.no_grad():
        for fam, prem, ded, pf in stories():
            ref = [as_fact(i["fact"]) for i in ded]
            up = R.f1_useful_premises(pf) if fam == "F1" else {}
            o_hist = []                                          # reference plan targets of the deductions said
            for j in range(len(ded)):
                ctx = [list(p) for p in prem] + [list(i["tokens"]) for i in ded[:j]]
                cands = sorted(V.valid_next(fam, pf, ref[:j]) - set(ref[:j]), key=str)
                if not cands:
                    continue
                zpast = ebar(ctx)
                tgt = ebar([list(ded[j]["tokens"])])[0]
                o_ref = int((zpast @ tgt).argmax())
                zc = ebar([so0(c) for c in cands])
                o_c = (zc @ zpast.T).argmax(1).tolist()
                hist = [BOS] + [t for idea in ctx for t in idea]; p = hist[-M:]
                pre = torch.full((1, M), PAD, dtype=torch.long); pre[0, M - len(p):] = torch.tensor(p); pre = pre.to(DEV)
                said = [w(i["tokens"]) for i in ded[:j]]
                ctx_facts = list(pf) + ref[:j]
                sub = None                                        # F1: object moved (useful premise = a "goes") or not
                init = None                                       # F1: premise stating the object's initial position
                if fam == "F1" and ref[j][0] == "objin" and ref[j][1] in up:
                    sub = "F1mov" if pf[up[ref[j][1]]][0] == "goes" else "F1imm"
                    init = next((i_ for i_, f_ in enumerate(pf) if f_[0] == "objin" and f_[1] == ref[j][1]), None)
                regimes = (("ref", {"o": o_ref}), ("drawn", {"mode": "sample"}), ("argmax", {"mode": "argmax"}),
                           *((("init", {"o": init}),) if init is not None else ()),   # control: choice without the place
                           # control: o = the useful premise (moved object: last "X goes to L" of its carrier)
                           *((("useful", {"o": up[ref[j][1]]}),) if sub is not None else ()),
                           ("random", {"o": RNG.randrange(len(ctx))}),    # ablation: a random past idea
                           ("none", {"o": -1}))                            # ablation: no plan for the predicted block
                regimes = [x for x in regimes if a.regimes is None or x[0] in a.regimes]
                for regime, kw in regimes:
                    pk, pv, lg, o_used = predict(ctx, len(prem), **kw)
                    out = decode(pre, pk, pv)
                    f_out = V.parse_sentence(w(out))[0]
                    vd = V.judge(fam, pf, said + [w(out)])[-1]["verdict"]
                    ok = vd.startswith("valid")
                    inc = collections.Counter()
                    inc["n"] += 1; inc["valid"] += ok; inc["o_is_ref"] += o_used == o_ref
                    inc["o_is_cand_o"] += o_used in o_c
                    # o = a premise that already served as the plan of a deduction said before (its deduction is said)
                    spent = 0 <= o_used < len(prem) and o_used in o_hist
                    inc["o_spent"] += spent; inc["o_spent_repeat"] += spent and vd == "repeat"; inc["repeat"] += vd == "repeat"
                    if fam == "F1" and o_used >= 0 and not ok and f_out is not None and f_out[0] == "objin":
                        objs = {x[1] for x in cands}; locs = {x[1]: x[2] for x in cands}
                        inc["chimera"] += f_out[1] in objs and any(l_ == f_out[2] for o_, l_ in locs.items() if o_ != f_out[1])
                        of = ctx_facts[o_used]                    # the place written = the place of the plan idea o?
                        inc["wrong_loc_is_o_loc"] += of[0] in ("objin", "at", "goes") and f_out[2] == of[-1]
                        inc["wrong_objin"] += 1
                    for key in [(fam, regime)] + ([(sub, regime)] if sub else []):
                        cnt[key].update(inc)
                    c = cnt[(fam, regime)]
                    if o_used < 0:
                        continue
                    zh = flat(pv)[0]
                    L[(fam, regime, "cos_nn_cand")].append(float((zc @ zh).max()))
                    L[(fam, regime, "copy")].append(float(zh @ tgt) - float(zpast[o_used] @ tgt))
                    if regime == "ref":
                        pr = torch.softmax(lg, -1)
                        c["head_acc"] += int(lg.argmax()) == o_ref
                        L[(fam, "head", "entropy")].append(float(-(pr * torch.log(pr.clamp_min(1e-12))).sum()))
                        L[(fam, "head", "mass_cand_o")].append(float(pr[sorted(set(o_c))].sum()))
                        L[(fam, "head", "chance_acc")].append(1.0 / len(ctx))                 # uniform over the past ideas
                        L[(fam, "head", "chance_mass")].append(len(set(o_c)) / len(ctx))
                    if len(ex) < 12 and regime == "drawn" and (len(ex) % 2 == 0) == ok:
                        ex.append({"fam": fam, "o": w(ctx[o_used]), "out": w(out), "verdict": vd})
                o_hist.append(o_ref)
    res = {f"{k[0]}|{k[1]}": dict(c) for k, c in cnt.items()}
    lists = {"|".join(k): st.mean(v) for k, v in L.items()}
    json.dump({"run": run, "counts": res, "means": lists, "examples": ex}, open(f"{run}/plan_eval.json", "w"), indent=1)
    print(f"{run}: " + " ".join(f"{k} {c['valid'] / c['n']:.3f}" for k, c in sorted(res.items())), flush=True)
    del model
    return res, lists, ex


allr = {r: run_one(r) for r in a.runs}
if a.summary:
    T = ["| run | famille | régime | n | valide | o = o de référence | o = o d'une candidate | chimères F1 | cos(Ẑ, candidate la plus proche) | recopie | o = prémisse déjà servie | répétitions (dont o servie) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    H = ["| run | famille | précision tête (argmax = o réf.) | hasard | entropie | masse sur les o des candidates | hasard |", "|---|---|---|---|---|---|---|"]
    for r, (res, lists, ex) in allr.items():
        nm = r.split("/")[-1]
        for fam in ("F1", "F1imm", "F1mov", "F2", "F3"):
            for regime in ("ref", "drawn", "argmax", "init", "useful", "random", "none"):
                c = res.get(f"{fam}|{regime}")
                if not c:
                    continue
                f = lambda x: f"{c.get(x, 0) / c['n']:.3f}"
                T.append(f"| {nm} | {fam} | {regime} | {c['n']} | {f('valid')} | {f('o_is_ref')} | {f('o_is_cand_o')} | "
                         f"{f('chimera') if fam == 'F1' else '—'} | {lists.get(f'{fam}|{regime}|cos_nn_cand', float('nan')):.3f} | "
                         f"{lists.get(f'{fam}|{regime}|copy', float('nan')):+.3f} | {f('o_spent')} | "
                         f"{c.get('repeat', 0)} ({c.get('o_spent_repeat', 0)}) |"
                         + (f" erreurs F1 au lieu de o : {c.get('wrong_loc_is_o_loc', 0)}/{c.get('wrong_objin', 0)}" if fam.startswith("F1") else ""))
            c = res.get(f"{fam}|ref")
            if c:
                H.append(f"| {nm} | {fam} | {c.get('head_acc', 0) / c['n']:.3f} | {lists.get(f'{fam}|head|chance_acc', float('nan')):.3f} | "
                         f"{lists.get(f'{fam}|head|entropy', float('nan')):.3f} | "
                         f"{lists.get(f'{fam}|head|mass_cand_o', float('nan')):.3f} | {lists.get(f'{fam}|head|chance_mass', float('nan')):.3f} |")
    E = ["", "Exemples (o tiré, contexte de référence) :"]
    for r, (_, _, ex) in allr.items():
        for e in ex[:4]:
            E.append(f"- {r.split('/')[-1]} {e['fam']} : o « {e['o']} » → « {e['out']} » ({e['verdict']})")
    open(a.summary, "w").write("\n".join(T + [""] + H + E) + "\n")
    print("\n".join(T + [""] + H + E))

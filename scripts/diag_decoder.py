"""Mémo §32.14 (a): is the decoder the source of the F2 errors at rank 3+ and of the F3 key/box confusions? No training.
B-type models, F2 and F3, reference context (premises + reference deductions < j), by rank of the step (1 / 2 / 3+).
 (i)  autoencoding: D(E_ema(reference fact j, "so" form 0)) parses to that fact? rate per rank and per relation
      (F2: relation word of the fact; F3: predicate canopen / gets-key / gets-item).
 (ii) P's prediction Ẑ: "idea right" = the nearest neighbour of Ẑ among the VALID CANDIDATES of the step is the fact
      the verifier would accept... precisely: NN_valid = argmax over the valid candidates of cos(Ẑ, E_ema(fact));
      NN_all = argmax over ALL the facts of the story (premises, every entailed fact, and the not-entailed
      alternatives with one argument swapped, as diag_decode_verify's invalid pool). "idea right" = NN_all is a valid
      candidate. Groups: idea right & sentence right / idea right & sentence wrong (decoder) / idea wrong. Per group:
      cos(Ẑ, NN_all), cos to the 2nd nearest of ALL facts, margin; and the same with the valid pool.
 (iii) for "idea right & sentence wrong": relation of NN_all (expected) vs relation of the decoded fact (written),
      with their frequencies among the training deductions.
 (iv) 5 examples with context.
 Complements (§32.14), on the "idea right" steps (NN_all valid), per family and rank:
 (v)   decoding error rate per bin of margin cos(Ẑ, NN_all) - cos(Ẑ, 2nd);
 (vi)  norm: Ẑ's values rescaled per slot to E's running slot norms (model.e_slot_norm), redecoded: sentence right?
 (vii) interpolation: Z_λ between E_ema(NN_all fact) (λ = 0) and Ẑ (λ = 1), per slot: keys normalize((1-λ) k_E + λ k̂),
       values = normalize((1-λ) v_E/|v_E| + λ v̂/|v̂|) × ((1-λ)|v_E| + λ|v̂|); decoded sentence = NN_all fact?
Space: E_ema values of the "so" form 0 sentence, per-slot normalised, flattened, L2 (diag_pred_cos space).
usage: uv run python scripts/diag_decoder.py <run> [<run> ...] [--n 30] [--summary out.md]
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
ap.add_argument("--n", type=int, default=30, help="stories per family")
ap.add_argument("--summary", default=None)
ap.add_argument("--strip", action="store_true", help="restated reference sentences \"since A , B\" given as \"so B\" "
                "(B in its clause form); outputs decoder_diag_strip.json")
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
FORCE = w2i["so"]
w = lambda toks: " ".join(vocab[t] for t in toks)
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)
rank = lambda j: "1" if j == 0 else ("2" if j == 1 else "3+")


def relation(f):
    if f is None:
        return "unparsed"
    if f[0] == "rel":
        return f[2]
    if f[0] == "gets":
        return "gets-" + (f[2][0] if isinstance(f[2], tuple) else "?")
    return f[0]


def train_relation_freq():
    tr = torch.load("data/reasoning/train.pt", weights_only=False)["docs"]
    c = collections.Counter()
    for d in tr[:20000]:
        for i in d["ideas"]:
            if i["step"] == "deduction" and d["family"] in ("F2", "F3"):
                c[relation(as_fact(i["fact"]))] += 1
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


FREQ = train_relation_freq()


def swapped(facts, prem, truth):
    """not-entailed alternatives: one argument of a fact replaced by a value seen in the same slot (diag_decode_verify)."""
    pool = collections.defaultdict(set)
    for f in list(prem) + list(truth):
        for pos in range(1, len(f)):
            pool[(f[0], pos)].add(f[pos])
    out = []
    for fact in facts:
        for pos in range(1, len(fact)):
            for e in sorted(pool.get((fact[0], pos), ()), key=str):
                if e != fact[pos]:
                    g = list(fact); g[pos] = e; g = tuple(g)
                    if g not in truth:
                        out.append(g)
    return list(dict.fromkeys(out))


def stories():
    per = collections.Counter()
    for doc in docs:
        fam = doc.get("family")
        if fam not in ("F2", "F3") or per[fam] >= a.n:
            continue
        prem = [i["tokens"] for i in doc["ideas"] if i["step"] == "premise"]
        ded = [i for i in doc["ideas"] if i["step"] == "deduction"]
        if len(ded) < 2:
            continue
        if a.strip:
            ded = [{**i, "tokens": [w2i[x] for x in R.render_with_template(as_fact(i["fact"]), 2 + int(i.get("form") or 0)).w]}
                   if i.get("restated") else i for i in ded]
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

    flat = lambda v: F.normalize(F.normalize(v.float(), dim=-1).flatten(1), dim=-1)
    zc = {}

    def zfacts(facts):
        todo = [f for f in facts if f not in zc]
        for s in range(0, len(todo), 256):
            ch = todo[s:s + 256]
            t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(f, 2).w] for f in ch])
            for f, z in zip(ch, flat(model.E_ema(t_, l_)[1])):
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

    ae = collections.defaultdict(collections.Counter)          # (fam, rank | rel) -> n, same
    grp = collections.defaultdict(lambda: collections.defaultdict(list))   # (fam, rank, group) -> measure -> values
    swaps = collections.Counter()
    comp = collections.defaultdict(collections.Counter)          # (fam, rank, measure) -> counts
    BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 2.0]
    LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
    esn = model.e_slot_norm.float()

    def mix(ka, va, kb, vb, lam):
        k_ = F.normalize((1 - lam) * ka.float() + lam * kb.float(), dim=-1)
        na, nb = va.float().norm(dim=-1, keepdim=True), vb.float().norm(dim=-1, keepdim=True)
        v_ = F.normalize((1 - lam) * va.float() / na.clamp_min(1e-6) + lam * vb.float() / nb.clamp_min(1e-6), dim=-1)
        return k_, v_ * ((1 - lam) * na + lam * nb)
    ex = []
    with torch.no_grad():
        for fam, prem, ded, pf in stories():
            truth = V.derivable(fam, pf)
            ref = [as_fact(i["fact"]) for i in ded]
            allf = list(dict.fromkeys(list(pf) + sorted(truth, key=str) + swapped(sorted(truth, key=str), pf, truth)))
            Zall = zfacts(allf)
            for j in range(len(ded)):
                ctx = [list(p) for p in prem] + [list(i["tokens"]) for i in ded[:j]]
                hist = [BOS] + [t for idea in ctx for t in idea]; p = hist[-M:]
                pre = torch.full((1, M), PAD, dtype=torch.long); pre[0, M - len(p):] = torch.tensor(p); pre = pre.to(DEV)
                rk = rank(j)
                # (i) autoencoding of the reference fact
                t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(ref[j], 2).w]])
                ka, va = model.E_ema(t_, l_)
                same = V.parse_sentence(w(decode(pre, ka, va)))[0] == ref[j]
                for key in ((fam, rk), (fam, "rel:" + relation(ref[j])), (fam, "all")):
                    ae[key]["n"] += 1; ae[key]["same"] += same
                # (ii) P's prediction
                cands = sorted(V.valid_next(fam, pf, ref[:j]) - set(ref[:j]), key=str)
                if not cands:
                    continue
                pk, pv = predict(ctx, len(prem))
                out = decode(pre, pk, pv)
                f_out = V.parse_sentence(w(out))[0]
                sent_ok = f_out in cands
                z = flat(pv)[0]
                sa = Zall @ z
                top = sa.topk(2)
                nn_all = allf[int(top.indices[0])]
                sv = zfacts(cands) @ z
                idea_ok = nn_all in cands
                g = "idea_wrong" if not idea_ok else ("idea_right_sent_right" if sent_ok else "idea_right_sent_wrong")
                for key in ((fam, rk, g), (fam, "all", g), ("all", "all", g)):
                    d_ = grp[key]
                    d_["n"].append(1)
                    d_["cos_nn_all"].append(float(top.values[0])); d_["cos_2nd_all"].append(float(top.values[1]))
                    d_["margin_all"].append(float(top.values[0] - top.values[1]))
                    d_["cos_nn_valid"].append(float(sv.max()))
                    if len(cands) > 1:
                        sv2 = sv.topk(2).values
                        d_["margin_valid"].append(float(sv2[0] - sv2[1]))
                if idea_ok:
                    mg = float(top.values[0] - top.values[1])
                    b_ = next(i for i in range(len(BINS) - 1) if mg < BINS[i + 1])
                    lab = f"margin[{BINS[b_]:.2f},{BINS[b_ + 1]:.2f})"
                    # (vi) values at E's slot norms
                    pv_n = F.normalize(pv.float(), dim=-1) * esn[None, :, None] if float(esn.sum()) > 0 else pv
                    ok_norm = V.parse_sentence(w(decode(pre, pk, pv_n.to(pv.dtype))))[0] == nn_all
                    t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(nn_all, 2).w]])
                    kE, vE = model.E_ema(t_, l_)
                    lam_ok = {}
                    for lam in LAMBDAS:
                        kz, vz = mix(kE, vE, pk, pv, lam)
                        lam_ok[lam] = V.parse_sentence(w(decode(pre, kz.to(pk.dtype), vz.to(pv.dtype))))[0] == nn_all
                    for key in ((fam, rk), (fam, "all")):
                        c_ = comp[key]
                        c_[lab + "|n"] += 1; c_[lab + "|err"] += not sent_ok
                        c_["n"] += 1; c_["sent_ok"] += sent_ok; c_["sent_is_nn"] += f_out == nn_all
                        c_["norm_ok"] += ok_norm
                        for lam, ok_ in lam_ok.items():
                            c_[f"lam{lam}"] += ok_
                if g == "idea_right_sent_wrong":
                    swaps[(fam, relation(nn_all), relation(f_out))] += 1
                if not idea_ok:
                    comp[(fam, rk)]["idea_wrong_nn_said"] += nn_all in ref[:j]
                    if len(ex) < 8:
                        ex.append({"fam": fam, "rank": rk, "context_last": [w(i["tokens"]) for i in ded[max(0, j - 2):j]],
                                   "idea_nn": " ".join(R.render_with_template(nn_all, 2).w), "decoded": w(out),
                                   "cos_nn": round(float(top.values[0]), 3), "margin": round(float(top.values[0] - top.values[1]), 3)})
    res = {"autoencode": {f"{k[0]}|{k[1]}": dict(c) for k, c in ae.items()},
           "groups": {"|".join(k): {m: (len(v) if m == "n" else st.mean(v)) for m, v in d_.items() if v} for k, d_ in grp.items()},
           "swaps": [{"fam": k[0], "expected": k[1], "written": k[2], "n": n, "freq_expected": FREQ.get(k[1], 0.0),
                      "freq_written": FREQ.get(k[2], 0.0)} for k, n in swaps.most_common()],
           "examples": ex, "complements": {f"{k[0]}|{k[1]}": dict(c_) for k, c_ in comp.items()},
           "e_slot_norm": esn.tolist()}
    json.dump({"run": run, **res}, open(f"{run}/decoder_diag{'_strip' if a.strip else ''}.json", "w"), indent=1)
    c = ae[("F2", "3+")]
    print(f"{run}: autoencode F2 rank 3+ {c['same'] / max(1, c['n']):.3f}  F3 all {ae[('F3', 'all')]['same'] / max(1, ae[('F3', 'all')]['n']):.3f}", flush=True)
    del model
    return res


allr = {r: run_one(r) for r in a.runs}
if a.summary:
    L = ["## (i) autoencodage D(E_ema(fait de référence)) = ce fait, contexte de référence", "",
         "| run | F2 r1 | F2 r2 | F2 r3+ | F3 r1 | F3 r2 | F3 r3+ |", "|---|---|---|---|---|---|---|"]
    for r, res in allr.items():
        cells = []
        for fam in ("F2", "F3"):
            for rk in ("1", "2", "3+"):
                c = res["autoencode"].get(f"{fam}|{rk}", {"n": 0})
                cells.append(f"{c.get('same', 0) / c['n']:.3f} (n={c['n']})" if c["n"] else "—")
        L.append(f"| {r.split('/')[-1]} | " + " | ".join(cells) + " |")
    L += ["", "autoencodage par relation :"]
    for r, res in allr.items():
        rels = {k: v for k, v in res["autoencode"].items() if "|rel:" in k}
        L.append(f"- {r.split('/')[-1]} : " + ", ".join(f"{k.replace('|rel:', ' ')} {v.get('same', 0) / v['n']:.2f} (n={v['n']})"
                                                     for k, v in sorted(rels.items())))
    L += ["", "## (ii) Ẑ par groupe (NN parmi TOUS les faits de l'histoire ; marge = cos 1er − cos 2e)", "",
          "| run | famille / rang | groupe | n | cos NN | cos 2e | marge | marge parmi valides |", "|---|---|---|---|---|---|---|---|"]
    for r, res in allr.items():
        for k, d_ in sorted(res["groups"].items()):
            fam, rk, g = k.split("|")
            if rk not in ("3+", "all"):
                continue
            mv = f"{d_['margin_valid']:.3f}" if "margin_valid" in d_ else "—"
            L.append(f"| {r.split('/')[-1]} | {fam} {rk} | {g} | {d_['n']} | {d_['cos_nn_all']:.3f} | {d_['cos_2nd_all']:.3f} | "
                     f"{d_['margin_all']:.3f} | {mv} |")
    L += ["", "## (iii) idée juste, phrase fausse : relation attendue → écrite (fréquence à l'entraînement)", ""]
    for r, res in allr.items():
        L.append(f"- {r.split('/')[-1]} : " + "; ".join(f"{s['fam']} {s['expected']} ({s['freq_expected']:.3f}) → {s['written']} "
                                                     f"({s['freq_written']:.3f}) ×{s['n']}" for s in res["swaps"][:6]))
    L += ["", "## (v)-(vii) pas « idée juste » (NN de Ẑ parmi tous les faits = candidate valide)", "",
          "| run | famille rang | n | phrase juste (= valide) | phrase = NN | valeurs aux normes de E : phrase = NN | λ=0 (autoenc. NN) | 0,25 | 0,5 | 0,75 | λ=1 (Ẑ) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, res in allr.items():
        for k, c_ in sorted(res["complements"].items()):
            n = c_.get("n", 0)
            if not n:
                continue
            f = lambda x: f"{c_.get(x, 0) / n:.3f}"
            L.append(f"| {r.split('/')[-1]} | {k.replace('|', ' ')} | {n} | {f('sent_ok')} | {f('sent_is_nn')} | {f('norm_ok')} | "
                     + " | ".join(f(f"lam{l}") for l in (0.0, 0.25, 0.5, 0.75, 1.0)) + " |")
    L += ["", "taux d'erreur du décodage par tranche de marge (NN − 2e), pas « idée juste » :"]
    for r, res in allr.items():
        for k, c_ in sorted(res["complements"].items()):
            if not k.endswith("3+") and not k.endswith("all"):
                continue
            bins = sorted({x.split("|")[0] for x in c_ if x.startswith("margin")})
            L.append(f"- {r.split('/')[-1]} {k.replace('|', ' ')} : " + ", ".join(
                f"{b} {c_[b + '|err']}/{c_[b + '|n']}" for b in bins))
    L += ["", "## (iv) exemples", ""]
    for r, res in allr.items():
        for e in res["examples"][:5]:
            L.append(f"- {r.split('/')[-1]} {e['fam']} rang {e['rank']} : contexte « {' | '.join(e['context_last'])} » ; "
                     f"idée (NN) « {e['idea_nn']} » ; écrit « {e['decoded']} » (cos {e['cos_nn']}, marge {e['margin']})")
    open(a.summary, "w").write("\n".join(L) + "\n")
    print("\n".join(L))

"""Mémo §32.21 (4): tipping point of the decoding along Ẑ -> E_ema(nearest INVALID fact), reference context, no training.
Per step with a valid decoding of P's prediction Ẑ: g = the not-entailed fact (one argument of an entailed fact swapped,
as diag_decode_verify's invalid pool) nearest to Ẑ (diag_pred_cos space); Z_λ = per slot, keys normalize((1-λ) k̂ + λ k_g),
values normalize((1-λ) v̂/|v̂| + λ v_g/|v_g|) × ((1-λ)|v̂| + λ|v_g|), λ = 0, 0.05, ..., 1; tipping λ* = the first λ whose
greedy decoding is no longer a valid next deduction (1.05 if never). Distribution of λ* per family; cos(Ẑ, g).
usage: uv run python scripts/diag_tipping.py <run> [<run> ...] [--n 30] [--summary out.md]
"""
import argparse, collections, json, os, statistics as st, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--n", type=int, default=30)
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}; FORCE = w2i["so"]
w = lambda t: " ".join(vocab[x] for x in t)
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)
LAMS = [i / 20 for i in range(21)]


def swapped(facts, prem, truth):
    pool = collections.defaultdict(set)
    for f in list(prem) + list(truth):
        for pos in range(1, len(f)):
            pool[(f[0], pos)].add(f[pos])
    # F3: the deduction predicates (canopen, gets) never appear in the premises: draw their arguments from the premises
    persons = {p[1] for p in prem if p[0] == "has"}
    boxes = {p[1] for p in prem if p[0] == "holds"} | {p[2] for p in prem if p[0] == "opens"}
    objs = {p[2] for p in prem if p[0] == "holds"} | {("key", p[1]) for p in prem if p[0] == "opens"}
    if persons:
        pool[("canopen", 1)] |= persons; pool[("canopen", 2)] |= boxes
        pool[("gets", 1)] |= persons; pool[("gets", 2)] |= objs
    out = []
    for fact in facts:
        for pos in range(1, len(fact)):
            for e in sorted(pool.get((fact[0], pos), ()), key=str):
                if e != fact[pos]:
                    g = list(fact); g[pos] = e; g = tuple(g)
                    if g not in truth:
                        out.append(g)
    return list(dict.fromkeys(out))


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV); model.eval()
    M = cfg["model"]["m_prefix"]
    endm = torch.zeros(len(vocab), dtype=torch.bool, device=DEV); endm[w2i["."]] = True

    def enc_tokens(ideas):
        L = max(len(t) for t in ideas)
        tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
        for i, t in enumerate(ideas):
            tok[i, :len(t)] = torch.tensor(t)
        return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)

    flat = lambda v: F.normalize(F.normalize(v.float(), dim=-1).flatten(1), dim=-1)

    def predict(story, n_prem):
        tok, lens = enc_tokens(story); k, v = model.E(tok, lens); n = len(story)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        return model.predict(k, v, {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                                    "idea_pos": torch.tensor([n], device=DEV), "idea_role": (torch.arange(n, device=DEV) >= n_prem).long()})

    def decode(prefix, pk, pv, max_len=48):
        toks = [FORCE]
        for s_ in range(1, max_len):
            win = torch.cat([prefix, torch.tensor([toks + [PAD]], device=DEV)], 1)
            logits, tr = model.dec(win, torch.tensor([s_ + 1], device=DEV), M, pk, pv, model.O(win, M))
            if endm[toks[-1]] or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)):
                break
            lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf"); toks.append(int(lg.argmax()))
        return toks

    def mix(ka, va, kb, vb, lam):
        k_ = F.normalize((1 - lam) * ka.float() + lam * kb.float(), dim=-1)
        na, nb = va.float().norm(dim=-1, keepdim=True), vb.float().norm(dim=-1, keepdim=True)
        v_ = F.normalize((1 - lam) * va.float() / na.clamp_min(1e-6) + lam * vb.float() / nb.clamp_min(1e-6), dim=-1)
        return k_.to(ka.dtype), (v_ * ((1 - lam) * na + lam * nb)).to(va.dtype)

    lam_of = collections.defaultdict(list); cosg = collections.defaultdict(list)
    per = collections.Counter()
    with torch.no_grad():
        for doc in docs:
            fam = doc.get("family")
            if per[fam] >= a.n:
                continue
            ded = [i for i in doc["ideas"] if i["step"] == "deduction"]
            if len(ded) < 2:
                continue
            per[fam] += 1
            prem = [i["tokens"] for i in doc["ideas"] if i["step"] == "premise"]
            pf = V.doc_premises(doc); truth = V.derivable(fam, pf); ref = [as_fact(i["fact"]) for i in ded]
            inval = swapped(sorted(truth, key=str), pf, truth)
            if not inval:
                continue
            t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(g, 2).w] for g in inval])
            kI, vI = model.E_ema(t_, l_); zI = flat(vI)
            for j in range(len(ded)):
                ctx = [list(p) for p in prem] + [list(i["tokens"]) for i in ded[:j]]
                cands = V.valid_next(fam, pf, ref[:j]) - set(ref[:j])
                hist = [BOS] + [t for idea in ctx for t in idea]; p = hist[-M:]
                pre = torch.full((1, M), PAD, dtype=torch.long); pre[0, M - len(p):] = torch.tensor(p); pre = pre.to(DEV)
                pk, pv = predict(ctx, len(prem))
                if V.parse_sentence(w(decode(pre, pk, pv)))[0] not in cands:
                    continue
                s = zI @ flat(pv)[0]; gi = int(s.argmax()); cosg[fam].append(float(s[gi]))
                star = 1.05
                for lam in LAMS[1:]:
                    kz, vz = mix(pk, pv, kI[gi:gi + 1], vI[gi:gi + 1], lam)
                    if V.parse_sentence(w(decode(pre, kz, vz)))[0] not in cands:
                        star = lam; break
                lam_of[fam].append(star)
    res = {}
    for fam, L in lam_of.items():
        q = sorted(L)
        res[fam] = {"n": len(L), "median": q[len(q) // 2], "q10": q[len(q) // 10], "q90": q[(9 * len(q)) // 10],
                    "mean": st.mean(L), "never": sum(x > 1 for x in L) / len(L), "cos_to_g": st.mean(cosg[fam]),
                    "hist": dict(collections.Counter(round(x, 2) for x in L))}
    json.dump({"run": run, "results": res}, open(f"{run}/tipping.json", "w"), indent=1)
    print(run + ": " + "  ".join(f"{f} median λ* {r['median']:.2f} (q10 {r['q10']:.2f}, q90 {r['q90']:.2f}, n={r['n']})" for f, r in sorted(res.items())), flush=True)
    del model
    return res


allr = {r: run_one(r) for r in a.runs}
if a.summary:
    L = ["| run | famille | n | λ* médian | q10 | q90 | moyenne | jamais basculé | cos(Ẑ, invalide le plus proche) |", "|---|---|---|---|---|---|---|---|---|"]
    for r, res in allr.items():
        for f in ("F1", "F2", "F3"):
            x = res.get(f)
            if x:
                L.append(f"| {r.split('/')[-1]} | {f} | {x['n']} | {x['median']:.2f} | {x['q10']:.2f} | {x['q90']:.2f} | {x['mean']:.3f} | {x['never']:.3f} | {x['cos_to_g']:.3f} |")
    open(a.summary, "w").write("\n".join(L) + "\n"); print("\n".join(L))

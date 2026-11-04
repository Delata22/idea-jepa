"""F2, per rank: is a wrong relation P's fault (the idea) or the decoder's (the sentence)?
For every F2 deduction B_j of the gen_eval stories whose antecedent A has another relation (reference context: premises +
reference deductions < j, story surfaces; at rank 1 this equals the free-generation context):
  idea      nearest of {B_j, B'_j (same people, A's relation), B''_j (third relation)} to P's prediction (E_ema "so" 0 values)
  sentence  greedy decoding of the prediction ("so" forced), verifier verdict given the reference deductions said
Crosstab per rank (1 / 2 / 3+): idea right & sentence valid / idea right & sentence wrong (decoder) / idea wrong (P) ...
usage: uv run python scripts/diag_rank_relation.py <run> [<run> ...] [--n 30]
"""
import argparse, collections, json, os, random, sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.data import reasoning as R, verifier as V
from ideajepa.train import load_model

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
w = lambda t: " ".join(vocab[x] for x in t)
RELS = ["parent", "grandparent", "sibling", "pibling"]
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)
PAD, BOS, EOS = 0, 1, 2


def stories():
    n = 0
    for doc in docs:
        if doc["family"] != "F2":
            continue
        ded = [i for i in doc["ideas"] if i.get("depth", -1) >= 0]
        if len(ded) < 2:
            continue
        n += 1
        if n > a.n:
            break
        yield doc


for run in a.runs:
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV); model.eval()
    M = cfg["model"]["m_prefix"]; rng = random.Random(0)
    is_end = torch.zeros(len(vocab), dtype=torch.bool, device=DEV); is_end[w2i["."]] = True

    def enc_tokens(ideas):
        L = max(map(len, ideas)); tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
        for i, t in enumerate(ideas): tok[i, :len(t)] = torch.tensor(t)
        return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)

    def zf(fs):
        t_, l_ = enc_tokens([[w2i[x] for x in R.render_with_template(f, 2).w] for f in fs])
        _, v_ = model.E_ema(t_, l_)
        return F.normalize(F.normalize(v_.float(), dim=-1).flatten(1), dim=-1)

    cnt = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        for doc in stories():
            prem_i = [i for i in doc["ideas"] if i.get("depth", -1) < 0]
            ded = [i for i in doc["ideas"] if i.get("depth", -1) >= 0]
            prem_facts = V.doc_premises(doc)
            facts = [as_fact(i["fact"]) for i in doc["ideas"]]
            n_prem = len(prem_i)
            for j, idea in enumerate(ded):
                B = as_fact(idea["fact"]); ii = n_prem + j
                A = (V.parse_sentence(w(idea["tokens"]))[1] if idea.get("restated") else R.antecedent("F2", B, facts[:ii]))
                if A is None or A[2] == B[2]:
                    continue
                ctx = [list(i["tokens"]) for i in prem_i] + [list(i["tokens"]) for i in ded[:j]]
                tok, lens = enc_tokens(ctx); k, v = model.E(tok, lens); n = len(ctx)
                di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
                pk, pv = model.predict(k, v, {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                                              "idea_pos": torch.tensor([n], device=DEV),
                                              "idea_role": (torch.arange(n, device=DEV) >= n_prem).long()})
                trip = [B, ("rel", B[1], A[2], B[3]), ("rel", B[1], rng.choice([x for x in RELS if x not in (B[2], A[2])]), B[3])]
                p = F.normalize(F.normalize(pv[0].float(), dim=-1).flatten(), dim=0)
                idea_cls = ["right", "relA", "other"][int((zf(trip) @ p).argmax())]
                hist = [BOS] + [t for x in ctx for t in x]; pp = hist[-M:]
                pre = torch.full((1, M), PAD, dtype=torch.long); pre[0, M - len(pp):] = torch.tensor(pp); pre = pre.to(DEV)
                toks = [w2i["so"]]
                for s_ in range(1, 48):
                    win = torch.cat([pre, torch.tensor([toks + [PAD]], device=DEV)], 1)
                    lg, tr = model.dec(win, torch.tensor([s_ + 1], device=DEV), M, pk, pv, model.O(win, M))
                    if is_end[toks[-1]] or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)):
                        break
                    l = lg[0, s_].float(); l[[PAD, BOS, EOS]] = float("-inf"); toks.append(int(l.argmax()))
                vd = V.judge("F2", prem_facts, [w(i["tokens"]) for i in ded[:j]] + [w(toks)])[-1]["verdict"]
                f_out = V.parse_sentence(w(toks))[0]
                sent_cls = "valid" if vd.startswith("valid") else (
                    "relA" if f_out is not None and len(f_out) == 4 and (f_out[1], f_out[3]) == (B[1], B[3]) and f_out[2] == A[2] else "wrong")
                rk = "1" if j == 0 else ("2" if j == 1 else "3+")
                for r_ in (rk, "all"):
                    cnt[r_][(idea_cls, sent_cls)] += 1
    out = {r_: {f"{i}|{s}": c for (i, s), c in cc.items()} for r_, cc in cnt.items()}
    json.dump({"run": run, "results": out}, open(f"{run}/rank_relation.json", "w"), indent=1)
    for r_ in ("1", "2", "3+", "all"):
        cc = cnt[r_]; tot = sum(cc.values())
        if not tot:
            continue
        ir = sum(v for (i, s), v in cc.items() if i == "right")
        dec_err = sum(v for (i, s), v in cc.items() if i == "right" and s != "valid")
        print(f"{Path(run).name} rank {r_:3s} n={tot:3d}  idea right {ir / tot:.2f} (relA {sum(v for (i, s), v in cc.items() if i == 'relA') / tot:.2f})"
              f"  sentence valid {sum(v for (i, s), v in cc.items() if s == 'valid') / tot:.2f}"
              f"  idea right & sentence wrong (decoder) {dec_err / tot:.2f}"
              f"  idea wrong & sentence wrong (P) {sum(v for (i, s), v in cc.items() if i != 'right' and s != 'valid') / tot:.2f}"
              f"  sentence with A's relation {sum(v for (i, s), v in cc.items() if s == 'relA') / tot:.2f}", flush=True)

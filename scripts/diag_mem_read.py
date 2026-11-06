"""What does P read in its memory? Inference only, reference context (premises + reference deductions < j,
story surfaces), one P pass per deduction step; the query block n = len(context) predicts the next idea.
 (1) constancy of the queries: mean cos between queries at fixed (layer, head, slot), across the steps of a story vs
     across stories (random pairs).
 (2) linear probe (logistic regression; half of the F1 stories to fit, half to test) from the value READ at the query
     block (raw delta-rule read, one layer, all heads and slots concatenated) to the current location of EACH object of
     the story (one probe per object name). Accuracy for the object of the reference deduction and for the other objects,
     vs the majority class of the test set.
 (3) memory swap: the query block reads the final state S of ANOTHER story's reference context (same family, next story);
     (3 bis) reads S = 0. Validity of the decoded sentence (F1 / F2 / F3) vs normal, same steps.
usage: uv run python scripts/diag_mem_read.py <run> [<run> ...] [--n 30] [--summary out.md]
"""
import argparse
import collections
import json
import os
import random
import statistics as st
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--n", type=int, default=30, help="stories per family (F1: x2 for the probe)")
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
FORCE = w2i["so"]
w = lambda t: " ".join(vocab[x] for x in t)
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)


def stories():
    per = collections.Counter()
    out = collections.defaultdict(list)
    for doc in docs:
        fam = doc.get("family")
        if per[fam] >= (2 * a.n if fam == "F1" else a.n):
            continue
        ded = [i for i in doc["ideas"] if i["step"] == "deduction"]
        if len(ded) < 2:
            continue
        per[fam] += 1
        out[fam].append(([i["tokens"] for i in doc["ideas"] if i["step"] == "premise"], ded, V.doc_premises(doc)))
    return out


STORIES = stories()


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    M, r = cfg["model"]["m_prefix"], model.P.r
    mems = list(model.P.mems)
    assert all(m.decay is None for m in mems), "state recurrence written for memories without decay"
    endm = torch.zeros(len(vocab), dtype=torch.bool, device=DEV); endm[w2i["."]] = True

    def enc_tokens(ideas):
        L = max(len(t) for t in ideas)
        tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
        for i, t in enumerate(ideas):
            tok[i, :len(t)] = torch.tensor(t)
        return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)

    def run_P(story, n_prem, read_S=None, capture=False):
        """P on the story; read_S: list per layer of a state [H, dv, dk] read by the query block (None = normal)."""
        tok, lens = enc_tokens(story)
        k, v = model.E(tok, lens)
        n = len(story)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                 "idea_pos": torch.tensor([n], device=DEV), "idea_role": (torch.arange(n, device=DEV) >= n_prem).long()}
        for li, m in enumerate(mems):
            m.capture = capture
            m.read_override = [(0, n * r, r, read_S[li])] if read_S is not None else None
        try:
            return model.predict(k, v, batch)
        finally:
            for m in mems:
                m.read_override = None; m.capture = False

    def states_and_reads(n):
        """after a capture pass: per layer, final state S [H, dv, dk], raw read at the query block [H, r, dv], queries."""
        Ss, reads, qs = [], [], []
        for m in mems:
            q, k = m.last_qk; v, beta = m.last_v, m.last_beta          # [1, H, T, d], beta [1, H, T]
            H, T = q.shape[1], q.shape[2]
            S = torch.zeros(H, v.shape[-1], k.shape[-1], device=DEV)
            for t in range(n * r):                                     # writes before the query block
                kt, vt, bt = k[0, :, t], v[0, :, t], beta[0, :, t, None, None]
                S = S - bt * torch.einsum("hvk,hk,hj->hvj", S, kt, kt) + bt * torch.einsum("hv,hk->hvk", vt, kt)
            Sq = S.clone()
            for t in range(n * r, T):                                  # the query block's own writes (as in the forward)
                kt, vt, bt = k[0, :, t], v[0, :, t], beta[0, :, t, None, None]
                Sq = Sq - bt * torch.einsum("hvk,hk,hj->hvj", Sq, kt, kt) + bt * torch.einsum("hv,hk->hvk", vt, kt)
            Ss.append(Sq)
            qb = q[0, :, n * r:(n + 1) * r]                               # [H, r, dk]
            reads.append(torch.einsum("hvk,hrk->hrv", S, qb))            # read of the state before the block
            qs.append(qb)
        return Ss, reads, qs

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

    def prefix_of(ctx):
        hist = [BOS] + [t for idea in ctx for t in idea]; p = hist[-M:]
        pre = torch.full((1, M), PAD, dtype=torch.long); pre[0, M - len(p):] = torch.tensor(p)
        return pre.to(DEV)

    Q = collections.defaultdict(list)          # (layer) -> list of (story id, [H, r, dk])
    probe_rows = []                            # (story id, layer reads, object, location, is_target)
    swap = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        # final states of every story's full reference context (for the swap)
        final_S = {}
        for fam, sts in STORIES.items():
            for si, (prem, ded, pf) in enumerate(sts[:a.n]):
                ctx = [list(p) for p in prem] + [list(i["tokens"]) for i in ded[:-1]]
                run_P(ctx, len(prem), capture=True)
                final_S[(fam, si)] = states_and_reads(len(ctx))[0]
        for fam, sts in STORIES.items():
            for si, (prem, ded, pf) in enumerate(sts):
                final = R.final_state_f1(pf) if fam == "F1" else None
                ref = [as_fact(i["fact"]) for i in ded]
                for j in range(len(ded)):
                    ctx = [list(p) for p in prem] + [list(i["tokens"]) for i in ded[:j]]
                    pk, pv = run_P(ctx, len(prem), capture=True)
                    own_S, reads, qs = states_and_reads(len(ctx))
                    for li, qb in enumerate(qs):
                        Q[li].append(((fam, si), qb.cpu()))
                    if fam == "F1":
                        for o, loc in final.items():
                            probe_rows.append(((fam, si), [x.flatten().cpu().numpy() for x in reads], o, loc, o == ref[j][1]))
                    if si >= a.n:
                        continue
                    pre = prefix_of(ctx)
                    said = [w(i["tokens"]) for i in ded[:j]]
                    other = final_S[(fam, (si + 1) % min(a.n, len(sts)))]
                    for cond, S_ in (("normal", None), ("self", own_S), ("swap", other), ("zero", [torch.zeros_like(x) for x in other])):
                        if S_ is not None:
                            pk, pv = run_P(ctx, len(prem), read_S=S_)
                        out = decode(pre, pk, pv)
                        vd = V.judge(fam, pf, said + [w(out)])[-1]["verdict"]
                        swap[(fam, cond)]["n"] += 1; swap[(fam, cond)]["valid"] += vd.startswith("valid")
    # (1) query constancy
    rng = random.Random(0)
    const = {}
    for li, rows in Q.items():
        within, across = [], []
        by = collections.defaultdict(list)
        for sid, qb in rows:
            by[sid].append(qb)
        for sid, qbs in by.items():
            for x in range(len(qbs) - 1):
                within.append(float((qbs[x] * qbs[x + 1]).sum(-1).mean()))
        for _ in range(2000):
            (s1, q1), (s2, q2) = rng.sample(rows, 2)
            if s1 != s2:
                across.append(float((q1 * q2).sum(-1).mean()))
        const[li] = {"within_story": st.mean(within), "across_stories": st.mean(across)}
    # (2) probe
    from sklearn.linear_model import LogisticRegression
    sids = sorted({x[0] for x in probe_rows}, key=str)
    fit = set(sids[: len(sids) // 2])
    probe = {}
    for li in range(len(mems)):
        acc = collections.defaultdict(list)
        for obj in sorted({x[2] for x in probe_rows}):
            rows = [x for x in probe_rows if x[2] == obj]
            tr = [x for x in rows if x[0] in fit]; te = [x for x in rows if x[0] not in fit]
            if len({x[3] for x in tr}) < 2 or not te:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(np.stack([x[1][li] for x in tr]), [x[3] for x in tr])
            pred = clf.predict(np.stack([x[1][li] for x in te]))
            maj = collections.Counter(x[3] for x in tr).most_common(1)[0][0]
            for x, p in zip(te, pred):
                g = "target" if x[4] else "other"
                acc[g].append(p == x[3]); acc[g + "_majority"].append(maj == x[3])
        probe[li] = {k: float(np.mean(v)) for k, v in acc.items()} | {"n_target": len(acc["target"]), "n_other": len(acc["other"])}
    res = {"query_constancy": const, "probe": probe,
           "swap": {f"{k[0]}|{k[1]}": {**dict(c), "valid_rate": c["valid"] / c["n"]} for k, c in swap.items()}}
    json.dump({"run": run, **res}, open(f"{run}/mem_read.json", "w"), indent=1)
    print(f"{run}: swap " + " ".join(f"{k} {v['valid_rate']:.3f}" for k, v in sorted(res["swap"].items())), flush=True)
    del model
    return res


allr = {r: run_one(r) for r in a.runs}
if a.summary:
    L = ["## (1) constance des requêtes (cos moyen à couche/tête/slot fixés)", "", "| run | couche | entre pas d'une histoire | entre histoires |", "|---|---|---|---|"]
    for r, res in allr.items():
        for li, c in sorted(res["query_constancy"].items()):
            L.append(f"| {r.split('/')[-1]} | {li} | {c['within_story']:.3f} | {c['across_stories']:.3f} |")
    L += ["", "## (2) sonde linéaire lecture → lieu de chaque objet (F1, test sur la moitié des histoires)", "",
          "| run | couche | objet de la déduction | classe majoritaire | autres objets | classe majoritaire |", "|---|---|---|---|---|---|"]
    for r, res in allr.items():
        for li, p in sorted(res["probe"].items()):
            L.append(f"| {r.split('/')[-1]} | {li} | {p.get('target', float('nan')):.3f} (n={p['n_target']}) | {p.get('target_majority', float('nan')):.3f} | "
                     f"{p.get('other', float('nan')):.3f} (n={p['n_other']}) | {p.get('other_majority', float('nan')):.3f} |")
    L += ["", "## (3) échange de mémoire / mémoire à zéro (lecture du bloc de requête), contexte de référence, % valides", "",
          "| run | F1 normal | F1 échange | F1 zéro | F2 normal | F2 échange | F2 zéro | F3 normal | F3 échange | F3 zéro |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r, res in allr.items():
        s = res["swap"]
        L.append(f"| {r.split('/')[-1]} | " + " | ".join(f"{s.get(f'{fam}|{c}', {}).get('valid_rate', float('nan')):.3f}"
                                                        for fam in ("F1", "F2", "F3") for c in ("normal", "swap", "zero")) + " |")
    open(a.summary, "w").write("\n".join(L) + "\n")
    print("\n".join(L))

"""Linear probe on the WHOLE memory state S (not the value read), F1, reference context, per memory layer.
At each F1 deduction step: S = state after the writes before the query block (delta recurrence, as diag_mem_read),
flattened, reduced by a fixed random projection to --dim; one logistic probe per object name and per target: (a) CURRENT
location of the object (given the premises), (b) INITIAL location. Half of the stories to fit, half to test. Accuracy for
immobile / moved objects (moved = its useful premise is a "goes") vs the majority class of the training rows.
usage: uv run python scripts/diag_mem_state_probe.py <run> [<run> ...] [--n 60] [--dim 512] [--summary out.md]
"""
import argparse, collections, os, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--n", type=int, default=60)
ap.add_argument("--dim", type=int, default=512); ap.add_argument("--summary", default=None)
ap.add_argument("--mlp", action="store_true", help="also a non-linear probe (one hidden layer of 128, ReLU)")
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w2i = {x: i for i, x in enumerate(vocab)}
w = lambda t: " ".join(vocab[x] for x in t)
FORCE = w2i["so"]
STORIES = []
for doc in docs:
    if doc.get("family") != "F1" or len(STORIES) >= a.n:
        continue
    ded = [i for i in doc["ideas"] if i["step"] == "deduction"]
    if ded:
        STORIES.append(([i["tokens"] for i in doc["ideas"] if i["step"] == "premise"], ded, V.doc_premises(doc)))


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

    rows = []   # (story, layer states, object, current loc, initial loc, moved)
    with torch.no_grad():
        for si, (prem, ded, pf) in enumerate(STORIES):
            final = R.final_state_f1(pf); up = R.f1_useful_premises(pf)
            init = {p[1]: p[2] for p in pf if p[0] == "objin"}
            for j in range(len(ded)):
                ctx = [list(p) for p in prem] + [list(i["tokens"]) for i in ded[:j]]
                run_P(ctx, len(prem), capture=True)
                n = len(ctx)
                Ss = []
                for m in mems:
                    q, k = m.last_qk; v, beta = m.last_v, m.last_beta
                    S = torch.zeros(q.shape[1], v.shape[-1], k.shape[-1], device=DEV)
                    for t in range(n * r):
                        kt, vt, bt = k[0, :, t], v[0, :, t], beta[0, :, t, None, None]
                        S = S - bt * torch.einsum("hvk,hk,hj->hvj", S, kt, kt) + bt * torch.einsum("hv,hk->hvk", vt, kt)
                    Ss.append(S.flatten().cpu())
                for o in final:
                    rows.append((si, Ss, o, final[o], init.get(o), up.get(o) is not None and pf[up[o]][0] == "goes"))
    from sklearn.linear_model import LogisticRegression
    fit = set(range(len(STORIES) // 2))
    rng = np.random.default_rng(0)
    res = {}
    for li in range(len(mems)):
        D = rows[0][1][li].numel()
        Pm = rng.standard_normal((D, a.dim)).astype(np.float32) / np.sqrt(a.dim)
        X = {id(r_): (r_[1][li].numpy() @ Pm) for r_ in rows}
        acc = collections.defaultdict(list)
        from sklearn.neural_network import MLPClassifier
        kinds = [("lin", lambda: LogisticRegression(max_iter=3000, C=0.1))]
        if a.mlp:
            kinds.append(("mlp", lambda: MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=2000, random_state=0)))
        for (pk_, mk_), (tname, ti) in [(k_, t_) for k_ in kinds for t_ in (("current", 3), ("initial", 4))]:
            for obj in sorted({r_[2] for r_ in rows}):
                rr = [r_ for r_ in rows if r_[2] == obj]
                tr = [r_ for r_ in rr if r_[0] in fit]; te = [r_ for r_ in rr if r_[0] not in fit]
                if len({r_[ti] for r_ in tr}) < 2 or not te:
                    continue
                clf = mk_().fit(np.stack([X[id(r_)] for r_ in tr]), [r_[ti] for r_ in tr])
                pred = clf.predict(np.stack([X[id(r_)] for r_ in te]))
                maj = collections.Counter(r_[ti] for r_ in tr).most_common(1)[0][0]
                for r_, p_ in zip(te, pred):
                    g = "moved" if r_[5] else "immobile"
                    acc[(pk_ + ":" + tname, g)].append(p_ == r_[ti]); acc[(pk_ + ":" + tname, g, "maj")].append(maj == r_[ti])
        res[li] = {"|".join(k): (float(np.mean(v)), len(v)) for k, v in acc.items()}
    print(run + ": " + "  ".join(f"L{li} cur imm {r_.get('lin:current|immobile', (0, 0))[0]:.2f} mov {r_.get('lin:current|moved', (0, 0))[0]:.2f}"
                                 for li, r_ in res.items()), flush=True)
    del model
    return res


allr = {r: run_one(r) for r in a.runs}
if a.summary:
    L = ["| run | sonde | couche | lieu COURANT immobiles (maj.) | lieu COURANT déplacés (maj.) | lieu INITIAL immobiles (maj.) | lieu INITIAL déplacés (maj.) |",
         "|---|---|---|---|---|---|---|"]
    for r, res in allr.items():
        for pk_ in ("lin", "mlp"):
            for li, x in res.items():
                if f"{pk_}:current|immobile" not in x:
                    continue
                f = lambda t, g: (f"{x.get(f'{pk_}:{t}|{g}', (float('nan'), 0))[0]:.2f} ({x.get(f'{pk_}:{t}|{g}|maj', (float('nan'), 0))[0]:.2f}, "
                                  f"n={x.get(f'{pk_}:{t}|{g}', (0, 0))[1]})")
                L.append(f"| {r.split('/')[-1]} | {pk_} | {li} | {f('current', 'immobile')} | {f('current', 'moved')} | {f('initial', 'immobile')} | {f('initial', 'moved')} |")
    open(a.summary, "w").write("\n".join(L) + "\n"); print("\n".join(L))

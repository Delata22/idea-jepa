"""Mémo §34 (reliable version of the memory-state probe): ONE probe for all objects, per memory layer, input
[random projection of the whole state S ; token embedding of the object], fitted on --n_fit TRAIN F1 stories and tested on
--n_test VAL stories. Targets: CURRENT location of the object (its final location given the premises) and its INITIAL
location. Reference context; S = the delta-rule state after the writes of the ideas before the block that predicts a
deduction (one P pass on the whole story: causal, so the states equal the step-by-step ones). Accuracy for immobile / moved
objects (moved = its useful premise is a "goes") vs the majority class of the fitting rows of the same group.
Probes: multinomial logistic regression (lin) and a one-hidden-layer MLP (256, ReLU).
usage: uv run python scripts/diag_mem_state_probe2.py <run> [<run> ...] [--n_fit 1000] [--n_test 300] [--dim 512] [--summary out.md]
"""
import argparse, collections, os, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V
from ideajepa.train import load_model

PAD = 0
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+")
ap.add_argument("--n_fit", type=int, default=1000); ap.add_argument("--n_test", type=int, default=300)
ap.add_argument("--dim", type=int, default=512); ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")


def stories(split, n):
    d = torch.load(f"data/reasoning/{split}.pt", weights_only=False)
    out = []
    for doc in d["docs"]:
        if doc.get("family") != "F1" or len(out) >= n:
            continue
        if any(i["step"] == "deduction" for i in doc["ideas"]):
            out.append(doc)
    return out, d["vocab"]


FIT, vocab = stories("train", a.n_fit)
TEST, _ = stories("val", a.n_test)
w2i = {x: i for i, x in enumerate(vocab)}


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    r = model.P.r
    mems = list(model.P.mems)
    assert all(m.decay is None for m in mems)
    rng = np.random.default_rng(0)
    proj = None

    def enc_tokens(ideas):
        L = max(len(t) for t in ideas)
        tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
        for i, t in enumerate(ideas):
            tok[i, :len(t)] = torch.tensor(t)
        return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)

    def rows_of(docs):
        nonlocal proj
        rows = []
        with torch.no_grad():
            for doc in docs:
                ideas = doc["ideas"]
                pf = V.doc_premises(doc)
                final = R.final_state_f1(pf); up = R.f1_useful_premises(pf)
                init = {p[1]: p[2] for p in pf if p[0] == "objin"}
                ded_blocks = [n for n, i in enumerate(ideas) if i["step"] == "deduction"]
                tok, lens = enc_tokens([i["tokens"] for i in ideas])
                k, v = model.E(tok, lens)
                n = len(ideas)
                di = torch.arange(n, device=DEV)[None]
                for m in mems:
                    m.capture = True
                model.P(k, v, di, role=torch.tensor([i["step"] == "deduction" for i in ideas], device=DEV).long())
                for m in mems:
                    m.capture = False
                per_layer = []
                for m in mems:
                    q, kk = m.last_qk; vv, beta = m.last_v, m.last_beta
                    S = torch.zeros(kk.shape[1], vv.shape[-1], kk.shape[-1], device=DEV)
                    snaps = {}
                    want = {b * r: b for b in ded_blocks}
                    for t in range(max(want) + 1):
                        if t in want:
                            snaps[want[t]] = S.flatten().cpu().numpy()
                        kt, vt, bt = kk[0, :, t], vv[0, :, t], beta[0, :, t, None, None]
                        S = S - bt * torch.einsum("hvk,hk,hj->hvj", S, kt, kt) + bt * torch.einsum("hv,hk->hvk", vt, kt)
                    per_layer.append(snaps)
                if proj is None:
                    D = per_layer[0][ded_blocks[0]].shape[0]
                    proj = rng.standard_normal((D, a.dim)).astype(np.float32) / np.sqrt(a.dim)
                emb = model.emb.weight.detach().float().cpu().numpy()
                for b in ded_blocks:
                    for o in final:
                        if o not in w2i:
                            continue
                        moved = up.get(o) is not None and pf[up[o]][0] == "goes"
                        rows.append(([per_layer[li][b] @ proj for li in range(len(mems))], emb[w2i[o]], final[o], init.get(o), moved))
        return rows

    fit, test = rows_of(FIT), rows_of(TEST)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    res = {}
    for li in range(len(mems)):
        Xf = np.stack([np.concatenate([x[0][li], x[1]]) for x in fit]); Xt = np.stack([np.concatenate([x[0][li], x[1]]) for x in test])
        sc = StandardScaler().fit(Xf); Xf, Xt = sc.transform(Xf), sc.transform(Xt)
        for tname, ti in (("current", 2), ("initial", 3)):
            yf = [x[ti] for x in fit]; yt = [x[ti] for x in test]
            keep_f = [i for i, y in enumerate(yf) if y is not None]; keep_t = [i for i, y in enumerate(yt) if y is not None]
            for pname, mk in (("lin", lambda: LogisticRegression(max_iter=2000, C=0.5)),
                              ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300, early_stopping=True, random_state=0))):
                clf = mk().fit(Xf[keep_f], [yf[i] for i in keep_f])
                pred = clf.predict(Xt[keep_t])
                for g in ("immobile", "moved"):
                    idx = [j for j, i in enumerate(keep_t) if test[i][4] == (g == "moved")]
                    fidx = [i for i in keep_f if fit[i][4] == (g == "moved")]
                    maj = collections.Counter(yf[i] for i in fidx).most_common(1)[0][0] if fidx else None
                    acc = float(np.mean([pred[j] == yt[keep_t[j]] for j in idx])) if idx else float("nan")
                    mj = float(np.mean([maj == yt[keep_t[j]] for j in idx])) if idx else float("nan")
                    res[(li, pname, tname, g)] = (acc, mj, len(idx))
    print(run + ": " + "  ".join(f"L{li} {p} cur-mov {res[(li, p, 'current', 'moved')][0]:.2f}" for li in range(len(mems)) for p in ("lin", "mlp")), flush=True)
    del model
    return res, len(fit), len(test)


allr = {r: run_one(r) for r in a.runs}
if a.summary:
    L = [f"Sonde unique [S projeté ({a.dim}) ; embedding de l'objet], ajustée sur {a.n_fit} histoires F1 d'entraînement, testée sur "
         f"{a.n_test} de validation ; précision (classe majoritaire du groupe) ; n = lignes de test", "",
         "| run | sonde | couche | COURANT immobiles | COURANT déplacés | INITIAL immobiles | INITIAL déplacés |", "|---|---|---|---|---|---|---|"]
    for r, (res, nf, nt) in allr.items():
        for p in ("lin", "mlp"):
            for li in sorted({k[0] for k in res}):
                f = lambda t, g: "{:.2f} ({:.2f}, n={})".format(*res[(li, p, t, g)])
                L.append(f"| {r.split('/')[-1]} | {p} | {li} | {f('current', 'immobile')} | {f('current', 'moved')} | {f('initial', 'immobile')} | {f('initial', 'moved')} |")
    open(a.summary, "w").write("\n".join(L) + "\n"); print("\n".join(L))

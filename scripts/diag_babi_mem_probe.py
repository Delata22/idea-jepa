"""bAbI qa2 memory-state probe (notes §35; protocol of diag_mem_state_probe2): ONE probe per memory layer of P,
input [random projection (--dim) of the whole delta-rule state S ; word embedding of the entity], S = the state after the
writes of every idea before the block that predicts the answer (one causal P pass on story + question). Targets: (a) the
CURRENT place of each PERSON (last move); (b) the CURRENT place of each OBJECT picked up at least once (its carrier's place
if held, else where it was put down; never written in the story, it must be deduced). Fitted on --n_fit TRAIN stories,
tested on --n_test VAL stories; accuracy vs the majority class of the fitting rows of the same group; linear and MLP.
usage: python scripts/diag_babi_mem_probe.py <run> [--n_fit 1000] [--n_test 500] [--dim 512] [--json out.json]"""
import argparse, collections, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from ideajepa.train import load_model

PAD = 0
MOVE = {"moved", "went", "journeyed", "travelled"}
TAKE = {"got", "picked", "grabbed", "took"}
DROP = {"dropped", "left", "discarded", "put"}
ap = argparse.ArgumentParser(); ap.add_argument("run")
ap.add_argument("--n_fit", type=int, default=1000); ap.add_argument("--n_test", type=int, default=500)
ap.add_argument("--dim", type=int, default=512); ap.add_argument("--json", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")


def world(stmts):
    """person -> current place; object -> current place (None if unknown)."""
    pos, holder, loc = {}, {}, {}
    for s in stmts:
        w = [x for x in s if x not in (".", "there", "back", "up", "down")]
        if len(w) < 3:
            continue
        who, verb, last = w[0], w[1], w[-1]
        if verb in MOVE:
            pos[who] = last
            for o, h in holder.items():
                if h == who:
                    loc[o] = last
        elif verb in TAKE:
            holder[last] = who; loc[last] = pos.get(who)
        elif verb in DROP:
            holder[last] = None; loc[last] = pos.get(who)
    return pos, loc


model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
model.eval()
r = model.P.r
mems = list(dict.fromkeys(model.P.mems))
assert all(m.decay is None for m in mems), "state recurrence below assumes no forgetting"
path = cfg["data"]["path"]
vocab = torch.load(f"{path}/train.pt", weights_only=False)["vocab"]
w2i = {x: i for i, x in enumerate(vocab)}
emb = model.emb.weight.detach().float().cpu().numpy()
rng = np.random.default_rng(0)
proj = None


def rows_of(split, n):
    global proj
    docs = torch.load(f"{path}/{split}.pt", weights_only=False)["docs"][:n]
    rows = []
    with torch.no_grad():
        for d in docs:
            ctx = [i["tokens"] for i in d["ideas"] if i["step"] != "deduction"]
            L = max(len(t) for t in ctx)
            tok = torch.full((len(ctx), L), PAD, dtype=torch.long)
            for i, t in enumerate(ctx):
                tok[i, :len(t)] = torch.tensor(t)
            tok = tok.to(DEV); lens = torch.tensor([len(t) for t in ctx], device=DEV)
            k, v = model.E(tok, lens)
            n = len(ctx)
            di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
            for m in mems:
                m.capture = True
            model.P(k, v, di, role=torch.zeros(n, dtype=torch.long, device=DEV), tokens=model._p_tokens(k),
                    words=model._p_words(k, {"idea_support": torch.zeros(n, dtype=torch.bool, device=DEV)}))
            snaps = []
            for m in mems:
                m.capture = False
                _, kk = m.last_qk; vv, beta = m.last_v, m.last_beta
                S = torch.zeros(kk.shape[1], vv.shape[-1], kk.shape[-1], device=DEV)
                for t in range(n * r):          # writes of positions < n (the block at position n predicts the answer)
                    kt, vt, bt = kk[0, :, t], vv[0, :, t], beta[0, :, t, None, None]
                    S = S - bt * torch.einsum("hvk,hk,hj->hvj", S, kt, kt) + bt * torch.einsum("hv,hk->hvk", vt, kt)
                snaps.append(S.flatten().cpu().numpy())
            if proj is None:
                proj = rng.standard_normal((snaps[0].shape[0], a.dim)).astype(np.float32) / np.sqrt(a.dim)
            pos, loc = world([[vocab[t] for t in c] for c in ctx[:-1]])
            feats = [s @ proj for s in snaps]
            for who, p in pos.items():
                rows.append((feats, emb[w2i[who]], p, "person"))
            for o, p in loc.items():
                if p is not None:
                    rows.append((feats, emb[w2i[o]], p, "object"))
    return rows


fit, test = rows_of("train", a.n_fit), rows_of("val", a.n_test)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
res = {}
for li in range(len(mems)):
    Xf = np.stack([np.concatenate([x[0][li], x[1]]) for x in fit]); Xt = np.stack([np.concatenate([x[0][li], x[1]]) for x in test])
    sc = StandardScaler().fit(Xf); Xf, Xt = sc.transform(Xf), sc.transform(Xt)
    yf = [x[2] for x in fit]; yt = [x[2] for x in test]
    for pname, mk in (("lin", lambda: LogisticRegression(max_iter=2000, C=0.5)),
                      ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300, early_stopping=True, random_state=0))):
        pred = mk().fit(Xf, yf).predict(Xt)
        for g in ("person", "object"):
            idx = [j for j, x in enumerate(test) if x[3] == g]
            maj = collections.Counter(yf[i] for i, x in enumerate(fit) if x[3] == g).most_common(1)[0][0]
            res[f"L{li}|{pname}|{g}"] = {"acc": float(np.mean([pred[j] == yt[j] for j in idx])),
                                        "majority": float(np.mean([maj == yt[j] for j in idx])), "n": len(idx)}
print(f"{a.run}: sonde [S ; entité], {a.n_fit} histoires d'ajustement (train), {a.n_test} de test (val)")
for li in range(len(mems)):
    print(f"  couche {li} : " + "  ".join(f"{p} {g} {res[f'L{li}|{p}|{g}']['acc']:.2f} (maj {res[f'L{li}|{p}|{g}']['majority']:.2f}, n={res[f'L{li}|{p}|{g}']['n']})"
                                       for p in ("lin", "mlp") for g in ("person", "object")))
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)

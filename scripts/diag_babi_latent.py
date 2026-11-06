"""Where does P's predicted answer idea sit, relative to its target? For the answer position of bAbI
stories: cos(Ẑ, E_ema(answer sentence)) on flattened [k ; v/|v|] slots, TRAIN vs VAL, correct vs wrong decoded answers
(from <run>/babi_outputs_<split>.jsonl written by babi_eval), decomposed into the "answer" part (the row space of a linear
6-class place probe fitted on the train targets E_ema(answer)) and its orthogonal complement; plus the mean cos between two
val TARGETS with the same answer place vs a different one.
usage: python scripts/diag_babi_latent.py <picked run> [--n 500] [--json out.json]"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model

PAD = 0
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--n", type=int, default=500)
ap.add_argument("--json", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
model.eval()
path = cfg["data"]["path"]


def flat(k, v):
    return torch.cat([k, F.normalize(v, dim=-1)], -1).reshape(k.shape[0], -1).float()


def enc(ideas, ema=False):
    L = max(len(t) for t in ideas)
    tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
    for i, t in enumerate(ideas):
        tok[i, :len(t)] = torch.tensor(t)
    return (model.E_ema if ema else model.E)(tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV))


def collect(split):
    docs = torch.load(f"{path}/{split}.pt", weights_only=False)["docs"][:a.n]
    outs = {r["i"]: r for r in map(json.loads, open(f"{a.run}/babi_outputs_{split}.jsonl"))}
    Z, T, ok, ans = [], [], [], []
    with torch.no_grad():
        for i, d in enumerate(docs):
            if i not in outs:
                continue
            ctx = [x["tokens"] for x in d["ideas"] if x["step"] != "deduction"]
            tgt = [x["tokens"] for x in d["ideas"] if x["step"] == "deduction"][0]
            k, v = enc(ctx)
            n = len(ctx)
            di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
            pk, pv = model.predict(k, v, {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                                          "idea_pos": torch.tensor([n], device=DEV),
                                          "idea_role": torch.zeros(n, dtype=torch.long, device=DEV),
                                          "idea_support": torch.zeros(n, dtype=torch.bool, device=DEV)})
            kt, vt = enc([tgt], ema=True)
            Z.append(flat(pk, pv)[0].cpu().numpy()); T.append(flat(kt, vt)[0].cpu().numpy())
            ok.append(outs[i]["ok"]); ans.append(d["answer"])
    return np.stack(Z), np.stack(T), np.array(ok), np.array(ans)


tr, va = collect("train"), collect("val")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=3000).fit(tr[1], tr[3])
Q, _ = np.linalg.qr(clf.coef_.T)                       # [D, 6] orthonormal basis of the probe's row space
proj = lambda X: X @ Q @ Q.T


def cos(A, B):
    return np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)).clip(1e-8)


res = {"run": a.run, "probe_val_acc_on_targets": float(clf.score(va[1], va[3]))}
for name, (Z, T, ok, _) in (("train", tr), ("val", va)):
    Pz, Pt = proj(Z), proj(T)
    for g, m in (("correct", ok), ("wrong", ~ok)):
        if m.sum() == 0:
            continue
        res[f"{name}|{g}"] = {"n": int(m.sum()), "cos_total": float(cos(Z[m], T[m]).mean()),
                              "cos_answer_part": float(cos(Pz[m], Pt[m]).mean()),
                              "cos_complement": float(cos(Z[m] - Pz[m], T[m] - Pt[m]).mean()),
                              "answer_part_energy_pred": float((np.linalg.norm(Pz[m], axis=1) / np.linalg.norm(Z[m], axis=1)).mean())}
T, ans = va[1], va[3]
C = (T / np.linalg.norm(T, axis=1, keepdims=True)) @ (T / np.linalg.norm(T, axis=1, keepdims=True)).T
same = ans[:, None] == ans[None, :]
iu = np.triu_indices(len(T), 1)
res["targets_same_answer_cos"] = float(C[iu][same[iu]].mean())
res["targets_diff_answer_cos"] = float(C[iu][~same[iu]].mean())
print(json.dumps(res, indent=1))
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)

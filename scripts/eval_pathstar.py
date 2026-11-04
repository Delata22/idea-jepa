"""Path-Star evaluation: solve rate (exact path, greedy, from the premises), position of the first error, and for
Idea-JEPA the commitment of P (cos of the predicted idea to E_ema of each arm's path).
usage: uv run python scripts/eval_pathstar.py <run> [--n 2000] [--split test]"""
import sys, json, argparse, collections, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.data.pathstar import arms
from ideajepa.utils.common import ROOT

PAD, BOS = 0, 1
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--split", default="test"); ap.add_argument("--bs", type=int, default=64)
a = ap.parse_args()
model, cfg, _ = load_model(f"{a.run}/ckpt.pt")
obj = torch.load(ROOT / cfg["data"]["path"] / f"{a.split}.pt", weights_only=False)
docs, vocab = obj["docs"][:a.n], obj["vocab"]; w2i = {w: i for i, w in enumerate(vocab)}
L = len(docs[0]["graph"]["path"])
node_ids = torch.tensor([w2i[str(i)] for i in range(len(vocab)) if str(i) in w2i], device="cuda")
only_nodes = torch.full((len(vocab),), float("-inf"), device="cuda"); only_nodes[node_ids] = 0.0


def flat(v):
    return F.normalize(F.normalize(v.float(), dim=-1).reshape(v.shape[0], -1), dim=-1)


@torch.no_grad()
def gen_gpt(chunk):
    seqs = [[BOS] + [t for i in d["ideas"][:-1] for t in i["tokens"]] for d in chunk]
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), PAD, dtype=torch.long, device="cuda")
    lens = torch.tensor([len(s) for s in seqs], device="cuda")
    for b, s in enumerate(seqs): x[b, T - len(s):] = torch.tensor(s)          # left padding: all end at T-1
    out = []
    for _ in range(L):
        lg = model.run(x)[:, -1].float() + only_nodes
        nxt = lg.argmax(-1); out.append(nxt); x = torch.cat([x, nxt[:, None]], 1)
    return torch.stack(out, 1).tolist(), None


@torch.no_grad()
def gen_im(chunk):
    B = len(chunk); m_prefix = model.cfg["m_prefix"]
    prem = [[i["tokens"] for i in d["ideas"][:-1]] for d in chunk]
    rows = [t for p in prem for t in p]
    Lr = max(len(t) for t in rows)
    tok = torch.full((len(rows), Lr), PAD, dtype=torch.long)
    for i, t in enumerate(rows): tok[i, :len(t)] = torch.tensor(t)
    k, v = model.E(tok.cuda(), torch.tensor([len(t) for t in rows], device="cuda"))
    n = max(len(p) for p in prem)
    di = torch.full((B, n + 1), -1, dtype=torch.long); r = 0
    for b, p in enumerate(prem):
        for j in range(len(p)): di[b, j] = r; r += 1
        di[b, len(p)] = 0
    batch = {"doc_index": di.cuda(), "idea_doc": torch.arange(B, device="cuda"),
             "idea_pos": torch.tensor([len(p) for p in prem], device="cuda")}
    pk, pv = model.predict(k, v, batch)
    prefix = torch.full((B, m_prefix), PAD, dtype=torch.long)
    for b, p in enumerate(prem):
        h = [BOS] + [t for x in p for t in x]; h = h[-m_prefix:]; prefix[b, m_prefix - len(h):] = torch.tensor(h)
    prefix = prefix.cuda(); toks = torch.zeros(B, 0, dtype=torch.long, device="cuda")
    for s_ in range(L):
        win = torch.cat([prefix, toks, torch.full((B, 1), PAD, dtype=torch.long, device="cuda")], 1)
        e = model.O(win, m_prefix)
        logits, _ = model.dec(win, torch.full((B,), s_ + 1, device="cuda"), m_prefix, pk, pv, e)
        nxt = (logits[:, s_].float() + only_nodes).argmax(-1); toks = torch.cat([toks, nxt[:, None]], 1)
    # commitment: cos of P's idea to E_ema of each arm's path
    commit = []
    for b, d in enumerate(chunk):
        A = arms(d["graph"]["edges"], d["graph"]["source"])
        at = torch.tensor([[w2i[str(x)] for x in arm] for arm in A], device="cuda")
        ka, va = model.E_ema(at, torch.full((len(A),), at.shape[1], device="cuda"))
        c = (flat(pv[b:b + 1]) @ flat(va).T)[0]
        true = A.index(d["graph"]["path"])
        others = torch.cat([c[:true], c[true + 1:]])
        cen = float((flat(pv[b:b + 1]) @ F.normalize(flat(va).mean(0), dim=0))[0])
        commit.append((float(c[true]), float(others.max()), cen))
    return toks.tolist(), commit


solved, first_err, commits = 0, collections.Counter(), []
gen = gen_im if isinstance(model, IdeaJEPA) else gen_gpt
for s in range(0, len(docs), a.bs):
    chunk = docs[s:s + a.bs]
    outs, cm = gen(chunk)
    if cm: commits += cm
    for d, o in zip(chunk, outs):
        ref = [w2i[str(x)] for x in d["graph"]["path"]]
        if o == ref: solved += 1; first_err["none"] += 1
        else: first_err[next(i for i in range(L) if o[i] != ref[i])] += 1
n = len(docs)
res = {"run": a.run.split("/")[-1], "n": n, "solve_rate": solved / n,
       "first_error_position": {str(k): v / n for k, v in sorted(first_err.items(), key=lambda kv: str(kv[0]))}}
if commits:
    res["cos_true_arm"] = sum(c[0] for c in commits) / n; res["cos_best_wrong_arm"] = sum(c[1] for c in commits) / n
    res["cos_centroid"] = sum(c[2] for c in commits) / n
    res["P_true_arm_closest"] = sum(c[0] > c[1] for c in commits) / n
print(json.dumps(res))
json.dump(res, open(f"{a.run}/pathstar_eval.json", "w"), indent=1)

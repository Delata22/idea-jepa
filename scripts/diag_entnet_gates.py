"""EntNet memory gates on bAbI qa2 (notes §35.22-23), no training, on --n validation stories:
(1) per memory layer / head / slot, the DOMINANT entity (person or object named by the ideas that open its write gate
    strongly: gate >= the slot's 75th percentile over all statement ideas) and the share of slots whose dominant entity
    covers >= 50 % of those strong writes;
(2) mean write gate of the slots dominated by an OBJECT o at the moves of o's CARRIER (while carrying o) vs at the moves of
    OTHER persons while o is carried (criterion of the maths session: >= 2x).
usage: python scripts/diag_entnet_gates.py <run> [--n 200] [--json out.json]"""
import argparse, collections, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from ideajepa.model import deltanet as DN
from ideajepa.train import load_model

PAD = 0
MOVE = {"moved", "went", "journeyed", "travelled"}
TAKE = {"got", "picked", "grabbed", "took"}
DROP = {"dropped", "left", "discarded", "put"}
PERSONS, OBJECTS = {"daniel", "john", "mary", "sandra"}, {"apple", "football", "milk"}
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--json", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
model.eval()
assert cfg["model"].get("idea_memory") == "entnet"
docs = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
vocab, docs = docs["vocab"], docs["docs"][:a.n]
L = len(model.P.mems)
rows = []   # (layer, gates [H, m], entities of the idea, event)
with torch.no_grad():
    for d in docs:
        ctx = [i["tokens"] for i in d["ideas"] if i["step"] != "deduction"]
        Lm = max(len(t) for t in ctx)
        tok = torch.full((len(ctx), Lm), PAD, dtype=torch.long)
        for i, t in enumerate(ctx):
            tok[i, :len(t)] = torch.tensor(t)
        tok = tok.to(DEV); lens = torch.tensor([len(t) for t in ctx], device=DEV)
        k, v = model.E(tok, lens)
        n = len(ctx)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        DN.entnet_scan.record = []
        model.P(k, v, di, role=torch.zeros(n, dtype=torch.long, device=DEV), tokens=model._p_tokens(k))
        rec = DN.entnet_scan.record; DN.entnet_scan.record = None
        steps = len(rec) // L                                   # one write step per input position, per layer
        holder = {}
        for i in range(n - 1):                                  # statements; idea i is written at input position i + 1
            w = [x for x in (vocab[t] for t in ctx[i]) if x not in (".", "there", "back", "up", "down")]
            ents = [x for x in w if x in PERSONS | OBJECTS]
            ev = None
            if w[1] in MOVE:
                ev = ("move", w[0], [o for o, h in holder.items() if h == w[0]], [o for o, h in holder.items() if h and h != w[0]])
            elif w[1] in TAKE:
                holder[w[-1]] = w[0]
            elif w[1] in DROP:
                holder[w[-1]] = None
            for li in range(L):
                rows.append((li, rec[li * steps + i + 1][0].numpy(), ents, ev))
res = {"run": a.run, "n_stories": len(docs)}
dom = {}
for li in range(L):
    G = np.stack([r[1] for r in rows if r[0] == li])            # [N, H, m]
    ents = [r[2] for r in rows if r[0] == li]
    H, m = G.shape[1:]
    net = 0
    for h in range(H):
        for j in range(m):
            thr = np.percentile(G[:, h, j], 75)
            strong = [e for g, e in zip(G[:, h, j], ents) if g >= thr]
            c = collections.Counter(x for e in strong for x in set(e))
            ent, cnt = c.most_common(1)[0] if c else (None, 0)
            share = cnt / max(len(strong), 1)
            dom[(li, h, j)] = (ent, share)
            net += share >= 0.5
    res[f"L{li}_net_share"] = net / (H * m)
    # chance: the share of the most frequent entity among ALL statement ideas (a gate blind to the content)
    call = collections.Counter(x for e in ents for x in set(e))
    res[f"L{li}_chance_share"] = call.most_common(1)[0][1] / len(ents) if call else 0.0
    res[f"L{li}_dominant_entities"] = dict(collections.Counter(e for (l_, _, _), (e, s) in dom.items() if l_ == li and s >= 0.5))
# (2) object slots at the carrier's moves vs other persons' moves
carrier_g, other_g = [], []
for li, g, ents, ev in rows:
    if ev is None:
        continue
    _, who, carried, carried_by_others = ev
    for (l_, h, j), (e, s) in dom.items():
        if l_ != li or s < 0.5 or e not in OBJECTS:
            continue
        if e in carried:
            carrier_g.append(g[h, j])
        elif e in carried_by_others:
            other_g.append(g[h, j])
res["object_slot_gate_carrier_moves"] = float(np.mean(carrier_g)) if carrier_g else None
res["object_slot_gate_other_moves"] = float(np.mean(other_g)) if other_g else None
res["ratio"] = (res["object_slot_gate_carrier_moves"] / max(res["object_slot_gate_other_moves"], 1e-9)
                if carrier_g and other_g else None)
res["n_carrier_moves_x_slots"], res["n_other_moves_x_slots"] = len(carrier_g), len(other_g)
print(json.dumps(res, indent=1))
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)

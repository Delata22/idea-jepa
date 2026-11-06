"""What does B3's per-token DeltaNet memory address when it writes the location of a MOVED object?
Reference text (val F1 stories): BOS + premises + reference deductions, cut just before the location word of each deduction
about a moved object (its useful premise is a "goes"). At that position, per block and head, the query q (normalised) is
compared to the keys k_t of all earlier tokens; top-1 token category:
  carrier_goes  the location word of the useful premise (the carrier's last "goes to L" that fixed the object's place)
  initial       the location word of the object's initial "the o is in L" premise
  other_loc     another location word          other   anything else
Reported separately for steps where B3's greedy next token is the right location and where it is wrong.
usage: uv run python scripts/diag_b3_moved.py <b3 run> [...] [--n 300] [--summary out.md]
"""
import argparse, collections, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V
from ideajepa.train import load_model

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
BOS = 1
LOCS = {w for w in vocab if w in set(R.LOCS)} if hasattr(R, "LOCS") else set()
as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f)

allres = {}
for run in a.runs:
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV); model.eval()
    mems = [b.mem for b in model.blocks]
    cnt = collections.defaultdict(collections.Counter)
    n_st = 0
    with torch.no_grad():
        for doc in docs:
            if doc.get("family") != "F1" or n_st >= a.n:
                continue
            n_st += 1
            ideas = doc["ideas"]
            pf = V.doc_premises(doc)
            up = R.f1_useful_premises(pf)
            prem_idx = [i for i, x in enumerate(ideas) if x["step"] == "premise"]
            final = R.final_state_f1(pf)
            seq, owner = [BOS], [(-1, -1)]
            for i, idea in enumerate(ideas):
                for j, t in enumerate(idea["tokens"]):
                    seq.append(t); owner.append((i, j))
            for i, idea in enumerate(ideas):
                f = as_fact(idea["fact"]) if idea.get("fact") is not None else None
                if idea["step"] != "deduction" or f is None or f[0] != "objin" or f[1] not in up or pf[up[f[1]]][0] != "goes":
                    continue
                toks = idea["tokens"]
                loc_j = len(toks) - 2                                          # "... <loc> ."
                if vocab[toks[loc_j]] != f[2]:
                    continue
                pos = owner.index((i, loc_j))                                  # position of the location word
                ug = prem_idx[up[f[1]]]                                        # useful premise (carrier goes to L)
                init_i = next((prem_idx[j] for j, p in enumerate(pf) if p[0] == "objin" and p[1] == f[1]), None)
                ug_pos = owner.index((ug, len(ideas[ug]["tokens"]) - 2))
                init_pos = owner.index((init_i, len(ideas[init_i]["tokens"]) - 2)) if init_i is not None else -1
                x = torch.tensor([seq[:pos]], device=DEV)
                for m in mems:
                    m.capture = True
                logits = model.run(x)
                for m in mems:
                    m.capture = False
                right = int(logits[0, -1].argmax()) == toks[loc_j]
                grp = "right" if right else "wrong"
                for li, m in enumerate(mems):
                    q, k = m.last_qk                                           # [1, H, T, dh], normalised
                    s = (q[0, :, -1:, :] * k[0, :, :pos - 1, :]).sum(-1)       # [H, T-1] earlier tokens only
                    top = s.argmax(-1).tolist()
                    for h, tp in enumerate(top):
                        w = vocab[seq[tp]]
                        cat = ("carrier_goes" if tp == ug_pos else "initial" if tp == init_pos else
                               "other_loc" if w in {p[2] for p in pf if p[0] in ("at", "objin", "goes")} else "other")
                        cnt[(grp, li, h)][cat] += 1; cnt[(grp, li, h)]["n"] += 1
                cnt[("steps", grp)]["n"] += 1
    allres[run] = cnt
    print(run, {g: cnt[("steps", g)]["n"] for g in ("right", "wrong")}, flush=True)
    del model

L = ["Par couche : tête qui adresse le plus souvent le « goes to L » du porteur ; part des pas (top-1 de q·k sur les jetons passés) "
     "pour cette tête : porteur / lieu initial / autre lieu / autre. Séparé selon que le jeton suivant prédit (glouton) est juste ou faux.", "",
     "| run | pas justes / faux | couche | meilleure tête | JUSTE : porteur / initial / autre lieu / autre | FAUX : porteur / initial / autre lieu / autre |",
     "|---|---|---|---|---|---|"]
for run, cnt in allres.items():
    nr, nw = cnt[("steps", "right")]["n"], cnt[("steps", "wrong")]["n"]
    layers = sorted({k[1] for k in cnt if k[0] in ("right", "wrong")})
    for li in layers:
        heads = sorted({k[2] for k in cnt if k[0] in ("right", "wrong") and k[1] == li})
        tot = lambda h: sum(cnt[(g, li, h)]["carrier_goes"] for g in ("right", "wrong"))
        h = max(heads, key=tot)
        def fmt(g):
            c = cnt[(g, li, h)]
            return " / ".join(f"{c[x] / c['n']:.2f}" for x in ("carrier_goes", "initial", "other_loc", "other")) if c["n"] else "—"
        L.append(f"| {run.split('/')[-1]} | {nr} / {nw} | {li} | {h} | {fmt('right')} | {fmt('wrong')} |")
out = "\n".join(L)
print(out)
if a.summary:
    open(a.summary, "w").write(out + "\n")

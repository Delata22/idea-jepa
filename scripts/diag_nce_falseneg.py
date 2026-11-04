"""Mémo §34.12: label-free InfoNCE (v3sd, in-batch negatives) — how many of an anchor's negatives are another surface of the
SAME fact (a false negative), and how close they end up. Val batches of the run's own view; the generator's fact labels are
used HERE only, to diagnose. Per family of the anchor: share of same-fact ideas among its in-batch negatives (other ideas of the
batch), mean cos (nce_embed space [k ; v/|v|]) of same-fact pairs vs different-fact pairs of the same story vs other stories.
usage: uv run python scripts/diag_nce_falseneg.py <run> [...] [--batches 8] [--summary out.md]
"""
import argparse, collections, os, sys
import torch
sys.path.insert(0, "src")
from ideajepa.data.dataset import to_device
from ideajepa.data.loaders import make_loader
from ideajepa.model.objective import nce_embed
from ideajepa.train import load_model

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--batches", type=int, default=8)
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
res = {}
for run in a.runs:
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV); model.eval()
    loader = make_loader(cfg, "val", False)
    orig = loader.collate
    fam_of = lambda d: ("F1" if any(i.get("fact") and i["fact"][0] in ("objin", "goes") for i in d["ideas"]) else
                        "F2" if any(i.get("fact") and i["fact"][0] == "rel" for i in d["ideas"]) else "F3")
    def coll(docs):
        b = orig(docs)
        facts = [(tuple(tuple(x) if isinstance(x, list) else x for x in i["fact"]) if i.get("fact") is not None else None)
                 for d in docs for i in d["ideas"]]
        b["_facts"] = facts; b["_fam"] = [fam_of(d) for d in docs for _ in d["ideas"]]
        return b
    loader.collate = coll
    acc = collections.defaultdict(list)
    with torch.no_grad():
        for bi, b in enumerate(loader.batches()):
            if bi >= a.batches:
                break
            facts, fam = b.pop("_facts"), b.pop("_fam")
            b = to_device(b, DEV)
            k, v = model.E(b["idea_tok"], b["idea_len"])
            z = nce_embed(k, v)
            doc = b["idea_doc"]
            sim = z @ z.T
            N = z.shape[0]
            for i in range(N):
                if facts[i] is None:
                    continue
                same = [j for j in range(N) if j != i and facts[j] == facts[i]]
                acc[(fam[i], "share_same_fact")].append(len(same) / (N - 1))
                for j in same:
                    acc[(fam[i], "cos_same_fact")].append(float(sim[i, j]))
                diff_story = [j for j in range(N) if j != i and doc[j] == doc[i] and facts[j] != facts[i] and facts[j] is not None]
                other = [j for j in range(N) if doc[j] != doc[i]][:50]
                if diff_story:
                    acc[(fam[i], "cos_diff_same_story")].append(float(sim[i, diff_story].mean()))
                if other:
                    acc[(fam[i], "cos_other_story")].append(float(sim[i, other].mean()))
    res[run] = {k_: sum(v_) / len(v_) for k_, v_ in acc.items() if v_}
    print(run, {k_: round(v_, 3) for k_, v_ in sorted(res[run].items())}, flush=True)
    del model
if a.summary:
    L = ["| run | famille | part des négatifs = même fait | cos même fait (faux négatifs) | cos faits différents même histoire | cos autres histoires |",
         "|---|---|---|---|---|---|"]
    for run, r in res.items():
        for f in ("F1", "F2", "F3"):
            g = lambda x: f"{r.get((f, x), float('nan')):.3f}"
            L.append(f"| {run.split('/')[-1]} | {f} | {g('share_same_fact')} | {g('cos_same_fact')} | {g('cos_diff_same_story')} | {g('cos_other_story')} |")
    open(a.summary, "w").write("\n".join(L) + "\n"); print("\n".join(L))

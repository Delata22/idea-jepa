"""Step 2: accuracy of P's relation head against the majority-class frequency, and
confusion matrix, per family, teacher forcing on the val stories of the run's own data view (compound + rel_source).
Rows = deductions. Relation classes none / justified_by / sequence; pointer accuracy on the justified_by rows.
usage: uv run python scripts/diag_rel_head.py <run> [<run> ...] [--batches 20] [--summary out.md]
"""
import argparse
import collections
import json
import os
import sys

import torch

sys.path.insert(0, "src")
from ideajepa.data.dataset import to_device
from ideajepa.data.loaders import make_loader
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--batches", type=int, default=20)
ap.add_argument("--summary", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
NAMES = ["none", "justified_by", "sequence"]


def family(doc):
    """chunk_docs drops the family key: read it from the predicates of the facts."""
    if doc.get("family"):
        return doc["family"]
    preds = {i["fact"][0] for i in doc["ideas"] if i.get("fact") is not None}
    return "F1" if preds & {"objin", "at", "goes", "takes", "drops"} else ("F2" if "rel" in preds else "F3")


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    assert model.cfg.get("rel_cond", False), "rel_cond model expected"
    loader = make_loader(cfg, "val", False)
    orig = loader.collate
    loader.collate = lambda docs: {**orig(docs), "_fam": [family(d) for d in docs for _ in d["ideas"]]}
    conf = collections.defaultdict(lambda: torch.zeros(3, 3, dtype=torch.long))   # fam -> [true, pred]
    ptr = collections.Counter()
    with torch.no_grad():
        for bi, batch in enumerate(loader.batches()):
            if bi >= a.batches:
                break
            fam = batch.pop("_fam")
            batch = to_device(batch, DEV)
            k, v = model.encode(batch)
            model.predict(k, v, batch)
            lr, lp = model.rel_logits()
            pr, pp = lr.argmax(-1).tolist(), lp.argmax(-1).tolist()
            tr, tp = batch["rel_id"].tolist(), batch["rel_off"].tolist()
            ded = batch["is_deduction"].tolist()
            for i in range(len(tr)):
                if not ded[i]:
                    continue
                for f in (fam[i], "all"):
                    conf[f][tr[i], pr[i]] += 1
                    if tr[i] == 1:
                        ptr[(f, "n")] += 1; ptr[(f, "ok")] += pp[i] == tp[i]
    out = {}
    for f, c in conf.items():
        n = int(c.sum()); true_freq = c.sum(1)
        out[f] = {"n": n, "acc": float(c.trace()) / n, "majority_freq": float(true_freq.max()) / n,
                  "majority_class": NAMES[int(true_freq.argmax())], "confusion_true_x_pred": c.tolist(),
                  "ptr_acc_justified": ptr[(f, "ok")] / ptr[(f, "n")] if ptr[(f, "n")] else None}
    json.dump({"run": run, "results": out}, open(f"{run}/rel_head.json", "w"), indent=1)
    o = out["all"]
    print(f"{run}: rel acc {o['acc']:.3f} vs majority {o['majority_freq']:.3f} ({o['majority_class']}), "
          f"pointer acc on justified_by {o['ptr_acc_justified']}", flush=True)
    del model
    return out


allr = {r: run_one(r) for r in a.runs}
if a.summary:
    lines = ["| run | family | n | head acc | majority freq (class) | pointer acc (justified_by) | confusion true×pred (none/jb/seq) |",
             "|---|---|---|---|---|---|---|"]
    for r, res in allr.items():
        for f in ["all", "F1", "F2", "F3"]:
            o = res.get(f)
            if o:
                pa = "—" if o["ptr_acc_justified"] is None else f"{o['ptr_acc_justified']:.3f}"
                lines.append(f"| {r.split('/')[-1]} | {f} | {o['n']} | {o['acc']:.3f} | {o['majority_freq']:.3f} ({o['majority_class']}) | "
                             f"{pa} | {o['confusion_true_x_pred']} |")
    open(a.summary, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

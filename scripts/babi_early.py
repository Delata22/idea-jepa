"""Early indicator (notes §35.25, maths session): at snapshots of a bAbI qa2 run, accuracy on the "object still carried" case
(a) on VALIDATION and on TRAIN (500 examples each, from <snapshot>/babi_outputs_{val,train}_early.jsonl written by
babi_eval --out_suffix _early), and the train - validation gap on that case. Announced criterion: success if >= 0.9 with a
gap <= 0.03 at step 3000; failure if the gap >= 0.10 from step 2000.
usage: python scripts/babi_early.py <run dir> [--steps 1000 2000 3000] [--data data/babi/qa2_10k] [--json out.json]
(run babi_eval on each snapshot first)"""
import argparse, json, sys
import torch
sys.path.insert(0, "scripts")
from babi_dropcase_lib import analyse

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--steps", nargs="+", type=int, default=[1000, 2000, 3000])
ap.add_argument("--data", default="data/babi/qa2_10k"); ap.add_argument("--json", default=None)
a = ap.parse_args()
res = {}
data = {s: torch.load(f"{a.data}/{s}.pt", weights_only=False) for s in ("val", "train")}
for st in a.steps:
    row = {}
    for split in ("val", "train"):
        d = data[split]; n = ok = 0
        for r in map(json.loads, open(f"{a.run}@{st}/babi_outputs_{split}_early.jsonl")):
            stm = [[d["vocab"][x] for x in i["tokens"]] for i in d["docs"][r["i"]]["ideas"] if i["step"] == "premise"]
            q = r["question"].split(); c = analyse(stm, q[q.index("the") + 1])
            if c is not None and c[0] == "a_carried":
                n += 1; ok += r["ok"]
        row[split] = ok / max(n, 1); row[f"n_{split}"] = n
    row["gap"] = row["train"] - row["val"]
    res[st] = row
    print(f"{a.run} pas {st}: cas porté val {row['val']:.3f} (n={row['n_val']}) train {row['train']:.3f} (n={row['n_train']}) écart {row['gap']:+.3f}")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)

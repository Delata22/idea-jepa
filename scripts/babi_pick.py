"""Pick the snapshot of a bAbI run with the best VALIDATION accuracy (babi_eval --json files), ties -> earliest step.
usage: python scripts/babi_pick.py <val.json> -> prints the chosen run dir; the curve goes to stderr"""
import json
import re
import sys

rows = json.load(open(sys.argv[1]))
def step(r):
    m = re.search(r"@(\d+)$", r["run"])
    return int(m.group(1)) if m else 10 ** 9
best = None
for r in sorted(rows, key=step):
    acc = sum(v["correct"] for k, v in r["results"].items() if "|" not in k) / sum(v["n"] for k, v in r["results"].items() if "|" not in k)
    print(f"  {r['run']}: val acc {acc:.3f}", file=sys.stderr)
    if best is None or acc > best[1]:
        best = (r["run"], acc)
print(best[0])

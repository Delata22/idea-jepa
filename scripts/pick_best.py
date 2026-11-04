"""Pick the best model of a gen_eval jsonl by free-generation validity (total over the stories); ties -> first listed.
usage: python scripts/pick_best.py <file.jsonl> <model> [<model> ...]   -> prints the best model name"""
import collections
import json
import sys

f, models = sys.argv[1], sys.argv[2:]
v, n = collections.Counter(), collections.Counter()
for line in open(f):
    r = json.loads(line)
    v[r["model"]] += r["n_valid"]; n[r["model"]] += r["n_deductions"]
scores = {m: v[m] / n[m] for m in models if n[m]}
for m in models:
    print(f"  {m}: {scores.get(m, float('nan')):.3f}", file=sys.stderr)
print(max(models, key=lambda m: scores.get(m, -1)))

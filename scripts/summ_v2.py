"""Aggregate runs/logs/v2c_breakdown.jsonl (deduction CE breakdown) by variant: mean ± std over seeds.
usage: uv run python scripts/summ_v2.py [variant_prefix ...]"""
import sys, json, statistics as st, collections
rows = collections.defaultdict(list)
for line in open("runs/logs/v2c_breakdown.jsonl"):
    line = line.strip()
    if not line or " " not in line: continue
    name, js = line.split(" ", 1)
    try: d = json.loads(js)
    except Exception: continue
    variant = name.rsplit("_s", 1)[0]
    rows[variant].append(d)
want = sys.argv[1:]
keys = ["all", "fam_F1", "fam_F2", "fam_F3", "implicit", "restated", "b1", "b2", "b3"]
print(f"{'variant':26s} n  " + " ".join(f"{k:>12s}" for k in keys))
for v in sorted(rows, key=lambda x: st.mean(r["all"] for r in rows[x])):
    if want and not any(v.startswith(w) for w in want): continue
    rs = rows[v]
    cells = [f"{st.mean([r[k] for r in rs]):.3f}±{st.pstdev([r[k] for r in rs]):.3f}" if len(rs) > 1 else f"{rs[0][k]:.3f}" for k in keys]
    print(f"{v:26s} {len(rs)}  " + " ".join(f"{c:>12s}" for c in cells))

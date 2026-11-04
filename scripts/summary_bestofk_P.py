"""Best-of-K: P (sampled decoding) vs the flow, on EXACTLY the same stories.

Reads every gen_eval jsonl of a sweep directory (P_T*_K*_oracle.jsonl from runs/after_chain.sh, P_K*_... from
sweep_bestofk.py, flow_K*_*.jsonl); keeps only the stories present in ALL the P files (the P sweep may use fewer
stories than the flow sweep) and recomputes the valid rate overall and per family for every file on that subset.
usage: uv run python scripts/summary_bestofk_P.py [runs/sweep_bestofk_s0]
"""
import collections
import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/sweep_bestofk_s0")
files = sorted(d.glob("*.jsonl"))
rows = {f.stem: [json.loads(l) for l in open(f)] for f in files}
p_files = [k for k in rows if k.startswith("P_")]
if not p_files:
    sys.exit(f"no P_*.jsonl in {d}")
common = set.intersection(*[{(r["story"], r["family"]) for r in rows[k]} for k in p_files])


def desc(name):
    m_t = re.search(r"T([0-9.]+)", name); m_k = re.search(r"K(\d+)", name)
    sel = "oracle" if "oracle" in name else ("energy" if "energy" in name else "first")
    return (name.split("_")[0], int(m_k.group(1)) if m_k else 1, sel, m_t.group(1) if m_t else ("greedy" if name.startswith("P") else "-"))


out = []
for name, rs in rows.items():
    c = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        if (r["story"], r["family"]) not in common:
            continue
        for fam in (r["family"], "all"):
            c[fam][0] += r["n_valid"]; c[fam][1] += r["n_deductions"]
    if not c["all"][1]:
        continue
    model, K, sel, T = desc(name)
    out.append((model, K, sel, T, {f: v / t for f, (v, t) in c.items()}, c["all"][1], name))
out.sort(key=lambda x: (x[0], x[3], x[1], x[2]))
lines = [f"Stories common to all P files: {len(common)} (per family: "
         + ", ".join(f"{f} {sum(1 for s in common if s[1] == f)}" for f in ('F1', 'F2', 'F3')) + ")",
         "", "| model | K | selector | temp | valid | F1 | F2 | F3 | deductions | file |", "|---|---|---|---|---|---|---|---|---|---|"]
for model, K, sel, T, v, n, name in out:
    lines.append(f"| {model} | {K} | {sel} | {T} | {v['all']:.1%} | {v.get('F1', 0):.0%} | {v.get('F2', 0):.0%} | "
                 f"{v.get('F3', 0):.0%} | {n} | {name} |")
(d / "summary_P_vs_flow.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

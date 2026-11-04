"""E-space and memory diagnostics side by side, mean ± std over the seeds of each run group (name without _sN).
Reads <run>/paraphrase_cos.json (diag_paraphrase_cos) and <run>/memory_values.json (diag_memory_values).
usage: uv run python scripts/summarize_space.py <run> [<run> ...]   (prints a markdown table)
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

ROWS = [
    ("E values: same deduction", "paraphrase_cos.json", ("values", "all", "same_deduction")),
    ("E values: premise <-> deduction", "paraphrase_cos.json", ("values", "all", "premise_vs_ded")),
    ("E values: different facts, same story", "paraphrase_cos.json", ("values", "all", "diff_same_story")),
    ("E values: 'X goes to L' <-> 'X is in L'", "paraphrase_cos.json", ("values", "F1_move", "pos_vs_goes")),
    ("   control (other place)", "paraphrase_cos.json", ("values", "F1_move", "pos_vs_goes_ctrl")),
    ("E keys: different facts, same story", "paraphrase_cos.json", ("keys", "all", "diff_same_story")),
    ("memory values: same template, different entities (target ~0.2)", "memory_values.json", ("memory", "all", "same_template_diff_entities")),
    ("memory values: same fact", "memory_values.json", ("memory", "all", "same_fact")),
    ("memory values: different templates", "memory_values.json", ("memory", "all", "diff_template")),
    ("E values (centred): same template, different entities", "memory_values.json", ("E", "all", "same_template_diff_entities")),
]


def get(run, fname, path):
    p = Path(run) / fname
    if not p.exists():
        p = Path(run) / fname.replace(".json", "_base_ref.json")
    if not p.exists():
        return None
    obj = json.loads(p.read_text())["results"]
    for k in path:
        if obj is None or k not in obj:
            return None
        obj = obj[k]
    return obj


groups = collections.defaultdict(list)
for run in sys.argv[1:]:
    name = Path(run).name
    groups[name.rsplit("_s", 1)[0] if "_s" in name and name[-1].isdigit() else name].append(run)
print("| measure | " + " | ".join(f"{g} (n={len(rs)})" for g, rs in groups.items()) + " |")
print("|---|" + "---|" * len(groups))
for label, fname, path in ROWS:
    cells = []
    for rs in groups.values():
        xs = [x for x in (get(r, fname, path) for r in rs) if x is not None]
        cells.append("–" if not xs else (f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) >= 2 else f"{xs[0]:.3f}"))
    print(f"| {label} | " + " | ".join(cells) + " |")

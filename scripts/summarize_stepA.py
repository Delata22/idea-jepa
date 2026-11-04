"""Stage A (§30) summary: the three diagnostics on the stage-A seeds vs the base reference, as one markdown table.

Reads, for each run dir, paraphrase_cos.json, gate_profile_recon.json and patch_midsentence.json (the base uses the
*_base_ref.json copies). Prints mean ± std over seeds.
usage: uv run python scripts/summarize_stepA.py runs/v3/v3a_stepA_s0 runs/v3/v3a_stepA_s1 runs/v3/v3a_stepA_s2
       [--base runs/v2/v2i_pwin_delta_s0]
"""
import argparse
import json
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--base", default="runs/v2/v2i_pwin_delta_s0")
a = ap.parse_args()


def load(d, name, base=False):
    p = Path(d) / (name.replace(".json", "_base_ref.json") if base else name)
    return json.loads(p.read_text()) if p.exists() else None


ROWS = [
    ("E values: same deduction", "paraphrase_cos.json", ("results", "values", "all", "same_deduction")),
    ("E values: premise <-> deduction", "paraphrase_cos.json", ("results", "values", "all", "premise_vs_ded")),
    ("E values: different facts, same story", "paraphrase_cos.json", ("results", "values", "all", "diff_same_story")),
    ("E values: different facts, other stories", "paraphrase_cos.json", ("results", "values", "all", "diff_other")),
    ("E values: 'X goes to L' <-> 'X is in L'", "paraphrase_cos.json", ("results", "values", "F1_move", "pos_vs_goes")),
    ("   control (other place)", "paraphrase_cos.json", ("results", "values", "F1_move", "pos_vs_goes_ctrl")),
    ("E values: 'X takes O' <-> 'the O is in L_X'", "paraphrase_cos.json", ("results", "values", "F1_move", "pos_vs_takes")),
    ("   control (other place)", "paraphrase_cos.json", ("results", "values", "F1_move", "pos_vs_takes_ctrl")),
    ("E flow space: same deduction", "paraphrase_cos.json", ("results", "flow", "all", "same_deduction")),
    ("E keys: same deduction", "paraphrase_cos.json", ("results", "keys", "all", "same_deduction")),
    ("E keys: different facts, same story", "paraphrase_cos.json", ("results", "keys", "all", "diff_same_story")),
    ("gate g: connector", "gate_profile_recon.json", ("by_class", "connector")),
    ("gate g: function word", "gate_profile_recon.json", ("by_class", "function")),
    ("gate g: relation word", "gate_profile_recon.json", ("by_class", "relation")),
    ("gate g: entity", "gate_profile_recon.json", ("by_class", "entity")),
    ("patching k=1: follows Z_B", "patch_midsentence.json", ("results", "Z_B|k=1|all", "follows_B")),
    ("patching k=3 compatible: follows Z_B", "patch_midsentence.json", ("results", "Z_B|k=3|all|compat", "follows_B")),
    ("patching k=3 incompatible: composition", "patch_midsentence.json", ("results", "Z_B|k=3|all|incompat", "composition")),
    ("patching k=3 incompatible: follows prefix A", "patch_midsentence.json", ("results", "Z_B|k=3|all|incompat", "follows_A")),
    ("patching k=5 compatible: follows Z_B", "patch_midsentence.json", ("results", "Z_B|k=5|all|compat", "follows_B")),
    ("patching k=5 incompatible: composition", "patch_midsentence.json", ("results", "Z_B|k=5|all|incompat", "composition")),
    ("patching k=5 incompatible: follows prefix A", "patch_midsentence.json", ("results", "Z_B|k=5|all|incompat", "follows_A")),
    ("patching control Z_A k=3: follows A", "patch_midsentence.json", ("results", "control_Z_A|k=3|all", "follows_A")),
]


def get(obj, path):
    for k in path:
        if obj is None or k not in obj:
            return None
        obj = obj[k]
    return obj


print(f"| measure | base ({Path(a.base).name}) | stage A (mean ± std, n={len(a.runs)}) | per seed |")
print("|---|---|---|---|")
for label, fname, path in ROWS:
    b = get(load(a.base, fname, base=True), path)
    vals = [get(load(r, fname), path) for r in a.runs]
    vals = [v for v in vals if v is not None]
    if not vals and b is None:
        continue
    if len(vals) >= 2:
        cell = f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}"
    elif vals:
        cell = f"{vals[0]:.3f}"
    else:
        cell = "–"
    print(f"| {label} | {b:.3f} | {cell} | " + ", ".join(f"{v:.3f}" for v in vals) + " |" if b is not None
          else f"| {label} | – | {cell} | " + ", ".join(f"{v:.3f}" for v in vals) + " |")

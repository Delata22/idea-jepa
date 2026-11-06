"""Best-of-K sweep in free generation — does choosing between modes help, and can we choose?

For each model (the flow head, and P as control) and K in --Ks, runs scripts/gen_eval.py with K candidates per
step and a selector:
  oracle  decode the K candidates, the verifier picks a valid one (upper bound of selection)
  energy  masked-predictor energy of the stage-A model (--energy_run), no truth information
K = 1 is the plain generation (no selection). The flow's K candidates are K flow samples; P is deterministic, so
its K candidates are K SAMPLED decodings of the same idea (--dec_temp), greedy at K = 1.
Writes <out>/<model>_K<k>_<select>.jsonl (per story) and <out>/summary.{json,md}; the cost of a configuration
is reported per deduction step: flow passes (K x flow_steps), decodes (K), energy passes (K).
usage: uv run python scripts/sweep_bestofk.py --flow_run runs/v2/v2l_flow2a_hidreg_s0 --p_run runs/v2/v2i_pwin_delta_s0
       --energy_run runs/v2/v2h_stageA2_s0 [--Ks 1 4 8 16] [--n 90] [--out runs/v2/bestofk_s0]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--flow_run", default=None)
ap.add_argument("--p_run", default=None)
ap.add_argument("--models", nargs="+", default=["flow", "P"], choices=["flow", "P"])
ap.add_argument("--energy_run", required=True)
ap.add_argument("--Ks", type=int, nargs="+", default=[1, 4, 8, 16])
ap.add_argument("--selects", nargs="+", default=["oracle", "energy"])
ap.add_argument("--n", type=int, default=90, help="stories (gen_eval --n)")
ap.add_argument("--flow_steps", type=int, default=32)
ap.add_argument("--flow_temp", type=float, default=0.5)
ap.add_argument("--dec_temp", type=float, default=0.7, help="P's decoding temperature for K > 1")
ap.add_argument("--dec_temps", type=float, nargs="+", default=None,
                help="P: sweep several decoding temperatures (each K, K = 1 included as a sampled control)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/v2/bestofk")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
TOTAL = re.compile(r"^\s+(\S+)\s+(F1: .*?)\s+total (\d+)/(\d+)")
FAM = re.compile(r"(F\d): (\d+)/(\d+)")
CHANGED = re.compile(r"candidate changed at (\d+)/(\d+) steps")

configs = []
for model, run in (("flow", a.flow_run), ("P", a.p_run)):
    if model not in a.models:
        continue
    assert run, f"--{model.lower()}_run is required for model {model}"
    temps = a.dec_temps if (model == "P" and a.dec_temps) else [None]
    for T in temps:
        for K in a.Ks:
            if K == 1 and T is not None and 1 in a.Ks and T == temps[0] and model == "P":
                configs.append((model, run, 1, "first", None))          # greedy reference (once)
            for sel in (["first"] if K == 1 else a.selects):
                if K == 1 and T is None and model == "P" and a.dec_temps:
                    continue
                configs.append((model, run, K, sel, T))

rows = []
for model, run, K, sel, T in configs:
    name = f"{model}_K{K}_{sel}" + (f"_T{T}" if T is not None else "")
    cmd = [sys.executable, "scripts/gen_eval.py", run, "--n", str(a.n), "--show", "0", "--seed", str(a.seed),
           "--sampler", "flow" if model == "flow" else "P", "--n_cand", str(K), "--select", sel,
           "--flow_steps", str(a.flow_steps), "--flow_temp", str(a.flow_temp), "--jsonl", str(out / f"{name}.jsonl")]
    if model == "P" and T is not None:
        cmd += ["--dec_temp", str(T)]                  # sampled decoding (K = 1 with T: control vs greedy)
    elif model == "P" and K > 1 and not a.dec_temps:
        cmd += ["--dec_temp", str(a.dec_temp)]
    if sel == "energy":
        cmd += ["--energy_run", a.energy_run]
    print(f"[{name}] {' '.join(cmd[1:])}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out / f"{name}.log").write_text(res.stdout + res.stderr)
    tot = next((m for m in map(TOTAL.match, res.stdout.splitlines()) if m), None)
    if tot is None:
        print(f"[{name}] FAILED (see {out / (name + '.log')})", flush=True)
        rows.append({"name": name, "failed": True}); continue
    fams = {f: int(v) / max(1, int(t)) for f, v, t in FAM.findall(tot.group(2))}
    ch = CHANGED.search(res.stdout)
    steps = int(tot.group(4))
    row = {"name": name, "model": model, "K": K, "select": sel, "dec_temp": T, "valid": int(tot.group(3)) / steps, "steps": steps,
           **{f"valid_{f}": v for f, v in fams.items()},
           "changed": (int(ch.group(1)) / max(1, int(ch.group(2)))) if ch else 0.0,
           "cost_flow_passes_per_step": K * a.flow_steps if model == "flow" else 0,
           "cost_decodes_per_step": K, "cost_energy_passes_per_step": K if sel == "energy" else 0}
    rows.append(row)
    print(f"[{name}] valid {row['valid']:.1%}  " + "  ".join(f"{f} {v:.0%}" for f, v in fams.items())
          + f"  changed {row['changed']:.0%}", flush=True)

tag = "_".join(a.models)
json.dump({"args": vars(a), "rows": rows}, open(out / f"summary_{tag}.json", "w"), indent=1)
lines = ["| model | K | selector | dec. temp | valid | F1 | F2 | F3 | candidate changed | flow passes / step | decodes / step |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r.get("failed"):
        lines.append(f"| {r['name']} | | | | FAILED | | | | | | |"); continue
    lines.append(f"| {r['model']} | {r['K']} | {r['select']} | {r.get('dec_temp') or 'greedy'} | {r['valid']:.1%} | {r.get('valid_F1', 0):.0%} | "
                 f"{r.get('valid_F2', 0):.0%} | {r.get('valid_F3', 0):.0%} | {r['changed']:.0%} | "
                 f"{r['cost_flow_passes_per_step']} | {r['cost_decodes_per_step']} |")
(out / f"summary_{tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

"""Final bAbI table (tasks 1-3, 10k training questions): test error per seed, for Idea-JEPA (minimal recipe) and the
size-matched references, all evaluated with the same protocol (scripts/eval_pick.sh: snapshot chosen on validation, then
test on 1000 questions, answer = word extracted from the generated sentence).

usage: python scripts/babi_results_table.py [--out reports/babi_results.md]
Missing runs are shown as "-" so the table can be produced while runs are still going."""
import argparse
import json
import os
import re

G = "runs/babi_scale"
SEEDS = (0, 1, 2)
# model label -> run name pattern ({t} = task, {s} = seed). References: lr and dropout chosen on qa2 validation, seed 0.
MODELS = {
    "Idea-JEPA (minimal, ours)": "bq{t}_min6k_d64_s{s}",
    "B0 transformer": "bq{t}_b0dp_s{s}",
    "B3 window 16 + delta memory": "bq{t}_b3dp_s{s}",
    # same references without dropout (first configuration tried; dropout 0.3 was then chosen on qa2 validation)
    "B0, no dropout": "bq{t}_b0small_s{s}",
    "B3, no dropout": "bq{t}_b3small_s{s}",
}


def load(name):
    p = f"{G}/{name}.test.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p))[0]
    main = next(v for k, v in r["results"].items() if "|" not in k)
    step = re.search(r"@(\d+)$", r["run"])
    params = None
    fj = f"runs/babi/{name}/final.json"
    if os.path.exists(fj):
        params = json.load(open(fj)).get("n_params")
    return {"err": main["err"], "step": int(step.group(1)) if step else 6000, "params": params, "res": r["results"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/babi_results.md")
    a = ap.parse_args()
    lines = ["| model | params | task | seed 0 | seed 1 | seed 2 | mean | seeds < 5 % |", "|---|---|---|---|---|---|---|---|"]
    for label, pat in MODELS.items():
        for t in ((1, 2) if "no dropout" in label else (1, 2, 3)):   # qa3 *small_s0 runs already use dropout 0.3
            rows = [load(pat.format(t=t, s=s)) for s in SEEDS]
            done = [r for r in rows if r]
            params = next((f"{r['params'] / 1e6:.2f} M" for r in done if r["params"]), "-")
            cells = [f"{100 * r['err']:.1f} %" if r else "-" for r in rows]
            mean = f"{100 * sum(r['err'] for r in done) / len(done):.1f} %" if done else "-"
            ok = f"{sum(r['err'] < 0.05 for r in done)}/{len(done)}" if done else "-"
            lines.append(f"| {label} | {params} | qa{t} | " + " | ".join(cells) + f" | {mean} | {ok} |")
    # qa3 errors by number of moves of the asked object (ours only)
    extra = ["", "qa3, Idea-JEPA: test error by number of times the object moved", "",
             "| seed | 1 move | 2 moves | 3+ moves |", "|---|---|---|---|"]
    for s in SEEDS:
        r = load(MODELS["Idea-JEPA (minimal, ours)"].format(t=3, s=s))
        if r:
            c = [r["res"].get(f"qa3|moves{m}") for m in ("1", "2", "3+")]
            extra.append(f"| {s} | " + " | ".join(f"{100 * x['err']:.1f} % (n={x['n']})" if x else "-" for x in c) + " |")
    text = "\n".join(lines + extra) + "\n"
    print(text)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(text)


if __name__ == "__main__":
    main()

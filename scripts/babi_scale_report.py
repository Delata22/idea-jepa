"""Report of the bAbI scale runs (30/09, notes §35): one row per run from runs/babi_scale/<run>.{val,test}.json, the run's
final.json / log / train.csv and <picked run>/babi_outputs_test.jsonl.
Columns: parameters, duration, L_jepa (validation) at step 1000, validation curve (every snapshot), picked step, test
error, accuracy by length bin (71+ = longer than any training story), error classes; then 4 correct + 4 wrong outputs
per run with an abbreviated context (supporting facts + question).
usage: python scripts/babi_scale_report.py <run name> [...] [--out runs/babi_scale/report.md]
"""
import argparse
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, "scripts")

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default=None)
ap.add_argument("--examples", type=int, default=4)
a = ap.parse_args()
G = Path("runs/babi_scale")
BINS = ["<=15", "16-30", "31-50", "51-70", "71-100", "101+"]   # before 30/09 16h: "71+" (shown under 71-100)


def step_of(run):
    m = re.search(r"@(\d+)$", run)
    return int(m.group(1)) if m else None


def jepa_at_1000(name):
    log = Path(f"runs/logs/{name}.log")
    if log.exists():
        m = re.search(r"eval step 1000: .*?L_jepa=([0-9.]+)", log.read_text())
        if m:
            return float(m.group(1))
    return None


lines, ex_lines = [], []
head = ("| run | param. | durée | ms/pas | L_jepa@1000 (val) | L_jepa réponses (test) | courbe val (exactitude, tous les 1000) | pas choisi | "
        "erreur test | exact. cas (a) porté | erreurs (a)/(b)/(c) | " + " | ".join(BINS) + " | erreurs (a/b/c/d/e/hors format) |")
lines += [head, "|" + "---|" * (head.count("|") - 1)]
for n in a.runs:
    fin = Path(f"runs/babi/{n}/final.json")
    if not fin.exists():
        lines.append(f"| {n} | pas fini |"); continue
    f = json.loads(fin.read_text())
    params, dur = f.get("n_params"), f.get("time", 0) / 60
    curve, picked, err, bins, errs, jans, case_acc, case_err = "", "", "", [""] * len(BINS), "", "—", "—", "—"
    vj, tj = G / f"{n}.val.json", G / f"{n}.test.json"
    if vj.exists():
        rows = json.loads(vj.read_text())
        fam = next(k for k in rows[0]["results"] if "|" not in k)
        pts = sorted(((step_of(r["run"]) or f["step"]), r["results"][fam]["acc"]) for r in rows)
        curve = " ".join(f"{acc:.2f}" for _, acc in pts)
    if tj.exists():
        t = json.loads(tj.read_text())[0]
        picked = str(step_of(t["run"]) or f["step"])
        res = t["results"]
        fam = next(k for k in res if "|" not in k)
        if f"{fam}|71+" in res:
            res[f"{fam}|71-100"] = res[f"{fam}|71+"]
        err = f"{res[fam]['err'] * 100:.1f} %"
        bins = [f"{res[f'{fam}|{b}']['acc']:.2f} (n={res[f'{fam}|{b}']['n']})" if f"{fam}|{b}" in res else "—" for b in BINS]
        q = res[fam]
        errs = "/".join(str(q.get(k, 0)) for k in ("err_a_initial_place", "err_b_earlier_carrier_place",
                                                   "err_c_other_actor_place", "err_d_other_place_in_story",
                                                   "err_e_place_not_in_story", "out_of_format"))
        if "L_jepa_answer" in q:
            jans = f"{q['L_jepa_answer']:.3f}"
        outs = Path(t["run"]) / "babi_outputs_test.jsonl"
        recs = [json.loads(x) for x in outs.read_text().splitlines()] if outs.exists() else []
        if recs and fam == "qa2":
            # notes §35.18: object state at the question, (a) carried / (b) dropped, carrier stayed / (c) dropped, carrier left
            import torch
            from babi_dropcase_lib import analyse
            dd = torch.load(f"{t['data']}/test.pt", weights_only=False)
            cn = {"a": [0, 0], "b": [0, 0], "c": [0, 0]}
            for r in recs:
                st = [[dd["vocab"][x] for x in i["tokens"]] for i in dd["docs"][r["i"]]["ideas"] if i["step"] == "premise"]
                qw = r["question"].split()
                res_ = analyse(st, qw[qw.index("the") + 1])
                if res_ is not None:
                    cn[res_[0][0]][0] += 1; cn[res_[0][0]][1] += r["ok"]
            case_acc = f"{cn['a'][1] / max(cn['a'][0], 1):.1%} (n={cn['a'][0]})"
            case_err = "/".join(str(cn[c][0] - cn[c][1]) for c in "abc")
        if recs:
            good = [r for r in recs if r["ok"]][:a.examples]
            bad = [r for r in recs if not r["ok"]][:a.examples]
            ex_lines.append(f"\n### {n} (pas {picked})")
            for r in good + bad:
                mark = "✓" if r["ok"] else f"✗ ({r['err']})"
                ex_lines.append(f"- {mark} [{r['n_ideas']} idées] appuis : « {' / '.join(r['support'])} » ; "
                                f"{r['question']} → « {r['generated']} » (attendu : {r['answer']})")
    jp = jepa_at_1000(n)
    msps = f"{f.get('time', 0) / max(f.get('step', 1), 1) * 1000:.0f}"
    lines.append(f"| {n} | {params / 1e6:.3f} M | {dur:.0f} min | {msps} | {jp if jp is not None else '—'} | {jans} | {curve} | {picked} | "
                 f"{err} | {case_acc} | {case_err} | " + " | ".join(bins) + f" | {errs} |")
text = "\n".join(lines + ex_lines) + "\n"
print(text)
if a.out:
    Path(a.out).write_text(text)

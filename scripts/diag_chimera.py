"""Chimera test: are free-generation false facts built from pieces of VALID candidates of the same step?
For each sentence judged "false" in gen_eval jsonl files: the valid next deductions at that step (verifier state rebuilt
from the model's own valid sentences); each component of the false fact (F2: person 1, relation, person 2; F1: object,
place; F3: person, colour, item...) is looked up at the same position in the valid candidates with the same predicate.
  chimera        every component comes from some valid candidate (never all from the same one: the fact is false)
  out_of_cands   at least one component appears in no valid candidate (or no candidate has the predicate)
Chance level: the same test against the candidate set of an error step of ANOTHER story (same family), averaged over
permutations. Split by number of valid candidates at the step (1 / 2 / 3+).
usage: uv run python scripts/diag_chimera.py <jsonl> [<jsonl> ...] [--models m1 m2 ...] [--perm 20]
"""
import argparse, collections, json, random, statistics as st, sys
import torch
sys.path.insert(0, "src")
from ideajepa.data import verifier as V

ap = argparse.ArgumentParser(); ap.add_argument("jsonl", nargs="+"); ap.add_argument("--models", nargs="*")
ap.add_argument("--perm", type=int, default=20); a = ap.parse_args()
docs = torch.load("data/reasoning/val.pt", weights_only=False)["docs"]


def comps(f):
    out = []
    for i, x in enumerate(f[1:], 1):
        if isinstance(x, tuple):
            out += [((i, j), y) for j, y in enumerate(x)]
        else:
            out.append(((i,), x))
    return out


def is_chimera(f, cands):
    same = [c for c in cands if c[0] == f[0]]
    if not same:
        return False
    pools = collections.defaultdict(set)
    for c in same:
        for pos, x in comps(c):
            pools[pos].add(x)
    return all(x in pools[pos] for pos, x in comps(f))


errs = collections.defaultdict(list)       # (model, family) -> [(story, fact, cands)]
seen = set()
for path in a.jsonl:
    for r in map(json.loads, open(path)):
        if a.models and r["model"] not in a.models:
            continue
        key = (r["model"], r["story"])
        if key in seen:
            continue
        seen.add(key)
        fam = r["family"]; prem = V.doc_premises(docs[r["story"]]); said = []
        for sent, vd in zip(r["sentences"], r["verdicts"]):
            f, _ = V.parse_sentence(sent)
            if vd == "false" and f is not None:
                errs[(r["model"], fam)].append((r["story"], f, sorted(V.valid_next(fam, prem, said), key=str)))
            if vd.startswith("valid") and f is not None:
                said.append(f)
rng = random.Random(0)
nb = lambda n: "1" if n == 1 else ("2" if n == 2 else "3+")
rows = collections.defaultdict(dict)
for (m, fam), E in sorted(errs.items()):
    E = [e for e in E if e[2]]
    if not E:
        continue
    ch = [is_chimera(f, c) for _, f, c in E]
    chance = []
    for _ in range(a.perm):
        for s, f, c in E:
            other = [e for e in E if e[0] != s]
            if other:
                chance.append(is_chimera(f, rng.choice(other)[2]))
    by_n = collections.defaultdict(list)
    for (s, f, c), x in zip(E, ch):
        by_n[nb(len(c))].append(x)
    rows[(m, fam)] = {"n_err": len(E), "chimera": sum(ch) / len(E), "chance": sum(chance) / max(1, len(chance)),
                      **{f"chimera_n{k}": (sum(v) / len(v), len(v)) for k, v in by_n.items()}}
print("| model | fam | false facts | chimera | chance (permutation) | chimera n=1 | n=2 | n=3+ |")
print("|---|---|---|---|---|---|---|---|")
for (m, fam), d in rows.items():
    cell = lambda k: f"{d[k][0]:.2f} ({d[k][1]})" if k in d else "–"
    print(f"| {m} | {fam} | {d['n_err']} | {d['chimera']:.2f} | {d['chance']:.2f} | {cell('chimera_n1')} | {cell('chimera_n2')} | {cell('chimera_n3+')} |")
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for (m, fam), d in rows.items():
    g = m.rsplit("_s", 1)[0]
    for k in ("chimera", "chance"):
        groups[(g, fam)][k].append(d[k])
print("\n| model group | fam | chimera | chance |\n|---|---|---|---|")
for (g, fam), d in sorted(groups.items()):
    f_ = lambda xs: f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"
    print(f"| {g} | {fam} | {f_(d['chimera'])} | {f_(d['chance'])} |")

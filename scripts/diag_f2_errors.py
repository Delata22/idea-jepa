"""Free-generation F2 (kinship) errors from a gen_eval jsonl: nature, first-error rank, propagation, bimodality.
For every non-valid F2 deduction, the verifier state is rebuilt from the model's own previous sentences:
  (reference facts: the valid-now facts first, then every entailed fact of the story)
  wrong_relation        same two people as a reference fact, other relation (e.g. uncle <-> grandfather)
  wrong_person_ante     same relation and one person of a valid-now fact; the other person is the middle person of
                        that fact's antecedent edge (the "since" A: y -[r2]-> z)
  wrong_person_other    same relation and one person of a valid-now fact; the other person is someone else
  premature / premise / repeat / unparsed   verifier verdicts;   other   none of the above
usage: uv run python scripts/diag_f2_errors.py runs/logs/day4_since_gen.jsonl [--models m1 m2 ...] [--examples 4]
"""
import argparse, collections, json, statistics as st, sys
import torch
sys.path.insert(0, "src")
from ideajepa.data import reasoning as R, verifier as V

ap = argparse.ArgumentParser(); ap.add_argument("jsonl"); ap.add_argument("--models", nargs="*"); ap.add_argument("--examples", type=int, default=4)
a = ap.parse_args()
docs = torch.load("data/reasoning/val.pt", weights_only=False)["docs"]
rows = [r for r in map(json.loads, open(a.jsonl)) if r["family"] == "F2" and (not a.models or r["model"] in a.models)]
KINDS = ["wrong_relation", "wrong_person_ante", "wrong_person_other", "premature", "premise", "repeat", "unparsed", "other"]


def classify(f, verdict, prem, said):
    if verdict != "false" or f is None:
        return verdict if verdict in KINDS else "other"
    now = sorted(V.valid_next("F2", prem, said), key=str) + sorted(V.derivable("F2", prem), key=str)
    for g in now:
        if (g[1], g[3]) == (f[1], f[3]) and g[2] != f[2]:
            return "wrong_relation"
    for g in now:
        if g[2] == f[2] and (g[1] == f[1]) != (g[3] == f[3]):
            wrong = f[3] if g[1] == f[1] else f[1]
            ante = R.antecedent("F2", g, list(prem) + list(said))
            return "wrong_person_ante" if ante is not None and wrong in (ante[1], ante[3]) else "wrong_person_other"
    return "other"


stat = collections.defaultdict(lambda: collections.defaultdict(list))
kinds = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(dict)
for r in rows:
    m = r["model"]; doc = docs[r["story"]]; prem = V.doc_premises(doc)
    said, ok_seq = [], []
    for sent, vd in zip(r["sentences"], r["verdicts"]):
        f, _ = V.parse_sentence(sent)
        ok = vd.startswith("valid")
        if not ok:
            kinds[m][classify(f, vd, prem, said)] += 1
        ok_seq.append(ok)
        if f is not None:
            said.append(f)
    n = len(ok_seq)
    stat[m]["all_ok"].append(all(ok_seq)); stat[m]["all_bad"].append(not any(ok_seq))
    stat[m]["mixed"].append(any(ok_seq) and not all(ok_seq)); stat[m]["valid"].append(sum(ok_seq) / n)
    if not all(ok_seq):
        i = ok_seq.index(False)
        stat[m]["first_err_rank"].append(i + 1)
        if i + 1 < n:
            stat[m]["valid_after_first_err"].append(sum(ok_seq[i + 1:]) / (n - i - 1))
    examples[m][r["story"]] = (r["sentences"], r["verdicts"])

models = sorted(stat)
groups = collections.defaultdict(list)
for m in models:
    groups[m.rsplit("_s", 1)[0]].append(m)
fmt = lambda xs: (f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}") if xs else "–"
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print("| measure | " + " | ".join(models) + " |"); print("|---|" + "---|" * len(models))
for key, lab in (("valid", "F2 valid (share)"), ("all_ok", "stories all correct"), ("all_bad", "stories all wrong"),
                 ("mixed", "stories mixed"), ("first_err_rank", "rank of first error (mean)"),
                 ("valid_after_first_err", "valid rate after the first error")):
    print(f"| {lab} | " + " | ".join(f"{mean([float(x) for x in stat[m][key]]):.2f}" for m in models) + " |")
for k in KINDS:
    print(f"| errors: {k} | " + " | ".join(str(kinds[m][k]) for m in models) + " |")
hist = {m: collections.Counter(stat[m]["first_err_rank"]) for m in models}
for rk in (1, 2, 3, 4):
    print(f"| first error at rank {rk}{'+' if rk == 4 else ''} (stories) | " + " | ".join(
        str(sum(v for k, v in hist[m].items() if (k >= 4 if rk == 4 else k == rk))) for m in models) + " |")
print("\n## Examples: B w3 all correct / since wrong (same story, same seed)")
shown = 0
for s_ in sorted({r["story"] for r in rows}):
    for seed in ("s0", "s1", "s2"):
        b, c = examples.get(f"v3b_w3_{seed}", {}).get(s_), examples.get(f"v3b_w3_since_{seed}", {}).get(s_)
        if b and c and all(v.startswith("valid") for v in b[1]) and not all(v.startswith("valid") for v in c[1]):
            prem = [" ".join(docs[s_]["ideas"][i]["tokens"] and []) for i in []]
            print(f"- story {s_} ({seed})\n  B w3 : " + " | ".join(b[0]) + "\n  since: " +
                  " | ".join(("✓ " if v.startswith("valid") else f"✗[{v}] ") + t for t, v in zip(*c)))
            shown += 1
            break
    if shown >= a.examples:
        break

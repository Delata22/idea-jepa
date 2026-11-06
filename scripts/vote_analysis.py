"""Why does the vote fail while the oracle reaches 96-97 %? From gen_eval --log_cands files (K samples of the
SAME idea Ẑ, decoder sampling at T = 0.7: P is deterministic, so all the diversity comes from the decoder).
Per step where the GREEDY decoding of Ẑ is wrong: (i) is there a valid sample among the K; (ii) the frequency rank of the
most frequent VALID fact among the samples (1st, 2nd, 3rd+, tied / single); (iii) share of the K samples that carry the SAME
fact as the greedy error; per group: F1 immobile / moved object (by the greedy fact's object), F2, F3. 5 examples.
usage: python scripts/vote_analysis.py <cands.jsonl> [...] [--summary out.md]
"""
import argparse
import collections
import json
import sys

import torch

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--summary", default=None)
a = ap.parse_args()
docs = torch.load("data/reasoning/val.pt", weights_only=False)["docs"]
res = collections.defaultdict(collections.Counter)
same_share = collections.defaultdict(list)
ex = []
for fn in a.files:
    for line in open(fn):
        r = json.loads(line)
        if r["greedy_verdict"].startswith("valid"):
            continue
        seed = r["story_key"][1]
        fam = r["family"]
        g = fam
        gf = V.parse_sentence(r["greedy"])[0]
        if fam == "F1" and gf is not None and gf[0] == "objin":
            pf = V.doc_premises(docs[r["story_key"][0]])
            up = R.f1_useful_premises(pf)
            g = "F1mov" if gf[1] in up and pf[up[gf[1]]][0] == "goes" else "F1imm"
        facts = [V.parse_sentence(c)[0] for c in r["cands"]]
        valid = [f for f, v in zip(facts, r["cand_verdicts"]) if v.startswith("valid")]
        cnt = collections.Counter(f for f in facts if f is not None)
        ranked = [f for f, _ in cnt.most_common()]
        for key in ((seed, g), (seed, "all"), ("all", g), ("all", "all")):
            c = res[key]
            c["n"] += 1
            c["valid_exists"] += bool(valid)
            if valid:
                best_valid = max(set(valid), key=lambda f: cnt[f])
                top = cnt.most_common(1)[0][1]
                rk = 1 + sum(1 for f in ranked if cnt[f] > cnt[best_valid])
                tie = sum(1 for f in ranked if cnt[f] == cnt[best_valid]) > 1
                c["rank_" + ("1" if rk == 1 and not tie else "tie" if rk == 1 else "2" if rk == 2 else "3+")] += 1
                c["valid_single"] += cnt[best_valid] == 1
            same_share[key].append(sum(f == gf for f in facts) / len(facts))
        if len(ex) < 5 and valid and gf in cnt and cnt[gf] >= len(facts) // 2:
            ex.append({"seed": seed, "group": g, "premises": r["premises"][-4:], "said": r["said"][-2:], "greedy": r["greedy"],
                       "cands": r["cands"], "verdicts": r["cand_verdicts"]})
L = ["Pas où la phrase GLOUTONNE de Ẑ est fausse ; K = 8 échantillons du décodeur (T = 0,7) sur la même idée Ẑ.", "",
     "| graine | groupe | pas | un valide parmi les 8 | fait valide le plus fréquent : 1er / ex æquo / 2e / 3e+ | valide unique (1 sur 8) | part des 8 = même erreur que le glouton |",
     "|---|---|---|---|---|---|---|"]
for key in sorted(res, key=lambda k: (str(k[0]), str(k[1]))):
    c = res[key]; n = c["n"]
    f = lambda x: f"{c[x] / n:.2f}"
    ss = same_share[key]
    L.append(f"| {key[0]} | {key[1]} | {n} | {f('valid_exists')} | {f('rank_1')} / {f('rank_tie')} / {f('rank_2')} / {f('rank_3+')} | "
             f"{f('valid_single')} | {sum(ss) / len(ss):.2f} |")
L += ["", "Exemples (le glouton se trompe, la même erreur fait au moins la moitié des 8, un valide existe) :"]
for e in ex:
    L.append(f"- {e['seed']} {e['group']} : … {' | '.join(e['premises'])} ; déjà dit : {' | '.join(e['said']) or '—'} ; glouton « {e['greedy']} » ; "
             "échantillons : " + " ; ".join(f"{c} [{v}]" for c, v in zip(e["cands"], e["verdicts"])))
out = "\n".join(L)
print(out)
if a.summary:
    open(a.summary, "w").write(out + "\n")

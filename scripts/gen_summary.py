"""Summary of gen_eval jsonl files (free generation) per model: total, F1 immobile / moved objects (moved = the useful
premise of the written object is a "goes", R.f1_useful_premises), F2, F3, error kinds, F1 chimeras (wrong "objin" fact whose
object and location both come from valid candidates of the step, from different ones).
Moved objects (§32.26): type of each WRONG F1 sentence on a moved object, first match in this order: (a) its INITIAL place;
(b) an EARLIER place of its carrier (the person of the useful "goes": initial "at" and earlier "goes"); (c) the final place
of another candidate object (chimera); (d) other. Per moved object of the story: omitted (no sentence about it), written
wrong only, written right. Examples with the premises about the object and its carrier.
usage: python scripts/gen_summary.py <file.jsonl> [...] [--summary out.md] [--examples 8]
"""
import argparse
import collections
import json
import sys

import torch

sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--summary", default=None)
ap.add_argument("--examples", type=int, default=8)
a = ap.parse_args()
data = torch.load("data/reasoning/val.pt", weights_only=False)
docs, vocab = data["docs"], data["vocab"]
wtxt = lambda i: " ".join(vocab[t] for t in i["tokens"])
cnt = collections.defaultdict(collections.Counter)
examples = []
for fn in a.files:
    tag = fn.rsplit("/", 1)[-1].replace(".jsonl", "")
    for line in open(fn):
        r = json.loads(line)
        m = f"{r['model']} [{tag}]"
        doc = docs[r["story"]]
        pf = V.doc_premises(doc)
        fam = r["family"]
        up = R.f1_useful_premises(pf) if fam == "F1" else {}
        final = R.final_state_f1(pf) if fam == "F1" else {}
        moved = {o for o, i in up.items() if pf[i][0] == "goes"}
        said = []
        about = collections.defaultdict(list)                  # moved object -> verdicts of the sentences about it
        for s, vd in zip(r["sentences"], r["verdicts"]):
            f = V.parse_sentence(s)[0]
            ok = vd.startswith("valid")
            keys = [fam, "all"]
            if fam == "F1" and f is not None and f[0] == "objin" and f[1] in up:
                keys.append("F1mov" if f[1] in moved else "F1imm")
            for k in keys:
                c = cnt[(m, k)]
                c["n"] += 1; c["valid"] += ok; c[vd if not ok else "ok"] += 1
            if fam == "F1" and not ok and f is not None and f[0] == "objin":
                cands = V.valid_next("F1", pf, said) - set(said)
                locs = {x[1]: x[2] for x in cands}
                chim = f[1] in locs and any(l_ == f[2] for o_, l_ in locs.items() if o_ != f[1])
                cnt[(m, "F1")]["chimera"] += chim
                if f[1] in moved:
                    o, L = f[1], f[2]
                    init = next((p[2] for p in pf if p[0] == "objin" and p[1] == o), None)
                    carrier = pf[up[o]][1]
                    last = up[o]
                    earlier = {p[2] for i, p in enumerate(pf) if i < last and p[1] == carrier and p[0] in ("at", "goes")}
                    kind = ("a_initial" if L == init else "b_carrier_earlier" if L in earlier else
                            "c_other_object" if L in {l_ for o_, l_ in final.items() if o_ != o} else "d_other")
                    cnt[(m, "F1mov")]["err_" + kind] += 1
                    if len(examples) < 4 * a.examples:
                        ctx = [wtxt(i) for i in doc["ideas"] if i["step"] == "premise" and
                               (o in wtxt(i).split() or carrier in wtxt(i).split())]
                        examples.append((m, kind, s, f"final {final[o]}", ctx))
            if fam == "F1" and f is not None and f[0] == "objin" and f[1] in moved:
                about[f[1]].append(ok)
            if f is not None:
                said.append(f)
        for o in moved:
            c = cnt[(m, "F1mov")]
            v = about.get(o, [])
            c["obj"] += 1
            c["obj_omitted"] += not v; c["obj_right"] += any(v); c["obj_wrong_only"] += bool(v) and not any(v)
L = ["| modèle [fichier] | total | F1 | F1 immobiles | F1 déplacés | F2 | F3 | faux | prématurées | répétitions | chimères F1 |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
M = ["", "Objets déplacés : type des phrases fausses (a lieu initial / b lieu antérieur du porteur / c lieu d'un autre objet / d autre) ; "
     "objets déplacés de l'histoire : omis / écrits faux seulement / écrits justes", "",
     "| modèle [fichier] | a | b | c | d | objets | omis | faux seulement | justes |", "|---|---|---|---|---|---|---|---|---|"]
for m in sorted({k[0] for k in cnt}):
    g = lambda k: cnt.get((m, k), collections.Counter())
    pct = lambda k: f"{100 * g(k)['valid'] / g(k)['n']:.0f} (n={g(k)['n']})" if g(k)["n"] else "—"
    a_ = g("all")
    L.append(f"| {m} | {pct('all')} | {pct('F1')} | {pct('F1imm')} | {pct('F1mov')} | {pct('F2')} | {pct('F3')} | "
             f"{a_['false']} | {a_['premature']} | {a_['repeat']} | {g('F1')['chimera']} |")
    mv = g("F1mov")
    if mv["obj"]:
        M.append(f"| {m} | {mv['err_a_initial']} | {mv['err_b_carrier_earlier']} | {mv['err_c_other_object']} | {mv['err_d_other']} | "
                 f"{mv['obj']} | {mv['obj_omitted']} | {mv['obj_wrong_only']} | {mv['obj_right']} |")
E = ["", "Exemples d'erreurs sur objets déplacés (prémisses de l'objet et de son porteur) :"]
seen = collections.Counter()
for m, kind, s, fin, ctx in examples:
    if seen[kind] >= 2 or len(E) > a.examples + 2:
        continue
    seen[kind] += 1
    E.append(f"- {m} [{kind}] « {s} » ({fin}) ← " + " | ".join(ctx))
out = "\n".join(L + M + E)
print(out)
if a.summary:
    open(a.summary, "w").write(out + "\n")

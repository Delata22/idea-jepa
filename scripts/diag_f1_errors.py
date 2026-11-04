"""F1 ("state") error diagnosis: when a model states a wrong final location, what went wrong?

Reads gen_eval --jsonl outputs (one row per story and model) and re-simulates every object's trajectory from the
true premises of v2 val. Every non-valid F1 sentence "the O is/ends up in the L" is classified as:
  wrong_object        O is not an object of the story
  repeat              O's final location was already stated
  initial_location    O moved, the model gives its starting location (no move tracked)
  intermediate        L is a place O passed through, neither the start nor the end (partial tracking)
  followed_holder     O was dropped, the model puts it where its last holder went AFTER dropping it
                      (believes the object kept following the person)
  other_object_final  L is the final location of ANOTHER object (binding confusion between objects)
  person_location     L is where some person ends up
  other_location      any other location of the story
  unparsed_or_other   not an objin sentence
usage: uv run python scripts/diag_f1_errors.py <gen_eval.jsonl> [more.jsonl ...]
"""
import collections
import json
import sys

import torch

sys.path.insert(0, "src")
from ideajepa.data import verifier as V
from ideajepa.data.reasoning import LOCS

docs = torch.load("data/reasoning/val.pt", weights_only=False)["docs"]


def trajectories(prem):
    """For every object: list of locations over time (start first), last holder and the holder's moves after the
    drop."""
    pos, where, holder, hist, last_holder, after_drop = {}, {}, {}, {}, {}, collections.defaultdict(list)
    for f in prem:
        if f[0] == "at":
            pos[f[1]] = f[2]
        elif f[0] == "objin" and f[1] not in where:
            where[f[1]] = f[2]; holder[f[1]] = None; hist[f[1]] = [f[2]]
        elif f[0] == "goes":
            pos[f[1]] = f[2]
            for o, h in holder.items():
                if h == f[1]:
                    where[o] = f[2]; hist[o].append(f[2])
            for o, h in last_holder.items():
                if h == f[1] and holder[o] is None:
                    after_drop[o].append(f[2])
        elif f[0] == "takes":
            holder[f[2]] = f[1]; last_holder[f[2]] = f[1]; after_drop[f[2]] = []
        elif f[0] == "drops":
            holder[f[2]] = None
    return where, hist, after_drop, pos


per_model = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        if r["family"] != "F1":
            continue
        d = docs[r["story"]]
        prem = V.doc_premises(d)
        final, hist, after_drop, people_pos = trajectories(prem)
        c = per_model[r["model"]]
        for sent, vd in zip(r["sentences"], r["verdicts"]):
            c["sentences"] += 1
            f0, _ = V.parse_sentence(sent)
            if f0 is not None and f0[0] == "objin" and f0[1] in hist:   # error rate by number of moves of the object
                mv = "moved_0" if len(hist[f0[1]]) == 1 else ("moved_1" if len(hist[f0[1]]) == 2 else "moved_2+")
                c[f"{mv}_n"] += 1
                c[f"{mv}_valid"] += vd.startswith("valid")
            if vd.startswith("valid"):
                c["valid"] += 1; continue
            f, _ = V.parse_sentence(sent)
            if vd == "repeat":
                cat = "repeat"
            elif f is None or f[0] != "objin":
                cat = "unparsed_or_other"
            else:
                o, l = f[1], f[2]
                if o not in final:
                    cat = "wrong_object"
                elif len(hist[o]) > 1 and l == hist[o][0]:
                    cat = "initial_location"
                elif l in hist[o][1:-1] or (len(hist[o]) > 1 and l in hist[o][:-1] and l != hist[o][0]):
                    cat = "intermediate"
                elif l in after_drop.get(o, []):
                    cat = "followed_holder"
                elif any(final[o2] == l for o2 in final if o2 != o):
                    cat = "other_object_final"      # binding confusion: the location of ANOTHER object
                elif l in people_pos.values():
                    cat = "person_location"         # where some person ends up
                else:
                    cat = "other_location"
            c[cat] += 1
            if f is not None and f[0] == "objin" and f[1] in final and vd != "repeat":
                # chance baseline: a wrong location drawn uniformly among the story's other locations
                locs = {x for f2 in prem for x in f2[1:] if isinstance(x, str) and x in LOCS} - {final[f[1]]}
                if locs:
                    c["chance_n"] += 1
                    c["chance_other_object_final"] += sum(any(final[o2] == l2 for o2 in final if o2 != f[1])
                                                          for l2 in locs) / len(locs)
                    c["chance_initial_location"] += (hist[f[1]][0] in locs and len(hist[f[1]]) > 1) / len(locs)
            if len(examples[(r["model"], cat)]) < 2:
                examples[(r["model"], cat)].append((sent, {o_: h for o_, h in hist.items()}, final))

cats = ["wrong_object", "repeat", "initial_location", "intermediate", "followed_holder", "other_object_final",
        "person_location", "other_location", "unparsed_or_other"]
for model, c in per_model.items():
    bad = c["sentences"] - c["valid"]
    print(f"{model}: {c['sentences']} F1 deductions, valid {c['valid'] / max(1, c['sentences']):.1%}; "
          f"errors ({bad}):")
    for cat in cats:
        if c[cat]:
            print(f"   {cat:18s} {c[cat]:4d}  ({c[cat] / max(1, bad):.0%} of errors)")
    print("   valid by number of moves of the object: " + ", ".join(
        f"{mv[6:]} moves {c[mv + '_valid'] / c[mv + '_n']:.0%} (n={c[mv + '_n']})" for mv in ("moved_0", "moved_1", "moved_2+")
        if c[mv + "_n"]))
    if c["chance_n"]:
        print(f"   chance baseline over {c['chance_n']} wrong-location errors (uniform wrong location): "
              f"other_object_final {c['chance_other_object_final'] / c['chance_n']:.0%}, "
              f"initial_location {c['chance_initial_location'] / c['chance_n']:.0%}")
    for cat in cats:
        for sent, hist, final in examples[(model, cat)][:1]:
            print(f"   e.g. [{cat}] «{sent}»  trajectories: " + "; ".join(f"{o}: {' → '.join(h)}" for o, h in hist.items()))

"""F1: silence or wrong statement? And which wrong location?
For every F1 story (gen_eval --jsonl rows): each object of the premises that the reference chain asks about is
  stated_valid / stated_wrong / omitted (never mentioned in the model's deductions), split by moved vs never moved.
For wrong locations on never-moved objects: is the wrong location the one mentioned LAST in the premises (any
sentence), the last one stated about ANOTHER object's position, a person's location, or other?
usage: uv run python scripts/diag_f1_omission.py <gen_eval.jsonl> [...]"""
import sys, json, re, collections, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from ideajepa.data import verifier as V

docs = torch.load("data/reasoning/val.pt", weights_only=False)["docs"]


def trajectories(prem):
    pos, where, holder, hist = {}, {}, {}, {}
    for f in prem:
        if f[0] == "at": pos[f[1]] = f[2]
        elif f[0] == "objin" and f[1] not in where: where[f[1]] = f[2]; holder[f[1]] = None; hist[f[1]] = [f[2]]
        elif f[0] == "goes":
            pos[f[1]] = f[2]
            for o, h in holder.items():
                if h == f[1]: where[o] = f[2]; hist[o].append(f[2])
        elif f[0] == "takes": holder[f[2]] = f[1]
        elif f[0] == "drops": holder[f[2]] = None
    return where, hist, pos


per = collections.defaultdict(collections.Counter)
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        if r["family"] != "F1": continue
        m = r["model"]; d = docs[r["story"]]
        prem = V.doc_premises(d)
        where, hist, pos = trajectories(prem)
        ref_objs = {tuple(i["fact"])[1] for i in d["ideas"] if i["step"] == "deduction"}
        said = {}
        for s in r["sentences"]:
            f, _ = V.parse_sentence(s)
            if f is not None and f[0] == "objin" and f[1] not in said: said[f[1]] = f[2]
        locs_in_order = [f[-1] for f in prem if f[0] in ("at", "objin", "goes")]
        obj_prem_locs = [f[2] for f in prem if f[0] == "objin"]
        for o in ref_objs:
            moved = "moved" if len(hist.get(o, [])) > 1 else "never_moved"
            if o not in said: per[m][f"{moved}_omitted"] += 1; continue
            if said[o] == where.get(o): per[m][f"{moved}_valid"] += 1; continue
            per[m][f"{moved}_wrong"] += 1
            if moved == "moved":
                wl = said[o]; h = hist[o]
                cat = ("initial" if wl == h[0] else "intermediate" if wl in h[1:-1] else
                       "other_object" if wl in [f[2] for f in prem if f[0] == "objin" and f[1] != o] else
                       "person" if wl in pos.values() else "other")
                per[m][f"moved_err_{cat}"] += 1
                places = sorted(set(f[-1] for f in prem if f[0] in ("at", "objin", "goes")) - {where[o]})
                if places: per[m]["chance_initial"] += (h[0] in places) / len(places); per[m]["chance_init_n"] += 1
            if moved == "never_moved":
                wl = said[o]
                other_obj_locs = [x for x in obj_prem_locs if x != where[o]]
                cat = ("last_location_mentioned" if locs_in_order and wl == locs_in_order[-1] else
                       "last_other_object_location" if other_obj_locs and wl == other_obj_locs[-1] else
                       "another_object_location" if wl in other_obj_locs else
                       "a_person_location" if wl in pos.values() else "other")
                per[m][f"wrong_{cat}"] += 1
                # chance of hitting the last mentioned location if the wrong location were uniform among story places
                ol = sorted(set(other_obj_locs))
                if ol: per[m]["chance_last_other"] += 1 / len(ol); per[m]["chance_other_n"] += 1
                per[m]["wrong_in_other_obj_locs"] += wl in other_obj_locs
                per[m]["hit_last_other_any"] += bool(other_obj_locs) and wl == other_obj_locs[-1]
                places = sorted(set(locs_in_order) - {where[o]})
                if places: per[m]["chance_last"] += (locs_in_order[-1] in places) / len(places); per[m]["chance_n"] += 1
for m, c in per.items():
    print(f"== {m}")
    for mv in ("moved", "never_moved"):
        tot = c[f"{mv}_valid"] + c[f"{mv}_wrong"] + c[f"{mv}_omitted"]
        if tot: print(f"   {mv:12s} n={tot}: valid {c[f'{mv}_valid']/tot:.0%}  wrong {c[f'{mv}_wrong']/tot:.0%}  omitted {c[f'{mv}_omitted']/tot:.0%}")
    mw = c["moved_wrong"]
    if mw:
        print("   wrong location on MOVED objects: " + "  ".join(f"{k[10:]} {c[k]/mw:.0%}" for k in sorted(c) if k.startswith("moved_err_"))
              + f"   (chance of the initial location if uniform among wrong places: {c['chance_initial']/max(c['chance_init_n'],1):.0%})")
    w = c["never_moved_wrong"]
    if w:
        print("   wrong location on never-moved objects: " + "  ".join(
            f"{k[6:]} {c[k]/w:.0%}" for k in sorted(c) if k.startswith("wrong_")) +
              f"   (chance of 'last mentioned' if uniform: {c['chance_last']/max(c['chance_n'],1):.0%}; "
              f"among wrong answers that are another object's location, share on the LAST such object "
              f"{c['hit_last_other_any']/max(c['wrong_in_other_obj_locs'],1):.0%} vs uniform {c['chance_last_other']/max(c['chance_other_n'],1):.0%})")

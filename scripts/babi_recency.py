"""Recency heuristic in bAbI qa2 errors: for the "c_other_actor_place" errors of a run (babi_outputs_test.jsonl),
is the actor whose final place was answered the LAST actor to have moved in the story? Compared with the same rate for a
random other actor (chance) and with how often the true carrier is the last mover in the correct answers.
usage: python scripts/babi_recency.py <picked run dir> [--data data/babi/qa2_10k] [--split test]"""
import argparse
import json
import random
import sys

import torch

sys.path.insert(0, "scripts")
MOVE = {"moved", "went", "journeyed", "travelled"}
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--data", default="data/babi/qa2_10k")
ap.add_argument("--split", default="test")
a = ap.parse_args()
d = torch.load(f"{a.data}/{a.split}.pt", weights_only=False)
docs, vocab = d["docs"], d["vocab"]
recs = [json.loads(x) for x in open(f"{a.run}/babi_outputs_{a.split}.jsonl")]
rng = random.Random(0)
n_c = last = chance = amb = 0
n_ok = ok_last = 0
for r in recs:
    st = [[vocab[t] for t in i["tokens"]] for i in docs[r["i"]]["ideas"] if i["step"] == "premise"]
    pos, movers = {}, []
    for s in st:
        if len(s) > 2 and s[1] in MOVE:
            pos[s[0]] = s[-2]; movers.append(s[0])
    if not movers:
        continue
    lastm = movers[-1]
    if r["err"] == "c_other_actor_place":
        chosen = [x for x, p in pos.items() if p == r["pred"]]
        n_c += 1
        last += lastm in chosen
        amb += len(chosen) > 1
        chance += len(chosen) / len(pos)               # a random actor lands on the answered place
    elif r["ok"]:
        n_ok += 1
        ok_last += pos.get(lastm) == r["answer"]
print(f"{a.run}: erreurs « autre acteur » {n_c} ; personnage choisi = DERNIER déplacé : {last} ({last / max(n_c, 1):.0%}) ; "
      f"hasard (acteur tiré au sort au même lieu) {chance / max(n_c, 1):.0%} ; lieu partagé par plusieurs acteurs {amb}")
print(f"  réponses justes : le lieu du dernier déplacé est la bonne réponse dans {ok_last}/{n_ok} ({ok_last / max(n_ok, 1):.0%})")

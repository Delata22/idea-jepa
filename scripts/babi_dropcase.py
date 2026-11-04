"""bAbI qa2 (notes §35.18): accuracy by the state of the queried object at the question — (a) still carried; (b) put down,
the ex-carrier has not moved since; (c) put down, the ex-carrier moved afterwards — share of the errors per case, and in
case (c) the share of errors answering the ex-carrier's CURRENT place. Second split: number of moves by OTHER persons between
the last action on the object (take / drop / carrier's move while holding it) and the question (0, 1-2, 3+).
usage: python scripts/babi_dropcase.py <picked run> [...] [--data data/babi/qa2_10k]"""
import argparse, collections, json
import torch

import sys
sys.path.insert(0, "scripts")
from babi_dropcase_lib import analyse

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--data", default="data/babi/qa2_10k")
a = ap.parse_args()
d = torch.load(f"{a.data}/test.pt", weights_only=False)
docs, vocab = d["docs"], d["vocab"]


for run in a.runs:
    recs = [json.loads(x) for x in open(f"{run}/babi_outputs_test.jsonl")]
    by_case, by_others = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    c_err_ex = 0
    n_err = sum(1 for r in recs if not r["ok"])
    for r in recs:
        st = [[vocab[t] for t in i["tokens"]] for i in docs[r["i"]]["ideas"] if i["step"] == "premise"]
        q = r["question"].split(); obj = q[q.index("the") + 1]
        res = analyse(st, obj)
        if res is None:
            continue
        case, others, ex_place = res
        by_case[case][0] += 1; by_case[case][1] += r["ok"]
        b = "0" if others == 0 else "1-2" if others <= 2 else "3+"
        by_others[b][0] += 1; by_others[b][1] += r["ok"]
        if case == "c_dropped_carrier_left" and not r["ok"]:
            c_err_ex += r["pred"] == ex_place
    print(run)
    for c in ("a_carried", "b_dropped_carrier_stayed", "c_dropped_carrier_left"):
        n, ok = by_case[c]
        print(f"  {c}: {ok}/{n} = {ok / max(n, 1):.1%}  | erreurs {n - ok} ({(n - ok) / max(n_err, 1):.0%} des {n_err})")
    nc = by_case["c_dropped_carrier_left"]; ec = nc[0] - nc[1]
    print(f"  (c) erreurs = lieu ACTUEL de l'ex-porteur : {c_err_ex}/{ec} = {c_err_ex / max(ec, 1):.0%}")
    print("  déplacements d'autres personnes depuis la dernière action sur l'objet : "
          + "  ".join(f"{b} {by_others[b][1]}/{by_others[b][0]} = {by_others[b][1] / max(by_others[b][0], 1):.1%}" for b in ("0", "1-2", "3+")))

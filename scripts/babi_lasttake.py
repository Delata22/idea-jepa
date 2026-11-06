"""bAbI qa2 recency test on the "take" template: accuracy split by whether the TRUE carrier of the queried
object (the last person who took it) is the author of the LAST take of the story (any object), and by whether that last
take is of the queried object itself; share of the errors in each case. Uses <run>/babi_outputs_test.jsonl.
usage: python scripts/babi_lasttake.py <picked run> [...] [--data data/babi/qa2_10k]"""
import argparse, json
import torch

TAKE = {"got", "picked", "grabbed", "took"}
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--data", default="data/babi/qa2_10k")
a = ap.parse_args()
d = torch.load(f"{a.data}/test.pt", weights_only=False)
docs, vocab = d["docs"], d["vocab"]
for run in a.runs:
    recs = [json.loads(x) for x in open(f"{run}/babi_outputs_test.jsonl")]
    cnt = {}
    for r in recs:
        st = [[vocab[t] for t in i["tokens"]] for i in docs[r["i"]]["ideas"] if i["step"] == "premise"]
        q = r["question"].split(); obj = q[q.index("the") + 1]
        takes = [(s[0], [x for x in s if x not in (".", "there", "up")][-1]) for s in st if len(s) > 2 and s[1] in TAKE]
        carrier = next((who for who, o in reversed(takes) if o == obj), None)
        if carrier is None or not takes:
            continue
        for key, cond in (("carrier_is_last_taker", takes[-1][0] == carrier), ("last_take_is_this_object", takes[-1][1] == obj)):
            c = cnt.setdefault((key, cond), [0, 0]); c[0] += 1; c[1] += r["ok"]
    n_err = sum(1 for r in recs if not r["ok"])
    print(run)
    for key in ("carrier_is_last_taker", "last_take_is_this_object"):
        yes, no = cnt.get((key, True), [0, 0]), cnt.get((key, False), [0, 0])
        err_no = no[0] - no[1]
        print(f"  {key}: oui {yes[1]}/{yes[0]} = {yes[1] / max(yes[0], 1):.1%}  |  non {no[1]}/{no[0]} = {no[1] / max(no[0], 1):.1%}"
              f"  |  part des erreurs dans « non » {err_no}/{err_no + yes[0] - yes[1]} = {err_no / max(err_no + yes[0] - yes[1], 1):.0%}")

"""CLUTRR (EMNLP 2019 release, authors' Google Drive archive) -> Idea-JEPA documents (notes §34.6 protocol).
Sets: data_089907f8 (train k = 2, 3; test k = 2..10) = "train23", data_db9b8f04 (train k = 2, 3, 4) = "train234". Clean stories
only (task 1.x). Text input: one idea per sentence (the story text as written), lower-cased, words and punctuation.
Names -> entity identifiers drawn at random PER STORY (ent_1 .. ent_20, same rule for every model), to avoid the unseen-names
trap of the test set. Question idea: "how is <B> related to <A> ?" for the query (A, B). Deductions (templates written by us):
a fact (X, r, Y) of the proof ("Y is X's r") is rendered "so <Y> is the <r> of <X> ." Two target variants:
  answer : the final fact only (one deduction);
  chain  : every fact derived in proof_state, bottom-up (leaves first), the final fact last.
Validation = last 10 % of the train file; test = one file per k. Output data/clutrr/<set>_<variant>/{train,val,test_k<k>}.pt,
with doc fields "answer" (relation word), "query" (A, B as entity ids), "proof_facts" (derived facts, entity ids),
"story_facts" (story edges as facts).
usage: python scripts/prepare_clutrr.py
"""
import ast
import csv
import glob
import random
import re
from pathlib import Path

import torch

SRC = Path("data/public/clutrr")
OUT = Path("data/clutrr")
SETS = {"train23": "data_089907f8", "train234": "data_db9b8f04"}
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
N_ENT = 20


def tokens(text):
    return re.findall(r"ent_\d+|[a-z]+|'s|[.,!?']", text.lower())


def sentences(story):
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", story.strip()) if s.strip()]
    return [p if p[-1] in ".!?" else p + " ." for p in parts]


def proof_facts(ps):
    """derived facts bottom-up (leaves first), final (the first dict's head) last."""
    out = []
    for d in reversed(ast.literal_eval(ps)):
        for head in d:
            if head not in out:
                out.append(head)
    return out


def convert(row, rng):
    names = sorted(set(re.findall(r"\[([^\]]+)\]", row["story"])))
    ids = rng.sample(range(1, N_ENT + 1), len(names))
    ent = {n: f"ent_{i}" for n, i in zip(names, ids)}
    story = re.sub(r"\[([^\]]+)\]", lambda m: ent[m.group(1)], row["story"])
    a, b = ast.literal_eval(row["query"])
    facts = [(ent[x], r, ent[y]) for x, r, y in proof_facts(row["proof_state"])]
    edges = ast.literal_eval(row["story_edges"]); rels = ast.literal_eval(row["edge_types"])
    return {"sents": sentences(story), "question": f"how is {ent[b]} related to {ent[a]} ?", "answer": row["target"],
            "query": (ent[a], ent[b]), "proof_facts": facts, "k": len(edges),
            "story_rel_count": len(rels)}


def render(f):
    x, r, y = f
    return f"so {y} is the {r} of {x} ."


def main():
    rng = random.Random(0)
    raw = {}
    for name, d in SETS.items():
        for f in glob.glob(str(SRC / d / "*.csv")):
            base = Path(f).stem
            split = "train" if base.endswith("_train") else "test"
            rows = list(csv.DictReader(open(f)))
            ex = [convert(r, rng) for r in rows]
            if split == "train":
                raw[(name, "train")] = ex
            else:
                k = int(base.split("_")[0].split(".")[1])
                raw[(name, f"test_k{k}")] = ex
    vocab = set()
    for ex in raw.values():
        for e in ex:
            for s in e["sents"] + [e["question"]] + [render(f) for f in e["proof_facts"]]:
                vocab.update(tokens(s))
    itos = SPECIAL + sorted(vocab | {f"ent_{i}" for i in range(1, N_ENT + 1)})
    w2i = {w: i for i, w in enumerate(itos)}
    OUT.mkdir(parents=True, exist_ok=True)

    def doc(e, variant):
        ideas = [{"tokens": [w2i[t] for t in tokens(s)], "step": "premise", "fact": None} for s in e["sents"]]
        ideas.append({"tokens": [w2i[t] for t in tokens(e["question"])], "step": "question", "fact": None})
        targets = e["proof_facts"] if variant == "chain" else e["proof_facts"][-1:]
        ideas += [{"tokens": [w2i[t] for t in tokens(render(f))], "step": "deduction", "fact": None} for f in targets]
        return {"ideas": ideas, "family": f"k{e['k']}", "answer": e["answer"], "query": e["query"],
                "proof_facts": e["proof_facts"]}

    for name in SETS:
        for variant in ("answer", "chain"):
            dd = OUT / f"{name}_{variant}"
            dd.mkdir(parents=True, exist_ok=True)
            torch.save({"vocab": itos}, dd / "meta.pt")
            tr = raw[(name, "train")]
            n_val = len(tr) // 10
            parts = {"train": tr[:-n_val], "val": tr[-n_val:]}
            parts.update({k: v for (n_, k), v in raw.items() if n_ == name and k.startswith("test_k")})
            for split, ex in parts.items():
                torch.save({"docs": [doc(e, variant) for e in ex], "vocab": itos}, dd / f"{split}.pt")
            print(f"{name}_{variant}: " + "  ".join(f"{k} {len(v)}" for k, v in sorted(parts.items(), key=lambda x: (len(x[0]), x[0]))))
    print(f"vocab {len(itos)} tokens")


if __name__ == "__main__":
    main()

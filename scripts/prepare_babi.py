"""bAbI v1.2 qa1-3 -> Idea-JEPA documents.
One example per question: the story statements before the question (earlier questions are not statements) are premise
ideas, one per sentence; the question is an idea of its own (step "question"); the answer is rewritten as ONE deduction
sentence with a fixed template per task (written by us, documented here and in the report):
  qa1, qa2  "Where is X?"                 -> "so X is in the <answer> ."
  qa3       "Where was the O before the L?" -> "so the O was in the <answer> before the L ."
Supporting-fact indices are kept in each document (field "support", story-statement indices) for diagnostics only, never
used in training. Lower-cased, word tokens + "." (as v2). Splits: official train / test; validation = the last 10 % of the
official train examples. Regimes: 1k (en/) and 10k (en-10k/). No truncation (every example kept whole).
One shared vocabulary for qa1-3 x {1k, 10k}: data/babi/meta.pt. Output: data/babi/<task>_<1k|10k>/{train,val,test}.pt.
usage: python scripts/prepare_babi.py
"""
import collections
import glob
import re
from pathlib import Path

import torch

SRC = Path("data/public/babi/tasks_1-20_v1-2")
OUT = Path("data/babi")
TASKS = {"qa1": "qa1_single-supporting-fact", "qa2": "qa2_two-supporting-facts", "qa3": "qa3_three-supporting-facts"}
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def words(s):
    return re.findall(r"[a-z]+|[.?]", s.lower())


def answer_sentence(task, question, answer):
    q = words(question)
    if task in ("qa1", "qa2"):                   # where is X ?
        assert q[:2] == ["where", "is"] and q[-1] == "?", question
        return ["so"] + q[2:-1] + ["is", "in", "the", answer.lower(), "."]
    assert q[:2] == ["where", "was"] and "before" in q and q[-1] == "?", question   # where was the O before the L ?
    b = q.index("before")
    return ["so"] + q[2:b] + ["was", "in", "the", answer.lower()] + q[b:-1] + ["."]


def examples(path, task):
    story = []
    for line in open(path):
        idx, rest = line.strip().split(" ", 1)
        if idx == "1":
            story = []
        if "\t" in rest:
            q, ans, sup = rest.split("\t")
            ids = [n for n, _ in story]
            yield {"task": task, "statements": [s for _, s in story], "question": q.strip(), "answer": ans.strip(),
                   "support": [ids.index(int(x)) for x in sup.split()]}
        else:
            story.append((int(idx), rest))


def main():
    raw = {}
    vocab = collections.Counter()
    for reg, d in (("1k", "en"), ("10k", "en-10k")):
        for t, stem in TASKS.items():
            for split in ("train", "test"):
                ex = list(examples(glob.glob(str(SRC / d / f"{stem}_{split}.txt"))[0], t))
                raw[(t, reg, split)] = ex
                for e in ex:
                    for s in e["statements"] + [e["question"]]:
                        vocab.update(words(s))
                    vocab.update(answer_sentence(t, e["question"], e["answer"]))
    itos = SPECIAL + sorted(vocab)
    w2i = {w: i for i, w in enumerate(itos)}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"vocab": itos, "templates": {"qa1": "so X is in the <answer> .", "qa2": "so X is in the <answer> .",
                                              "qa3": "so the O was in the <answer> before the L ."}}, OUT / "meta.pt")

    def doc(e):
        ideas = [{"tokens": [w2i[w] for w in words(s)], "step": "premise", "fact": None} for s in e["statements"]]
        ideas.append({"tokens": [w2i[w] for w in words(e["question"])], "step": "question", "fact": None})
        ideas.append({"tokens": [w2i[w] for w in answer_sentence(e["task"], e["question"], e["answer"])],
                      "step": "deduction", "fact": None})
        return {"ideas": ideas, "family": e["task"], "answer": e["answer"].lower(), "support": e["support"]}

    for (t, reg, split), ex in raw.items():
        if split != "train":
            continue
        n_val = len(ex) // 10
        parts = {"train": ex[:-n_val], "val": ex[-n_val:], "test": raw[(t, reg, "test")]}
        dd = OUT / f"{t}_{reg}"
        dd.mkdir(parents=True, exist_ok=True)
        torch.save({"vocab": itos, "templates": None}, dd / "meta.pt")
        for name, ee in parts.items():
            torch.save({"docs": [doc(e) for e in ee], "vocab": itos}, dd / f"{name}.pt")
        print(f"{t}_{reg}: " + "  ".join(f"{k} {len(v)}" for k, v in parts.items()))
    print(f"vocab {len(itos)} words ({len(itos) - len(SPECIAL)} + {len(SPECIAL)} special)")


if __name__ == "__main__":
    main()

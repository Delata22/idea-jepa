"""Synthetic v2.1 options: default output unchanged, skip mode exact, dense distractors inert, marker."""
import random
from pathlib import Path

import pytest
import torch

from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V

VAL = Path("data/reasoning/val.pt")


def story_docs(n, seed=11, **kw):
    vocab = R.build_vocab()
    w2i = {w: i for i, w in enumerate(vocab)}
    idx = R.IdeaIndex()
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        fam = ["F1", "F2", "F3"][i % 3]
        d = R.realise(R.generate_story(rng, fam, (3, 7), None, **kw), rng, w2i, idx)
        d["family"] = fam
        docs.append(d)
    return vocab, docs


def text(vocab, toks):
    return " ".join(vocab[t] for t in toks)


@pytest.mark.skipif(not VAL.exists(), reason="needs data/reasoning/val.pt")
def test_default_generation_unchanged():
    """With the v2.1 options off, the generator reproduces the existing v2 val split token for token."""
    ref = torch.load(VAL, weights_only=False)
    vocab, docs = story_docs(300, seed=1)  # val split = seed + 1 with seed 0
    assert vocab[:len(ref["vocab"])] == ref["vocab"]
    n_fixed = 0
    for a, b in zip(docs, ref["docs"][:300]):
        assert len(a["ideas"]) == len(b["ideas"])
        for x, y in zip(a["ideas"], b["ideas"]):
            if x["tokens"] != y["tokens"]:  # only restated antecedents may differ (bug fixed 26/09)
                assert x["restated"] and y["restated"] and x["fact"] == y["fact"]
                n_fixed += 1
    assert n_fixed < 0.1 * sum(len(d["ideas"]) for d in docs)


def test_restated_antecedent_is_the_licensing_premise():
    """Removing the restated antecedent from the stated facts must make the deduction non-derivable."""
    vocab, docs = story_docs(600)
    n = 0
    for d in docs:
        stated = []
        for i in d["ideas"]:
            if i["step"] == "deduction" and i["restated"]:
                f, ante = V.parse_sentence(text(vocab, i["tokens"]))
                assert f == i["fact"] and ante in stated
                n_prem = sum(1 for x in d["ideas"] if x["step"] == "premise")
                without = [x for x in stated if x != ante]
                if d["family"] != "F1":
                    assert f not in R.family_closure(d["family"], without, n_prem - 1), (d["family"], f, ante)
                n += 1
            stated.append(i["fact"])
    assert n > 50


def test_skip_mode_exact():
    vocab, docs = story_docs(240, mode="skip")
    for d in docs:
        prem = V.doc_premises(d)
        ded = [i for i in d["ideas"] if i["step"] == "deduction"]
        assert {i["fact"] for i in ded} == R.derive_sinks(d["family"], prem)
        said = []
        for i in ded:
            assert V.valid_next(d["family"], prem, said, skip=True) == set(i["candidates"])
            said.append(i["fact"])
        verdicts = V.judge(d["family"], prem, [text(vocab, i["tokens"]) for i in ded], skip=True)
        assert all(v["verdict"] == "valid" for v in verdicts)
        if d["family"] == "F1":  # no chaining: skip = all final locations
            assert len(ded) == len(R.derive_all("F1", prem))


def test_dense_distractors_are_inert():
    vocab, docs = story_docs(240, dense=4)
    n_extra = 0
    for d in docs:
        prem = V.doc_premises(d)
        useful = [i["fact"] for i in d["ideas"] if i["step"] == "premise" and not i["distractor"]]
        assert R.derive_all(d["family"], useful) == R.derive_all(d["family"], prem)
        n_extra += sum(i["distractor"] for i in d["ideas"])
        ded = [i for i in d["ideas"] if i["step"] == "deduction"]
        assert all(v["verdict"] == "valid" for v in
                   V.judge(d["family"], prem, [text(vocab, i["tokens"]) for i in ded]))
    assert n_extra > 2 * len(docs)  # clearly denser than v2 (1-2 distractors per F2/F3 story)


def test_marker_once_at_boundary():
    vocab, docs = story_docs(60, marker=True)
    for d in docs:
        steps = [i["step"] for i in d["ideas"]]
        k = steps.index("marker")
        assert steps.count("marker") == 1 and all(s == "premise" for s in steps[:k])
        assert all(s == "deduction" for s in steps[k + 1:])
        assert text(vocab, d["ideas"][k]["tokens"]) == "let us think ."
        assert V.parse_sentence("let us think .")[0] == ("marker",)


def test_distinct_final_ids_in_v21():
    vocab = R.build_vocab(); w2i = {w: i for i, w in enumerate(vocab)}
    rng = random.Random(3); idx = R.IdeaIndex(); clash = 0
    for _ in range(200):
        d = R.realise(R.generate_story(rng, "F1", (3, 7), None, marker=True), rng, w2i, idx, distinct_final_ids=True)
        prem_ids = {i["idea_id"] for i in d["ideas"] if i["step"] == "premise"}
        prem_facts = {i["fact"] for i in d["ideas"] if i["step"] == "premise"}
        for i in d["ideas"]:
            if i["step"] == "deduction":
                assert i["idea_id"] not in prem_ids
                clash += i["fact"] in prem_facts
    assert clash > 0  # the situation does occur (object never moved)

"""Exact verifier: round-trip parse on the real v2 data, reference chains are valid, wrong answers caught."""
from pathlib import Path

import pytest
import torch

from ideajepa.data import verifier as V

VAL = Path("data/reasoning/val.pt")
pytestmark = pytest.mark.skipif(not VAL.exists(), reason="generate data/reasoning first")


@pytest.fixture(scope="module")
def data():
    d = torch.load(VAL, weights_only=False)
    return d["vocab"], d["docs"][:600]


def text(vocab, toks):
    return " ".join(vocab[t] for t in toks)


def test_parse_roundtrip(data):
    vocab, docs = data
    n = 0
    for d in docs:
        for i in d["ideas"]:
            f, _ = V.parse_sentence(text(vocab, i["tokens"]))
            assert f == i["fact"], (text(vocab, i["tokens"]), f, i["fact"])
            n += 1
    assert n > 5000


def test_reference_chain_valid_and_frontier_exact(data):
    vocab, docs = data
    for d in docs:
        prem = V.doc_premises(d)
        ded = [i for i in d["ideas"] if i["step"] == "deduction"]
        verdicts = V.judge(d["family"], prem, [text(vocab, i["tokens"]) for i in ded])
        assert all(v["verdict"] == "valid" for v in verdicts), (d["family"], verdicts)
        said = []
        for i in ded:  # the verifier's frontier is exactly the generator's candidate set
            assert V.valid_next(d["family"], prem, said) == set(i["candidates"])
            said.append(i["fact"])
        assert not V.valid_next(d["family"], prem, said)  # chain complete


def test_wrong_answers(data):
    vocab, docs = data
    d = next(d for d in docs if d["family"] == "F3" and sum(i["step"] == "deduction" for i in d["ideas"]) >= 3)
    prem = V.doc_premises(d)
    ded = [text(vocab, i["tokens"]) for i in d["ideas"] if i["step"] == "deduction"]
    first_prem = text(vocab, next(i["tokens"] for i in d["ideas"] if i["step"] == "premise"))
    out = V.judge("F3", prem, [ded[0], ded[0], ded[2], "so zzz can fly .", first_prem,
                               "so kate gets the crown ."])
    assert [o["verdict"] for o in out][:5] == ["valid", "repeat", "premature", "unparsed", "premise"]
    assert out[5]["verdict"] in ("false", "unparsed")
    # a false statement does not unlock anything: after it, the frontier is unchanged
    assert out[2]["n_valid"] == out[1]["n_valid"]


def test_expert_does_not_derive_from_premature_statements(data):
    """A premature (true but unlicensed) statement is excluded from the targets but unlocks nothing."""
    vocab, docs = data
    checked = 0
    for d in docs:
        if d["family"] == "F1":
            continue
        prem = V.doc_premises(d)
        ded = [i for i in d["ideas"] if i["step"] == "deduction"]
        if len(ded) < 3:
            continue
        # find a deduction that is not in the initial frontier (premature if said first)
        first = V.valid_next(d["family"], prem, [])
        late = next((i for i in ded if i["fact"] not in first), None)
        if late is None:
            continue
        targets, verd = V.expert_targets(d["family"], prem, [text(vocab, late["tokens"])])
        assert verd[0]["verdict"] == "premature"
        assert late["fact"] not in targets            # said -> never taught again
        assert targets == first - {late["fact"]}      # nothing derived from it
        checked += 1
    assert checked > 50



def test_schema_contains_every_true_deduction(data):
    vocab, docs = data
    ratio = []
    for d in docs:
        prem = V.doc_premises(d)
        schema = set(V.schema_facts(d["family"], prem))
        truth = V.derivable(d["family"], prem)
        assert truth <= schema, (d["family"], truth - schema)
        ratio.append(len(schema) / max(1, len(truth)))
    assert sum(ratio) / len(ratio) > 2  # the pool is much larger than the true facts: no truth information

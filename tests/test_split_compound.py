"""Step 1: data.compound (conclusion | split) turns a restated "since A , B ." into the proposition B
(+ pointer to A) or into A (context only) then B; the verifier scores a split A as context, not as a sentence."""
from pathlib import Path

import pytest
import torch

from ideajepa.data import verifier as V
from ideajepa.data.reasoning import split_compound

VAL = Path("data/reasoning/val.pt")
pytestmark = pytest.mark.skipif(not VAL.exists(), reason="needs data/reasoning")


@pytest.fixture(scope="module")
def data():
    d = torch.load(VAL, weights_only=False)
    docs = [x for x in d["docs"] if any(i.get("restated") for i in x["ideas"])][:60]
    assert docs
    return docs, d["vocab"]


def text(idea, vocab):
    return " ".join(vocab[t] for t in idea["tokens"])


def fact(f):
    return tuple(tuple(x) if isinstance(x, list) else x for x in f) if f is not None else None


def test_conclusion_drops_since_and_points_to_A(data):
    docs, vocab = data
    for doc in docs:
        before = [text(i, vocab) for i in doc["ideas"]]
        new = split_compound(doc, "conclusion", vocab)
        assert [text(i, vocab) for i in doc["ideas"]] == before           # input untouched
        assert len(new["ideas"]) == len(doc["ideas"])
        for old, i in zip(doc["ideas"], new["ideas"]):
            s = text(i, vocab)
            assert not s.startswith("since ")
            assert fact(i["fact"]) == fact(old["fact"]) and i["idea_id"] == old["idea_id"]
            assert V.parse_sentence(s)[0] == fact(old["fact"])            # B still parses to the same fact
            if old.get("restated"):
                A = V.parse_sentence(text(old, vocab))[1]
                assert i["rel"] == "justified_by" and i["restated"] is False
                assert i["rel_ptr"] >= 0 and fact(new["ideas"][i["rel_ptr"]]["fact"]) == A   # A already in the story
            else:
                assert i["rel_ptr"] == -1
                assert i["rel"] == ("sequence" if old.get("step") == "deduction" and old.get("connector") == 1 else "none")


def test_split_inserts_A_as_context(data):
    docs, vocab = data
    for doc in docs:
        new = split_compound(doc, "split", vocab)
        n_rest = sum(bool(i.get("restated")) for i in doc["ideas"])
        assert len(new["ideas"]) == len(doc["ideas"]) + n_rest
        for j, i in enumerate(new["ideas"]):
            if i.get("split_context"):
                assert i["train_w"] == 0.0 and i["rel_ptr"] == -1 and i["candidates"] is None
                B = new["ideas"][j + 1]
                assert B["rel"] == "justified_by" and B["rel_ptr"] == j
                assert V.parse_sentence(text(i, vocab))[0] == fact(i["fact"])
                stated = [x for x in new["ideas"][:j] if fact(x.get("fact")) == fact(i["fact"])]
                assert stated and i["idea_id"] == stated[-1]["idea_id"]   # A's id, not B's
                assert i["idea_id"] != B["idea_id"]


def test_judge_context_mask(data):
    docs, vocab = data
    doc = split_compound(docs[0], "split", vocab)
    prem = [fact(i["fact"]) for i in doc["ideas"] if i["step"] == "premise"]
    sents = [text(i, vocab) for i in doc["ideas"] if i["step"] != "premise"]
    mask = [bool(i.get("split_context")) for i in doc["ideas"] if i["step"] != "premise"]
    assert any(mask)
    out = V.judge(doc["family"], prem, sents, context_mask=mask)
    assert [o["verdict"] == "context" for o in out] == mask
    assert all(o["verdict"] == "valid" for o, m in zip(out, mask) if not m)   # gold story: every B stays valid
    plain = V.judge(doc["family"], prem, [s for s, m in zip(sents, mask) if not m])
    assert [o["verdict"] for o in plain] == [o["verdict"] for o, m in zip(out, mask) if not m]


def test_step2_rel_labels_and_model(data):
    """Step 2: rel_id / rel_off in the batch; P's prediction for idea n does not see r_n (only r_<n), the decoder
    does; the relation loss trains; rule arm = justified_by on every F2/F3 deduction with a rule antecedent."""
    from ideajepa.data.dataset import Collator
    from ideajepa.model.ideajepa import IdeaJEPA
    from ideajepa.model.objective import idea_losses
    from test_core import LOSS, cfg
    docs, vocab = data
    w2i = {w: i for i, w in enumerate(vocab)}
    conv = [split_compound(d, "conclusion", vocab) for d in docs[:6]]
    rule = [split_compound(d, "conclusion", vocab, "rule") for d in docs[:6]]
    from ideajepa.data.reasoning import final_state_f1
    rule += [split_compound(d, "conclusion", vocab, "rule") for d in
             [x for x in torch.load(VAL, weights_only=False)["docs"] if x["family"] == "F1"][:5]]
    for d in rule:
        for i in d["ideas"]:
            if i["step"] == "deduction":
                assert i["rel"] == "justified_by" and 0 <= i["rel_ptr"]
                if d["family"] == "F1":   # pointer = the premise that carries the object's final location
                    pf = [fact(x["fact"]) for x in d["ideas"] if x["step"] == "premise"]
                    assert d["ideas"][i["rel_ptr"]]["fact"][-1] == final_state_f1(pf)[i["fact"][1]]
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(conv)
    flat = [i for d in conv for i in d["ideas"]]
    pos = [j for d in conv for j in range(len(d["ideas"]))]
    REL = {"none": 0, "justified_by": 1, "sequence": 2}
    assert b["rel_id"].tolist() == [REL[i["rel"]] for i in flat]
    assert b["rel_off"].tolist() == [p - i["rel_ptr"] if i["rel_ptr"] >= 0 else 0 for i, p in zip(flat, pos)]
    torch.manual_seed(0)
    model = IdeaJEPA(cfg(vocab_size=len(vocab), max_idea_len=24, max_window=27, max_ideas=24, stop_head=True,
                             p_window=2, idea_memory="delta", rel_cond=True)).eval()
    k, v = model.encode(b)
    pk, _ = model.predict(k, v, b)
    row = next(r for r, i in enumerate(flat) if i["rel"] == "justified_by")
    b2 = {**b, "rel_id": b["rel_id"].clone(), "rel_off": b["rel_off"].clone()}
    b2["rel_id"][row] = 0; b2["rel_off"][row] = 0
    pk2, _ = model.predict(k, v, b2)
    assert torch.allclose(pk[row], pk2[row], atol=1e-5)                 # no leak of r_n into the prediction of n
    assert not torch.allclose(pk[row + 1], pk2[row + 1], atol=1e-5)     # r_n is seen by the next prediction
    lg, _ = model.decode(b, k, v)
    lg2, _ = model.decode(b2, k, v)
    assert not torch.allclose(lg[row], lg2[row])                        # the decoder reads r of the idea it writes
    model.train()
    out, aux = idea_losses(model, b, dict(LOSS, multi_form=True))
    assert float(out["L_rel"]) > 0 and "rel_acc" in aux["stats"]
    out["loss"].backward()
    assert float(model.rel_head.weight.grad.abs().sum()) > 0 and float(model.P.rel_emb.weight.grad.abs().sum()) > 0


def test_dropout_positives_label_free(data):
    """data.nce_positives dropout: the positive of each idea is its own sentence; false-negative keys = sentence identity
    (no generator fact); two E passes differ (enc_dropout) while E_ema is deterministic in train mode."""
    from ideajepa.data.dataset import Collator
    from ideajepa.model.ideajepa import IdeaJEPA
    from ideajepa.model.objective import idea_losses
    from test_core import LOSS, cfg
    docs, vocab = data
    w2i = {w: i for i, w in enumerate(vocab)}
    conv = [split_compound(d, "split", vocab) for d in docs[:6]]
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0, nce_positives="dropout")(conv)
    for j, r in enumerate(b["pos_idea"].tolist()):
        L = int(b["pos_len"][j])
        assert b["pos_tok"][j, :L].tolist() == b["idea_tok"][r, :int(b["idea_len"][r])].tolist()
        assert int(b["fact_key"][r]) == int(b["pos_key"][r])
    torch.manual_seed(0)
    m = IdeaJEPA(cfg(vocab_size=len(vocab), max_idea_len=24, max_window=27, max_ideas=28, stop_head=True,
                         p_window=2, idea_memory="delta", enc_dropout=0.1)).train()
    k1, v1 = m.E(b["idea_tok"], b["idea_len"]); k2, v2 = m.E(b["idea_tok"], b["idea_len"])
    assert not torch.allclose(v1, v2)
    e1 = m.E_ema(b["idea_tok"], b["idea_len"])[1]; e2 = m.E_ema(b["idea_tok"], b["idea_len"])[1]
    assert torch.allclose(e1, e2)
    out, _ = idea_losses(m, dict(b), dict(LOSS, multi_form=True, w_nce=0.3, nce_temp=0.1))
    assert float(out["L_nce"]) > 0


def test_nce_negatives_other_docs_only(data):
    """loss.nce_neg other_docs: the InfoNCE negatives of an anchor are only ideas of OTHER stories (no fact label)."""
    from ideajepa.data.dataset import Collator
    from ideajepa.model import objective as O
    from ideajepa.model.ideajepa import IdeaJEPA
    from test_core import cfg
    docs, vocab = data
    w2i = {w: i for i, w in enumerate(vocab)}
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0, nce_positives="dropout")(
        [split_compound(d, "split", vocab) for d in docs[:4]])
    torch.manual_seed(0)
    m = IdeaJEPA(cfg(vocab_size=len(vocab), max_idea_len=24, max_window=27, max_ideas=28, stop_head=True,
                         p_window=2, idea_memory="delta", enc_dropout=0.1)).train()
    k, v = m.encode(b)
    seen = {}
    orig = torch.Tensor.masked_fill
    def spy(t, mask, val):
        seen.setdefault("masks", []).append(mask.clone())
        return orig(t, mask, val)
    torch.Tensor.masked_fill = spy
    try:
        L = O._info_nce(m, b, k, v, 0.1, "other_docs")
    finally:
        torch.Tensor.masked_fill = orig
    doc = b["idea_doc"][b["pos_idea"]]
    P = len(doc)
    same = (doc[:, None] == doc[None, :]) & ~torch.eye(P, dtype=torch.bool)
    mask = seen["masks"][0]                                   # [P, 2P]: positives block then anchors block
    assert bool(mask[:, P:][same].all()) and bool(mask[:, :P][same].all()) and float(L) > 0

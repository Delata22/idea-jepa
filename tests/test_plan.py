"""Plan variable: targets = nearest past idea in E_ema space; the pointer head never sees o; o drives P."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ideajepa.data.dataset import Collator
from ideajepa.data.reasoning import split_compound
from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.model.objective import idea_losses, plan_targets
from test_core import LOSS, cfg

VAL = Path("data/reasoning/val.pt")
pytestmark = pytest.mark.skipif(not VAL.exists(), reason="needs data/reasoning")


@pytest.fixture(scope="module")
def setup():
    d = torch.load(VAL, weights_only=False)
    vocab = d["vocab"]
    w2i = {w: i for i, w in enumerate(vocab)}
    docs = [split_compound(x, "conclusion", vocab) for x in d["docs"][:6]]
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(docs)
    torch.manual_seed(0)
    model = IdeaJEPA(cfg(vocab_size=len(vocab), max_idea_len=24, max_window=27, max_ideas=24, stop_head=True,
                             p_window=2, idea_memory="delta", plan_cond=True)).eval()
    return model, b


def test_plan_targets_brute_force(setup):
    model, b = setup
    _, vt = model.E_ema(b["idea_tok"], b["idea_len"])
    o = plan_targets(vt, b["doc_index"])
    z = F.normalize(F.normalize(vt.float(), dim=-1).flatten(1), dim=-1)
    di = b["doc_index"]
    for bi in range(di.shape[0]):
        rows = [int(x) for x in di[bi] if x >= 0]
        assert o[rows[0]] == -1
        for n in range(1, len(rows)):
            sims = [float(z[rows[n]] @ z[rows[j]]) for j in range(n)]
            assert int(o[rows[n]]) == max(range(n), key=lambda j: sims[j])


def test_no_leak_and_conditioning(setup):
    model, b = setup
    k, v = model.encode(b)
    _, vt = model.E_ema(b["idea_tok"], b["idea_len"])
    o = plan_targets(vt, b["doc_index"])
    o2 = torch.where(o > 0, o - 1, o)                     # another past idea
    pk1, _ = model.predict(k, v, {**b, "plan_o": o}); lg1 = model.plan_logits.clone()
    pk2, _ = model.predict(k, v, {**b, "plan_o": o2}); lg2 = model.plan_logits.clone()
    assert torch.equal(torch.isinf(lg1), torch.isinf(lg2))
    assert torch.allclose(lg1[~torch.isinf(lg1)], lg2[~torch.isinf(lg2)])   # the head never sees o
    changed = (o != o2)
    assert changed.any() and not torch.allclose(pk1[changed], pk2[changed], atol=1e-5)   # o drives the prediction
    # inference without plan_o: o drawn among the past ideas
    model.plan_mode = "argmax"
    model.predict(k, v, {kk: vv for kk, vv in b.items() if kk != "plan_o"})
    ch = model.plan_choice
    n_idx = torch.arange(ch.shape[1])[None].expand_as(ch)
    assert ((ch < n_idx) | (ch == -1)).all() and (ch[:, 0] == -1).all()
    model.plan_mode = "sample"


def test_plan_loss_trains(setup):
    model, b = setup
    model.train()
    out, aux = idea_losses(model, dict(b), dict(LOSS, multi_form=True, w_plan=1.0))
    assert float(out["L_plan"]) > 0 and "plan_acc" in aux["stats"]
    out["loss"].backward()
    assert float(model.plan_q.weight.grad.abs().sum()) > 0 and float(model.P.plan_emb.grad.abs().sum()) > 0
    model.eval()


def test_inference_plan_past_targets_and_next(setup):
    """inference: past blocks get the label-free targets of the ideas already there; only the predicted block's o
    comes from the head (or plan_next); the start block never gets a plan."""
    model, b = setup
    k, v = model.encode(b)
    story = {kk: vv for kk, vv in b.items() if kk != "plan_o"}
    story["plan_next"] = torch.zeros_like(b["idea_pos"])
    model.predict(k, v, story)
    ch = model.plan_choice
    past = plan_targets(v, b["doc_index"])
    di = b["doc_index"]
    for bi in range(di.shape[0]):
        for n in range(di.shape[1]):
            if di[bi, n] < 0:
                continue
            is_pred = bool(((b["idea_doc"] == bi) & (b["idea_pos"] == n)).any())
            assert int(ch[bi, n]) == (0 if is_pred else int(past[di[bi, n]]))
    assert (ch[:, 0] <= 0).all()


def test_detached_heads_and_output_injection():
    """rel_head_detach / plan_head_detach: the heads' losses send no gradient into P; plan_inject output: o changes the
    prediction but not the memory writes."""
    d = torch.load(VAL, weights_only=False)
    vocab = d["vocab"]; w2i = {w: i for i, w in enumerate(vocab)}
    docs = [split_compound(x, "conclusion", vocab, "rule") for x in d["docs"][:6]]
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(docs)
    for extra, loss_key, head in (({"rel_cond": True, "rel_head_detach": True}, "L_rel", "rel_head"),
                                  ({"plan_cond": True, "plan_head_detach": True}, "L_plan", "plan_q"),
                                  ({"plan_cond": True, "plan_head_detach": True, "plan_inject": "output"}, "L_plan", "plan_q")):
        torch.manual_seed(0)
        m = IdeaJEPA(cfg(vocab_size=len(vocab), max_idea_len=24, max_window=27, max_ideas=24, stop_head=True,
                             p_window=2, idea_memory="delta", **extra)).train()
        from ideajepa.model import objective as O
        bb = dict(b)
        idea_losses(m, bb, dict(LOSS, multi_form=True))
        L = O._rel_loss(m, bb)[0] if loss_key == "L_rel" else O._plan_loss(m, bb)[0]
        m.zero_grad(); L.backward()
        assert float(getattr(m, head).weight.grad.abs().sum()) > 0
        assert all(p_.grad is None or float(p_.grad.abs().sum()) == 0 for p_ in m.P.blocks.parameters())
        assert all(p_.grad is None or float(p_.grad.abs().sum()) == 0 for p_ in m.E.parameters())
    m.eval()
    for mm in m.P.mems:
        mm.capture = True
    k, v = m.encode(b)
    _, vt = m.E_ema(b["idea_tok"], b["idea_len"])
    o = plan_targets(vt, b["doc_index"])
    pk1, _ = m.predict(k, v, {**b, "plan_o": o}); k1 = m.P.mems[0].last_qk[1].clone()
    pk2, _ = m.predict(k, v, {**b, "plan_o": torch.where(o > 0, o - 1, o)}); k2 = m.P.mems[0].last_qk[1].clone()
    assert torch.allclose(k1, k2) and not torch.allclose(pk1, pk2)

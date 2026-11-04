"""Phase 2 options (branch phase2): K-hypothesis idea head (k_hyp) and flow head (flow_head / flow_only)."""
import torch

from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.model.objective import flow_only_step, idea_losses
from test_core import LOSS, batch_of, cfg, rand_docs

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make(**kw):
    torch.manual_seed(0)
    return IdeaJEPA(cfg(**kw)).to(DEV)


def test_khyp_shapes_selection_and_loss():
    m = make(k_hyp=3)
    b = batch_of(rand_docs(3))
    k, v = m.encode(b)
    pk, pv = m.predict(k, v, b)
    pk_all, pv_all, sc = m.predict_hyp(k, v, b)
    NI = k.shape[0]
    assert pk_all.shape == (NI, 3, 4, 64) and sc.shape == (NI, 3)
    best = sc.argmax(-1)
    assert torch.allclose(pk, pk_all[torch.arange(NI), best]) and torch.allclose(pv, pv_all[torch.arange(NI), best])
    out, aux = idea_losses(m, b, {**LOSS, "w_score": 0.1}, p_gen=0.5)
    assert torch.isfinite(out["loss"]) and out["L_score"] > 0
    out["loss"].backward()
    assert m.P.score.weight.grad is not None and m.P.head.weight.grad.abs().sum() > 0
    assert "khyp_winner_entropy" in aux["stats"] and "khyp_score_acc" in aux["stats"]
    # the masked predictor keeps a single output
    m2 = make(k_hyp=3, mask_predictor=True)
    assert m2.PM.K == 1


def test_flow_head_joint_loss_and_detached_context():
    m = make(flow_head=True)
    b = batch_of(rand_docs(3))
    out, _ = idea_losses(m, b, {**LOSS, "w_flow": 1.0}, p_gen=0.5)
    assert out["L_flow"] > 0
    m.zero_grad()
    out["L_flow"].requires_grad_(False)
    loss, _ = idea_losses(m, b, {**LOSS, "w_flow": 1.0, "w_jepa": 0.0}, p_gen=0.0, force_path="recon")
    # isolate the flow term: its gradient reaches the flow head but not P (context detached by default)
    from ideajepa.model.objective import _flow_loss
    k, v = m.encode(b)
    kt, vt = m.E_ema(b["idea_tok"], b["idea_len"])
    pk, pv = m.predict(k, v, b)
    m.zero_grad()
    _flow_loss(m, b, {}, kt, vt, pk, pv).backward()
    assert m.Fh.inp.weight.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.P.parameters())


def test_flow_only_step_trains_only_the_flow_head():
    m = make(flow_head=True)
    b = batch_of(rand_docs(3))
    m.zero_grad()
    out, _ = flow_only_step(m, b, {})
    out["loss"].backward()
    assert m.Fh.inp.weight.grad.abs().sum() > 0
    others = [p for n, p in m.named_parameters() if not n.startswith("Fh.")]
    assert all(p.grad is None for p in others)
    out2, _ = idea_losses(m, b, {**LOSS, "flow_only": True}, train=True)  # routed to the fast path
    assert set(out2) == {"loss", "L_tok", "L_flow"}


def test_predict_flow_and_eval_stats():
    m = make(flow_head=True).eval()
    b = batch_of(rand_docs(3))
    k, v = m.encode(b)
    pk, pv = m.predict(k, v, b)
    kf, vf = m.predict_flow(k, v, b, n_steps=4)
    assert kf.shape == pk.shape and vf.shape == pv.shape
    assert torch.allclose(kf.norm(dim=-1), torch.ones_like(kf[..., 0]), atol=1e-4)
    assert torch.allclose(vf.norm(dim=-1), pv.norm(dim=-1), atol=1e-4)   # value norms come from P
    _, aux = idea_losses(m, b, {**LOSS, "flow_eval_steps": 2}, force_path="recon", train=False)
    for key in ("ce_ded_P", "ce_ded_flow", "cos_ded_P", "cos_ded_flow"):
        assert key in aux["stats"]


def test_init_from_checkpoint_without_flow_head():
    base = make()
    m = make(flow_head=True)
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [] and missing and all(k.startswith("Fh.") for k in missing)



import pytest


@pytest.mark.parametrize("ctx,dim", [("reg", 128), ("hidden", 64), ("hidden+reg", 192)])
def test_flow_context_modes(ctx, dim):
    m = make(flow_head=True, flow_ctx=ctx).eval()
    b = batch_of(rand_docs(3))
    k, v = m.encode(b)
    pk, pv = m.predict(k, v, b)
    c = m.flow_context(pk, pv)
    assert c.shape == (k.shape[0], 4, dim) and m.Fh.ctx_in.in_features == dim
    if ctx != "reg":  # the hidden context is P's slot state, not the regression output
        pos = (b["idea_doc"], b["idea_pos"])
        assert torch.allclose(c[..., :64], m.P.last_hidden[pos])
    kf, vf = m.predict_flow(k, v, b, n_steps=2)           # interface unchanged
    assert kf.shape == pk.shape and torch.allclose(vf.norm(dim=-1), pv.norm(dim=-1), atol=1e-4)


def test_flow_hidden_context_is_detached_in_2a():
    m = make(flow_head=True, flow_ctx="hidden+reg")
    b = batch_of(rand_docs(3))
    m.zero_grad()
    out, _ = flow_only_step(m, b, {})
    out["loss"].backward()
    assert m.Fh.ctx_in.weight.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in m.named_parameters() if not n.startswith("Fh."))



def test_flow_eval_splits_by_number_of_candidates():
    m = make(flow_head=True).eval()
    docs = rand_docs(3)
    for d in docs:
        for j, i in enumerate(d["ideas"]):
            i["step"] = "deduction" if j else "premise"
            i["b_eff"] = 0 if j == 0 else (1 if j % 2 else 2)
    b = batch_of(docs)
    assert "b_eff" in b and "is_deduction" in b
    _, aux = idea_losses(m, b, {**LOSS, "flow_eval_steps": 2}, force_path="recon", train=False)
    for key in ("ce_ded_P_n1", "ce_ded_P_n2+", "ce_ded_flow_n1", "ce_ded_flow_n2+"):
        assert key in aux["stats"]


def test_flow_space_scaled_roundtrip_and_loss():
    m = make(flow_head=True, flow_space="scaled")
    b = batch_of(rand_docs(3))
    k, v = m.encode(b)
    m.e_slot_norm.copy_(v.detach().norm(dim=-1).mean(0))
    z = m.flow_target(k, v)
    k2, v2 = m.flow_decode(z)
    assert torch.allclose(k2, torch.nn.functional.normalize(k, dim=-1), atol=1e-5)
    assert torch.allclose(v2, v, atol=1e-4)                       # the norm is modelled, not supplied
    m.zero_grad()
    out, _ = flow_only_step(m, b, {})
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    out["loss"].backward()
    m.eval()
    kf, vf = m.predict_flow(k, v, b, n_steps=2)
    assert kf.shape == k.shape and vf.shape == v.shape


def test_flow_norms_E_uses_running_norms():
    m = make(flow_head=True, flow_norms="E").eval()
    b = batch_of(rand_docs(3))
    k, v = m.encode(b)
    ref = torch.arange(1, 5, dtype=torch.float32, device=m.e_slot_norm.device)  # the model may live on cuda
    m.e_slot_norm.copy_(ref)
    kf, vf = m.predict_flow(k, v, b, n_steps=2)
    assert torch.allclose(vf.norm(dim=-1), ref.expand_as(vf[..., 0]), atol=1e-4)

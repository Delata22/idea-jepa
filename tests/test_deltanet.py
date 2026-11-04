import torch

from ideajepa.model.deltanet import delta_rule_chunked, delta_rule_naive


def test_chunked_matches_naive():
    torch.manual_seed(0)
    B, H, T, dk, dv = 2, 3, 37, 8, 5
    q = torch.nn.functional.normalize(torch.randn(B, H, T, dk, dtype=torch.float64), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, dk, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, dv, dtype=torch.float64)
    beta = torch.rand(B, H, T, dtype=torch.float64)
    ref = delta_rule_naive(q, k, v, beta)
    for c in (4, 16, 64):
        assert torch.allclose(delta_rule_chunked(q, k, v, beta, chunk=c), ref, atol=1e-8)


def test_causal():
    torch.manual_seed(1)
    q, k = (torch.nn.functional.normalize(torch.randn(1, 1, 20, 4), dim=-1) for _ in range(2))
    v, beta = torch.randn(1, 1, 20, 3), torch.rand(1, 1, 20)
    o1 = delta_rule_chunked(q, k, v, beta, chunk=8)
    v2 = v.clone(); v2[:, :, 12] += 5
    o2 = delta_rule_chunked(q, k, v2, beta, chunk=8)
    assert torch.allclose(o1[:, :, :12], o2[:, :, :12], atol=1e-6)


def _rand(B=2, H=3, T=37, dk=8, dv=5, seed=0):
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, H, T, dk, dtype=torch.float64), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, dk, dtype=torch.float64), dim=-1)
    return q, k, torch.randn(B, H, T, dv, dtype=torch.float64), torch.rand(B, H, T, dtype=torch.float64)


def test_gated_chunked_matches_naive():
    """Forget gate (Gated DeltaNet): fixed alpha, per-head alpha and per-step alpha, several chunk sizes."""
    q, k, v, beta = _rand()
    B, H, T = beta.shape
    for la in (torch.full((B, H, T), torch.log(torch.tensor(0.9, dtype=torch.float64)).item(), dtype=torch.float64),
               torch.log(torch.tensor([0.8, 0.9, 0.95], dtype=torch.float64))[None, :, None].expand(B, H, T),
               torch.log(torch.rand(B, H, T, dtype=torch.float64) * 0.5 + 0.5)):
        ref = delta_rule_naive(q, k, v, beta, log_alpha=la)
        for c in (4, 16, 64):
            assert torch.allclose(delta_rule_chunked(q, k, v, beta, chunk=c, log_alpha=la), ref, atol=1e-8)


def test_gated_alpha_one_is_plain_delta_rule():
    q, k, v, beta = _rand(seed=3)
    zero = torch.zeros_like(beta)
    assert torch.allclose(delta_rule_chunked(q, k, v, beta, chunk=8, log_alpha=zero),
                          delta_rule_chunked(q, k, v, beta, chunk=8), atol=1e-10)


def test_gated_forgets():
    """With a strong forget gate, an early write has (almost) no effect on a late output."""
    q, k, v, beta = _rand(T=40, seed=4)
    v2 = v.clone(); v2[:, :, 0] += 10.0
    la = torch.full_like(beta, torch.log(torch.tensor(0.5, dtype=torch.float64)).item())
    d_forget = (delta_rule_chunked(q, k, v, beta, 16, la) - delta_rule_chunked(q, k, v2, beta, 16, la))[:, :, -1].abs().max()
    d_plain = (delta_rule_chunked(q, k, v, beta, 16) - delta_rule_chunked(q, k, v2, beta, 16))[:, :, -1].abs().max()
    assert d_forget < 1e-6 < d_plain


def test_memory_module_decay_options():
    from ideajepa.model.deltanet import DeltaMemory
    torch.manual_seed(0)
    x = torch.randn(2, 12, 32)
    for mode in ("delta", "sum"):
        for decay in (None, 0.9, "learned"):
            mem = DeltaMemory(32, 4, mode=mode, decay=decay)
            y = mem(x)
            assert y.shape == x.shape and torch.isfinite(y).all()
            if decay == "learned":
                y.sum().backward()
                assert mem.decay_logit.grad is not None



def test_decay_every_applies_once_per_block():
    """decay_every = r: alpha acts once per idea (r slot tokens) -> alpha per idea, not alpha^r (float32 module)."""
    from ideajepa.model.deltanet import DeltaMemory
    torch.manual_seed(0)
    x = torch.randn(1, 16, 16)
    base = DeltaMemory(16, 2, mode="delta", decay=None)
    outs = {}
    for name, kw in (("none", dict(decay=None)), ("a1_blk", dict(decay=1.0, decay_every=4)),
                     ("tok", dict(decay=0.5, decay_every=1)), ("blk", dict(decay=0.5, decay_every=4))):
        m = DeltaMemory(16, 2, mode="delta", **kw)
        m.load_state_dict(base.state_dict())
        with torch.no_grad():
            outs[name] = m(x)
    assert torch.allclose(outs["none"], outs["a1_blk"], atol=1e-5)         # alpha = 1: no effect
    d_tok = (outs["tok"] - outs["none"]).abs().mean()
    d_blk = (outs["blk"] - outs["none"]).abs().mean()
    assert 0 < d_blk < d_tok                                              # per-idea forgetting is milder

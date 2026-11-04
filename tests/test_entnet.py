"""idea_memory "entnet" (notes §34.3): causal slot memory with fixed learned keys and content-gated delta writes."""
import torch

from ideajepa.model.deltanet import DeltaMemory, entnet_scan


def test_entnet_causal_and_trains():
    torch.manual_seed(0)
    m = DeltaMemory(64, 4, mode="entnet", n_slots=16, decay_every=4)
    x = torch.randn(2, 12, 64, requires_grad=True)
    y = m(x)
    x2 = x.detach().clone(); x2[:, 8:] += torch.randn(2, 4, 64)          # change the future blocks only (block = 4)
    y2 = m(x2)
    assert torch.allclose(y[:, :8], y2[:, :8], atol=1e-5) and not torch.allclose(y[:, 8:], y2[:, 8:])
    y.sum().backward()
    assert float(m.slot_keys.grad.abs().sum()) > 0 and float(x.grad.abs().sum()) > 0


def test_entnet_slot_overwrite():
    """beta = 1, gate saturated by aligned keys: a slot takes the new value (delta overwrite)."""
    H, dh = 1, 4
    W = torch.eye(dh)[None, :2] * 10                                         # 2 slots, keys e0, e1
    k = torch.tensor([[[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]]])                   # both tokens address slot 0
    v = torch.tensor([[[[1.0, 0, 0, 0], [0, 1.0, 0, 0]]]])
    beta = torch.ones(1, 1, 2)
    o = entnet_scan(k, k, v, beta, W, scale=50.0)
    assert o[0, 0, 1, 1] > 0.5 and o[0, 0, 1, 0] < 0.5                      # the second value replaced the first


def test_entnet_gate_sum_per_slot_at_most_one():
    """block writes: the gates of the r tokens of an idea sum to <= 1 per slot (no overshoot); saturated gates are rescaled."""
    torch.manual_seed(0)
    B, H, T, dh = 2, 2, 16, 8
    q = torch.nn.functional.normalize(torch.randn(B, H, T, dh), dim=-1)
    k = torch.nn.functional.normalize(torch.ones(B, H, T, dh), dim=-1)          # every token addresses the same way
    v = torch.randn(B, H, T, dh)
    W = torch.ones(H, 4, dh) * 5                                                # keys aligned with k: gates ~ 1
    entnet_scan(q, k, v, torch.ones(B, H, T), W, block=8)
    s = entnet_scan.last_gate_sum
    assert float(s.max()) <= 1.0 + 1e-5 and float(s.max()) > 0.99             # saturated -> rescaled to exactly 1
    # with a single token per step nothing changes (sum = g beta <= 1 already)
    entnet_scan(q, k, v, torch.full((B, H, T), 0.3), W, block=1)
    assert float(entnet_scan.last_gate_sum.max()) <= 0.3 + 1e-5

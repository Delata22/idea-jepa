"""FlowHead (rectified-flow idea head): shapes, initialisation, interpolation path, conditioning. CPU."""
import torch
import torch.nn.functional as F

from ideajepa.model.flow_head import FlowHead, from_target, to_target

torch.set_num_threads(2)
CFG = dict(d=32, n_heads=4, r=2, d_slot=4, flow_layers=2)


def test_shapes_and_normalisation():
    torch.manual_seed(0)
    fh = FlowHead(CFG)
    k, v = F.normalize(torch.randn(5, 2, 4), dim=-1), 3 * torch.randn(5, 2, 4)
    z1 = to_target(k, v)
    assert z1.shape == (5, 2, 8)
    k2, v2 = from_target(z1)
    assert torch.allclose(k2, k, atol=1e-6) and torch.allclose(v2, F.normalize(v, dim=-1), atol=1e-6)
    c = torch.randn(5, 2, 8)
    assert fh.velocity(z1, torch.rand(5), c).shape == (5, 2, 8)
    assert fh.sample(c, n_steps=4).shape == (5, 2, 8)
    assert torch.isfinite(fh.loss(z1, c))


def test_zero_initial_velocity():
    """Output layer is zero-initialised: an untrained head returns its (scaled) starting noise."""
    torch.manual_seed(0)
    fh = FlowHead(CFG).eval()
    c = torch.randn(3, 2, 8)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    z, path = fh.sample(c, n_steps=5, generator=g1, return_path=True)
    assert torch.allclose(z, fh.noise((3, 2, 8), "cpu", generator=g2)) and len(path) == 6


def test_linear_path_with_ideal_velocity_is_exact():
    """With the ideal velocity z1 - z0, Euler integration reaches z1 exactly (the path is straight)."""
    torch.manual_seed(0)
    z0, z1 = torch.randn(4, 2, 8), torch.randn(4, 2, 8)
    z = z0.clone()
    for i in range(16):
        tau = i / 16
        zt = (1 - tau) * z0 + tau * z1
        assert torch.allclose(z, zt, atol=1e-5)
        z = z + (z1 - z0) / 16
    assert torch.allclose(z, z1, atol=1e-5)


def test_running_std_tracks_targets():
    torch.manual_seed(0)
    fh = FlowHead({**CFG, "std_momentum": 0.0}).train()
    z1 = torch.randn(256, 2, 8) * torch.linspace(0.1, 2.0, 8)
    fh.loss(z1, torch.randn(256, 2, 8))
    assert torch.allclose(fh.z_std, torch.linspace(0.1, 2.0, 8), rtol=0.2)


def test_conditional_overfit_two_contexts():
    """Two contexts, each with its own fixed target idea: after a short training, sampling from each
    context lands on its own target (conditioning works; no averaging between the two)."""
    torch.manual_seed(0)
    fh = FlowHead(CFG)
    targets = to_target(F.normalize(torch.randn(2, 2, 4), dim=-1), torch.randn(2, 2, 4))
    ctx = torch.randn(2, 2, 8)
    opt = torch.optim.Adam(fh.parameters(), 3e-3)
    for _ in range(400):
        idx = torch.randint(0, 2, (64,))
        loss = fh.loss(targets[idx], ctx[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    fh.eval()
    for i in range(2):
        s = fh.sample(ctx[i:i + 1].expand(50, -1, -1), n_steps=16)
        err_own = (s - targets[i]).norm(dim=-1).mean()
        err_other = (s - targets[1 - i]).norm(dim=-1).mean()
        err_mean = (s - targets.mean(0)).norm(dim=-1).mean()
        assert err_own < 0.25 and err_own < 0.3 * err_other and err_own < err_mean

"""Toy for the "averaging problem": a constant context, a target that is a mixture of 3 far-apart Gaussians.
MSE regression predicts the centre of mass (outside every mode); a rectified-flow head samples inside the
modes. CPU, small MLPs, a few seconds."""
import math

import torch
import torch.nn as nn

torch.set_num_threads(2)  # the machine is shared with training jobs

MODES = torch.tensor([[4.0, 0.0], [-2.0, 3.4641], [-2.0, -3.4641]])  # equilateral, radius 4
SIGMA = 0.25


def sample_target(n, g):
    idx = torch.randint(0, 3, (n,), generator=g)
    return MODES[idx] + SIGMA * torch.randn(n, 2, generator=g)


def nearest_mode(x):
    d = torch.cdist(x, MODES)
    return d.min(1).values, d.argmin(1)


class VelocityMLP(nn.Module):
    """F(z_tau, tau, c) with a constant context c (a learned vector, as a conditioned P output would be)."""

    def __init__(self, h=128):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(8))
        self.net = nn.Sequential(nn.Linear(2 + 16 + 8, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, z, tau):
        f = torch.arange(8, dtype=z.dtype) * math.log(1000.0) / 8
        ang = tau[:, None] * torch.exp(f)[None] * 10
        emb = torch.cat([ang.sin(), ang.cos()], -1)
        return self.net(torch.cat([z, emb, self.c.expand(len(z), -1)], -1))


def train_mse(g, steps=600):
    head = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 2))
    c = torch.zeros(1, 8)
    opt = torch.optim.Adam(head.parameters(), 1e-2)
    for _ in range(steps):
        y = sample_target(256, g)
        loss = (head(c.expand(len(y), -1)) - y).pow(2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head(c).detach()[0]


def train_flow(g, steps=3000):
    torch.manual_seed(0)
    F_ = VelocityMLP()
    opt = torch.optim.Adam(F_.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        z1 = sample_target(512, g)
        z0 = torch.randn(512, 2, generator=g)
        tau = torch.sigmoid(torch.randn(512, generator=g))          # logit-normal
        zt = (1 - tau[:, None]) * z0 + tau[:, None] * z1
        loss = (F_(zt, tau) - (z1 - z0)).pow(2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return F_


@torch.no_grad()
def euler_sample(F_, n, g, n_steps=16):
    z = torch.randn(n, 2, generator=g)
    for i in range(n_steps):
        tau = torch.full((n,), i / n_steps)
        z = z + F_(z, tau) / n_steps
    return z


def test_mse_regression_predicts_centre_of_mass():
    g = torch.Generator().manual_seed(0)
    pred = train_mse(g)
    assert pred.norm() < 0.5                           # ~ centre of mass (0, 0)
    dist, _ = nearest_mode(pred[None])
    assert dist.item() > 3.0                           # far from every mode: an "average idea" nobody meant


def test_rectified_flow_samples_inside_modes():
    g = torch.Generator().manual_seed(1)
    F_ = train_flow(g)
    z = euler_sample(F_, 4000, g)
    dist, which = nearest_mode(z)
    inside = dist < 4 * SIGMA
    assert inside.float().mean() > 0.90                # > 90 % of samples land in a mode
    share = torch.bincount(which[inside], minlength=3).float() / inside.sum()
    assert share.min() > 0.2                           # all three modes are covered (no mode collapse)

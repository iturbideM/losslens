"""Minimal tests for the critical paths. Runs on CPU in a few seconds."""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import losslens
from losslens._directions import make_directions, trainable_params

torch.manual_seed(0)
QUIET = dict(_progress=False, device="cpu")


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn(self.conv(x))), 2)
        return self.fc(self.drop(x.flatten(1)))


class Quadratic(nn.Module):
    """A 'model' that is just a weight matrix; the loss is defined in each test."""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 6))


def image_data(n=64):
    return torch.randn(n, 3, 16, 16), torch.randint(0, 10, (n,))


DUMMY = (torch.zeros(8, 1), torch.zeros(8))


def test_directions_filter_normalized_and_bias_zero():
    params = trainable_params(SmallCNN())
    dx, dy = make_directions(params, seed=0)
    for p, u, v in zip(params, dx, dy):
        if p.dim() <= 1:
            assert torch.all(u == 0) and torch.all(v == 0)
        else:
            w_norm = p.detach().flatten(1).norm(dim=1)
            assert torch.allclose(u.flatten(1).norm(dim=1), w_norm, rtol=1e-4)
            assert torch.allclose(v.flatten(1).norm(dim=1), w_norm, rtol=1e-4)


def test_bowl_matches_formula_and_is_centered():
    model = Quadratic()
    W0 = model.W.detach().clone()
    dx, dy = make_directions(trainable_params(model), seed=0)
    surf = losslens.surface(model, DUMMY, forward=lambda m, b: ((m.W - W0) ** 2).sum(),
                            resolution=5, **QUIET)
    for iy, b in enumerate(surf.y):
        for ix, a in enumerate(surf.x):
            expected = ((a * dx[0] + b * dy[0]) ** 2).sum().item()
            assert surf.Z[iy, ix] == pytest.approx(expected, rel=1e-4, abs=1e-6)
    assert np.unravel_index(np.argmin(surf.Z), surf.Z.shape) == (2, 2)


def test_orientation_x_is_columns():
    model = Quadratic()
    W0 = model.W.detach().clone()
    dx, dy = make_directions(trainable_params(model), seed=0)
    # Part of dx orthogonal to dy: the loss then changes only along x, so all rows match.
    v = dx[0] - (dx[0] * dy[0]).sum() / (dy[0] * dy[0]).sum() * dy[0]
    surf = losslens.surface(model, DUMMY, forward=lambda m, b: ((m.W - W0) * v).sum(),
                            resolution=5, **QUIET)
    assert np.allclose(surf.Z, surf.Z[0:1, :], atol=1e-5)
    assert np.all(np.diff(surf.Z[0]) > 0)


def test_student_model_is_untouched():
    net = SmallCNN().train()
    before = copy.deepcopy(net.state_dict())
    losslens.surface(net, image_data(), loss_fn=nn.CrossEntropyLoss(), resolution=3, **QUIET)
    assert net.training
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_center_loss_equals_direct_loss():
    net = SmallCNN()
    X, y = image_data(50)
    surf = losslens.surface(net, (X, y), loss_fn=nn.CrossEntropyLoss(), resolution=3, **QUIET)
    with torch.no_grad():
        direct = F.cross_entropy(net.eval()(X), y).item()
    assert surf.center_loss == pytest.approx(direct, rel=1e-5)
    assert surf.Z[1, 1] == pytest.approx(direct, rel=1e-5)


def test_plots_summary_compare_and_save_load(tmp_path):
    a = losslens.surface(SmallCNN(), image_data(), loss_fn=nn.CrossEntropyLoss(),
                         resolution=5, **QUIET)
    a.plot(); a.plot(log=True); a.plot_3d(); a.plot_slice()
    s = a.summary(show=False)
    assert s["center_loss"] == a.center_loss
    a.save(tmp_path / "a.npz")
    b = losslens.load(tmp_path / "a.npz")
    assert np.array_equal(a.Z, b.Z) and a.meta == b.meta
    losslens.compare({"a": a, "b": b})

"""Tests for TorchSGDFitter."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyapprox.generative.flowmatching.fitters.torch_sgd_fitter import (
    TorchSGDFitter,
)
from pyapprox.generative.flowmatching.linear_path import LinearPath
from pyapprox.generative.flowmatching.nn_vf import MLPVelocityField
from pyapprox.generative.flowmatching.ode_adapter import integrate_flow
from pyapprox.generative.flowmatching.samplers import (
    GaussianSourceSampler,
    WeightedEmpiricalSampler,
)
from pyapprox.ode.explicit_steppers.heun import HeunStepper
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd


def _make_torch_bkd():
    torch.set_default_dtype(torch.float64)
    return TorchBkd()


def _make_shift_samplers(shift=2.0, n_pool=500, seed=0):
    """Source N(0,1) and target N(shift, 1) via empirical pool."""
    rng = torch.Generator().manual_seed(seed)
    pool = torch.randn(1, n_pool, generator=rng, dtype=torch.float64) + shift
    w = torch.ones(n_pool, dtype=torch.float64)
    source = GaussianSourceSampler(d=1, seed=seed + 1)
    target = WeightedEmpiricalSampler(pool, w, seed=seed + 2)
    return source, target


class TestTorchSGDFitter:
    def test_shift_transport_statistics(self, bkd):
        """SGD-trained VF transports N(0,1) to N(shift,1) in mean."""
        torch_bkd = _make_torch_bkd()
        shift = 2.0
        source, target = _make_shift_samplers(shift=shift, n_pool=5000)
        path = LinearPath(torch_bkd)
        torch.manual_seed(0)
        vf = MLPVelocityField(
            nvars_in=2, nqoi=1, hidden_dims=[64, 64], bkd=bkd,
        )

        fitter = TorchSGDFitter(
            lr=1e-3, n_steps=10000, batch_size=512, seed=42,
        )
        result = fitter.fit(vf, path, source, target)
        fitted = result.surrogate()

        nsamples = 2000
        x0 = bkd.asarray(np.random.RandomState(99).randn(1, nsamples).tolist())
        x1 = integrate_flow(
            fitted, x0, 0.0, 1.0, n_steps=100, bkd=bkd,
            stepper_cls=HeunStepper,
        )
        transported_mean = float(bkd.mean(x1))
        transported_std = float(bkd.std(x1))
        assert abs(transported_mean - shift) < 0.05
        assert abs(transported_std - 1.0) < 0.05

    def test_seed_reproducibility(self, bkd):
        """Same seed produces same result."""
        torch_bkd = _make_torch_bkd()
        path = LinearPath(torch_bkd)

        losses = []
        for _ in range(2):
            source, target = _make_shift_samplers(shift=1.0, seed=0)
            torch.manual_seed(42)
            vf = MLPVelocityField(
                nvars_in=2, nqoi=1, hidden_dims=[8], bkd=bkd,
            )
            fitter = TorchSGDFitter(
                lr=1e-3, n_steps=50, batch_size=64, seed=123,
            )
            result = fitter.fit(vf, path, source, target)
            losses.append(result.training_loss())
        assert losses[0] == losses[1]

    def test_wrong_path_backend_raises(self):
        """Passing a non-torch path raises TypeError."""
        np_bkd = NumpyBkd()
        numpy_path = LinearPath(np_bkd)
        torch_bkd = _make_torch_bkd()
        source, target = _make_shift_samplers()
        vf = MLPVelocityField(
            nvars_in=2, nqoi=1, hidden_dims=[8], bkd=torch_bkd,
        )
        fitter = TorchSGDFitter(lr=1e-3, n_steps=1)
        with pytest.raises(TypeError, match="TorchBkd"):
            fitter.fit(vf, numpy_path, source, target)

    def test_density_end_to_end(self, bkd):
        """Trained SGD model works with compute_flow_density."""
        from pyapprox.generative.flowmatching.density import (
            compute_flow_density,
        )

        torch_bkd = _make_torch_bkd()
        source, target = _make_shift_samplers(shift=1.0, n_pool=500)
        path = LinearPath(torch_bkd)
        torch.manual_seed(0)
        vf = MLPVelocityField(
            nvars_in=2, nqoi=1, hidden_dims=[16, 16], bkd=bkd,
        )

        fitter = TorchSGDFitter(
            lr=1e-3, n_steps=1000, batch_size=128, seed=42,
        )
        result = fitter.fit(vf, path, source, target)
        fitted = result.surrogate()

        x_grid = bkd.reshape(bkd.linspace(-4, 4, 50), (1, -1))
        q = compute_flow_density(
            fitted, x_grid, bkd, n_steps=50, scheme="heun",
        )
        assert q.shape == (1, 50)
        assert bkd.all_bool(bkd.isfinite(q))
        assert bkd.all_bool(q >= 0)

"""Tests for TorchQuadratureFitter."""

import pytest

torch = pytest.importorskip("torch")

from pyapprox.generative.flowmatching.fitters.torch_quadrature_fitter import (
    TorchQuadratureFitter,
)
from pyapprox.generative.flowmatching.linear_path import LinearPath
from pyapprox.generative.flowmatching.nn_vf import MLPVelocityField
from pyapprox.generative.flowmatching.quad_data import (
    build_flow_matching_quad_data,
    gauss_hermite_rule,
    gauss_legendre_rule,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd


def _make_torch_bkd():
    torch.set_default_dtype(torch.float64)
    return TorchBkd()


def _make_linear_quad_data(bkd, n_t=5, n_x=10):
    """Quad data for the identity transport x1 = x0 (u_t = 0)."""
    t_rule = gauss_legendre_rule(bkd)
    x0_rule = gauss_hermite_rule(bkd)
    return build_flow_matching_quad_data(
        t_rule, x0_rule, forward_map=lambda x: x, n_t=n_t, n_x=n_x, bkd=bkd,
    )


def _make_shift_quad_data(bkd, shift=2.0, n_t=5, n_x=10):
    """Quad data for the shift transport x1 = x0 + shift (u_t = shift)."""
    t_rule = gauss_legendre_rule(bkd)
    x0_rule = gauss_hermite_rule(bkd)
    return build_flow_matching_quad_data(
        t_rule, x0_rule, forward_map=lambda x: x + shift,
        n_t=n_t, n_x=n_x, bkd=bkd,
    )


class TestTorchQuadratureFitter:
    def test_identity_transport_zero_loss(self, bkd):
        """For x1=x0, target field u_t=0. NN should learn near-zero VF."""
        torch_bkd = _make_torch_bkd()
        qd = _make_linear_quad_data(bkd, n_t=5, n_x=15)
        path = LinearPath(torch_bkd)
        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[16, 16], bkd=bkd)

        fitter = TorchQuadratureFitter(
            lr=1e-3, n_epochs=500, seed=42,
        )
        result = fitter.fit(vf, path, qd)
        assert result.training_loss() < 1e-4

    def test_shift_transport_recovery(self, bkd):
        """For x1=x0+c, target field u_t=c. NN should recover constant VF."""
        torch_bkd = _make_torch_bkd()
        shift = 2.0
        qd = _make_shift_quad_data(bkd, shift=shift, n_t=5, n_x=15)
        path = LinearPath(torch_bkd)
        torch.manual_seed(0)
        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[32, 32], bkd=bkd)

        fitter = TorchQuadratureFitter(
            lr=1e-3, n_epochs=1000, seed=42,
        )
        result = fitter.fit(vf, path, qd)
        assert result.training_loss() < 1e-3

        fitted = result.surrogate()
        t_test = bkd.reshape(bkd.linspace(0.1, 0.9, 10), (1, -1))
        x_test = bkd.reshape(bkd.linspace(-2.0, 2.0, 10), (1, -1))
        vf_in = bkd.vstack([
            bkd.tile(t_test, (1, 10)),
            bkd.repeat(x_test, 10),
        ])
        v_pred = fitted(vf_in)
        expected = bkd.full(v_pred.shape, shift)
        bkd.assert_allclose(v_pred, expected, rtol=0.02)

    def test_seed_reproducibility(self, bkd):
        """Same seed produces same result."""
        torch_bkd = _make_torch_bkd()
        qd = _make_shift_quad_data(bkd, shift=1.0)
        path = LinearPath(torch_bkd)

        losses = []
        for _ in range(2):
            torch.manual_seed(42)
            vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[8], bkd=bkd)
            fitter = TorchQuadratureFitter(
                lr=1e-3, n_epochs=50, seed=123,
            )
            result = fitter.fit(vf, path, qd)
            losses.append(result.training_loss())
        assert losses[0] == losses[1]

    def test_mini_batch(self, bkd):
        """Mini-batch training runs without error and reduces loss."""
        torch_bkd = _make_torch_bkd()
        qd = _make_shift_quad_data(bkd, shift=1.0, n_t=5, n_x=15)
        path = LinearPath(torch_bkd)
        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[16], bkd=bkd)

        fitter = TorchQuadratureFitter(
            lr=1e-3, n_epochs=200, batch_size=20, seed=42,
        )
        result = fitter.fit(vf, path, qd)
        assert result.training_loss() < 1.0

    def test_wrong_path_backend_raises(self):
        """Passing a non-torch path raises TypeError."""
        np_bkd = NumpyBkd()
        numpy_path = LinearPath(np_bkd)
        torch_bkd = _make_torch_bkd()
        qd = _make_linear_quad_data(torch_bkd)
        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[8], bkd=torch_bkd)
        fitter = TorchQuadratureFitter(lr=1e-3, n_epochs=1)
        with pytest.raises(TypeError, match="TorchBkd"):
            fitter.fit(vf, numpy_path, qd)

    def test_density_end_to_end(self, bkd):
        """Trained NN can be used with compute_flow_density."""
        from pyapprox.generative.flowmatching.density import (
            compute_flow_density,
        )

        torch_bkd = _make_torch_bkd()
        qd = _make_shift_quad_data(bkd, shift=1.0, n_t=5, n_x=15)
        path = LinearPath(torch_bkd)
        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[16, 16], bkd=bkd)

        fitter = TorchQuadratureFitter(
            lr=1e-3, n_epochs=500, seed=42,
        )
        result = fitter.fit(vf, path, qd)
        fitted = result.surrogate()

        x_grid = bkd.reshape(bkd.linspace(-4, 4, 50), (1, -1))
        q = compute_flow_density(
            fitted, x_grid, bkd, n_steps=50, scheme="heun",
        )
        assert q.shape == (1, 50)
        assert bkd.all_bool(bkd.isfinite(q))
        assert bkd.all_bool(q >= 0)

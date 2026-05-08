"""Tests for MLPVelocityField."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyapprox.generative.flowmatching.nn_vf import MLPVelocityField
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    BatchDerivativeChecker,
)


class TestMLPVelocityField:
    def test_forward_shape(self, bkd):
        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[16, 16], bkd=bkd)
        x = bkd.asarray(np.random.randn(2, 50))
        y = vf(x)
        assert y.shape == (1, 50)

    def test_forward_multiqoi_shape(self, bkd):
        vf = MLPVelocityField(nvars_in=3, nqoi=2, hidden_dims=[16], bkd=bkd)
        x = bkd.asarray(np.random.randn(3, 30))
        y = vf(x)
        assert y.shape == (2, 30)

    def test_jacobian_batch_shape(self, bkd):
        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[16, 16], bkd=bkd)
        x = bkd.asarray(np.random.randn(2, 50))
        jac = vf.jacobian_batch(x)
        assert jac.shape == (50, 1, 2)

    def test_jacobian_batch_fd(self, bkd):
        """Verify jacobian_batch via BatchDerivativeChecker."""
        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[16, 16], bkd=bkd)
        np.random.seed(42)
        samples = bkd.asarray(np.random.randn(2, 5))
        checker = BatchDerivativeChecker(vf, samples)
        errors = checker.check_jacobian_batch(verbosity=0)
        ratio = checker.error_ratio(errors)
        assert float(ratio) < 1e-6

    def test_jacobian_batch_fd_multiqoi(self, bkd):
        """Verify jacobian_batch for multi-output VF."""
        vf = MLPVelocityField(nvars_in=3, nqoi=2, hidden_dims=[16], bkd=bkd)
        np.random.seed(0)
        samples = bkd.asarray(np.random.randn(3, 5))
        checker = BatchDerivativeChecker(vf, samples)
        errors = checker.check_jacobian_batch(verbosity=0)
        ratio = checker.error_ratio(errors)
        assert float(ratio) < 1e-6

    def test_interface(self, bkd):
        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[8], bkd=bkd)
        assert vf.nvars() == 2
        assert vf.nqoi() == 1
        assert vf.bkd() is bkd

    def test_invalid_activation_raises(self, bkd):
        with pytest.raises(ValueError, match="Unknown activation"):
            MLPVelocityField(
                nvars_in=2, nqoi=1, hidden_dims=[8], bkd=bkd, activation="gelu"
            )

    def test_density_pipeline(self, bkd):
        """MLPVelocityField works with compute_flow_density."""
        from pyapprox.generative.flowmatching.density import compute_flow_density

        vf = MLPVelocityField(nvars_in=2, nqoi=1, hidden_dims=[16, 16], bkd=bkd)
        x_grid = bkd.reshape(bkd.linspace(-2, 2, 20), (1, -1))
        q = compute_flow_density(vf, x_grid, bkd, n_steps=10, scheme="euler")
        assert q.shape == (1, 20)
        assert bkd.all_bool(bkd.isfinite(q))

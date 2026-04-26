"""Tests for SUPN MSE fitter.

Tests validate end-to-end fitting convergence and accuracy.
"""

import numpy as np
import pytest

from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.surrogates.supn import create_supn
from pyapprox.surrogates.supn.fitters import SUPNMSEFitter


class TestSUPNMSEFitter:
    """Tests for SUPNMSEFitter."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_rejects_non_supn(self, bkd) -> None:
        """Test that fitter raises TypeError for non-SUPN."""
        fitter = SUPNMSEFitter(bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        with pytest.raises(TypeError, match="SUPNMSEFitter only works with SUPN"):
            fitter.fit("not_a_supn", samples, values)

    def test_shape_validation_samples(self, bkd) -> None:
        """Test ValueError for wrong sample dimension."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        fitter = SUPNMSEFitter(bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        with pytest.raises(ValueError, match="samples has 3 variables"):
            fitter.fit(supn, samples, values)

    def test_shape_validation_values(self, bkd) -> None:
        """Test ValueError for wrong value dimension."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        fitter = SUPNMSEFitter(bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        values = bkd.asarray(np.random.randn(3, 10))

        with pytest.raises(ValueError, match="values has 3 QoIs"):
            fitter.fit(supn, samples, values)

    def test_fits_own_output(self, bkd) -> None:
        """Test SUPN can fit its own output (exact recovery)."""
        supn = create_supn(nvars=1, width=2, max_level=2, bkd=bkd)
        true_params = bkd.asarray(np.random.randn(supn.nparams()))
        supn_true = supn.with_params(true_params)

        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        values = supn_true(samples)

        # Start from different initial params
        supn_init = supn.with_params(
            bkd.asarray(np.random.randn(supn.nparams()) * 0.1)
        )

        fitter = SUPNMSEFitter(bkd)
        result = fitter.fit(supn_init, samples, values)

        assert result.final_loss() < 1e-10

    def test_fit_sin(self, bkd) -> None:
        """Test fitting sin(pi*x) on [-1, 1]."""
        supn = create_supn(nvars=1, width=5, max_level=4, bkd=bkd)

        nsamples = 50
        x = np.random.uniform(-1, 1, (1, nsamples))
        samples = bkd.asarray(x)
        values = bkd.asarray(np.sin(np.pi * x))

        fitter = SUPNMSEFitter(bkd)
        fitter.set_optimizer(
            ScipyTrustConstrOptimizer(verbosity=0, maxiter=2000)
        )
        result = fitter.fit(supn, samples, values)

        assert result.final_loss() < 1e-6

        # Check on test points
        x_test = np.linspace(-1, 1, 30).reshape(1, -1)
        test_samples = bkd.asarray(x_test)
        test_values = bkd.asarray(np.sin(np.pi * x_test))
        pred = result(test_samples)
        bkd.assert_allclose(pred, test_values, atol=1e-2)

    def test_fit_2d(self, bkd) -> None:
        """Test fitting exp(x1-0.7)*sin(1.3*x2) on [-1,1]^2."""
        supn = create_supn(nvars=2, width=8, max_level=3, bkd=bkd)

        nsamples = 100
        x = np.random.uniform(-1, 1, (2, nsamples))
        samples = bkd.asarray(x)
        values = bkd.asarray(
            np.exp(x[0:1] - 0.7) * np.sin(1.3 * x[1:2])
        )

        fitter = SUPNMSEFitter(bkd)
        fitter.set_optimizer(
            ScipyTrustConstrOptimizer(verbosity=0, maxiter=3000)
        )
        result = fitter.fit(supn, samples, values)

        assert result.final_loss() < 1e-4

    def test_fit_multi_qoi(self, bkd) -> None:
        """Test fitting with nqoi > 1."""
        nqoi = 2
        supn = create_supn(nvars=1, width=4, max_level=3, bkd=bkd, nqoi=nqoi)

        nsamples = 50
        x = np.random.uniform(-1, 1, (1, nsamples))
        samples = bkd.asarray(x)
        values = bkd.asarray(np.vstack([
            np.sin(np.pi * x),
            np.cos(np.pi * x),
        ]))

        fitter = SUPNMSEFitter(bkd)
        fitter.set_optimizer(
            ScipyTrustConstrOptimizer(verbosity=0, maxiter=2000)
        )
        result = fitter.fit(supn, samples, values)

        assert result.final_loss() < 1e-4

    def test_result_properties(self, bkd) -> None:
        """Test SUPNFitterResult properties."""
        supn = create_supn(nvars=1, width=2, max_level=2, bkd=bkd)
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = SUPNMSEFitter(bkd)
        result = fitter.fit(supn, samples, values)

        assert result.surrogate() is not None
        assert result.params().shape == (supn.nparams(),)
        assert isinstance(result.final_loss(), float)
        assert isinstance(result.converged(), bool)
        assert result.optimizer_result() is not None

    def test_1d_values_auto_reshape(self, bkd) -> None:
        """Test that 1D values array is automatically reshaped."""
        supn = create_supn(nvars=1, width=2, max_level=2, bkd=bkd)
        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        values = bkd.asarray(np.random.randn(nsamples))  # 1D

        fitter = SUPNMSEFitter(bkd)
        result = fitter.fit(supn, samples, values)
        assert result.surrogate() is not None

    def test_custom_optimizer(self, bkd) -> None:
        """Test setting a custom optimizer."""
        supn = create_supn(nvars=1, width=2, max_level=2, bkd=bkd)
        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = SUPNMSEFitter(bkd)
        optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=50)
        fitter.set_optimizer(optimizer)
        assert fitter.optimizer() is optimizer

        result = fitter.fit(supn, samples, values)
        assert result.surrogate() is not None

    def test_immutability(self, bkd) -> None:
        """Test that fitting does not modify the original SUPN."""
        supn = create_supn(nvars=1, width=2, max_level=2, bkd=bkd)
        original_params = supn._flatten_params()

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = SUPNMSEFitter(bkd)
        fitter.fit(supn, samples, values)

        bkd.assert_allclose(supn._flatten_params(), original_params)

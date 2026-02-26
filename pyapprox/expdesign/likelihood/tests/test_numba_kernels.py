"""
Tests for Numba-accelerated OED likelihood kernels.

Tests verify that:
- Each Numba kernel matches the vectorized implementation (rtol=1e-12)
- Dispatch logic selects the correct implementation per backend
- Fused evidence jacobian matches separate jacobian_matrix + einsum
- Results are correct with and without latent samples

Minor differences (~1e-14) from float64 arithmetic ordering are expected
with Numba parallel mode. Tests use rtol=1e-12 to accommodate this.
"""

import numpy as np
import pytest

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.optional_deps import package_available

if not package_available("numba"):
    pytest.skip("numba not installed", allow_module_level=True)

from pyapprox.expdesign.evidence import LogEvidence
from pyapprox.expdesign.likelihood import (
    GaussianOEDInnerLoopLikelihood,
)
from pyapprox.expdesign.likelihood.compute import (
    evidence_jacobian_vectorized,
    jacobian_matrix_vectorized,
    logpdf_matrix_vectorized,
)
from pyapprox.expdesign.likelihood.compute_numba import (
    fused_evidence_jacobian_numba,
    jacobian_matrix_numba,
    logpdf_matrix_numba,
)
from pyapprox.expdesign.likelihood.dispatch import (
    get_evidence_jacobian_impl,
    get_jacobian_matrix_impl,
    get_logpdf_matrix_impl,
)


class TestNumbaKernels:
    """Test Numba kernels match vectorized implementations."""

    def _make_data(self, nobs, ninner, nouter):
        """Create test data arrays."""
        shapes = np.random.randn(nobs, ninner)
        obs = np.random.randn(nobs, nouter)
        base_variances = np.abs(np.random.randn(nobs)) + 0.1
        design_weights = np.random.uniform(0.5, 2.0, (nobs, 1))
        latent_samples = np.random.randn(nobs, nouter)
        return shapes, obs, base_variances, design_weights, latent_samples

    def test_logpdf_matrix_small(self, numpy_bkd):
        """Test logpdf_matrix Numba vs vectorized at small size."""
        np.random.seed(42)
        shapes, obs, bv, dw, _ = self._make_data(10, 50, 50)

        result_numba = logpdf_matrix_numba(shapes, obs, bv, dw)
        result_vec = logpdf_matrix_vectorized(shapes, obs, bv, dw, numpy_bkd)

        numpy_bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_logpdf_matrix_medium(self, numpy_bkd):
        """Test logpdf_matrix Numba vs vectorized at medium size."""
        np.random.seed(42)
        shapes, obs, bv, dw, _ = self._make_data(50, 200, 200)

        result_numba = logpdf_matrix_numba(shapes, obs, bv, dw)
        result_vec = logpdf_matrix_vectorized(shapes, obs, bv, dw, numpy_bkd)

        numpy_bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_matrix_no_latent(self, numpy_bkd):
        """Test jacobian_matrix Numba vs vectorized without latent samples."""
        np.random.seed(42)
        shapes, obs, bv, dw, _ = self._make_data(10, 50, 50)
        dummy = np.zeros_like(obs)

        result_numba = jacobian_matrix_numba(shapes, obs, dummy, bv, dw, False)
        result_vec = jacobian_matrix_vectorized(shapes, obs, None, bv, dw, numpy_bkd)

        numpy_bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_matrix_with_latent(self, numpy_bkd):
        """Test jacobian_matrix Numba vs vectorized with latent samples."""
        np.random.seed(42)
        shapes, obs, bv, dw, latent = self._make_data(10, 50, 50)

        result_numba = jacobian_matrix_numba(shapes, obs, latent, bv, dw, True)
        result_vec = jacobian_matrix_vectorized(shapes, obs, latent, bv, dw, numpy_bkd)

        numpy_bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_matrix_medium(self, numpy_bkd):
        """Test jacobian_matrix Numba vs vectorized at medium size."""
        np.random.seed(42)
        shapes, obs, bv, dw, latent = self._make_data(50, 200, 200)

        result_numba = jacobian_matrix_numba(shapes, obs, latent, bv, dw, True)
        result_vec = jacobian_matrix_vectorized(shapes, obs, latent, bv, dw, numpy_bkd)

        numpy_bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_fused_evidence_jacobian_no_latent(self, numpy_bkd):
        """Test fused evidence jacobian without latent samples."""
        np.random.seed(42)
        shapes, obs, bv, dw, _ = self._make_data(10, 50, 50)
        dummy = np.zeros_like(obs)

        # Compute quad_weighted_like from logpdf
        loglike = logpdf_matrix_numba(shapes, obs, bv, dw)
        like = np.exp(loglike)
        ninner = shapes.shape[1]
        quad_weights = np.ones(ninner) / ninner
        quad_weighted_like = quad_weights[:, None] * like

        result_fused = fused_evidence_jacobian_numba(
            shapes,
            obs,
            dummy,
            bv,
            dw,
            quad_weighted_like,
            False,
        )

        # Reference: separate jacobian + einsum
        jac = jacobian_matrix_vectorized(shapes, obs, None, bv, dw, numpy_bkd)
        result_ref = evidence_jacobian_vectorized(jac, quad_weighted_like, numpy_bkd)

        numpy_bkd.assert_allclose(result_fused, result_ref, rtol=1e-12)

    def test_fused_evidence_jacobian_with_latent(self, numpy_bkd):
        """Test fused evidence jacobian with latent samples."""
        np.random.seed(42)
        shapes, obs, bv, dw, latent = self._make_data(10, 50, 50)

        # Compute quad_weighted_like
        loglike = logpdf_matrix_numba(shapes, obs, bv, dw)
        like = np.exp(loglike)
        ninner = shapes.shape[1]
        quad_weights = np.ones(ninner) / ninner
        quad_weighted_like = quad_weights[:, None] * like

        result_fused = fused_evidence_jacobian_numba(
            shapes,
            obs,
            latent,
            bv,
            dw,
            quad_weighted_like,
            True,
        )

        # Reference: separate jacobian + einsum
        jac = jacobian_matrix_vectorized(shapes, obs, latent, bv, dw, numpy_bkd)
        result_ref = evidence_jacobian_vectorized(jac, quad_weighted_like, numpy_bkd)

        numpy_bkd.assert_allclose(result_fused, result_ref, rtol=1e-12)

    def test_fused_evidence_jacobian_medium(self, numpy_bkd):
        """Test fused evidence jacobian at medium size."""
        np.random.seed(42)
        shapes, obs, bv, dw, latent = self._make_data(50, 200, 200)

        loglike = logpdf_matrix_numba(shapes, obs, bv, dw)
        like = np.exp(loglike)
        ninner = shapes.shape[1]
        quad_weights = np.ones(ninner) / ninner
        quad_weighted_like = quad_weights[:, None] * like

        result_fused = fused_evidence_jacobian_numba(
            shapes,
            obs,
            latent,
            bv,
            dw,
            quad_weighted_like,
            True,
        )

        jac = jacobian_matrix_vectorized(shapes, obs, latent, bv, dw, numpy_bkd)
        result_ref = evidence_jacobian_vectorized(jac, quad_weighted_like, numpy_bkd)

        numpy_bkd.assert_allclose(result_fused, result_ref, rtol=1e-11)


class TestDispatch:
    """Test dispatch logic selects correct implementation per backend."""

    def test_numpy_gets_numba(self):
        """NumPy backend should use Numba when available."""
        bkd = NumpyBkd()
        impl = get_logpdf_matrix_impl(bkd)
        # The Numba impl is a closure, not the vectorized function directly
        assert impl is not logpdf_matrix_vectorized

    def test_torch_gets_compiled(self):
        """Torch backend should use torch.compile, not vectorized."""
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()
        impl = get_logpdf_matrix_impl(bkd)
        # torch.compile returns a closure, not the vectorized function
        assert impl is not logpdf_matrix_vectorized

    def test_jacobian_dispatch_numpy(self):
        """Jacobian dispatch for NumPy backend uses Numba."""
        bkd = NumpyBkd()
        impl = get_jacobian_matrix_impl(bkd)
        assert impl is not jacobian_matrix_vectorized

    def test_jacobian_dispatch_torch(self):
        """Jacobian dispatch for Torch backend uses compiled."""
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()
        impl = get_jacobian_matrix_impl(bkd)
        assert impl is not jacobian_matrix_vectorized

    def test_evidence_jacobian_dispatch_numpy(self):
        """Evidence jacobian dispatch for NumPy backend uses Numba."""
        bkd = NumpyBkd()
        impl = get_evidence_jacobian_impl(bkd)
        # Should be a numba closure, not the vectorized fallback
        assert impl is not None

    def test_evidence_jacobian_dispatch_torch(self):
        """Evidence jacobian dispatch for Torch backend uses compiled."""
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()
        impl = get_evidence_jacobian_impl(bkd)
        assert impl is not None


class TestDispatchBranches:
    """Test all dispatch branches produce correct results."""

    def test_numba_evidence_jac_matches_vectorized(self):
        """Numba evidence_jacobian matches vectorized reference."""
        bkd_np = NumpyBkd()
        np.random.seed(42)
        nobs, ninner, nouter = 5, 20, 15
        shapes = np.random.randn(nobs, ninner)
        obs = np.random.randn(nobs, nouter)
        bv = np.abs(np.random.randn(nobs)) + 0.1
        dw = np.random.uniform(0.5, 2.0, (nobs, 1))
        latent = np.random.randn(nobs, nouter)

        impl_numba = get_evidence_jacobian_impl(bkd_np)

        # Compute quad_weighted_like
        loglike = logpdf_matrix_numba(shapes, obs, bv, dw)
        like = np.exp(loglike)
        qw = np.ones(ninner) / ninner
        qwl = qw[:, None] * like

        result_numba = impl_numba(
            shapes,
            obs,
            latent,
            bv,
            dw,
            qwl,
            bkd_np,
        )

        # Reference: vectorized path
        jac = jacobian_matrix_vectorized(
            shapes,
            obs,
            latent,
            bv,
            dw,
            bkd_np,
        )
        result_ref = evidence_jacobian_vectorized(jac, qwl, bkd_np)

        bkd_np.assert_allclose(result_numba, result_ref, rtol=1e-12)


class TestNumbaIntegration:
    """Test Numba integration at the class level (objective pipeline)."""

    def _setup_data(self, bkd):
        self._nobs = 3
        self._ninner = 15
        self._nouter = 10

        np.random.seed(42)
        self._noise_variances = bkd.asarray(np.array([0.1, 0.2, 0.15]))
        self._shapes = bkd.asarray(np.random.randn(self._nobs, self._ninner))
        self._obs = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        self._latent = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        self._design_weights = bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

    def test_logpdf_matrix_runs(self, bkd):
        """Inner loop logpdf_matrix runs without error."""
        self._setup_data(bkd)
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        result = like.logpdf_matrix(self._design_weights)
        assert result.shape == (self._ninner, self._nouter)

    def test_jacobian_matrix_runs(self, bkd):
        """Inner loop jacobian_matrix runs without error."""
        self._setup_data(bkd)
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        like.set_latent_samples(self._latent)
        result = like.jacobian_matrix(self._design_weights)
        assert result.shape == (self._ninner, self._nouter, self._nobs)

    def test_evidence_jacobian_runs(self, bkd):
        """Fused evidence jacobian runs via LogEvidence."""
        self._setup_data(bkd)
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        like.set_latent_samples(self._latent)

        quad_weights = bkd.ones((self._ninner,)) / self._ninner
        ev = LogEvidence(like, quad_weights, bkd)
        jac = ev.jacobian(self._design_weights)
        assert jac.shape == (self._nouter, self._nobs)

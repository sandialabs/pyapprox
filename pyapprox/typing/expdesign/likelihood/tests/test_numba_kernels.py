"""
Tests for Numba-accelerated OED likelihood kernels.

Tests verify that:
- Each Numba kernel matches the vectorized implementation (rtol=1e-12)
- Dispatch logic selects the correct implementation per backend
- use_numba=False forces the vectorized path
- Fused evidence jacobian matches separate jacobian_matrix + einsum
- Results are correct with and without latent samples

Minor differences (~1e-14) from float64 arithmetic ordering are expected
with Numba parallel mode. Tests use rtol=1e-12 to accommodate this.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.likelihood.compute import (
    logpdf_matrix_vectorized,
    jacobian_matrix_vectorized,
    evidence_jacobian_vectorized,
)
from pyapprox.typing.expdesign.likelihood.compute_numba import (
    logpdf_matrix_numba,
    jacobian_matrix_numba,
    fused_evidence_jacobian_numba,
)
from pyapprox.typing.expdesign.likelihood.dispatch import (
    get_logpdf_matrix_impl,
    get_jacobian_matrix_impl,
    get_evidence_jacobian_impl,
)
from pyapprox.typing.expdesign.likelihood import (
    GaussianOEDInnerLoopLikelihood,
)
from pyapprox.typing.expdesign.evidence import LogEvidence


class TestNumbaKernels(unittest.TestCase):
    """Test Numba kernels match vectorized implementations."""

    def setUp(self):
        self._bkd = NumpyBkd()
        np.random.seed(42)

    def _make_data(self, nobs, ninner, nouter):
        """Create test data arrays."""
        shapes = np.random.randn(nobs, ninner)
        obs = np.random.randn(nobs, nouter)
        base_variances = np.abs(np.random.randn(nobs)) + 0.1
        design_weights = np.random.uniform(0.5, 2.0, (nobs, 1))
        latent_samples = np.random.randn(nobs, nouter)
        return shapes, obs, base_variances, design_weights, latent_samples

    def test_logpdf_matrix_small(self):
        """Test logpdf_matrix Numba vs vectorized at small size."""
        shapes, obs, bv, dw, _ = self._make_data(10, 50, 50)

        result_numba = logpdf_matrix_numba(shapes, obs, bv, dw)
        result_vec = logpdf_matrix_vectorized(shapes, obs, bv, dw, self._bkd)

        self._bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_logpdf_matrix_medium(self):
        """Test logpdf_matrix Numba vs vectorized at medium size."""
        shapes, obs, bv, dw, _ = self._make_data(50, 200, 200)

        result_numba = logpdf_matrix_numba(shapes, obs, bv, dw)
        result_vec = logpdf_matrix_vectorized(shapes, obs, bv, dw, self._bkd)

        self._bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_matrix_no_latent(self):
        """Test jacobian_matrix Numba vs vectorized without latent samples."""
        shapes, obs, bv, dw, _ = self._make_data(10, 50, 50)
        dummy = np.zeros_like(obs)

        result_numba = jacobian_matrix_numba(
            shapes, obs, dummy, bv, dw, False
        )
        result_vec = jacobian_matrix_vectorized(
            shapes, obs, None, bv, dw, self._bkd
        )

        self._bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_matrix_with_latent(self):
        """Test jacobian_matrix Numba vs vectorized with latent samples."""
        shapes, obs, bv, dw, latent = self._make_data(10, 50, 50)

        result_numba = jacobian_matrix_numba(
            shapes, obs, latent, bv, dw, True
        )
        result_vec = jacobian_matrix_vectorized(
            shapes, obs, latent, bv, dw, self._bkd
        )

        self._bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_matrix_medium(self):
        """Test jacobian_matrix Numba vs vectorized at medium size."""
        shapes, obs, bv, dw, latent = self._make_data(50, 200, 200)

        result_numba = jacobian_matrix_numba(
            shapes, obs, latent, bv, dw, True
        )
        result_vec = jacobian_matrix_vectorized(
            shapes, obs, latent, bv, dw, self._bkd
        )

        self._bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_fused_evidence_jacobian_no_latent(self):
        """Test fused evidence jacobian without latent samples."""
        shapes, obs, bv, dw, _ = self._make_data(10, 50, 50)
        dummy = np.zeros_like(obs)

        # Compute quad_weighted_like from logpdf
        loglike = logpdf_matrix_numba(shapes, obs, bv, dw)
        like = np.exp(loglike)
        ninner = shapes.shape[1]
        quad_weights = np.ones(ninner) / ninner
        quad_weighted_like = quad_weights[:, None] * like

        result_fused = fused_evidence_jacobian_numba(
            shapes, obs, dummy, bv, dw, quad_weighted_like, False,
        )

        # Reference: separate jacobian + einsum
        jac = jacobian_matrix_vectorized(
            shapes, obs, None, bv, dw, self._bkd
        )
        result_ref = evidence_jacobian_vectorized(jac, quad_weighted_like, self._bkd)

        self._bkd.assert_allclose(result_fused, result_ref, rtol=1e-12)

    def test_fused_evidence_jacobian_with_latent(self):
        """Test fused evidence jacobian with latent samples."""
        shapes, obs, bv, dw, latent = self._make_data(10, 50, 50)

        # Compute quad_weighted_like
        loglike = logpdf_matrix_numba(shapes, obs, bv, dw)
        like = np.exp(loglike)
        ninner = shapes.shape[1]
        quad_weights = np.ones(ninner) / ninner
        quad_weighted_like = quad_weights[:, None] * like

        result_fused = fused_evidence_jacobian_numba(
            shapes, obs, latent, bv, dw, quad_weighted_like, True,
        )

        # Reference: separate jacobian + einsum
        jac = jacobian_matrix_vectorized(
            shapes, obs, latent, bv, dw, self._bkd
        )
        result_ref = evidence_jacobian_vectorized(jac, quad_weighted_like, self._bkd)

        self._bkd.assert_allclose(result_fused, result_ref, rtol=1e-12)

    def test_fused_evidence_jacobian_medium(self):
        """Test fused evidence jacobian at medium size."""
        shapes, obs, bv, dw, latent = self._make_data(50, 200, 200)

        loglike = logpdf_matrix_numba(shapes, obs, bv, dw)
        like = np.exp(loglike)
        ninner = shapes.shape[1]
        quad_weights = np.ones(ninner) / ninner
        quad_weighted_like = quad_weights[:, None] * like

        result_fused = fused_evidence_jacobian_numba(
            shapes, obs, latent, bv, dw, quad_weighted_like, True,
        )

        jac = jacobian_matrix_vectorized(
            shapes, obs, latent, bv, dw, self._bkd
        )
        result_ref = evidence_jacobian_vectorized(jac, quad_weighted_like, self._bkd)

        self._bkd.assert_allclose(result_fused, result_ref, rtol=1e-11)


class TestDispatch(unittest.TestCase):
    """Test dispatch logic selects correct implementation per backend."""

    def test_numpy_gets_numba(self):
        """NumPy backend should use Numba when available and requested."""
        bkd = NumpyBkd()
        impl = get_logpdf_matrix_impl(bkd, use_numba=True)
        # The Numba impl is a closure, not the vectorized function directly
        self.assertIsNot(impl, logpdf_matrix_vectorized)

    def test_numpy_no_numba_gets_vectorized(self):
        """NumPy backend with use_numba=False should use vectorized."""
        bkd = NumpyBkd()
        impl = get_logpdf_matrix_impl(bkd, use_numba=False)
        self.assertIs(impl, logpdf_matrix_vectorized)

    def test_torch_gets_vectorized(self):
        """Torch backend should always use vectorized."""
        bkd = TorchBkd()
        impl = get_logpdf_matrix_impl(bkd, use_numba=True)
        self.assertIs(impl, logpdf_matrix_vectorized)

    def test_jacobian_dispatch_numpy(self):
        """Jacobian dispatch for NumPy backend."""
        bkd = NumpyBkd()
        impl_numba = get_jacobian_matrix_impl(bkd, use_numba=True)
        impl_vec = get_jacobian_matrix_impl(bkd, use_numba=False)
        self.assertIsNot(impl_numba, jacobian_matrix_vectorized)
        self.assertIs(impl_vec, jacobian_matrix_vectorized)

    def test_evidence_jacobian_dispatch_numpy(self):
        """Evidence jacobian dispatch for NumPy backend."""
        bkd = NumpyBkd()
        impl_numba = get_evidence_jacobian_impl(bkd, use_numba=True)
        impl_vec = get_evidence_jacobian_impl(bkd, use_numba=False)
        # Both are closures, but they should be different objects
        self.assertIsNot(impl_numba, impl_vec)


class TestNumbaIntegration(Generic[Array], unittest.TestCase):
    """Test Numba integration at the class level (objective pipeline)."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 3
        self._ninner = 15
        self._nouter = 10

        np.random.seed(42)
        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.2, 0.15])
        )
        self._shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._obs = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._latent = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._design_weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

    def test_numba_vs_no_numba_logpdf(self):
        """Inner loop logpdf_matrix matches with and without Numba."""
        like_numba = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd, use_numba=True,
        )
        like_nonumba = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd, use_numba=False,
        )

        for like in (like_numba, like_nonumba):
            like.set_shapes(self._shapes)
            like.set_observations(self._obs)

        result_numba = like_numba.logpdf_matrix(self._design_weights)
        result_nonumba = like_nonumba.logpdf_matrix(self._design_weights)

        self._bkd.assert_allclose(result_numba, result_nonumba, rtol=1e-12)

    def test_numba_vs_no_numba_jacobian(self):
        """Inner loop jacobian_matrix matches with and without Numba."""
        like_numba = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd, use_numba=True,
        )
        like_nonumba = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd, use_numba=False,
        )

        for like in (like_numba, like_nonumba):
            like.set_shapes(self._shapes)
            like.set_observations(self._obs)
            like.set_latent_samples(self._latent)

        result_numba = like_numba.jacobian_matrix(self._design_weights)
        result_nonumba = like_nonumba.jacobian_matrix(self._design_weights)

        self._bkd.assert_allclose(result_numba, result_nonumba, rtol=1e-12)

    def test_numba_vs_no_numba_evidence_jacobian(self):
        """Fused evidence jacobian matches non-fused path."""
        like_numba = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd, use_numba=True,
        )
        like_nonumba = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd, use_numba=False,
        )

        for like in (like_numba, like_nonumba):
            like.set_shapes(self._shapes)
            like.set_observations(self._obs)
            like.set_latent_samples(self._latent)

        quad_weights = self._bkd.ones((self._ninner,)) / self._ninner

        ev_numba = LogEvidence(like_numba, quad_weights, self._bkd)
        ev_nonumba = LogEvidence(like_nonumba, quad_weights, self._bkd)

        jac_numba = ev_numba.jacobian(self._design_weights)
        jac_nonumba = ev_nonumba.jacobian(self._design_weights)

        self._bkd.assert_allclose(jac_numba, jac_nonumba, rtol=1e-10)

    def test_use_numba_flag(self):
        """Test that use_numba flag is accessible."""
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd, use_numba=False,
        )
        self.assertFalse(like.use_numba())

        like2 = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd, use_numba=True,
        )
        self.assertTrue(like2.use_numba())


class TestNumbaIntegrationNumpy(TestNumbaIntegration[NDArray[Any]]):
    """NumPy backend integration tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestNumbaIntegrationTorch(TestNumbaIntegration[torch.Tensor]):
    """PyTorch backend integration tests.

    Torch backend always uses the vectorized path, so Numba vs non-Numba
    should produce identical results (same code path).
    """

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()

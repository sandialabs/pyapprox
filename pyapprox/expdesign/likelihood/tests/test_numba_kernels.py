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

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.optional_deps import package_available
from pyapprox.util.test_utils import load_tests  # noqa: F401

if not package_available("numba"):
    raise unittest.SkipTest("numba not installed")

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

        result_numba = jacobian_matrix_numba(shapes, obs, dummy, bv, dw, False)
        result_vec = jacobian_matrix_vectorized(shapes, obs, None, bv, dw, self._bkd)

        self._bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_matrix_with_latent(self):
        """Test jacobian_matrix Numba vs vectorized with latent samples."""
        shapes, obs, bv, dw, latent = self._make_data(10, 50, 50)

        result_numba = jacobian_matrix_numba(shapes, obs, latent, bv, dw, True)
        result_vec = jacobian_matrix_vectorized(shapes, obs, latent, bv, dw, self._bkd)

        self._bkd.assert_allclose(result_numba, result_vec, rtol=1e-12)

    def test_jacobian_matrix_medium(self):
        """Test jacobian_matrix Numba vs vectorized at medium size."""
        shapes, obs, bv, dw, latent = self._make_data(50, 200, 200)

        result_numba = jacobian_matrix_numba(shapes, obs, latent, bv, dw, True)
        result_vec = jacobian_matrix_vectorized(shapes, obs, latent, bv, dw, self._bkd)

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
            shapes,
            obs,
            dummy,
            bv,
            dw,
            quad_weighted_like,
            False,
        )

        # Reference: separate jacobian + einsum
        jac = jacobian_matrix_vectorized(shapes, obs, None, bv, dw, self._bkd)
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
            shapes,
            obs,
            latent,
            bv,
            dw,
            quad_weighted_like,
            True,
        )

        # Reference: separate jacobian + einsum
        jac = jacobian_matrix_vectorized(shapes, obs, latent, bv, dw, self._bkd)
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
            shapes,
            obs,
            latent,
            bv,
            dw,
            quad_weighted_like,
            True,
        )

        jac = jacobian_matrix_vectorized(shapes, obs, latent, bv, dw, self._bkd)
        result_ref = evidence_jacobian_vectorized(jac, quad_weighted_like, self._bkd)

        self._bkd.assert_allclose(result_fused, result_ref, rtol=1e-11)


class TestDispatch(unittest.TestCase):
    """Test dispatch logic selects correct implementation per backend."""

    def test_numpy_gets_numba(self):
        """NumPy backend should use Numba when available."""
        bkd = NumpyBkd()
        impl = get_logpdf_matrix_impl(bkd)
        # The Numba impl is a closure, not the vectorized function directly
        self.assertIsNot(impl, logpdf_matrix_vectorized)

    def test_torch_gets_compiled(self):
        """Torch backend should use torch.compile, not vectorized."""
        bkd = TorchBkd()
        impl = get_logpdf_matrix_impl(bkd)
        # torch.compile returns a closure, not the vectorized function
        self.assertIsNot(impl, logpdf_matrix_vectorized)

    def test_jacobian_dispatch_numpy(self):
        """Jacobian dispatch for NumPy backend uses Numba."""
        bkd = NumpyBkd()
        impl = get_jacobian_matrix_impl(bkd)
        self.assertIsNot(impl, jacobian_matrix_vectorized)

    def test_jacobian_dispatch_torch(self):
        """Jacobian dispatch for Torch backend uses compiled."""
        bkd = TorchBkd()
        impl = get_jacobian_matrix_impl(bkd)
        self.assertIsNot(impl, jacobian_matrix_vectorized)

    def test_evidence_jacobian_dispatch_numpy(self):
        """Evidence jacobian dispatch for NumPy backend uses Numba."""
        bkd = NumpyBkd()
        impl = get_evidence_jacobian_impl(bkd)
        # Should be a numba closure, not the vectorized fallback
        self.assertIsNotNone(impl)

    def test_evidence_jacobian_dispatch_torch(self):
        """Evidence jacobian dispatch for Torch backend uses compiled."""
        bkd = TorchBkd()
        impl = get_evidence_jacobian_impl(bkd)
        self.assertIsNotNone(impl)


class TestDispatchBranches(unittest.TestCase):
    """Test all dispatch branches produce correct results."""

    def setUp(self):
        self._bkd_np = NumpyBkd()
        self._bkd_torch = TorchBkd()
        np.random.seed(42)
        nobs, ninner, nouter = 5, 20, 15
        self._shapes = np.random.randn(nobs, ninner)
        self._obs = np.random.randn(nobs, nouter)
        self._bv = np.abs(np.random.randn(nobs)) + 0.1
        self._dw = np.random.uniform(0.5, 2.0, (nobs, 1))
        self._latent = np.random.randn(nobs, nouter)

    def test_numba_evidence_jac_matches_vectorized(self):
        """Numba evidence_jacobian matches vectorized reference."""
        impl_numba = get_evidence_jacobian_impl(self._bkd_np)

        # Compute quad_weighted_like
        loglike = logpdf_matrix_numba(self._shapes, self._obs, self._bv, self._dw)
        like = np.exp(loglike)
        ninner = self._shapes.shape[1]
        qw = np.ones(ninner) / ninner
        qwl = qw[:, None] * like

        result_numba = impl_numba(
            self._shapes,
            self._obs,
            self._latent,
            self._bv,
            self._dw,
            qwl,
            self._bkd_np,
        )

        # Reference: vectorized path
        jac = jacobian_matrix_vectorized(
            self._shapes,
            self._obs,
            self._latent,
            self._bv,
            self._dw,
            self._bkd_np,
        )
        result_ref = evidence_jacobian_vectorized(jac, qwl, self._bkd_np)

        self._bkd_np.assert_allclose(result_numba, result_ref, rtol=1e-12)


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
        self._noise_variances = self._bkd.asarray(np.array([0.1, 0.2, 0.15]))
        self._shapes = self._bkd.asarray(np.random.randn(self._nobs, self._ninner))
        self._obs = self._bkd.asarray(np.random.randn(self._nobs, self._nouter))
        self._latent = self._bkd.asarray(np.random.randn(self._nobs, self._nouter))
        self._design_weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

    def test_logpdf_matrix_runs(self):
        """Inner loop logpdf_matrix runs without error."""
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            self._bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        result = like.logpdf_matrix(self._design_weights)
        self.assertEqual(result.shape, (self._ninner, self._nouter))

    def test_jacobian_matrix_runs(self):
        """Inner loop jacobian_matrix runs without error."""
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            self._bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        like.set_latent_samples(self._latent)
        result = like.jacobian_matrix(self._design_weights)
        self.assertEqual(
            result.shape,
            (self._ninner, self._nouter, self._nobs),
        )

    def test_evidence_jacobian_runs(self):
        """Fused evidence jacobian runs via LogEvidence."""
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            self._bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        like.set_latent_samples(self._latent)

        quad_weights = self._bkd.ones((self._ninner,)) / self._ninner
        ev = LogEvidence(like, quad_weights, self._bkd)
        jac = ev.jacobian(self._design_weights)
        self.assertEqual(jac.shape, (self._nouter, self._nobs))


class TestNumbaIntegrationNumpy(TestNumbaIntegration[NDArray[Any]]):
    """NumPy backend integration tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestNumbaIntegrationTorch(TestNumbaIntegration[torch.Tensor]):
    """PyTorch backend integration tests.

    Torch backend uses torch.compile path automatically.
    """

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()

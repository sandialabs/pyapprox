"""
Tests verifying torch.compile dispatch correctness.

Each test class uses a single fixed shape (nobs=5, ninner=20, nouter=15)
to ensure only one torch.compile compilation per class, keeping warmup
overhead manageable.

With automatic dispatch, Torch backend always gets torch.compile. These
tests verify compiled results match vectorized reference results computed
using the vectorized implementations directly.
"""

import unittest

import numpy as np
import torch

from pyapprox.expdesign.likelihood import (
    GaussianOEDInnerLoopLikelihood,
)
from pyapprox.expdesign.likelihood.compute import (
    jacobian_matrix_vectorized,
    logpdf_matrix_vectorized,
)
from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests, slow_test  # noqa: F401

# Fixed dimensions — shared across all test classes to minimize recompilation.
_NOBS = 5
_NINNER = 20
_NOUTER = 15


@slow_test
class TestCompiledLogpdfMatrix(unittest.TestCase):
    """Verify compiled logpdf_matrix matches vectorized reference."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

        self._noise_variances = self._bkd.asarray(
            np.random.uniform(0.05, 0.3, (_NOBS,))
        )
        self._shapes = self._bkd.asarray(np.random.randn(_NOBS, _NINNER))
        self._obs = self._bkd.asarray(np.random.randn(_NOBS, _NOUTER))

    def test_compiled_matches_vectorized(self):
        """torch.compile logpdf_matrix matches vectorized computation."""
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            self._bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)

        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (_NOBS, 1)))

        compiled_result = like.logpdf_matrix(weights)
        vec_result = logpdf_matrix_vectorized(
            self._shapes,
            self._obs,
            self._noise_variances,
            weights,
            self._bkd,
        )

        self._bkd.assert_allclose(compiled_result, vec_result, rtol=1e-12)


@slow_test
class TestCompiledJacobianMatrix(unittest.TestCase):
    """Verify compiled jacobian_matrix matches vectorized reference."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

        self._noise_variances = self._bkd.asarray(
            np.random.uniform(0.05, 0.3, (_NOBS,))
        )
        self._shapes = self._bkd.asarray(np.random.randn(_NOBS, _NINNER))
        self._obs = self._bkd.asarray(np.random.randn(_NOBS, _NOUTER))
        self._latent_samples = self._bkd.asarray(np.random.randn(_NOBS, _NOUTER))

    def test_compiled_matches_vectorized_no_latent(self):
        """torch.compile jacobian matches vectorized without latent samples."""
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            self._bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)

        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (_NOBS, 1)))

        compiled_result = like.jacobian_matrix(weights)
        vec_result = jacobian_matrix_vectorized(
            self._shapes,
            self._obs,
            None,
            self._noise_variances,
            weights,
            self._bkd,
        )

        self._bkd.assert_allclose(compiled_result, vec_result, rtol=1e-12)

    def test_compiled_matches_vectorized_with_latent(self):
        """torch.compile jacobian matches vectorized with latent samples."""
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            self._bkd,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        like.set_latent_samples(self._latent_samples)

        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (_NOBS, 1)))

        compiled_result = like.jacobian_matrix(weights)
        vec_result = jacobian_matrix_vectorized(
            self._shapes,
            self._obs,
            self._latent_samples,
            self._noise_variances,
            weights,
            self._bkd,
        )

        self._bkd.assert_allclose(compiled_result, vec_result, rtol=1e-12)


@slow_test
class TestCompiledKLObjective(unittest.TestCase):
    """Verify compiled KL objective produces correct results."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def _create_objective(self):
        bkd = self._bkd
        noise_variances = bkd.asarray(np.random.uniform(0.05, 0.3, (_NOBS,)))
        inner_loglike = GaussianOEDInnerLoopLikelihood(
            noise_variances,
            bkd,
        )
        inner_shapes = bkd.asarray(np.random.randn(_NOBS, _NINNER))
        outer_shapes = bkd.asarray(np.random.randn(_NOBS, _NOUTER))
        latent_samples = bkd.asarray(np.random.randn(_NOBS, _NOUTER))

        return KLOEDObjective(
            inner_loglike,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,
            None,
            bkd,
        )

    def test_objective_runs(self):
        """torch.compile KL objective evaluates without error."""
        np.random.seed(42)
        obj = self._create_objective()
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (_NOBS, 1)))
        result = obj(weights)
        self.assertEqual(result.shape, (1, 1))

    def test_jacobian_runs(self):
        """torch.compile KL objective jacobian evaluates without error."""
        np.random.seed(42)
        obj = self._create_objective()
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (_NOBS, 1)))
        result = obj.jacobian(weights)
        self.assertEqual(result.shape, (1, _NOBS))


@slow_test
class TestCompiledAutograd(unittest.TestCase):
    """Verify torch.autograd works through the compiled path."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        # torch.compile donated buffers conflict with autograd jacobian
        import torch._functorch.config as _ftconfig

        _ftconfig.donated_buffer = False
        self._bkd = TorchBkd()
        np.random.seed(42)

    def _create_objective(self):
        bkd = self._bkd
        noise_variances = bkd.asarray(np.random.uniform(0.05, 0.3, (_NOBS,)))
        inner_loglike = GaussianOEDInnerLoopLikelihood(
            noise_variances,
            bkd,
        )
        inner_shapes = bkd.asarray(np.random.randn(_NOBS, _NINNER))
        outer_shapes = bkd.asarray(np.random.randn(_NOBS, _NOUTER))
        latent_samples = bkd.asarray(np.random.randn(_NOBS, _NOUTER))

        return KLOEDObjective(
            inner_loglike,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,
            None,
            bkd,
        )

    def test_autograd_jacobian_matches_analytical(self):
        """torch.autograd.functional.jacobian matches analytical through compiled
        path."""
        obj = self._create_objective()
        np.random.seed(123)
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (_NOBS, 1)))

        analytical_jac = obj.jacobian(weights)

        def forward(w):
            return obj(w).flatten()

        w = weights.clone().requires_grad_(True)
        autograd_jac = torch.autograd.functional.jacobian(forward, w)
        autograd_jac = autograd_jac.squeeze(-1)

        self._bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

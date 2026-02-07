"""
Tests verifying torch.compile dispatch correctness.

Each test class uses a single fixed shape (nobs=5, ninner=20, nouter=15)
to ensure only one torch.compile compilation per class, keeping warmup
overhead manageable.

Tests compare compiled results against non-compiled (eager) results to
verify that the torch.compile path produces numerically identical output.
"""

import unittest

import numpy as np
import torch

from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401

from pyapprox.typing.expdesign.likelihood import (
    GaussianOEDInnerLoopLikelihood,
)
from pyapprox.typing.expdesign.evidence import Evidence, LogEvidence
from pyapprox.typing.expdesign.objective import KLOEDObjective


# Fixed dimensions — shared across all test classes to minimize recompilation.
_NOBS = 5
_NINNER = 20
_NOUTER = 15


@slow_test
class TestCompiledLogpdfMatrix(unittest.TestCase):
    """Verify compiled logpdf_matrix matches eager (non-compiled)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

        self._noise_variances = self._bkd.asarray(
            np.random.uniform(0.05, 0.3, (_NOBS,))
        )
        self._shapes = self._bkd.asarray(
            np.random.randn(_NOBS, _NINNER)
        )
        self._obs = self._bkd.asarray(
            np.random.randn(_NOBS, _NOUTER)
        )

    def _create_likelihood(self, use_torch_compile):
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd,
            use_numba=False, use_torch_compile=use_torch_compile,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        return like

    def test_compiled_matches_eager(self):
        """torch.compile logpdf_matrix matches eager computation."""
        eager = self._create_likelihood(use_torch_compile=False)
        compiled = self._create_likelihood(use_torch_compile=True)

        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (_NOBS, 1))
        )

        eager_result = eager.logpdf_matrix(weights)
        compiled_result = compiled.logpdf_matrix(weights)

        self._bkd.assert_allclose(compiled_result, eager_result, rtol=1e-12)


@slow_test
class TestCompiledJacobianMatrix(unittest.TestCase):
    """Verify compiled jacobian_matrix matches eager (non-compiled)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

        self._noise_variances = self._bkd.asarray(
            np.random.uniform(0.05, 0.3, (_NOBS,))
        )
        self._shapes = self._bkd.asarray(
            np.random.randn(_NOBS, _NINNER)
        )
        self._obs = self._bkd.asarray(
            np.random.randn(_NOBS, _NOUTER)
        )
        self._latent_samples = self._bkd.asarray(
            np.random.randn(_NOBS, _NOUTER)
        )

    def _create_likelihood(self, use_torch_compile):
        like = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd,
            use_numba=False, use_torch_compile=use_torch_compile,
        )
        like.set_shapes(self._shapes)
        like.set_observations(self._obs)
        return like

    def test_compiled_matches_eager_no_latent(self):
        """torch.compile jacobian matches eager without latent samples."""
        eager = self._create_likelihood(use_torch_compile=False)
        compiled = self._create_likelihood(use_torch_compile=True)

        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (_NOBS, 1))
        )

        eager_result = eager.jacobian_matrix(weights)
        compiled_result = compiled.jacobian_matrix(weights)

        self._bkd.assert_allclose(compiled_result, eager_result, rtol=1e-12)

    def test_compiled_matches_eager_with_latent(self):
        """torch.compile jacobian matches eager with latent samples."""
        eager = self._create_likelihood(use_torch_compile=False)
        compiled = self._create_likelihood(use_torch_compile=True)

        eager.set_latent_samples(self._latent_samples)
        compiled.set_latent_samples(self._latent_samples)

        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (_NOBS, 1))
        )

        eager_result = eager.jacobian_matrix(weights)
        compiled_result = compiled.jacobian_matrix(weights)

        self._bkd.assert_allclose(compiled_result, eager_result, rtol=1e-12)


@slow_test
class TestCompiledKLObjective(unittest.TestCase):
    """Verify compiled KL objective matches eager (non-compiled)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def _create_objective(self, use_torch_compile):
        bkd = self._bkd
        noise_variances = bkd.asarray(
            np.random.uniform(0.05, 0.3, (_NOBS,))
        )
        inner_loglike = GaussianOEDInnerLoopLikelihood(
            noise_variances, bkd,
            use_numba=False, use_torch_compile=use_torch_compile,
        )
        inner_shapes = bkd.asarray(np.random.randn(_NOBS, _NINNER))
        outer_shapes = bkd.asarray(np.random.randn(_NOBS, _NOUTER))
        latent_samples = bkd.asarray(np.random.randn(_NOBS, _NOUTER))

        return KLOEDObjective(
            inner_loglike, outer_shapes, latent_samples, inner_shapes,
            None, None, bkd,
        )

    def test_compiled_value_matches_eager(self):
        """torch.compile KL objective value matches eager."""
        np.random.seed(42)
        eager = self._create_objective(use_torch_compile=False)
        np.random.seed(42)
        compiled = self._create_objective(use_torch_compile=True)

        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (_NOBS, 1))
        )

        eager_result = eager(weights)
        compiled_result = compiled(weights)

        self._bkd.assert_allclose(compiled_result, eager_result, rtol=1e-10)

    def test_compiled_jacobian_matches_eager(self):
        """torch.compile KL objective jacobian matches eager."""
        np.random.seed(42)
        eager = self._create_objective(use_torch_compile=False)
        np.random.seed(42)
        compiled = self._create_objective(use_torch_compile=True)

        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (_NOBS, 1))
        )

        eager_result = eager.jacobian(weights)
        compiled_result = compiled.jacobian(weights)

        self._bkd.assert_allclose(compiled_result, eager_result, rtol=1e-10)


@slow_test
class TestCompiledAutograd(unittest.TestCase):
    """Verify torch.autograd works through the compiled path."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def _create_objective(self):
        bkd = self._bkd
        noise_variances = bkd.asarray(
            np.random.uniform(0.05, 0.3, (_NOBS,))
        )
        inner_loglike = GaussianOEDInnerLoopLikelihood(
            noise_variances, bkd,
            use_numba=False, use_torch_compile=True,
        )
        inner_shapes = bkd.asarray(np.random.randn(_NOBS, _NINNER))
        outer_shapes = bkd.asarray(np.random.randn(_NOBS, _NOUTER))
        latent_samples = bkd.asarray(np.random.randn(_NOBS, _NOUTER))

        return KLOEDObjective(
            inner_loglike, outer_shapes, latent_samples, inner_shapes,
            None, None, bkd,
        )

    def test_autograd_jacobian_matches_analytical(self):
        """torch.autograd.functional.jacobian matches analytical through compiled path."""
        obj = self._create_objective()
        np.random.seed(123)
        weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (_NOBS, 1))
        )

        analytical_jac = obj.jacobian(weights)

        def forward(w):
            return obj(w).flatten()

        w = weights.clone().requires_grad_(True)
        autograd_jac = torch.autograd.functional.jacobian(forward, w)
        autograd_jac = autograd_jac.squeeze(-1)

        self._bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

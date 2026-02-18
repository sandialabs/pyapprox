"""
Tests for ModelBasedLogLikelihood.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.probability.likelihood import (
    DiagonalGaussianLogLikelihood,
    GaussianLogLikelihood,
    ModelBasedLogLikelihood,
)
from pyapprox.typing.probability.covariance import (
    DenseCholeskyCovarianceOperator,
)
from pyapprox.typing.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


class LinearModel(Generic[Array]):
    """Simple linear model y = A @ x for testing.

    Has analytical jacobian: J = A (constant).
    """

    def __init__(self, A: Array, bkd: Backend[Array]) -> None:
        self._A = A
        self._bkd = bkd
        self._nqoi = A.shape[0]
        self._nvars = A.shape[1]

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        return self._A @ samples

    def jacobian(self, sample: Array) -> Array:
        return self._A


class LinearModelNoJacobian(Generic[Array]):
    """Linear model without jacobian method."""

    def __init__(self, A: Array, bkd: Backend[Array]) -> None:
        self._A = A
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._A.shape[1]

    def nqoi(self) -> int:
        return self._A.shape[0]

    def __call__(self, samples: Array) -> Array:
        return self._A @ samples


class TestModelBasedDiagonalGaussian(Generic[Array], unittest.TestCase):
    """Tests for ModelBasedLogLikelihood with DiagonalGaussianLogLikelihood."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # Linear model: 2 inputs -> 2 outputs
        self._A = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        self._model = LinearModel(self._A, self._bkd)
        self._noise_var = self._bkd.asarray([0.01, 0.02])
        self._noise_lik = DiagonalGaussianLogLikelihood(
            self._noise_var, self._bkd
        )
        self._composed = ModelBasedLogLikelihood(
            self._model, self._noise_lik, self._bkd
        )
        # Set observations
        self._obs = self._bkd.asarray([[1.0], [2.0]])
        self._composed.set_observations(self._obs)

    def test_logpdf_matches_manual(self) -> None:
        """logpdf(p) equals noise_lik.logpdf(model(p))."""
        params = self._bkd.asarray([[0.5], [0.3]])
        model_out = self._model(params)
        expected = self._noise_lik.logpdf(model_out)
        result = self._composed.logpdf(params)
        self._bkd.assert_allclose(result, expected)

    def test_call_is_logpdf(self) -> None:
        """__call__ is an alias for logpdf."""
        params = self._bkd.asarray([[0.5], [0.3]])
        self._bkd.assert_allclose(
            self._composed(params), self._composed.logpdf(params)
        )

    def test_logpdf_shape(self) -> None:
        """logpdf returns shape (1, nsamples)."""
        params = self._bkd.asarray([[0.5, 1.0], [0.3, 0.7]])
        result = self._composed.logpdf(params)
        self.assertEqual(result.shape, (1, 2))

    def test_logpdf_batch(self) -> None:
        """logpdf works with multiple samples."""
        nsamples = 5
        np.random.seed(42)
        params_np = np.random.randn(2, nsamples)
        params = self._bkd.asarray(params_np)
        result = self._composed.logpdf(params)
        self.assertEqual(result.shape, (1, nsamples))
        # Verify each sample individually
        for i in range(nsamples):
            p_i = self._bkd.asarray(params_np[:, i:i+1])
            expected_i = self._composed.logpdf(p_i)
            self._bkd.assert_allclose(
                self._bkd.asarray(result[:, i:i+1]),
                expected_i,
            )

    def test_rvs_shape(self) -> None:
        """rvs returns shape (nobs, nsamples)."""
        params = self._bkd.asarray([[0.5], [0.3]])
        np.random.seed(42)
        samples = self._composed.rvs(params)
        self.assertEqual(samples.shape, (2, 1))

    def test_rvs_multiple_samples(self) -> None:
        """rvs with nsamples > 1."""
        params = self._bkd.asarray([[0.5], [0.3]])
        np.random.seed(42)
        samples = self._composed.rvs(params, nsamples=3)
        self.assertEqual(samples.shape, (2, 3))

    def test_jacobian_derivative_checker(self) -> None:
        """Validate jacobian via DerivativeChecker."""
        # Wrap as FunctionWithJacobianFromCallable for DerivativeChecker
        def logpdf_fn(params: Array) -> Array:
            return self._composed.logpdf(params)

        def jac_fn(sample: Array) -> Array:
            return self._composed.jacobian(sample)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=2,
            fun=logpdf_fn, jacobian=jac_fn,
            bkd=self._bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = self._bkd.asarray([[0.5], [0.3]])
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        self._bkd.assert_allclose(
            self._bkd.asarray([float(ratio)]),
            self._bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_jacobian_matches_manual_chain_rule(self) -> None:
        """Jacobian matches manual chain rule computation."""
        sample = self._bkd.asarray([[0.5], [0.3]])
        model_out = self._model(sample)
        # gradient: (nobs, 1)
        grad = self._noise_lik.gradient(model_out)
        # model jacobian: (nobs, nvars) = A
        J_model = self._A
        # chain rule: grad^T @ J_model = (1, nobs) @ (nobs, nvars) = (1, nvars)
        expected = grad.T @ J_model
        result = self._composed.jacobian(sample)
        self._bkd.assert_allclose(result, expected)

    def test_gradient_shape(self) -> None:
        """gradient returns shape (nvars, 1)."""
        sample = self._bkd.asarray([[0.5], [0.3]])
        result = self._composed.gradient(sample)
        self.assertEqual(result.shape, (2, 1))

    def test_gradient_is_jacobian_transposed(self) -> None:
        """gradient is jacobian transposed."""
        sample = self._bkd.asarray([[0.5], [0.3]])
        self._bkd.assert_allclose(
            self._composed.gradient(sample),
            self._composed.jacobian(sample).T,
        )

    def test_logpdf_vectorized_shape(self) -> None:
        """logpdf_vectorized returns shape (n_params, n_obs)."""
        params = self._bkd.asarray([[0.5, 1.0], [0.3, 0.7]])
        obs = self._bkd.asarray([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])
        result = self._composed.logpdf_vectorized(params, obs)
        self.assertEqual(result.shape, (2, 3))

    def test_logpdf_vectorized_matches_loop(self) -> None:
        """logpdf_vectorized matches looped single evaluations."""
        np.random.seed(42)
        params = self._bkd.asarray(np.random.randn(2, 3))
        obs = self._bkd.asarray(np.random.randn(2, 4) * 0.1 + 1.0)
        result = self._composed.logpdf_vectorized(params, obs)
        self.assertEqual(result.shape, (3, 4))
        # Check against loop
        for i in range(3):
            for j in range(4):
                p_i = self._bkd.asarray(params[:, i:i+1])
                o_j = obs[:, j:j+1]
                self._noise_lik.set_observations(o_j)
                expected_ij = self._noise_lik.logpdf(self._model(p_i))
                self._bkd.assert_allclose(
                    self._bkd.asarray([[float(result[i, j])]]),
                    expected_ij,
                    rtol=1e-12,
                )
        # Restore original observations
        self._noise_lik.set_observations(self._obs)

    def test_set_design_weights_delegates(self) -> None:
        """set_design_weights delegates to noise likelihood."""
        weights = self._bkd.asarray([1.0, 0.5])
        self._composed.set_design_weights(weights)
        # Verify effect: logpdf should differ with weights
        params = self._bkd.asarray([[0.5], [0.3]])
        logpdf_weighted = self._composed.logpdf(params)
        # Reset
        self._noise_lik._design_weights = None
        logpdf_unweighted = self._composed.logpdf(params)
        # They should differ
        self.assertFalse(
            np.allclose(
                float(logpdf_weighted), float(logpdf_unweighted),
                atol=1e-14,
            )
        )

    def test_nvars(self) -> None:
        """nvars returns model's nvars."""
        self.assertEqual(self._composed.nvars(), 2)

    def test_nobs(self) -> None:
        """nobs returns noise likelihood's nobs."""
        self.assertEqual(self._composed.nobs(), 2)

    def test_model_accessor(self) -> None:
        """model() returns the wrapped model."""
        self.assertIs(self._composed.model(), self._model)

    def test_noise_likelihood_accessor(self) -> None:
        """noise_likelihood() returns the wrapped noise likelihood."""
        self.assertIs(self._composed.noise_likelihood(), self._noise_lik)


class TestModelBasedDiagonalGaussianNumpy(
    TestModelBasedDiagonalGaussian[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestModelBasedDiagonalGaussianTorch(
    TestModelBasedDiagonalGaussian[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestModelBasedDenseGaussian(Generic[Array], unittest.TestCase):
    """Tests for ModelBasedLogLikelihood with GaussianLogLikelihood."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        # Linear model: 2 inputs -> 3 outputs
        self._A = self._bkd.asarray(
            [[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]]
        )
        self._model = LinearModel(self._A, self._bkd)
        # Correlated noise covariance
        cov_np = np.array(
            [[0.04, 0.01, 0.005],
             [0.01, 0.03, 0.008],
             [0.005, 0.008, 0.02]]
        )
        noise_cov_op = DenseCholeskyCovarianceOperator(
            self._bkd.asarray(cov_np), self._bkd
        )
        self._noise_lik = GaussianLogLikelihood(noise_cov_op, self._bkd)
        self._composed = ModelBasedLogLikelihood(
            self._model, self._noise_lik, self._bkd
        )
        self._obs = self._bkd.asarray([[1.0], [2.0], [1.5]])
        self._composed.set_observations(self._obs)

    def test_logpdf_matches_manual(self) -> None:
        """logpdf(p) equals noise_lik.logpdf(model(p))."""
        params = self._bkd.asarray([[0.5], [0.3]])
        model_out = self._model(params)
        expected = self._noise_lik.logpdf(model_out)
        result = self._composed.logpdf(params)
        self._bkd.assert_allclose(result, expected)

    def test_logpdf_shape(self) -> None:
        """logpdf returns shape (1, nsamples)."""
        params = self._bkd.asarray([[0.5, 1.0, -0.3], [0.3, 0.7, 0.1]])
        result = self._composed.logpdf(params)
        self.assertEqual(result.shape, (1, 3))

    def test_jacobian_derivative_checker(self) -> None:
        """Validate jacobian via DerivativeChecker."""
        def logpdf_fn(params: Array) -> Array:
            return self._composed.logpdf(params)

        def jac_fn(sample: Array) -> Array:
            return self._composed.jacobian(sample)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=2,
            fun=logpdf_fn, jacobian=jac_fn,
            bkd=self._bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = self._bkd.asarray([[0.5], [0.3]])
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        self._bkd.assert_allclose(
            self._bkd.asarray([float(ratio)]),
            self._bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_jacobian_matches_manual_chain_rule(self) -> None:
        """Jacobian matches manual chain rule computation."""
        sample = self._bkd.asarray([[0.5], [0.3]])
        model_out = self._model(sample)
        grad = self._noise_lik.gradient(model_out)
        J_model = self._A
        expected = grad.T @ J_model
        result = self._composed.jacobian(sample)
        self._bkd.assert_allclose(result, expected)

    def test_rvs_shape(self) -> None:
        """rvs returns shape (nobs, nsamples)."""
        params = self._bkd.asarray([[0.5], [0.3]])
        np.random.seed(42)
        samples = self._composed.rvs(params)
        self.assertEqual(samples.shape, (3, 1))

    def test_logpdf_vectorized_shape(self) -> None:
        """logpdf_vectorized returns shape (n_params, n_obs)."""
        params = self._bkd.asarray([[0.5, 1.0], [0.3, 0.7]])
        obs = self._bkd.asarray(
            [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [1.5, 1.6, 1.7]]
        )
        result = self._composed.logpdf_vectorized(params, obs)
        self.assertEqual(result.shape, (2, 3))

    def test_nobs(self) -> None:
        """nobs matches noise likelihood."""
        self.assertEqual(self._composed.nobs(), 3)


class TestModelBasedDenseGaussianNumpy(
    TestModelBasedDenseGaussian[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestModelBasedDenseGaussianTorch(
    TestModelBasedDenseGaussian[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestModelBasedValidation(Generic[Array], unittest.TestCase):
    """Tests for validation and error handling."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_dimension_mismatch_raises(self) -> None:
        """ValueError when model.nqoi() != noise_likelihood.nobs()."""
        model = FunctionFromCallable(
            nqoi=3, nvars=2,
            fun=lambda x: self._bkd.zeros((3, x.shape[1])),
            bkd=self._bkd,
        )
        noise_var = self._bkd.asarray([0.01, 0.02])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, self._bkd)
        with self.assertRaises(ValueError):
            ModelBasedLogLikelihood(model, noise_lik, self._bkd)

    def test_jacobian_unavailable_without_model_jacobian(self) -> None:
        """No jacobian/gradient on composed object when model lacks jacobian."""
        A = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        model = LinearModelNoJacobian(A, self._bkd)
        noise_var = self._bkd.asarray([0.01, 0.02])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, self._bkd)
        composed = ModelBasedLogLikelihood(model, noise_lik, self._bkd)
        self.assertFalse(hasattr(composed, "jacobian"))
        self.assertFalse(hasattr(composed, "gradient"))
        # logpdf should still work
        composed.set_observations(self._bkd.asarray([[1.0], [2.0]]))
        result = composed.logpdf(self._bkd.asarray([[0.5], [0.3]]))
        self.assertEqual(result.shape, (1, 1))

    def test_rvs_available_with_gaussian(self) -> None:
        """rvs is available when noise likelihood has rvs."""
        A = self._bkd.asarray([[1.0, 2.0]])
        model = LinearModelNoJacobian(A, self._bkd)
        noise_var = self._bkd.asarray([0.01])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, self._bkd)
        composed = ModelBasedLogLikelihood(model, noise_lik, self._bkd)
        self.assertTrue(hasattr(composed, "rvs"))

    def test_logpdf_vectorized_available(self) -> None:
        """logpdf_vectorized is available for DiagonalGaussian."""
        A = self._bkd.asarray([[1.0, 2.0]])
        model = LinearModelNoJacobian(A, self._bkd)
        noise_var = self._bkd.asarray([0.01])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, self._bkd)
        composed = ModelBasedLogLikelihood(model, noise_lik, self._bkd)
        self.assertTrue(hasattr(composed, "logpdf_vectorized"))

    def test_set_design_weights_available(self) -> None:
        """set_design_weights is available for DiagonalGaussian."""
        A = self._bkd.asarray([[1.0, 2.0]])
        model = LinearModelNoJacobian(A, self._bkd)
        noise_var = self._bkd.asarray([0.01])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, self._bkd)
        composed = ModelBasedLogLikelihood(model, noise_lik, self._bkd)
        self.assertTrue(hasattr(composed, "set_design_weights"))


class TestModelBasedValidationNumpy(
    TestModelBasedValidation[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestModelBasedValidationTorch(
    TestModelBasedValidation[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestModelBasedAutograd(unittest.TestCase):
    """Test autograd compatibility (Torch only)."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        A = self._bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        self._model = LinearModel(A, self._bkd)
        noise_var = self._bkd.asarray([0.01, 0.02])
        self._noise_lik = DiagonalGaussianLogLikelihood(
            noise_var, self._bkd
        )
        self._composed = ModelBasedLogLikelihood(
            self._model, self._noise_lik, self._bkd
        )
        obs = self._bkd.asarray([[1.0], [2.0]])
        self._composed.set_observations(obs)

    def test_autograd_jacobian_matches_analytical(self) -> None:
        """torch.autograd.functional.jacobian matches analytical jacobian."""
        sample = torch.tensor([[0.5], [0.3]], dtype=torch.float64)

        def logpdf_for_autograd(params: torch.Tensor) -> torch.Tensor:
            return self._composed.logpdf(params).squeeze()

        autograd_jac = torch.autograd.functional.jacobian(
            logpdf_for_autograd, sample
        )
        # autograd_jac has shape (nvars, 1) from the (nvars, 1) input
        autograd_jac_2d = autograd_jac.reshape(1, -1)

        analytical_jac = self._composed.jacobian(sample)

        self._bkd.assert_allclose(
            analytical_jac, autograd_jac_2d, rtol=1e-10
        )


if __name__ == "__main__":
    unittest.main()

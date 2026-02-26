"""
Tests for ModelBasedLogLikelihood.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.probability.covariance import (
    DenseCholeskyCovarianceOperator,
)
from pyapprox.probability.likelihood import (
    DiagonalGaussianLogLikelihood,
    GaussianLogLikelihood,
    ModelBasedLogLikelihood,
)


class LinearModel:
    """Simple linear model y = A @ x for testing.

    Has analytical jacobian: J = A (constant).
    """

    def __init__(self, A, bkd) -> None:
        self._A = A
        self._bkd = bkd
        self._nqoi = A.shape[0]
        self._nvars = A.shape[1]

    def bkd(self):
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, samples):
        return self._A @ samples

    def jacobian(self, sample):
        return self._A


class LinearModelNoJacobian:
    """Linear model without jacobian method."""

    def __init__(self, A, bkd) -> None:
        self._A = A
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self) -> int:
        return self._A.shape[1]

    def nqoi(self) -> int:
        return self._A.shape[0]

    def __call__(self, samples):
        return self._A @ samples


class TestModelBasedDiagonalGaussian:
    """Tests for ModelBasedLogLikelihood with DiagonalGaussianLogLikelihood."""

    def _setup(self, bkd):
        # Linear model: 2 inputs -> 2 outputs
        A = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        model = LinearModel(A, bkd)
        noise_var = bkd.asarray([0.01, 0.02])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, bkd)
        composed = ModelBasedLogLikelihood(
            model, noise_lik, bkd
        )
        # Set observations
        obs = bkd.asarray([[1.0], [2.0]])
        composed.set_observations(obs)
        return A, model, noise_var, noise_lik, composed, obs

    def test_logpdf_matches_manual(self, bkd) -> None:
        """logpdf(p) equals noise_lik.logpdf(model(p))."""
        _, model, _, noise_lik, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5], [0.3]])
        model_out = model(params)
        expected = noise_lik.logpdf(model_out)
        result = composed.logpdf(params)
        bkd.assert_allclose(result, expected)

    def test_call_is_logpdf(self, bkd) -> None:
        """__call__ is an alias for logpdf."""
        _, _, _, _, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5], [0.3]])
        bkd.assert_allclose(composed(params), composed.logpdf(params))

    def test_logpdf_shape(self, bkd) -> None:
        """logpdf returns shape (1, nsamples)."""
        _, _, _, _, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5, 1.0], [0.3, 0.7]])
        result = composed.logpdf(params)
        assert result.shape == (1, 2)

    def test_logpdf_batch(self, bkd) -> None:
        """logpdf works with multiple samples."""
        _, _, _, _, composed, _ = self._setup(bkd)
        nsamples = 5
        np.random.seed(42)
        params_np = np.random.randn(2, nsamples)
        params = bkd.asarray(params_np)
        result = composed.logpdf(params)
        assert result.shape == (1, nsamples)
        # Verify each sample individually
        for i in range(nsamples):
            p_i = bkd.asarray(params_np[:, i : i + 1])
            expected_i = composed.logpdf(p_i)
            bkd.assert_allclose(
                bkd.asarray(result[:, i : i + 1]),
                expected_i,
            )

    def test_rvs_shape(self, bkd) -> None:
        """rvs returns shape (nobs, nsamples)."""
        _, _, _, _, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5], [0.3]])
        np.random.seed(42)
        samples = composed.rvs(params)
        assert samples.shape == (2, 1)

    def test_rvs_multiple_samples(self, bkd) -> None:
        """rvs with nsamples > 1."""
        _, _, _, _, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5], [0.3]])
        np.random.seed(42)
        samples = composed.rvs(params, nsamples=3)
        assert samples.shape == (2, 3)

    def test_jacobian_derivative_checker(self, bkd) -> None:
        """Validate jacobian via DerivativeChecker."""
        _, _, _, _, composed, _ = self._setup(bkd)

        # Wrap as FunctionWithJacobianFromCallable for DerivativeChecker
        def logpdf_fn(params):
            return composed.logpdf(params)

        def jac_fn(sample):
            return composed.jacobian(sample)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=logpdf_fn,
            jacobian=jac_fn,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray([[0.5], [0.3]])
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        bkd.assert_allclose(
            bkd.asarray([float(ratio)]),
            bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_jacobian_matches_manual_chain_rule(self, bkd) -> None:
        """Jacobian matches manual chain rule computation."""
        A, model, _, noise_lik, composed, _ = self._setup(bkd)
        sample = bkd.asarray([[0.5], [0.3]])
        model_out = model(sample)
        # gradient: (nobs, 1)
        grad = noise_lik.gradient(model_out)
        # model jacobian: (nobs, nvars) = A
        J_model = A
        # chain rule: grad^T @ J_model = (1, nobs) @ (nobs, nvars) = (1, nvars)
        expected = grad.T @ J_model
        result = composed.jacobian(sample)
        bkd.assert_allclose(result, expected)

    def test_gradient_shape(self, bkd) -> None:
        """gradient returns shape (nvars, 1)."""
        _, _, _, _, composed, _ = self._setup(bkd)
        sample = bkd.asarray([[0.5], [0.3]])
        result = composed.gradient(sample)
        assert result.shape == (2, 1)

    def test_gradient_is_jacobian_transposed(self, bkd) -> None:
        """gradient is jacobian transposed."""
        _, _, _, _, composed, _ = self._setup(bkd)
        sample = bkd.asarray([[0.5], [0.3]])
        bkd.assert_allclose(
            composed.gradient(sample),
            composed.jacobian(sample).T,
        )

    def test_logpdf_vectorized_shape(self, bkd) -> None:
        """logpdf_vectorized returns shape (n_params, n_obs)."""
        _, _, _, _, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5, 1.0], [0.3, 0.7]])
        obs = bkd.asarray([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])
        result = composed.logpdf_vectorized(params, obs)
        assert result.shape == (2, 3)

    def test_logpdf_vectorized_matches_loop(self, bkd) -> None:
        """logpdf_vectorized matches looped single evaluations."""
        A, model, _, noise_lik, composed, obs = self._setup(bkd)
        np.random.seed(42)
        params = bkd.asarray(np.random.randn(2, 3))
        obs_test = bkd.asarray(np.random.randn(2, 4) * 0.1 + 1.0)
        result = composed.logpdf_vectorized(params, obs_test)
        assert result.shape == (3, 4)
        # Check against loop
        for i in range(3):
            for j in range(4):
                p_i = bkd.asarray(params[:, i : i + 1])
                o_j = obs_test[:, j : j + 1]
                noise_lik.set_observations(o_j)
                expected_ij = noise_lik.logpdf(model(p_i))
                bkd.assert_allclose(
                    bkd.asarray([[float(result[i, j])]]),
                    expected_ij,
                    rtol=1e-12,
                )
        # Restore original observations
        noise_lik.set_observations(obs)

    def test_set_design_weights_delegates(self, bkd) -> None:
        """set_design_weights delegates to noise likelihood."""
        _, _, _, noise_lik, composed, _ = self._setup(bkd)
        weights = bkd.asarray([1.0, 0.5])
        composed.set_design_weights(weights)
        # Verify effect: logpdf should differ with weights
        params = bkd.asarray([[0.5], [0.3]])
        logpdf_weighted = composed.logpdf(params)
        # Reset
        noise_lik._design_weights = None
        logpdf_unweighted = composed.logpdf(params)
        # They should differ
        assert not np.allclose(
            float(logpdf_weighted),
            float(logpdf_unweighted),
            atol=1e-14,
        )

    def test_nvars(self, bkd) -> None:
        """nvars returns model's nvars."""
        _, _, _, _, composed, _ = self._setup(bkd)
        assert composed.nvars() == 2

    def test_nobs(self, bkd) -> None:
        """nobs returns noise likelihood's nobs."""
        _, _, _, _, composed, _ = self._setup(bkd)
        assert composed.nobs() == 2

    def test_model_accessor(self, bkd) -> None:
        """model() returns the wrapped model."""
        _, model, _, _, composed, _ = self._setup(bkd)
        assert composed.model() is model

    def test_noise_likelihood_accessor(self, bkd) -> None:
        """noise_likelihood() returns the wrapped noise likelihood."""
        _, _, _, noise_lik, composed, _ = self._setup(bkd)
        assert composed.noise_likelihood() is noise_lik


class TestModelBasedDenseGaussian:
    """Tests for ModelBasedLogLikelihood with GaussianLogLikelihood."""

    def _setup(self, bkd):
        # Linear model: 2 inputs -> 3 outputs
        A = bkd.asarray([[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]])
        model = LinearModel(A, bkd)
        # Correlated noise covariance
        cov_np = np.array(
            [[0.04, 0.01, 0.005], [0.01, 0.03, 0.008], [0.005, 0.008, 0.02]]
        )
        noise_cov_op = DenseCholeskyCovarianceOperator(
            bkd.asarray(cov_np), bkd
        )
        noise_lik = GaussianLogLikelihood(noise_cov_op, bkd)
        composed = ModelBasedLogLikelihood(
            model, noise_lik, bkd
        )
        obs = bkd.asarray([[1.0], [2.0], [1.5]])
        composed.set_observations(obs)
        return A, model, noise_lik, composed, obs

    def test_logpdf_matches_manual(self, bkd) -> None:
        """logpdf(p) equals noise_lik.logpdf(model(p))."""
        _, model, noise_lik, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5], [0.3]])
        model_out = model(params)
        expected = noise_lik.logpdf(model_out)
        result = composed.logpdf(params)
        bkd.assert_allclose(result, expected)

    def test_logpdf_shape(self, bkd) -> None:
        """logpdf returns shape (1, nsamples)."""
        _, _, _, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5, 1.0, -0.3], [0.3, 0.7, 0.1]])
        result = composed.logpdf(params)
        assert result.shape == (1, 3)

    def test_jacobian_derivative_checker(self, bkd) -> None:
        """Validate jacobian via DerivativeChecker."""
        _, _, _, composed, _ = self._setup(bkd)

        def logpdf_fn(params):
            return composed.logpdf(params)

        def jac_fn(sample):
            return composed.jacobian(sample)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=logpdf_fn,
            jacobian=jac_fn,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray([[0.5], [0.3]])
        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        bkd.assert_allclose(
            bkd.asarray([float(ratio)]),
            bkd.asarray([0.0]),
            atol=1e-6,
        )

    def test_jacobian_matches_manual_chain_rule(self, bkd) -> None:
        """Jacobian matches manual chain rule computation."""
        A, model, noise_lik, composed, _ = self._setup(bkd)
        sample = bkd.asarray([[0.5], [0.3]])
        model_out = model(sample)
        grad = noise_lik.gradient(model_out)
        J_model = A
        expected = grad.T @ J_model
        result = composed.jacobian(sample)
        bkd.assert_allclose(result, expected)

    def test_rvs_shape(self, bkd) -> None:
        """rvs returns shape (nobs, nsamples)."""
        _, _, _, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5], [0.3]])
        np.random.seed(42)
        samples = composed.rvs(params)
        assert samples.shape == (3, 1)

    def test_logpdf_vectorized_shape(self, bkd) -> None:
        """logpdf_vectorized returns shape (n_params, n_obs)."""
        _, _, _, composed, _ = self._setup(bkd)
        params = bkd.asarray([[0.5, 1.0], [0.3, 0.7]])
        obs = bkd.asarray([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [1.5, 1.6, 1.7]])
        result = composed.logpdf_vectorized(params, obs)
        assert result.shape == (2, 3)

    def test_nobs(self, bkd) -> None:
        """nobs matches noise likelihood."""
        _, _, _, composed, _ = self._setup(bkd)
        assert composed.nobs() == 3


class TestModelBasedValidation:
    """Tests for validation and error handling."""

    def test_dimension_mismatch_raises(self, bkd) -> None:
        """ValueError when model.nqoi() != noise_likelihood.nobs()."""
        model = FunctionFromCallable(
            nqoi=3,
            nvars=2,
            fun=lambda x: bkd.zeros((3, x.shape[1])),
            bkd=bkd,
        )
        noise_var = bkd.asarray([0.01, 0.02])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, bkd)
        with pytest.raises(ValueError):
            ModelBasedLogLikelihood(model, noise_lik, bkd)

    def test_jacobian_unavailable_without_model_jacobian(self, bkd) -> None:
        """No jacobian/gradient on composed object when model lacks jacobian."""
        A = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        model = LinearModelNoJacobian(A, bkd)
        noise_var = bkd.asarray([0.01, 0.02])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, bkd)
        composed = ModelBasedLogLikelihood(model, noise_lik, bkd)
        assert not hasattr(composed, "jacobian")
        assert not hasattr(composed, "gradient")
        # logpdf should still work
        composed.set_observations(bkd.asarray([[1.0], [2.0]]))
        result = composed.logpdf(bkd.asarray([[0.5], [0.3]]))
        assert result.shape == (1, 1)

    def test_rvs_available_with_gaussian(self, bkd) -> None:
        """rvs is available when noise likelihood has rvs."""
        A = bkd.asarray([[1.0, 2.0]])
        model = LinearModelNoJacobian(A, bkd)
        noise_var = bkd.asarray([0.01])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, bkd)
        composed = ModelBasedLogLikelihood(model, noise_lik, bkd)
        assert hasattr(composed, "rvs")

    def test_logpdf_vectorized_available(self, bkd) -> None:
        """logpdf_vectorized is available for DiagonalGaussian."""
        A = bkd.asarray([[1.0, 2.0]])
        model = LinearModelNoJacobian(A, bkd)
        noise_var = bkd.asarray([0.01])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, bkd)
        composed = ModelBasedLogLikelihood(model, noise_lik, bkd)
        assert hasattr(composed, "logpdf_vectorized")

    def test_set_design_weights_available(self, bkd) -> None:
        """set_design_weights is available for DiagonalGaussian."""
        A = bkd.asarray([[1.0, 2.0]])
        model = LinearModelNoJacobian(A, bkd)
        noise_var = bkd.asarray([0.01])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, bkd)
        composed = ModelBasedLogLikelihood(model, noise_lik, bkd)
        assert hasattr(composed, "set_design_weights")


class TestModelBasedAutograd:
    """Test autograd compatibility (Torch only)."""

    def test_autograd_jacobian_matches_analytical(self, bkd) -> None:
        """torch.autograd.functional.jacobian matches analytical jacobian."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        if bkd.__class__.__name__ != "TorchBkd":
            pytest.skip("torch-only test")

        A = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        model = LinearModel(A, bkd)
        noise_var = bkd.asarray([0.01, 0.02])
        noise_lik = DiagonalGaussianLogLikelihood(noise_var, bkd)
        composed = ModelBasedLogLikelihood(model, noise_lik, bkd)
        obs = bkd.asarray([[1.0], [2.0]])
        composed.set_observations(obs)

        sample = torch.tensor([[0.5], [0.3]], dtype=torch.float64)

        def logpdf_for_autograd(params):
            return composed.logpdf(params).squeeze()

        autograd_jac = torch.autograd.functional.jacobian(logpdf_for_autograd, sample)
        # autograd_jac has shape (nvars, 1) from the (nvars, 1) input
        autograd_jac_2d = autograd_jac.reshape(1, -1)

        analytical_jac = composed.jacobian(sample)

        bkd.assert_allclose(analytical_jac, autograd_jac_2d, rtol=1e-10)

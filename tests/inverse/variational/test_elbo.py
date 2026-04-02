"""
Tests for ELBOObjective and make_single_problem_elbo.

Uses ConditionalGaussian with degree-0 BasisExpansion as the variational
distribution (constant parameters = single-problem VI).
"""

import numpy as np
import pytest

from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.inverse.variational.elbo import (
    ELBOObjective,
    make_single_problem_elbo,
)
from pyapprox.probability.conditional.gaussian import ConditionalGaussian
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.protocols import Backend


def _make_degree0_expansion(bkd: Backend, coeff: float = 0.0) -> BasisExpansion:
    """Create a degree-0 BasisExpansion (constant function)."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(1, 0, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=1)
    exp.set_coefficients(bkd.asarray([[coeff]]))
    return exp


class TestELBOBase:
    """Base test class for ELBOObjective."""

    def _make_cond_gaussian(
        self, bkd, mean: float = 0.0, log_stdev: float = 0.0
    ) -> ConditionalGaussian:
        """Create a ConditionalGaussian with constant mean and log_stdev."""
        mean_func = _make_degree0_expansion(bkd, mean)
        log_stdev_func = _make_degree0_expansion(bkd, log_stdev)
        return ConditionalGaussian(mean_func, log_stdev_func, bkd)

    def _make_simple_elbo(self, bkd) -> ELBOObjective:
        """Create a simple ELBO for testing."""
        cond = self._make_cond_gaussian(bkd)
        prior = GaussianMarginal(0.0, 1.0, bkd)

        def log_likelihood_fn(z):
            obs = bkd.ones((1, 1))
            diff = z - obs
            return -0.5 * diff**2

        np.random.seed(42)
        nsamples = 50
        base_samples = bkd.asarray(np.random.normal(0, 1, (1, nsamples)))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        return make_single_problem_elbo(
            cond,
            log_likelihood_fn,
            prior,
            base_samples,
            weights,
            bkd,
        )

    def test_elbo_returns_correct_shape(self, bkd) -> None:
        elbo = self._make_simple_elbo(bkd)
        params = bkd.zeros((elbo.nvars(), 1))
        result = elbo(params)
        assert result.shape == (1, 1)

    def test_elbo_satisfies_function_protocol(self, bkd) -> None:
        elbo = self._make_simple_elbo(bkd)
        assert isinstance(elbo, FunctionProtocol)

    def test_elbo_nqoi(self, bkd) -> None:
        elbo = self._make_simple_elbo(bkd)
        assert elbo.nqoi() == 1

    def test_elbo_nvars(self, bkd) -> None:
        elbo = self._make_simple_elbo(bkd)
        # degree-0 expansion: 1 coeff for mean + 1 coeff for log_stdev = 2
        assert elbo.nvars() == 2

    def test_elbo_deterministic(self, bkd) -> None:
        """Calling ELBO twice with same params gives same result."""
        elbo = self._make_simple_elbo(bkd)
        params = bkd.zeros((elbo.nvars(), 1))
        v1 = elbo(params)
        v2 = elbo(params)
        bkd.assert_allclose(v1, v2, rtol=1e-12)


class TestELBONumpy(TestELBOBase):
    """NumPy-specific ELBO tests."""

    def test_elbo_no_jacobian_numpy(self, numpy_bkd) -> None:
        elbo = self._make_simple_elbo(numpy_bkd)
        assert not hasattr(elbo, "jacobian")


class TestELBOTorch:
    """Torch-specific ELBO tests."""

    def _make_cond_gaussian(
        self, bkd, mean: float = 0.0, log_stdev: float = 0.0
    ) -> ConditionalGaussian:
        mean_func = _make_degree0_expansion(bkd, mean)
        log_stdev_func = _make_degree0_expansion(bkd, log_stdev)
        return ConditionalGaussian(mean_func, log_stdev_func, bkd)

    def _make_simple_elbo(self, bkd) -> ELBOObjective:
        cond = self._make_cond_gaussian(bkd)
        prior = GaussianMarginal(0.0, 1.0, bkd)

        def log_likelihood_fn(z):
            obs = bkd.ones((1, 1))
            diff = z - obs
            return -0.5 * diff**2

        np.random.seed(42)
        nsamples = 50
        base_samples = bkd.asarray(np.random.normal(0, 1, (1, nsamples)))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        return make_single_problem_elbo(
            cond,
            log_likelihood_fn,
            prior,
            base_samples,
            weights,
            bkd,
        )

    @pytest.fixture
    def torch_bkd(self):
        import torch

        from pyapprox.util.backends.torch import TorchBkd
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

    def test_elbo_jacobian_shape(self, torch_bkd) -> None:
        bkd = torch_bkd
        elbo = self._make_simple_elbo(bkd)
        assert hasattr(elbo, "jacobian")
        params = bkd.zeros((elbo.nvars(), 1))
        jac = elbo.jacobian(params)
        assert jac.shape == (1, elbo.nvars())

    def test_elbo_gradient_derivative_checker(self, torch_bkd) -> None:
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = torch_bkd
        elbo = self._make_simple_elbo(bkd)
        checker = DerivativeChecker(elbo)
        sample = bkd.zeros((elbo.nvars(), 1))
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = checker.error_ratio(errors[0])
        assert float(bkd.flatten(ratio)[0]) <= 1e-5


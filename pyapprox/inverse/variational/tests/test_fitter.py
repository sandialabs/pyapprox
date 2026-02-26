"""Tests for VariationalFitter and VIFitResult."""

import numpy as np
import pytest

from pyapprox.inverse.variational.elbo import make_single_problem_elbo
from pyapprox.inverse.variational.fitter import (
    VariationalFitter,
    VIFitResult,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.probability.conditional.gaussian import (
    ConditionalGaussian,
)
from pyapprox.probability.conditional.joint import (
    ConditionalIndependentJoint,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.likelihood.gaussian import (
    DiagonalGaussianLogLikelihood,
    MultiExperimentLogLikelihood,
)
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions.base import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.test_utils import slow_test


def _make_degree0_expansion(bkd, coeff: float = 0.0):
    """Create a degree-0 BasisExpansion (constant function)."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(1, 0, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=1)
    exp.set_coefficients(bkd.asarray([[coeff]]))
    return exp


def _make_simple_elbo(bkd):
    """Create a minimal ELBO for testing (1D Gaussian, single problem)."""
    mean_func = _make_degree0_expansion(bkd)
    log_stdev_func = _make_degree0_expansion(bkd)
    cond = ConditionalGaussian(mean_func, log_stdev_func, bkd)
    var_dist = ConditionalIndependentJoint([cond], bkd)
    prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

    noise_variances = bkd.full((1,), 0.5)
    base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
    observations = bkd.asarray([[2.0]])
    multi_lik = MultiExperimentLogLikelihood(base_lik, observations, bkd)

    def log_likelihood_fn(z):
        return multi_lik.logpdf(z)

    np.random.seed(42)
    base_samples = bkd.asarray(np.random.normal(0, 1, (1, 200)))
    weights = bkd.full((1, 200), 1.0 / 200)

    elbo = make_single_problem_elbo(
        var_dist,
        log_likelihood_fn,
        prior,
        base_samples,
        weights,
        bkd,
    )
    return elbo, var_dist


class TestVariationalFitterBase:

    def test_fit_returns_vi_result(self, bkd) -> None:
        """fit() returns VIFitResult with correct types and shapes."""
        elbo, _ = _make_simple_elbo(bkd)
        fitter = VariationalFitter(bkd)
        result = fitter.fit(elbo)

        assert isinstance(result, VIFitResult)
        assert isinstance(result.neg_elbo(), float)
        assert result.initial_params().shape == (elbo.nvars(), 1)
        assert result.optimized_params().shape == (elbo.nvars(), 1)
        assert result.optimization_result() is not None

    def test_fit_default_init_guess_zeros(self, bkd) -> None:
        """When init_guess=None, zeros are used."""
        elbo, _ = _make_simple_elbo(bkd)
        fitter = VariationalFitter(bkd)
        result = fitter.fit(elbo)

        expected = bkd.zeros((elbo.nvars(), 1))
        bkd.assert_allclose(result.initial_params(), expected, rtol=1e-12)

    @pytest.mark.slow_on("TorchBkd")
    def test_fit_custom_init_guess(self, bkd) -> None:
        """Explicit init_guess is recorded in result."""
        elbo, _ = _make_simple_elbo(bkd)
        fitter = VariationalFitter(bkd)
        init = bkd.full((elbo.nvars(), 1), 0.1)
        result = fitter.fit(elbo, init_guess=init)

        bkd.assert_allclose(result.initial_params(), init, rtol=1e-12)

    def test_fit_custom_optimizer(self, bkd) -> None:
        """Pass a custom optimizer, verify fit() works."""
        elbo, _ = _make_simple_elbo(bkd)
        opt = ScipyTrustConstrOptimizer(maxiter=100, gtol=1e-6)
        fitter = VariationalFitter(bkd, optimizer=opt)
        result = fitter.fit(elbo)

        assert isinstance(result, VIFitResult)

    def test_bounds_method_on_elbo(self, bkd) -> None:
        """elbo.bounds() returns correct shape matching hyp_list bounds."""
        elbo, var_dist = _make_simple_elbo(bkd)
        bounds = elbo.bounds()
        expected = var_dist.hyp_list().get_active_bounds()

        assert bounds.shape == (elbo.nvars(), 2)
        bkd.assert_allclose(bounds, expected, rtol=1e-12)

    @slow_test
    def test_fit_pushes_params(self, bkd) -> None:
        """After fit, var_dist params match result.optimized_params()."""
        elbo, var_dist = _make_simple_elbo(bkd)
        fitter = VariationalFitter(bkd)
        result = fitter.fit(elbo)

        active_vals = var_dist.hyp_list().get_active_values()
        optimized = result.optimized_params()[:, 0]
        bkd.assert_allclose(active_vals, optimized, rtol=1e-12)

    @slow_test
    def test_fit_respects_bounds(self, bkd) -> None:
        """Optimized params stay within hyp_list bounds."""
        elbo, var_dist = _make_simple_elbo(bkd)

        # Set tight bounds on the first hyperparameter
        hp = var_dist.hyp_list().hyperparameters()[0]
        lower = bkd.full((hp.nparams(),), -2.0)
        upper = bkd.full((hp.nparams(),), 2.0)
        new_bounds = bkd.stack([lower, upper], axis=1)
        hp.set_bounds(new_bounds)

        fitter = VariationalFitter(bkd)
        result = fitter.fit(elbo)

        bounds = elbo.bounds()
        optimized = result.optimized_params()[:, 0]
        for i in range(elbo.nvars()):
            val = float(optimized[i])
            lo = float(bounds[i, 0])
            hi = float(bounds[i, 1])
            assert val >= lo - 1e-10
            assert val <= hi + 1e-10

    def test_repr(self, bkd) -> None:
        """VIFitResult.__repr__ produces a string."""
        elbo, _ = _make_simple_elbo(bkd)
        fitter = VariationalFitter(bkd)
        result = fitter.fit(elbo)

        r = repr(result)
        assert "VIFitResult" in r
        assert "neg_elbo=" in r

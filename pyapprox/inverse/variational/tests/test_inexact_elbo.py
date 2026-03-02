"""Tests for InexactELBOObjective and inexact ELBO factory functions."""


import numpy as np
import pytest

from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.inverse.variational.inexact_elbo import (
    make_inexact_discrete_group_elbo,
    make_inexact_single_problem_elbo,
)
from pyapprox.optimization.minimize.inexact.fixed import (
    FixedSampleStrategy,
)
from pyapprox.optimization.minimize.inexact.monte_carlo import (
    MonteCarloSAAStrategy,
)
from pyapprox.optimization.minimize.inexact.protocols import (
    InexactDifferentiable,
    InexactEvaluable,
)
from pyapprox.probability.conditional.gaussian import ConditionalGaussian
from pyapprox.probability.conditional.joint import (
    ConditionalIndependentJoint,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.test_utils import slow_test


def _make_degree0_expansion(bkd: Backend, coeff: float = 0.0) -> BasisExpansion:
    """Create a degree-0 BasisExpansion (constant function)."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(1, 0, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=1)
    exp.set_coefficients(bkd.asarray([[coeff]]))
    return exp


def _make_cond_gaussian(
    bkd: Backend, mean: float = 0.0, log_stdev: float = 0.0,
) -> ConditionalGaussian:
    """Create a ConditionalGaussian with constant mean and log_stdev."""
    mean_func = _make_degree0_expansion(bkd, mean)
    log_stdev_func = _make_degree0_expansion(bkd, log_stdev)
    return ConditionalGaussian(mean_func, log_stdev_func, bkd)


def _extract_gaussian_params(cond: ConditionalGaussian, bkd: Backend) -> tuple:
    """Extract (mean, stdev) from a fitted ConditionalGaussian."""
    dummy_x = bkd.zeros((cond.nvars(), 1))
    mean = cond._mean_func(dummy_x)[0, 0]
    log_stdev = cond._log_stdev_func(dummy_x)[0, 0]
    return mean, bkd.exp(log_stdev)


def _make_fixed_strategy(bkd, nsamples=50, seed=42):
    """Create a FixedSampleStrategy with N(0,1) samples."""
    np.random.seed(seed)
    samples = bkd.asarray(np.random.normal(0, 1, (1, nsamples)))
    weights = bkd.full((nsamples,), 1.0 / nsamples)
    return FixedSampleStrategy(samples, weights, bkd)


def _make_mc_strategy(bkd, n_max=1000, seed=42):
    """Create a MonteCarloSAAStrategy with N(0,1) base samples."""
    np.random.seed(seed)
    base = bkd.asarray(np.random.normal(0, 1, (1, n_max)))
    return MonteCarloSAAStrategy(base, bkd, scale_factor=1.0)


def _make_simple_inexact_elbo(bkd, strategy=None):
    """Create a simple InexactELBOObjective for testing."""
    cond = _make_cond_gaussian(bkd)
    prior = GaussianMarginal(0.0, 1.0, bkd)

    def log_likelihood_fn(z):
        obs = bkd.ones((1, 1))
        diff = z - obs
        return -0.5 * diff**2

    if strategy is None:
        strategy = _make_fixed_strategy(bkd)

    return make_inexact_single_problem_elbo(
        cond,
        log_likelihood_fn,
        prior,
        strategy,
        bkd,
    )


class TestInexactELBOBasic:
    """Basic tests for InexactELBOObjective."""

    def test_returns_correct_shape(self, bkd) -> None:
        elbo = _make_simple_inexact_elbo(bkd)
        params = bkd.zeros((elbo.nvars(), 1))
        result = elbo(params)
        assert result.shape == (1, 1)

    def test_inexact_value_returns_correct_shape(self, bkd) -> None:
        elbo = _make_simple_inexact_elbo(bkd)
        params = bkd.zeros((elbo.nvars(), 1))
        result = elbo.inexact_value(params, 0.5)
        assert result.shape == (1, 1)

    def test_satisfies_function_protocol(self, bkd) -> None:
        elbo = _make_simple_inexact_elbo(bkd)
        assert isinstance(elbo, FunctionProtocol)

    def test_satisfies_inexact_evaluable(self, bkd) -> None:
        elbo = _make_simple_inexact_elbo(bkd)
        assert isinstance(elbo, InexactEvaluable)

    def test_nqoi(self, bkd) -> None:
        elbo = _make_simple_inexact_elbo(bkd)
        assert elbo.nqoi() == 1

    def test_nvars(self, bkd) -> None:
        elbo = _make_simple_inexact_elbo(bkd)
        # degree-0 expansion: 1 coeff for mean + 1 coeff for log_stdev = 2
        assert elbo.nvars() == 2

    def test_deterministic(self, bkd) -> None:
        """Calling ELBO twice with same params gives same result."""
        elbo = _make_simple_inexact_elbo(bkd)
        params = bkd.zeros((elbo.nvars(), 1))
        v1 = elbo(params)
        v2 = elbo(params)
        bkd.assert_allclose(v1, v2, rtol=1e-12)

    def test_bounds_shape(self, bkd) -> None:
        elbo = _make_simple_inexact_elbo(bkd)
        bounds = elbo.bounds()
        assert bounds.shape == (elbo.nvars(), 2)


class TestInexactELBONeutrality:
    """InexactELBO + FixedStrategy matches standard ELBOObjective."""

    def test_matches_standard_elbo(self, bkd) -> None:
        """InexactELBOObjective with FixedSampleStrategy matches ELBOObjective."""
        from pyapprox.inverse.variational.elbo import make_single_problem_elbo

        np.random.seed(42)
        nsamples = 50
        base_samples = bkd.asarray(np.random.normal(0, 1, (1, nsamples)))
        base_weights_1d = bkd.full((nsamples,), 1.0 / nsamples)
        base_weights_2d = bkd.full((1, nsamples), 1.0 / nsamples)

        prior = GaussianMarginal(0.0, 1.0, bkd)

        def log_likelihood_fn(z):
            obs = bkd.ones((1, 1))
            diff = z - obs
            return -0.5 * diff**2

        # Standard ELBO
        cond_std = _make_cond_gaussian(bkd)
        elbo_std = make_single_problem_elbo(
            cond_std, log_likelihood_fn, prior,
            base_samples, base_weights_2d, bkd,
        )

        # Inexact ELBO with FixedSampleStrategy
        cond_inex = _make_cond_gaussian(bkd)
        strategy = FixedSampleStrategy(base_samples, base_weights_1d, bkd)
        elbo_inex = make_inexact_single_problem_elbo(
            cond_inex, log_likelihood_fn, prior, strategy, bkd,
        )

        params = bkd.zeros((elbo_std.nvars(), 1))
        bkd.assert_allclose(elbo_std(params), elbo_inex(params), rtol=1e-12)

    def test_call_equals_inexact_value_tol_zero(self, bkd) -> None:
        """__call__ and inexact_value(tol=0) return the same result."""
        elbo = _make_simple_inexact_elbo(bkd)
        params = bkd.zeros((elbo.nvars(), 1))
        bkd.assert_allclose(
            elbo(params),
            elbo.inexact_value(params, 0.0),
            rtol=1e-12,
        )


class TestInexactELBOWithMC:
    """Tests with MonteCarloSAAStrategy for tolerance-dependent behavior."""

    def test_inexact_value_different_tols(self, bkd) -> None:
        """inexact_value at different tols does not error."""
        strategy = _make_mc_strategy(bkd, n_max=1000)
        elbo = _make_simple_inexact_elbo(bkd, strategy=strategy)
        params = bkd.zeros((elbo.nvars(), 1))
        v1 = elbo.inexact_value(params, 1.0)
        v2 = elbo.inexact_value(params, 0.01)
        assert v1.shape == (1, 1)
        assert v2.shape == (1, 1)

    def test_convergence_as_tol_shrinks(self, bkd) -> None:
        """Inexact ELBO approaches exact (tol=0) as tol decreases."""
        strategy = _make_mc_strategy(bkd, n_max=5000)
        elbo = _make_simple_inexact_elbo(bkd, strategy=strategy)
        params = bkd.zeros((elbo.nvars(), 1))

        exact = elbo.inexact_value(params, 0.0)
        coarse = elbo.inexact_value(params, 1.0)
        fine = elbo.inexact_value(params, 0.01)

        coarse_err = float(
            bkd.to_numpy(bkd.abs(coarse - exact))[0, 0]
        )
        fine_err = float(
            bkd.to_numpy(bkd.abs(fine - exact))[0, 0]
        )
        # Fine should generally be closer to exact
        assert fine_err <= coarse_err + 1e-10


class TestInexactELBOConjugateRecovery:
    """Test that inexact ELBO recovers correct posterior on conjugate problem."""

    @slow_test
    def test_gaussian_1d_conjugate_fixed_strategy(self, bkd) -> None:
        """Inexact ELBO + FixedSampleStrategy recovers Gaussian conjugate."""
        from pyapprox.inverse.conjugate.gaussian import (
            DenseGaussianConjugatePosterior,
        )
        from pyapprox.inverse.variational.fitter import VariationalFitter
        from pyapprox.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        from pyapprox.probability.likelihood.gaussian import (
            DiagonalGaussianLogLikelihood,
            MultiExperimentLogLikelihood,
        )

        obs_matrix = bkd.asarray([[1.0]])
        observations = bkd.asarray([[2.0]])
        noise_var = 0.5
        nsamples = 1000

        # Exact conjugate posterior
        prior_mean = bkd.reshape(bkd.asarray([0.0]), (1, 1))
        prior_cov = bkd.asarray([[1.0]])
        noise_cov = bkd.asarray([[noise_var]])
        conjugate = DenseGaussianConjugatePosterior(
            obs_matrix, prior_mean, prior_cov, noise_cov, bkd,
        )
        conjugate.compute(observations)
        exact_mean = bkd.flatten(conjugate.posterior_mean())
        exact_var = bkd.diag(conjugate.posterior_covariance())

        # VI setup
        var_dist = ConditionalIndependentJoint(
            [_make_cond_gaussian(bkd)], bkd,
        )
        prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        noise_variances = bkd.full((1,), noise_var)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        multi_lik = MultiExperimentLogLikelihood(base_lik, observations, bkd)

        def log_likelihood_fn(z):
            return multi_lik.logpdf(obs_matrix @ z)

        np.random.seed(42)
        base_samples = bkd.asarray(np.random.normal(0, 1, (1, nsamples)))
        base_weights = bkd.full((nsamples,), 1.0 / nsamples)
        strategy = FixedSampleStrategy(base_samples, base_weights, bkd)

        elbo = make_inexact_single_problem_elbo(
            var_dist, log_likelihood_fn, prior, strategy, bkd,
        )

        optimizer = ScipyTrustConstrOptimizer(maxiter=300, gtol=1e-8)
        fitter = VariationalFitter(bkd, optimizer=optimizer)
        fitter.fit(elbo)

        # Extract recovered params
        cond = var_dist._conditionals[0]
        vi_mean, vi_stdev = _extract_gaussian_params(cond, bkd)

        bkd.assert_allclose(
            bkd.asarray([vi_mean]), exact_mean, atol=0.15,
        )
        bkd.assert_allclose(
            bkd.asarray([vi_stdev ** 2]), exact_var, rtol=0.3,
        )

    @slow_test
    def test_gaussian_1d_conjugate_mc_strategy(self, bkd) -> None:
        """Inexact ELBO + MonteCarloSAAStrategy recovers Gaussian conjugate."""
        from pyapprox.inverse.conjugate.gaussian import (
            DenseGaussianConjugatePosterior,
        )
        from pyapprox.inverse.variational.fitter import VariationalFitter
        from pyapprox.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        from pyapprox.probability.likelihood.gaussian import (
            DiagonalGaussianLogLikelihood,
            MultiExperimentLogLikelihood,
        )

        obs_matrix = bkd.asarray([[1.0]])
        observations = bkd.asarray([[2.0]])
        noise_var = 0.5

        # Exact conjugate posterior
        prior_mean = bkd.reshape(bkd.asarray([0.0]), (1, 1))
        prior_cov = bkd.asarray([[1.0]])
        noise_cov = bkd.asarray([[noise_var]])
        conjugate = DenseGaussianConjugatePosterior(
            obs_matrix, prior_mean, prior_cov, noise_cov, bkd,
        )
        conjugate.compute(observations)
        exact_mean = bkd.flatten(conjugate.posterior_mean())
        exact_var = bkd.diag(conjugate.posterior_covariance())

        # VI setup
        var_dist = ConditionalIndependentJoint(
            [_make_cond_gaussian(bkd)], bkd,
        )
        prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        noise_variances = bkd.full((1,), noise_var)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        multi_lik = MultiExperimentLogLikelihood(base_lik, observations, bkd)

        def log_likelihood_fn(z):
            return multi_lik.logpdf(obs_matrix @ z)

        np.random.seed(42)
        base_samples = bkd.asarray(np.random.normal(0, 1, (1, 2000)))
        strategy = MonteCarloSAAStrategy(base_samples, bkd, scale_factor=1.0)

        elbo = make_inexact_single_problem_elbo(
            var_dist, log_likelihood_fn, prior, strategy, bkd,
        )

        optimizer = ScipyTrustConstrOptimizer(maxiter=300, gtol=1e-8)
        fitter = VariationalFitter(bkd, optimizer=optimizer)
        fitter.fit(elbo)

        # Extract recovered params
        cond = var_dist._conditionals[0]
        vi_mean, vi_stdev = _extract_gaussian_params(cond, bkd)

        bkd.assert_allclose(
            bkd.asarray([vi_mean]), exact_mean, atol=0.2,
        )
        bkd.assert_allclose(
            bkd.asarray([vi_stdev ** 2]), exact_var, rtol=0.4,
        )


    @slow_test
    def test_gaussian_2d_conjugate_sparse_grid_strategy(self, bkd) -> None:
        """2D Gaussian conjugate recovery via sparse grid QuadratureStrategy."""
        from pyapprox.inverse.conjugate.gaussian import (
            DenseGaussianConjugatePosterior,
        )
        from pyapprox.inverse.variational.fitter import VariationalFitter
        from pyapprox.optimization.minimize.inexact.quadrature import (
            QuadratureStrategy,
        )
        from pyapprox.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )
        from pyapprox.probability.likelihood.gaussian import (
            DiagonalGaussianLogLikelihood,
            MultiExperimentLogLikelihood,
        )
        from pyapprox.surrogates.affine.indices import LinearGrowthRule
        from pyapprox.surrogates.sparsegrids import (
            GaussLagrangeFactory,
            ParameterizedIsotropicSparseGridQuadratureRule,
            TensorProductSubspaceFactory,
        )

        nlatent = 2
        obs_matrix = bkd.eye(nlatent)
        observations = bkd.asarray([[2.0], [1.0]])
        noise_var = 0.5

        # Exact conjugate posterior
        prior_mean = bkd.zeros((nlatent, 1))
        prior_cov = bkd.eye(nlatent)
        noise_cov = noise_var * bkd.eye(nlatent)
        conjugate = DenseGaussianConjugatePosterior(
            obs_matrix, prior_mean, prior_cov, noise_cov, bkd,
        )
        conjugate.compute(observations)
        exact_mean = bkd.flatten(conjugate.posterior_mean())
        exact_var = bkd.diag(conjugate.posterior_covariance())

        # VI setup — mean-field: 2 independent ConditionalGaussians
        conditionals = [_make_cond_gaussian(bkd) for _ in range(nlatent)]
        var_dist = ConditionalIndependentJoint(conditionals, bkd)
        prior = IndependentJoint(
            [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nlatent)],
            bkd,
        )

        noise_variances = bkd.full((nlatent,), noise_var)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        multi_lik = MultiExperimentLogLikelihood(base_lik, observations, bkd)

        def log_likelihood_fn(z):
            return multi_lik.logpdf(obs_matrix @ z)

        # Sparse grid strategy: 2D Gauss-Hermite (standard normal)
        std_normal = GaussianMarginal(0.0, 1.0, bkd)
        growth = LinearGrowthRule(scale=1, shift=1)
        sg_factories = [GaussLagrangeFactory(std_normal, bkd)] * nlatent
        sg_tp_factory = TensorProductSubspaceFactory(bkd, sg_factories, growth)
        sg_rule = ParameterizedIsotropicSparseGridQuadratureRule(
            bkd, sg_tp_factory,
        )
        sg_strategy = QuadratureStrategy(
            sg_rule, bkd, min_level=1, max_level=4,
        )

        elbo_sg = make_inexact_single_problem_elbo(
            var_dist, log_likelihood_fn, prior, sg_strategy, bkd,
        )

        optimizer = ScipyTrustConstrOptimizer(maxiter=300, gtol=1e-8)
        fitter = VariationalFitter(bkd, optimizer=optimizer)
        fitter.fit(elbo_sg)

        # Extract recovered params for each component
        vi_means = []
        vi_vars = []
        for cond in conditionals:
            m, s = _extract_gaussian_params(cond, bkd)
            vi_means.append(m)
            vi_vars.append(s ** 2)

        bkd.assert_allclose(
            bkd.asarray(vi_means), exact_mean, atol=0.15,
        )
        bkd.assert_allclose(
            bkd.asarray(vi_vars), exact_var, rtol=0.3,
        )



class TestInexactELBOTorch:
    """Torch-specific tests for autograd jacobian."""

    @pytest.fixture
    def torch_bkd(self):
        import torch

        from pyapprox.util.backends.torch import TorchBkd
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

    def test_has_jacobian(self, torch_bkd) -> None:
        bkd = torch_bkd
        elbo = _make_simple_inexact_elbo(bkd)
        assert hasattr(elbo, "jacobian")

    def test_has_inexact_jacobian(self, torch_bkd) -> None:
        bkd = torch_bkd
        elbo = _make_simple_inexact_elbo(bkd)
        assert isinstance(elbo, InexactDifferentiable)

    def test_jacobian_shape(self, torch_bkd) -> None:
        bkd = torch_bkd
        elbo = _make_simple_inexact_elbo(bkd)
        params = bkd.zeros((elbo.nvars(), 1))
        jac = elbo.jacobian(params)
        assert jac.shape == (1, elbo.nvars())

    def test_inexact_jacobian_shape(self, torch_bkd) -> None:
        bkd = torch_bkd
        elbo = _make_simple_inexact_elbo(bkd)
        params = bkd.zeros((elbo.nvars(), 1))
        jac = elbo.inexact_jacobian(params, 0.5)
        assert jac.shape == (1, elbo.nvars())

    def test_jacobian_derivative_checker(self, torch_bkd) -> None:
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = torch_bkd
        elbo = _make_simple_inexact_elbo(bkd)
        checker = DerivativeChecker(elbo)
        sample = bkd.zeros((elbo.nvars(), 1))
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = checker.error_ratio(errors[0])
        assert float(bkd.flatten(ratio)[0]) <= 1e-5

    def test_jacobian_equals_inexact_jacobian_tol_zero(self, torch_bkd) -> None:
        bkd = torch_bkd
        elbo = _make_simple_inexact_elbo(bkd)
        params = bkd.zeros((elbo.nvars(), 1))
        bkd.assert_allclose(
            elbo.jacobian(params),
            elbo.inexact_jacobian(params, 0.0),
            rtol=1e-12,
        )


class TestInexactELBONoJacobianNumpy:
    """NumPy-specific: no jacobian available."""

    def test_no_jacobian_numpy(self, numpy_bkd) -> None:
        elbo = _make_simple_inexact_elbo(numpy_bkd)
        assert not hasattr(elbo, "jacobian")

    def test_no_inexact_jacobian_numpy(self, numpy_bkd) -> None:
        """inexact_jacobian is the autograd one, not available on numpy."""
        elbo = _make_simple_inexact_elbo(numpy_bkd)
        # InexactDifferentiable checks for inexact_jacobian attribute
        assert not isinstance(elbo, InexactDifferentiable)


class TestInexactDiscreteGroupELBO:
    """Tests for make_inexact_discrete_group_elbo."""

    def test_discrete_group_shape(self, bkd) -> None:
        """Discrete-group inexact ELBO returns correct shape."""
        cond = ConditionalIndependentJoint(
            [_make_cond_gaussian(bkd)], bkd,
        )
        prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        def log_lik_0(z):
            return -0.5 * (z - 1.0) ** 2

        def log_lik_1(z):
            return -0.5 * (z - 2.0) ** 2

        strategy = _make_fixed_strategy(bkd, nsamples=30)
        labels = bkd.asarray([[-1.0, 1.0]])  # 2 groups, 1 label dim

        elbo = make_inexact_discrete_group_elbo(
            cond, [log_lik_0, log_lik_1], prior, strategy, bkd,
            labels=labels,
        )

        params = bkd.zeros((elbo.nvars(), 1))
        result = elbo(params)
        assert result.shape == (1, 1)

    def test_discrete_group_deterministic(self, bkd) -> None:
        """Discrete-group ELBO is deterministic."""
        cond = ConditionalIndependentJoint(
            [_make_cond_gaussian(bkd)], bkd,
        )
        prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        def log_lik_0(z):
            return -0.5 * (z - 1.0) ** 2

        strategy = _make_fixed_strategy(bkd, nsamples=30)
        labels = bkd.asarray([[0.0]])

        elbo = make_inexact_discrete_group_elbo(
            cond, [log_lik_0], prior, strategy, bkd,
            labels=labels,
        )

        params = bkd.zeros((elbo.nvars(), 1))
        v1 = elbo(params)
        v2 = elbo(params)
        bkd.assert_allclose(v1, v2, rtol=1e-12)

    def test_validation_errors(self, bkd) -> None:
        """Factory raises on invalid inputs."""
        cond = _make_cond_gaussian(bkd)
        prior = GaussianMarginal(0.0, 1.0, bkd)
        strategy = _make_fixed_strategy(bkd)

        with pytest.raises(ValueError, match="Either 'labels'"):
            make_inexact_discrete_group_elbo(
                cond, [lambda z: z], prior, strategy, bkd,
            )

        with pytest.raises(ValueError, match="Expected 2"):
            make_inexact_discrete_group_elbo(
                cond, [lambda z: z, lambda z: z], prior, strategy, bkd,
                labels=bkd.asarray([[0.0]]),  # 1 column, need 2
            )

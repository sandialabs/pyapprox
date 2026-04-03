"""
Tests for recovering conjugate posteriors via variational inference.

These tests verify that optimizing the ELBO with conditional distributions
(ConditionalGaussian, ConditionalBeta, ConditionalIndependentJoint)
recovers the exact conjugate posterior.
"""

# TODO: check if their is any redundancy in tests here and in test_amortized
# if so consolidate. Either way update file doc strings of tests
# to state what is unique to the test contained in them

import math

import numpy as np

from pyapprox.inverse.conjugate.beta import BetaConjugatePosterior
from pyapprox.inverse.conjugate.gaussian import (
    DenseGaussianConjugatePosterior,
)
from pyapprox.inverse.variational.elbo import (
    make_single_problem_elbo,
)
from pyapprox.inverse.variational.fitter import VariationalFitter
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.probability.conditional.beta import ConditionalBeta
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
from pyapprox.probability.univariate.beta import BetaMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.protocols import Backend
from tests._helpers.markers import slow_test


def _make_degree0_expansion(bkd: Backend, coeff: float = 0.0) -> BasisExpansion:
    """Create a degree-0 BasisExpansion (constant function).

    Returns a function f(x) = coeff for any x, with 1 conditioning variable
    and 1 tunable coefficient.
    """
    marginals = [UniformMarginal(-1.0, 1.0, bkd)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(1, 0, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=1)
    exp.set_coefficients(bkd.asarray([[coeff]]))
    return exp


def _make_cond_gaussian(
    bkd: Backend, mean: float = 0.0, log_stdev: float = 0.0
) -> ConditionalGaussian:
    """Create a ConditionalGaussian with constant mean and log_stdev."""
    mean_func = _make_degree0_expansion(bkd, mean)
    log_stdev_func = _make_degree0_expansion(bkd, log_stdev)
    return ConditionalGaussian(mean_func, log_stdev_func, bkd)


def _make_cond_beta(
    bkd: Backend, log_alpha: float = 0.0, log_beta: float = 0.0
) -> ConditionalBeta:
    """Create a ConditionalBeta with constant log_alpha and log_beta."""
    log_alpha_func = _make_degree0_expansion(bkd, log_alpha)
    log_beta_func = _make_degree0_expansion(bkd, log_beta)
    return ConditionalBeta(log_alpha_func, log_beta_func, bkd)


def _extract_gaussian_params(cond: ConditionalGaussian, bkd: Backend) -> tuple:
    """Extract (mean, stdev) from a fitted ConditionalGaussian.

    For degree-0 expansions, the param functions are constant so
    the conditioning input value doesn't matter.
    """
    dummy_x = bkd.zeros((cond.nvars(), 1))
    mean = cond._mean_func(dummy_x)[0, 0]
    log_stdev = cond._log_stdev_func(dummy_x)[0, 0]
    return mean, bkd.exp(log_stdev)


def _extract_beta_params(cond: ConditionalBeta, bkd: Backend) -> tuple:
    """Extract (alpha, beta) from a fitted ConditionalBeta."""
    dummy_x = bkd.zeros((cond.nvars(), 1))
    log_alpha = cond._log_alpha_func(dummy_x)[0, 0]
    log_beta = cond._log_beta_func(dummy_x)[0, 0]
    return bkd.exp(log_alpha), bkd.exp(log_beta)


class TestGaussianConjugateRecoveryBase:
    """Base test class for Gaussian conjugate recovery."""

    def _make_cond_gaussian_joint(self, bkd, nvars: int) -> ConditionalIndependentJoint:
        """Create a variational distribution for nvars dimensions.

        Always returns a ConditionalIndependentJoint, even for nvars=1.
        """
        conditionals = [_make_cond_gaussian(bkd) for _ in range(nvars)]
        return ConditionalIndependentJoint(conditionals, bkd)

    def _run_vi_recovery(
        self,
        bkd,
        nvars: int,
        obs_matrix,
        prior_mean_vals: list,
        prior_var_vals: list,
        noise_var: float,
        observations,
        nsamples: int = 500,
        maxiter: int = 200,
    ) -> tuple:
        """Run VI and return (vi_mean, vi_var, exact_mean, exact_var)."""
        nobs = obs_matrix.shape[0]

        # Exact conjugate posterior
        prior_mean_arr = bkd.reshape(bkd.asarray(prior_mean_vals), (nvars, 1))
        prior_cov_arr = bkd.diag(bkd.asarray(prior_var_vals))
        noise_cov_arr = noise_var * bkd.eye(nobs)
        conjugate = DenseGaussianConjugatePosterior(
            obs_matrix,
            prior_mean_arr,
            prior_cov_arr,
            noise_cov_arr,
            bkd,
        )
        conjugate.compute(observations)
        exact_mean = conjugate.posterior_mean()
        exact_cov = conjugate.posterior_covariance()

        # VI setup using conditional distributions
        var_dist = self._make_cond_gaussian_joint(bkd, nvars)

        # Prior: IndependentJoint of GaussianMarginals
        prior_marginals = [
            GaussianMarginal(m, math.sqrt(v), bkd)
            for m, v in zip(prior_mean_vals, prior_var_vals)
        ]
        prior = IndependentJoint(prior_marginals, bkd)

        # Build log-likelihood using MultiExperimentLogLikelihood
        noise_variances = bkd.full((nobs,), noise_var)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        multi_lik = MultiExperimentLogLikelihood(
            base_lik,
            observations,
            bkd,
        )

        def log_likelihood_fn(z):
            return multi_lik.logpdf(obs_matrix @ z)

        np.random.seed(42)
        base_samples = bkd.asarray(np.random.normal(0, 1, (nvars, nsamples)))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        elbo = make_single_problem_elbo(
            var_dist,
            log_likelihood_fn,
            prior,
            base_samples,
            weights,
            bkd,
        )

        optimizer = ScipyTrustConstrOptimizer(maxiter=maxiter, gtol=1e-8)
        fitter = VariationalFitter(bkd, optimizer=optimizer)
        fitter.fit(elbo)

        # Extract recovered mean and variance from each conditional
        means = []
        variances = []
        for cond in var_dist._conditionals:
            m, s = _extract_gaussian_params(cond, bkd)
            means.append(m)
            variances.append(s**2)
        vi_mean = bkd.asarray(means)
        vi_var = bkd.asarray(variances)

        return vi_mean, vi_var, exact_mean, exact_cov

    @slow_test
    def test_gaussian_1d_conjugate(self, bkd) -> None:
        """1D linear model: A=[[1]], prior N(0,1), noise_var=0.5, obs=[2.0]."""
        obs_matrix = bkd.asarray([[1.0]])
        observations = bkd.asarray([[2.0]])

        vi_mean, vi_var, exact_mean, exact_cov = self._run_vi_recovery(
            bkd,
            nvars=1,
            obs_matrix=obs_matrix,
            prior_mean_vals=[0.0],
            prior_var_vals=[1.0],
            noise_var=0.5,
            observations=observations,
            nsamples=1000,
            maxiter=300,
        )

        exact_mean_flat = bkd.flatten(exact_mean)
        exact_var_diag = bkd.diag(exact_cov)

        bkd.assert_allclose(vi_mean, exact_mean_flat, atol=0.15)
        bkd.assert_allclose(vi_var, exact_var_diag, rtol=0.3)

    @slow_test
    def test_gaussian_2d_conjugate(self, bkd) -> None:
        """2D linear model: A=[[1,0],[0,1]], prior N(0,I), noise_var=0.5."""
        obs_matrix = bkd.eye(2)
        observations = bkd.asarray([[1.5], [2.5]])

        vi_mean, vi_var, exact_mean, exact_cov = self._run_vi_recovery(
            bkd,
            nvars=2,
            obs_matrix=obs_matrix,
            prior_mean_vals=[0.0, 0.0],
            prior_var_vals=[1.0, 1.0],
            noise_var=0.5,
            observations=observations,
            nsamples=1000,
            maxiter=300,
        )

        exact_mean_flat = bkd.flatten(exact_mean)
        exact_var_diag = bkd.diag(exact_cov)

        bkd.assert_allclose(vi_mean, exact_mean_flat, atol=0.15)
        bkd.assert_allclose(vi_var, exact_var_diag, rtol=0.3)

    @slow_test
    def test_gaussian_more_data_closer(self, bkd) -> None:
        """More observations should make VI closer to exact posterior."""
        obs_matrix = bkd.asarray([[1.0]])

        # Few observations
        obs_few = bkd.asarray([[2.0]])
        vi_mean_few, vi_var_few, exact_mean_few, exact_cov_few = self._run_vi_recovery(
            bkd,
            nvars=1,
            obs_matrix=obs_matrix,
            prior_mean_vals=[0.0],
            prior_var_vals=[1.0],
            noise_var=0.5,
            observations=obs_few,
            nsamples=1000,
            maxiter=300,
        )

        # Many observations (3 experiments)
        obs_many = bkd.asarray([[2.0, 1.8, 2.2]])
        vi_mean_many, vi_var_many, exact_mean_many, exact_cov_many = (
            self._run_vi_recovery(
                bkd,
                nvars=1,
                obs_matrix=obs_matrix,
                prior_mean_vals=[0.0],
                prior_var_vals=[1.0],
                noise_var=0.5,
                observations=obs_many,
                nsamples=1000,
                maxiter=300,
            )
        )

        # More data -> smaller posterior variance
        exact_var_few = exact_cov_few[0, 0]
        exact_var_many = exact_cov_many[0, 0]
        assert float(bkd.flatten(exact_var_many)[0]) < float(
            bkd.flatten(exact_var_few)[0]
        )

        assert float(bkd.flatten(vi_var_many)[0]) < float(
            bkd.flatten(vi_var_few)[0]
        )


class TestBetaBernoulliRecoveryBase:
    """Test recovering Beta-Bernoulli conjugate posterior via VI."""

    @slow_test
    def test_beta_bernoulli_conjugate(self, bkd) -> None:
        """Prior Beta(2,2), obs=[1,1,0,1,0] -> exact Beta(5,4)."""
        prior_alpha, prior_beta = 2.0, 2.0
        obs_list = [1.0, 1.0, 0.0, 1.0, 0.0]

        # Exact conjugate posterior
        conjugate = BetaConjugatePosterior(prior_alpha, prior_beta, bkd)
        conjugate.compute(bkd.asarray([obs_list]))
        exact_mean = conjugate.posterior_mean()

        # VI setup using ConditionalBeta with degree-0 expansion
        # Initialize log_alpha = log(prior_alpha), log_beta = log(prior_beta)
        cond_beta = _make_cond_beta(
            bkd,
            log_alpha=math.log(prior_alpha),
            log_beta=math.log(prior_beta),
        )

        # Prior distribution for ELBO
        prior_marginal = BetaMarginal(prior_alpha, prior_beta, bkd)
        prior = prior_marginal

        # Bernoulli log-likelihood: sum_i obs_i*log(p) + (1-obs_i)*log(1-p)
        obs = bkd.asarray([obs_list])  # (1, 5)

        def log_likelihood_fn(z):
            # z: (1, N) -- probability values in (0, 1)
            p = bkd.clip(z, 1e-8, 1.0 - 1e-8)
            log_p = bkd.log(p)  # (1, N)
            log_1mp = bkd.log(1.0 - p)  # (1, N)
            n_success = bkd.sum(obs)
            n_fail = obs.shape[1] - n_success
            result = n_success * log_p + n_fail * log_1mp
            return result  # (1, N)

        nsamples = 2000
        np.random.seed(42)
        base_samples = bkd.asarray(np.random.uniform(0.01, 0.99, (1, nsamples)))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        elbo = make_single_problem_elbo(
            cond_beta,
            log_likelihood_fn,
            prior,
            base_samples,
            weights,
            bkd,
        )

        fitter = VariationalFitter(bkd)
        fitter.fit(elbo)

        # Extract recovered alpha/beta
        recovered_alpha, recovered_beta = _extract_beta_params(cond_beta, bkd)
        recovered_mean = recovered_alpha / (recovered_alpha + recovered_beta)

        # Posterior mean should match
        bkd.assert_allclose(
            bkd.asarray([recovered_mean]),
            bkd.asarray([exact_mean]),
            rtol=0.15,
        )


class TestGaussianEquivalenceBase:
    """Test that ConditionalIndependentJoint with a single ConditionalGaussian
    produces identical results to a standalone ConditionalGaussian.

    Both parameterise the same diagonal Gaussian and both have analytical KL,
    so the ELBO objectives are functionally identical and converged optima
    should agree to near machine precision.
    """

    def test_logpdf_equivalence(self, bkd) -> None:
        """logpdf from single vs joint-wrapped conditional agrees."""
        # Single ConditionalGaussian
        cond_single = _make_cond_gaussian(bkd, mean=1.0, log_stdev=math.log(0.5))

        # ConditionalIndependentJoint wrapping one ConditionalGaussian
        cond_inner = _make_cond_gaussian(bkd, mean=1.0, log_stdev=math.log(0.5))
        cond_joint = ConditionalIndependentJoint([cond_inner], bkd)

        x = bkd.zeros((1, 4))  # dummy conditioning
        y = bkd.asarray([[-1.0, 0.0, 1.0, 2.0]])

        logp_single = cond_single.logpdf(x, y)
        logp_joint = cond_joint.logpdf(x, y)
        bkd.assert_allclose(logp_single, logp_joint, rtol=1e-12)

    def test_reparameterize_equivalence(self, bkd) -> None:
        """reparameterize from single vs joint-wrapped conditional agrees."""
        cond_single = _make_cond_gaussian(bkd, mean=1.0, log_stdev=math.log(0.5))

        cond_inner = _make_cond_gaussian(bkd, mean=1.0, log_stdev=math.log(0.5))
        cond_joint = ConditionalIndependentJoint([cond_inner], bkd)

        x = bkd.zeros((1, 50))  # dummy conditioning
        np.random.seed(42)
        base = bkd.asarray(np.random.randn(1, 50))

        z_single = cond_single.reparameterize(x, base)
        z_joint = cond_joint.reparameterize(x, base)
        bkd.assert_allclose(z_single, z_joint, rtol=1e-12)

    @slow_test
    def test_converged_optima_equivalence(self, bkd) -> None:
        """Single conditional and joint-wrapped conditional converge to same result."""
        obs_matrix = bkd.asarray([[1.0]])
        observations = bkd.asarray([[2.0]])
        noise_var = 0.5
        nsamples = 1000

        noise_variances = bkd.full((1,), noise_var)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        multi_lik = MultiExperimentLogLikelihood(
            base_lik,
            observations,
            bkd,
        )

        def log_likelihood_fn(z):
            return multi_lik.logpdf(obs_matrix @ z)

        np.random.seed(42)
        base_samples = bkd.asarray(np.random.normal(0, 1, (1, nsamples)))
        weights = bkd.full((1, nsamples), 1.0 / nsamples)

        # --- Single ConditionalGaussian ---
        cond_single = _make_cond_gaussian(bkd)
        prior_single = GaussianMarginal(0.0, 1.0, bkd)
        elbo_single = make_single_problem_elbo(
            cond_single,
            log_likelihood_fn,
            prior_single,
            base_samples,
            weights,
            bkd,
        )
        optimizer = ScipyTrustConstrOptimizer(maxiter=300, gtol=1e-8)
        fitter_single = VariationalFitter(bkd, optimizer=optimizer)
        fitter_single.fit(elbo_single)
        single_mean, single_stdev = _extract_gaussian_params(cond_single, bkd)

        # --- ConditionalIndependentJoint([ConditionalGaussian]) ---
        cond_inner = _make_cond_gaussian(bkd)
        cond_joint = ConditionalIndependentJoint([cond_inner], bkd)
        prior_joint = IndependentJoint(
            [GaussianMarginal(0.0, 1.0, bkd)],
            bkd,
        )
        elbo_joint = make_single_problem_elbo(
            cond_joint,
            log_likelihood_fn,
            prior_joint,
            base_samples,
            weights,
            bkd,
        )
        fitter_joint = VariationalFitter(bkd, optimizer=optimizer)
        fitter_joint.fit(elbo_joint)
        joint_mean, joint_stdev = _extract_gaussian_params(cond_inner, bkd)

        # Both optimised the same objective -> tight tolerance
        bkd.assert_allclose(
            bkd.asarray([single_mean]),
            bkd.asarray([joint_mean]),
            rtol=1e-8,
        )
        bkd.assert_allclose(
            bkd.asarray([single_stdev]),
            bkd.asarray([joint_stdev]),
            rtol=1e-8,
        )

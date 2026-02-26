"""
Tests for amortized variational inference using conditional distributions
with higher-degree BasisExpansion parameter functions.

Amortized VI: degree > 0 BasisExpansion maps labels -> distribution params.
The same variational distribution simultaneously approximates posteriors
for multiple groups (identified by labels).

All tests that require optimization or gradients are Torch-only, since
NumPy does not yet support analytical or autograd derivatives.
"""

import pytest

import numpy as np

from pyapprox.expdesign.quadrature.gaussian import (
    GaussianQuadratureSampler,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.inverse.conjugate.gaussian import (
    DenseGaussianConjugatePosterior,
)
from pyapprox.inverse.variational.elbo import (
    make_discrete_group_elbo,
)
from pyapprox.inverse.variational.fitter import VariationalFitter
from pyapprox.probability.conditional.gaussian import (
    ConditionalGaussian,
)
from pyapprox.probability.conditional.joint import (
    ConditionalIndependentJoint,
)
from pyapprox.probability.conditional.multivariate_gaussian import (
    ConditionalDenseCholGaussian,
    ConditionalLowRankCholGaussian,
)
from pyapprox.probability.covariance import (
    DenseCholeskyCovarianceOperator,
)
from pyapprox.probability.gaussian.dense import (
    DenseCholeskyMultivariateGaussian,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.likelihood.gaussian import (
    DiagonalGaussianLogLikelihood,
    GaussianLogLikelihood,
    MultiExperimentLogLikelihood,
)
from pyapprox.probability.univariate import UniformMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions.base import BasisExpansion
from pyapprox.surrogates.affine.expansions.pce import (
    create_pce_from_marginals,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.test_utils import slow_test


def _make_expansion(bkd: Backend, degree: int, coeff: float = 0.0) -> BasisExpansion:
    """Create a BasisExpansion with given degree (1D input, 1 QoI).

    For degree d, the expansion has d+1 terms.
    Constant term initialized to coeff, rest to 0.
    """
    marginals = [UniformMarginal(-1.0, 1.0, bkd)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(1, degree, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=1)
    coeffs_np = np.zeros((degree + 1, 1))
    coeffs_np[0, 0] = coeff
    exp.set_coefficients(bkd.asarray(coeffs_np))
    return exp


def _make_cond_gaussian_degree(
    bkd: Backend, degree: int, mean: float = 0.0, log_stdev: float = 0.0
) -> ConditionalGaussian:
    """Create a ConditionalGaussian with degree-d parameter functions."""
    mean_func = _make_expansion(bkd, degree, mean)
    log_stdev_func = _make_expansion(bkd, degree, log_stdev)
    return ConditionalGaussian(mean_func, log_stdev_func, bkd)


def _gauss_hermite_nodes_weights(bkd: Backend, npoints: int):
    """Return Gauss-Hermite quadrature nodes and weights for N(0,1).

    Uses GaussianQuadratureSampler from expdesign.

    Returns
    -------
    nodes : Array, shape (1, npoints)
    weights : Array, shape (1, npoints)
    """
    sampler = GaussianQuadratureSampler(nvars=1, bkd=bkd, npoints_1d=npoints)
    nodes, weights = sampler.sample(0)
    # nodes: (1, npoints), weights: (npoints,)
    return nodes, bkd.reshape(weights, (1, -1))


def _make_expansion_nd_nqoi(
    bkd: Backend,
    nvars_in: int,
    degree: int,
    nqoi: int,
    coeff: float = 0.0,
):
    """Create a BasisExpansion with given degree, input dim, and nqoi."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars_in)]
    exp = create_pce_from_marginals(marginals, degree, bkd, nqoi=nqoi)
    nterms = exp.nterms()
    coeffs_np = np.zeros((nterms, nqoi))
    coeffs_np[0, :] = coeff
    exp.set_coefficients(bkd.asarray(coeffs_np))
    return exp


def _make_cond_dense_chol_nd(
    bkd: Backend,
    nvars_in: int,
    d: int,
    degree: int,
) -> ConditionalDenseCholGaussian:
    """Create a ConditionalDenseCholGaussian with nvars_in-dim input."""
    mean_func = _make_expansion_nd_nqoi(bkd, nvars_in, degree, nqoi=d)
    log_chol_diag_func = _make_expansion_nd_nqoi(bkd, nvars_in, degree, nqoi=d)
    n_offdiag = d * (d - 1) // 2
    chol_offdiag_func = None
    if d > 1:
        chol_offdiag_func = _make_expansion_nd_nqoi(
            bkd,
            nvars_in,
            degree,
            nqoi=n_offdiag,
        )
    return ConditionalDenseCholGaussian(
        mean_func,
        log_chol_diag_func,
        chol_offdiag_func,
        bkd,
    )


def _make_cond_low_rank_nd(
    bkd: Backend,
    nvars_in: int,
    d: int,
    rank: int,
    degree: int,
    log_diag_lower_bound: float = -6.0,
) -> ConditionalLowRankCholGaussian:
    """Create a ConditionalLowRankCholGaussian with nvars_in-dim input.

    Sets a lower bound on log_diag_func coefficients to prevent D from
    becoming too small, which would make Sigma = D^2 + VV^T numerically
    singular during Cholesky factorization in reparameterize().
    """
    mean_func = _make_expansion_nd_nqoi(bkd, nvars_in, degree, nqoi=d)
    log_diag_func = _make_expansion_nd_nqoi(bkd, nvars_in, degree, nqoi=d)
    # Set lower bounds on log_diag coefficients via public set_bounds() API
    for hp in log_diag_func.hyp_list().hyperparameters():
        bounds = hp.get_bounds()
        lower = bkd.full((bounds.shape[0],), log_diag_lower_bound)
        new_bounds = bkd.stack([lower, bounds[:, 1]], axis=1)
        hp.set_bounds(new_bounds)
    factor_func = None
    if rank > 0:
        factor_func = _make_expansion_nd_nqoi(
            bkd,
            nvars_in,
            degree,
            nqoi=d * rank,
        )
    return ConditionalLowRankCholGaussian(
        mean_func,
        log_diag_func,
        factor_func,
        rank,
        bkd,
    )


@pytest.fixture()
def torch_bkd():
    """Lazy-import TorchBkd fixture for Torch-only tests."""
    import torch

    from pyapprox.util.backends.torch import TorchBkd

    torch.set_default_dtype(torch.float64)
    return TorchBkd()


class TestAmortizedBase:
    """Base test class for amortized VI.

    Contains only tests that do not require optimization or gradients,
    so they work with both NumPy and Torch backends.
    """

    def test_amortized_polynomial_shape(self, bkd) -> None:
        """Degree-2 expansion, verify ELBO callable with label nodes."""
        degree = 2

        cond = _make_cond_gaussian_degree(bkd, degree)
        var_dist = ConditionalIndependentJoint([cond], bkd)
        prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        K = 3
        labels = bkd.asarray([[-0.5, 0.0, 0.5]])  # (1, K)

        noise_var = 0.5
        noise_variances = bkd.full((1,), noise_var)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        log_lik_fns = []
        for k in range(K):
            obs_k = bkd.asarray([[float(k + 1)]])
            multi_lik = MultiExperimentLogLikelihood(base_lik, obs_k, bkd)
            log_lik_fns.append(multi_lik.logpdf)

        npoints = 15
        base_nodes, base_weights = _gauss_hermite_nodes_weights(bkd, npoints)

        elbo = make_discrete_group_elbo(
            var_dist,
            log_lik_fns,
            prior,
            labels,
            base_nodes,
            base_weights,
            bkd,
        )

        # nvars = nterms_mean + nterms_logstdev = (degree+1)*2 = 6
        assert elbo.nqoi() == 1
        assert elbo.nvars() == 2 * (degree + 1)

        params = bkd.zeros((elbo.nvars(), 1))
        result = elbo(params)
        assert result.shape == (1, 1)

        assert isinstance(elbo, FunctionProtocol)

    def test_make_discrete_group_elbo_joint_nodes_shape(self, bkd) -> None:
        """Verify joint nodes have correct shape (nlabel_dims + nbase, K*M)."""
        degree = 1
        cond = _make_cond_gaussian_degree(bkd, degree)
        var_dist = ConditionalIndependentJoint([cond], bkd)
        prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        K = 4
        M = 10
        labels = bkd.asarray([[0.1, 0.2, 0.3, 0.4]])  # (1, K)
        base_nodes, base_weights = _gauss_hermite_nodes_weights(bkd, M)

        noise_variances = bkd.full((1,), 1.0)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        log_lik_fns = []
        for k in range(K):
            obs_k = bkd.asarray([[float(k)]])
            multi_lik = MultiExperimentLogLikelihood(base_lik, obs_k, bkd)
            log_lik_fns.append(multi_lik.logpdf)

        elbo = make_discrete_group_elbo(
            var_dist,
            log_lik_fns,
            prior,
            labels,
            base_nodes,
            base_weights,
            bkd,
        )

        # K*M = 4*10 = 40 joint quadrature points
        assert elbo._joint_nodes.shape[1] == K * M
        # nlabel_dims=1, nbase=1 -> 2 rows
        assert elbo._joint_nodes.shape[0] == 2
        # Weights should sum to 1
        weight_sum = bkd.sum(elbo._joint_weights)
        bkd.assert_allclose(
            bkd.asarray([weight_sum]),
            bkd.asarray([1.0]),
            rtol=1e-12,
        )

    def test_elbo_deterministic(self, bkd) -> None:
        """Calling amortized ELBO twice with same params gives same result."""
        degree = 1
        cond = _make_cond_gaussian_degree(bkd, degree)
        var_dist = ConditionalIndependentJoint([cond], bkd)
        prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        K = 2
        labels = bkd.asarray([[-0.5, 0.5]])
        npoints = 10
        base_nodes, base_weights = _gauss_hermite_nodes_weights(bkd, npoints)

        noise_variances = bkd.full((1,), 1.0)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        log_lik_fns = []
        for k in range(K):
            obs_k = bkd.asarray([[float(k)]])
            multi_lik = MultiExperimentLogLikelihood(base_lik, obs_k, bkd)
            log_lik_fns.append(multi_lik.logpdf)

        elbo = make_discrete_group_elbo(
            var_dist,
            log_lik_fns,
            prior,
            labels,
            base_nodes,
            base_weights,
            bkd,
        )
        params = bkd.zeros((elbo.nvars(), 1))
        v1 = elbo(params)
        v2 = elbo(params)
        bkd.assert_allclose(v1, v2, rtol=1e-12)


class TestAmortizedTorchOnly:
    """Torch-only tests including optimization and gradient checks.

    All tests that require the optimizer or autograd jacobian live here,
    since NumPy does not yet support analytical or autograd derivatives.
    """

    # --- Generalization test infrastructure ---

    def _run_generalization_test_multivariate(
        self,
        torch_bkd,
        obs_matrix,
        noise_cov,
        prior_cov,
        n_obs: int,
        var_dist_factory,
        nprior_quad: int = 3,
        nbase_quad: int = 50,
        atol: float = 1e-4,
    ) -> None:
        """Generalization test using multivariate variational distribution.

        Generates training groups from the noisy linear model:
            y_i = A z_k + eps_i,  eps_i ~ N(0, noise_cov)
        where z_k values come from Gauss-Hermite quadrature over the
        prior z ~ N(0, prior_cov).

        Labels encode the sufficient statistic bar_y_k = mean(y_{k,i})
        mapped to [-1,1]^nobs_dim via an affine transform.

        Verifies posterior mean and full covariance matrix at unseen
        test data against exact conjugate posterior.

        Mathematical note
        -----------------
        For a Gaussian linear model y = A z + eps with prior z ~ N(mu0, C0)
        and noise eps ~ N(0, R), the conjugate posterior mean is:

            mu_post = C_post (C0^{-1} mu0 + A^T R^{-1} S)

        where S = sum_i y_i and C_post = (C0^{-1} + n A^T R^{-1} A)^{-1}.
        For fixed n, the map S -> mu_post is *affine*. A degree-1 PCE can
        represent this affine map exactly.

        Parameters
        ----------
        var_dist_factory : callable
            (bkd, nlabel_dims, nlatent, degree) -> var_dist
        """
        bkd = torch_bkd
        nobs_dim = obs_matrix.shape[0]
        nlatent = obs_matrix.shape[1]
        nlabel_dims = nobs_dim
        prior_mean = bkd.zeros((nlatent, 1))

        # --- Prior quadrature ---
        prior_sampler = GaussianQuadratureSampler(
            nvars=nlatent,
            bkd=bkd,
            npoints_1d=nprior_quad,
        )
        prior_nodes, _ = prior_sampler.sample(0)
        prior_stds = bkd.sqrt(bkd.diag(prior_cov))
        z_train = prior_nodes * bkd.reshape(prior_stds, (nlatent, 1))
        K = z_train.shape[1]

        # --- Generate noisy observations ---
        np.random.seed(42)
        noise_L = np.linalg.cholesky(
            np.array(
                [
                    [float(noise_cov[i, j]) for j in range(nobs_dim)]
                    for i in range(nobs_dim)
                ]
            )
        )
        all_obs = []
        for k in range(K):
            mean_k = bkd.dot(obs_matrix, z_train[:, k : k + 1])
            eps_np = noise_L @ np.random.randn(nobs_dim, n_obs)
            eps = bkd.asarray(eps_np)
            obs_k = bkd.tile(mean_k, (1, n_obs)) + eps
            all_obs.append(obs_k)

        # --- Compute observation means and map to labels ---
        bar_y_list = []
        for obs_k in all_obs:
            bar_y_k = bkd.reshape(bkd.sum(obs_k, axis=1) / n_obs, (nobs_dim, 1))
            bar_y_list.append(bar_y_k)
        bar_y_all = bkd.hstack(bar_y_list)

        bar_y_min = bkd.reshape(
            bkd.asarray([float(bar_y_all[d, :].min()) for d in range(nobs_dim)]),
            (nobs_dim, 1),
        )
        bar_y_max = bkd.reshape(
            bkd.asarray([float(bar_y_all[d, :].max()) for d in range(nobs_dim)]),
            (nobs_dim, 1),
        )
        bar_y_mid = 0.5 * (bar_y_min + bar_y_max)
        bar_y_scale = 0.5 * (bar_y_max - bar_y_min)
        bar_y_scale = bkd.where(
            bar_y_scale > 1e-12,
            bar_y_scale,
            bkd.ones_like(bar_y_scale),
        )
        train_labels = (bar_y_all - bar_y_mid) / bar_y_scale

        # --- Build per-group likelihoods ---
        noise_cov_op = DenseCholeskyCovarianceOperator(noise_cov, bkd)
        base_lik = GaussianLogLikelihood(noise_cov_op, bkd)
        log_lik_fns = []
        for obs_k in all_obs:
            multi_lik = MultiExperimentLogLikelihood(base_lik, obs_k, bkd)

            def _make_log_lik(ml):
                def log_lik(z):
                    return ml.logpdf(bkd.dot(obs_matrix, z))

                return log_lik

            log_lik_fns.append(_make_log_lik(multi_lik))

        # --- Variational distribution (multivariate) ---
        degree = 1
        var_dist = var_dist_factory(bkd, nlabel_dims, nlatent, degree)

        # Prior: multivariate Gaussian
        prior = DenseCholeskyMultivariateGaussian(
            prior_mean,
            prior_cov,
            bkd,
        )

        # --- Gauss-Hermite quadrature for ELBO base samples ---
        nbase = var_dist.base_distribution().nvars()
        base_sampler = GaussianQuadratureSampler(
            nvars=nbase,
            bkd=bkd,
            npoints_1d=nbase_quad,
        )
        base_nodes, weights_1d = base_sampler.sample(0)
        base_weights = bkd.reshape(weights_1d, (1, -1))

        elbo = make_discrete_group_elbo(
            var_dist,
            log_lik_fns,
            prior,
            train_labels,
            base_nodes,
            base_weights,
            bkd,
        )

        # --- Optimize ---
        fitter = VariationalFitter(bkd)
        np.random.seed(0)
        init_guess = bkd.asarray(
            np.random.randn(elbo.nvars(), 1) * 0.01,
        )
        fitter.fit(elbo, init_guess=init_guess)

        # --- Verify at unseen test data ---
        np.random.seed(123)
        test_sampler = GaussianQuadratureSampler(
            nvars=nlatent,
            bkd=bkd,
            npoints_1d=nprior_quad + 1,
        )
        test_nodes, _ = test_sampler.sample(0)
        z_test = test_nodes * bkd.reshape(prior_stds, (nlatent, 1))
        K_test = z_test.shape[1]

        conjugate = DenseGaussianConjugatePosterior(
            obs_matrix,
            prior_mean,
            prior_cov,
            noise_cov,
            bkd,
        )

        for t in range(K_test):
            mean_t = bkd.dot(obs_matrix, z_test[:, t : t + 1])
            eps_np = noise_L @ np.random.randn(nobs_dim, n_obs)
            eps = bkd.asarray(eps_np)
            obs_test = bkd.tile(mean_t, (1, n_obs)) + eps

            bar_y_t = bkd.reshape(bkd.sum(obs_test, axis=1) / n_obs, (nobs_dim, 1))
            test_label = (bar_y_t - bar_y_mid) / bar_y_scale

            max_abs = float(bkd.max(bkd.abs(test_label)))
            if max_abs > 1.0:
                continue

            # Exact conjugate posterior
            conjugate.compute(obs_test)
            exact_mean = conjugate.posterior_mean()
            exact_cov_mat = conjugate.posterior_covariance()

            # Evaluate variational distribution
            vi_mean = var_dist.reparameterize(
                test_label,
                bkd.zeros((nbase, 1)),
            )
            bkd.assert_allclose(vi_mean, exact_mean, atol=atol)

            vi_cov = var_dist.covariance(test_label)  # (1, d, d)
            bkd.assert_allclose(vi_cov[0], exact_cov_mat, atol=atol)

    # --- Generalization tests (Torch-only, require optimizer) ---

    @slow_test
    def test_amortized_generalization_1d(self, torch_bkd) -> None:
        """1D generalization using ConditionalDenseCholGaussian(d=1).

        1D latent z ~ N(0,1), obs y = z + eps, eps ~ N(0, 0.5).
        Training groups from Gauss-Hermite quadrature over prior.
        Degree-1 PCE exactly represents the affine map S -> mu_post.
        """
        bkd = torch_bkd

        def factory(bkd, nlabel_dims, nlatent, degree):
            return _make_cond_dense_chol_nd(bkd, nlabel_dims, nlatent, degree)

        self._run_generalization_test_multivariate(
            torch_bkd,
            obs_matrix=bkd.asarray([[1.0]]),
            noise_cov=bkd.asarray([[0.5]]),
            prior_cov=bkd.asarray([[1.0]]),
            n_obs=2,
            var_dist_factory=factory,
            nprior_quad=3,
            nbase_quad=50,
        )

    @slow_test
    def test_amortized_generalization_2d_coupled(self, torch_bkd) -> None:
        """2D generalization with non-diagonal A and non-diagonal noise.

        2D latent z ~ N(0, I_2).
        Non-diagonal obs_matrix A = [[1, 0.5], [0.3, 1]] couples latent
        dims: y = A z + eps.
        Non-diagonal noise_cov = [[0.5, 0.1], [0.1, 0.4]] correlates
        observation errors.

        Uses ConditionalDenseCholGaussian to capture the full posterior
        covariance including off-diagonal terms.
        """
        bkd = torch_bkd

        def factory(bkd, nlabel_dims, nlatent, degree):
            return _make_cond_dense_chol_nd(bkd, nlabel_dims, nlatent, degree)

        self._run_generalization_test_multivariate(
            torch_bkd,
            obs_matrix=bkd.asarray([[1.0, 0.5], [0.3, 1.0]]),
            noise_cov=bkd.asarray([[0.5, 0.1], [0.1, 0.4]]),
            prior_cov=bkd.asarray([[1.0, 0.0], [0.0, 1.0]]),
            n_obs=2,
            var_dist_factory=factory,
            nprior_quad=4,
            nbase_quad=5,
            atol=1e-4,
        )

    @slow_test
    def test_amortized_generalization_2d_lowrank(self, torch_bkd) -> None:
        """2D generalization using low-rank (rank=d) variational dist.

        Same problem as test_amortized_generalization_2d_coupled but
        using ConditionalLowRankCholGaussian with rank=d=2, which has
        sufficient capacity to represent any covariance.
        """
        bkd = torch_bkd

        def factory(bkd, nlabel_dims, nlatent, degree):
            return _make_cond_low_rank_nd(
                bkd,
                nlabel_dims,
                nlatent,
                rank=nlatent,
                degree=degree,
            )

        self._run_generalization_test_multivariate(
            torch_bkd,
            obs_matrix=bkd.asarray([[1.0, 0.5], [0.3, 1.0]]),
            noise_cov=bkd.asarray([[0.5, 0.1], [0.1, 0.4]]),
            prior_cov=bkd.asarray([[1.0, 0.0], [0.0, 1.0]]),
            n_obs=2,
            var_dist_factory=factory,
            nprior_quad=4,
            nbase_quad=5,
            atol=1e-4,
        )

    # --- DerivativeChecker tests (Torch-only, require autograd) ---

    @slow_test
    def test_amortized_elbo_derivative_checker(self, torch_bkd) -> None:
        """DerivativeChecker on ELBO with ConditionalIndependentJoint."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = torch_bkd
        degree = 1
        cond = _make_cond_gaussian_degree(bkd, degree)
        var_dist = ConditionalIndependentJoint([cond], bkd)
        prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        K = 2
        labels = bkd.asarray([[-0.5, 0.5]])
        npoints = 10
        base_nodes, base_weights = _gauss_hermite_nodes_weights(bkd, npoints)

        noise_variances = bkd.full((1,), 1.0)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        log_lik_fns = []
        for k in range(K):
            obs_k = bkd.asarray([[float(k)]])
            multi_lik = MultiExperimentLogLikelihood(base_lik, obs_k, bkd)
            log_lik_fns.append(multi_lik.logpdf)

        elbo = make_discrete_group_elbo(
            var_dist,
            log_lik_fns,
            prior,
            labels,
            base_nodes,
            base_weights,
            bkd,
        )

        assert hasattr(elbo, "jacobian")
        checker = DerivativeChecker(elbo)
        sample = bkd.zeros((elbo.nvars(), 1))
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = checker.error_ratio(errors[0])
        assert float(bkd.flatten(ratio)[0]) <= 1e-5

    def test_elbo_derivative_checker_dense_chol(self, torch_bkd) -> None:
        """DerivativeChecker on ELBO with ConditionalDenseCholGaussian."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = torch_bkd
        d = 2
        nlabel_dims = 2
        degree = 1
        var_dist = _make_cond_dense_chol_nd(bkd, nlabel_dims, d, degree)
        prior = DenseCholeskyMultivariateGaussian(
            bkd.zeros((d, 1)),
            bkd.eye(d),
            bkd,
        )

        K = 2
        labels = bkd.asarray([[-0.5, 0.5], [-0.3, 0.3]])
        nbase_quad = 3
        nbase = var_dist.base_distribution().nvars()
        base_sampler = GaussianQuadratureSampler(
            nvars=nbase,
            bkd=bkd,
            npoints_1d=nbase_quad,
        )
        base_nodes, weights_1d = base_sampler.sample(0)
        base_weights = bkd.reshape(weights_1d, (1, -1))

        noise_cov = bkd.eye(d)
        noise_cov_op = DenseCholeskyCovarianceOperator(noise_cov, bkd)
        base_lik = GaussianLogLikelihood(noise_cov_op, bkd)
        log_lik_fns = []
        for k in range(K):
            obs_k = bkd.asarray([[float(k)], [float(k + 0.5)]])
            multi_lik = MultiExperimentLogLikelihood(base_lik, obs_k, bkd)
            log_lik_fns.append(multi_lik.logpdf)

        elbo = make_discrete_group_elbo(
            var_dist,
            log_lik_fns,
            prior,
            labels,
            base_nodes,
            base_weights,
            bkd,
        )

        assert hasattr(elbo, "jacobian")
        checker = DerivativeChecker(elbo)
        sample = bkd.zeros((elbo.nvars(), 1))
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = checker.error_ratio(errors[0])
        assert float(bkd.flatten(ratio)[0]) <= 1e-5

    def test_elbo_derivative_checker_low_rank(self, torch_bkd) -> None:
        """DerivativeChecker on ELBO with ConditionalLowRankCholGaussian."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = torch_bkd
        d = 2
        rank = 2
        nlabel_dims = 2
        degree = 1
        var_dist = _make_cond_low_rank_nd(
            bkd,
            nlabel_dims,
            d,
            rank,
            degree,
        )
        prior = DenseCholeskyMultivariateGaussian(
            bkd.zeros((d, 1)),
            bkd.eye(d),
            bkd,
        )

        K = 2
        labels = bkd.asarray([[-0.5, 0.5], [-0.3, 0.3]])
        nbase_quad = 3
        nbase = var_dist.base_distribution().nvars()
        base_sampler = GaussianQuadratureSampler(
            nvars=nbase,
            bkd=bkd,
            npoints_1d=nbase_quad,
        )
        base_nodes, weights_1d = base_sampler.sample(0)
        base_weights = bkd.reshape(weights_1d, (1, -1))

        noise_cov = bkd.eye(d)
        noise_cov_op = DenseCholeskyCovarianceOperator(noise_cov, bkd)
        base_lik = GaussianLogLikelihood(noise_cov_op, bkd)
        log_lik_fns = []
        for k in range(K):
            obs_k = bkd.asarray([[float(k)], [float(k + 0.5)]])
            multi_lik = MultiExperimentLogLikelihood(base_lik, obs_k, bkd)
            log_lik_fns.append(multi_lik.logpdf)

        elbo = make_discrete_group_elbo(
            var_dist,
            log_lik_fns,
            prior,
            labels,
            base_nodes,
            base_weights,
            bkd,
        )

        assert hasattr(elbo, "jacobian")
        checker = DerivativeChecker(elbo)
        sample = bkd.zeros((elbo.nvars(), 1))
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = checker.error_ratio(errors[0])
        assert float(bkd.flatten(ratio)[0]) <= 1e-5

    def test_amortized_jacobian_shape(self, torch_bkd) -> None:
        """Verify Jacobian shape for amortized ELBO."""
        bkd = torch_bkd
        degree = 1
        cond = _make_cond_gaussian_degree(bkd, degree)
        var_dist = ConditionalIndependentJoint([cond], bkd)
        prior = IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)], bkd)

        K = 2
        labels = bkd.asarray([[-0.5, 0.5]])
        npoints = 8
        base_nodes, base_weights = _gauss_hermite_nodes_weights(bkd, npoints)

        noise_variances = bkd.full((1,), 1.0)
        base_lik = DiagonalGaussianLogLikelihood(noise_variances, bkd)
        log_lik_fns = []
        for k in range(K):
            obs_k = bkd.asarray([[float(k)]])
            multi_lik = MultiExperimentLogLikelihood(base_lik, obs_k, bkd)
            log_lik_fns.append(multi_lik.logpdf)

        elbo = make_discrete_group_elbo(
            var_dist,
            log_lik_fns,
            prior,
            labels,
            base_nodes,
            base_weights,
            bkd,
        )

        assert hasattr(elbo, "jacobian")
        params = bkd.zeros((elbo.nvars(), 1))
        jac = elbo.jacobian(params)
        assert jac.shape == (1, elbo.nvars())

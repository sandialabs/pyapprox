"""Integration test: conditional flow matching + log-normal QoI push-forward.

Uses DenseGaussianConjugatePosterior as oracle.

Setup:
  Prior p(x) = N(0, I_d)
  Likelihood p(y|x) = N(Hx, Sigma_lik)
  Posterior p(x|y) = N(mu_post(y), Sigma_post)
  QoI g(x) = exp(a'x)
  g(x)|y ~ LogNormal(a'mu_post(y), a'Sigma_post a)

For d=1: g(x) = exp(x), exactly LogNormal(mu_post, sigma_post^2).
Known moments:
  E[g] = exp(mu + sigma^2/2)
  Var[g] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)

Paired linear coupling: x0 = z, x1 = L_post @ z + mu_post(y).
"""

import numpy as np
import pytest

from pyapprox.inverse.conjugate.gaussian import (
    DenseGaussianConjugatePosterior,
)
from pyapprox.pde.time.explicit_steppers.heun import HeunResidual
from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.flowmatching.cfm_loss import CFMLoss
from pyapprox.surrogates.flowmatching.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.flowmatching.linear_path import LinearPath
from pyapprox.surrogates.flowmatching.ode_adapter import (
    integrate_flow,
)
from pyapprox.surrogates.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.util.test_utils import slow_test


def _conjugate_params(d, m):
    """Generate fixed conjugate problem parameters."""
    np.random.seed(42)
    H_np = np.random.randn(m, d) * 0.5
    Sigma_lik_np = 0.1 * np.eye(m)
    return H_np, Sigma_lik_np


def _build_lognormal_setup(bkd, d, m, degree, n_per_dim=6):
    """Build paired conditional flow matching for Gaussian conjugate.

    Identical to test_gaussian_conjugate setup. Returns additionally
    the QoI direction vector a and analytical log-normal parameters.

    Returns vf, path, loss, quad_data, conjugate, test_y, a_np.
    """
    H_np, Sigma_lik_np = _conjugate_params(d, m)

    mu_prior = bkd.array(np.zeros((d, 1)).tolist())
    Sigma_prior = bkd.array(np.eye(d).tolist())
    H = bkd.array(H_np.tolist())
    Sigma_lik = bkd.array(Sigma_lik_np.tolist())

    conjugate = DenseGaussianConjugatePosterior(
        observation_matrix=H,
        prior_mean=mu_prior,
        prior_covariance=Sigma_prior,
        noise_covariance=Sigma_lik,
        bkd=bkd,
    )

    # Compute posterior covariance (independent of y) and its Cholesky
    dummy_y = bkd.array(np.zeros((m, 1)).tolist())
    conjugate.compute(dummy_y)
    Sigma_post_np = bkd.to_numpy(conjugate.posterior_covariance())
    L_post_np = np.linalg.cholesky(Sigma_post_np)

    # Quadrature over (t, z_1..z_d, y_1..y_m)
    quad_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    quad_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    quad_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * m
    quad_bases_1d = create_bases_1d(quad_marginals, bkd)
    quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
    nvars = 1 + d + m
    quad_pts, quad_wts = quad_basis.tensor_product_quadrature([n_per_dim] * nvars)

    t_all = quad_pts[0:1, :]
    z_all = quad_pts[1 : 1 + d, :]
    y_all = quad_pts[1 + d :, :]

    # Paired coupling: x0 = z, x1 = L_post @ z + mu_post(y)
    n_quad = quad_pts.shape[1]
    y_all_np = bkd.to_numpy(y_all)
    z_all_np = bkd.to_numpy(z_all)

    x1_np = np.zeros((d, n_quad))
    for i in range(n_quad):
        y_i = bkd.array(y_all_np[:, i : i + 1].tolist())
        conjugate.compute(y_i)
        mu_post_np = bkd.to_numpy(conjugate.posterior_mean())
        x1_np[:, i : i + 1] = L_post_np @ z_all_np[:, i : i + 1] + mu_post_np

    x0_all = z_all
    x1_all = bkd.array(x1_np.tolist())
    c_all = y_all

    # VF basis
    vf_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    vf_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    vf_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * m
    vf_bases_1d = create_bases_1d(vf_marginals, bkd)
    indices = compute_hyperbolic_indices(1 + d + m, degree, 1.0, bkd)
    vf_basis = OrthonormalPolynomialBasis(vf_bases_1d, bkd, indices)
    vf = BasisExpansion(vf_basis, bkd, nqoi=d)

    path = LinearPath(bkd)
    loss = CFMLoss(bkd)
    quad_data = FlowMatchingQuadData(
        t=t_all,
        x0=x0_all,
        x1=x1_all,
        weights=quad_wts,
        bkd=bkd,
        c=c_all,
    )

    # QoI direction: a = [1, 0, ..., 0] for simplicity (exp(x_1))
    np.random.seed(123)
    a_np = np.zeros((d, 1))
    a_np[0, 0] = 1.0

    # Test observation
    np.random.seed(999)
    test_y_np = H_np @ np.random.randn(d, 1) + np.random.randn(m, 1) * 0.1
    test_y = bkd.array(test_y_np.tolist())

    return vf, path, loss, quad_data, conjugate, test_y, a_np


def _analytical_lognormal_moments(conjugate, test_y, a_np, bkd):
    """Compute analytical LogNormal moments for g(x) = exp(a'x).

    Returns (mean, variance) as floats.
    """
    conjugate.compute(test_y)
    mu_post_np = bkd.to_numpy(conjugate.posterior_mean())  # (d, 1)
    Sigma_post_np = bkd.to_numpy(conjugate.posterior_covariance())  # (d, d)

    # a'mu_post and a'Sigma_post a
    mu_ln = float(a_np.T @ mu_post_np)  # scalar
    sigma2_ln = float(a_np.T @ Sigma_post_np @ a_np)  # scalar

    # LogNormal moments
    mean = np.exp(mu_ln + sigma2_ln / 2.0)
    variance = (np.exp(sigma2_ln) - 1.0) * np.exp(2 * mu_ln + sigma2_ln)
    return mean, variance


class TestLogNormalQoI:
    @pytest.mark.parametrize("d,m", [(1, 1)])
    def test_loss_decreases_with_degree(self, bkd, d: int, m: int) -> None:
        """Training loss decreases monotonically with polynomial degree."""
        degrees = [1, 2, 3, 4]
        losses = []
        for deg in degrees:
            vf, path, loss, qd, _, _, _ = _build_lognormal_setup(
                bkd,
                d,
                m,
                deg,
            )
            result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
            losses.append(result.training_loss())

        for i in range(len(losses) - 1):
            assert losses[i] > losses[i + 1], (
                f"Loss did not decrease from degree {degrees[i]} "
                f"({losses[i]:.2e}) to {degrees[i + 1]} "
                f"({losses[i + 1]:.2e})"
            )
        assert losses[-1] < 1e-4

    @pytest.mark.parametrize("d,m", [(1, 1)])
    @slow_test
    def test_lognormal_mean(self, bkd, d: int, m: int) -> None:
        """ODE-pushed QoI samples approximate LogNormal mean."""
        deg = 4
        vf, path, loss, qd, conjugate, test_y, a_np = _build_lognormal_setup(
            bkd, d, m, deg
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        fitted_vf = result.surrogate()

        # Analytical log-normal moments
        expected_mean, _ = _analytical_lognormal_moments(
            conjugate,
            test_y,
            a_np,
            bkd,
        )

        # Generate posterior samples via ODE
        np.random.seed(456)
        nsamples = 5000
        x0_np = np.random.randn(d, nsamples)
        x0_samples = bkd.array(x0_np.tolist())

        test_y_np = bkd.to_numpy(test_y)
        c_np = np.tile(test_y_np, (1, nsamples))
        c_samples = bkd.array(c_np.tolist())

        x1_samples = integrate_flow(
            fitted_vf,
            x0_samples,
            0.0,
            1.0,
            n_steps=50,
            bkd=bkd,
            c=c_samples,
            stepper_cls=HeunResidual,
        )

        # Push forward: g = exp(a' @ x1)
        x1_np = bkd.to_numpy(x1_samples)
        g_np = np.exp(a_np.T @ x1_np)  # (1, nsamples)
        sample_mean = float(np.mean(g_np))

        assert abs(sample_mean - expected_mean) < expected_mean * 0.15, (
            f"Sample mean {sample_mean:.4f} vs analytical {expected_mean:.4f}"
        )

    @pytest.mark.parametrize("d,m", [(1, 1)])
    @slow_test
    def test_lognormal_variance(self, bkd, d: int, m: int) -> None:
        """ODE-pushed QoI samples approximate LogNormal variance."""
        deg = 4
        vf, path, loss, qd, conjugate, test_y, a_np = _build_lognormal_setup(
            bkd, d, m, deg
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        fitted_vf = result.surrogate()

        _, expected_var = _analytical_lognormal_moments(
            conjugate,
            test_y,
            a_np,
            bkd,
        )

        np.random.seed(456)
        nsamples = 5000
        x0_np = np.random.randn(d, nsamples)
        x0_samples = bkd.array(x0_np.tolist())

        test_y_np = bkd.to_numpy(test_y)
        c_np = np.tile(test_y_np, (1, nsamples))
        c_samples = bkd.array(c_np.tolist())

        x1_samples = integrate_flow(
            fitted_vf,
            x0_samples,
            0.0,
            1.0,
            n_steps=50,
            bkd=bkd,
            c=c_samples,
            stepper_cls=HeunResidual,
        )

        x1_np = bkd.to_numpy(x1_samples)
        g_np = np.exp(a_np.T @ x1_np)
        sample_var = float(np.var(g_np))

        assert abs(sample_var - expected_var) < expected_var * 0.3, (
            f"Sample variance {sample_var:.4f} vs analytical {expected_var:.4f}"
        )

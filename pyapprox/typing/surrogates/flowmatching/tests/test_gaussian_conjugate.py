"""Integration test: conditional Gaussian flow matching.

Uses DenseGaussianConjugatePosterior as oracle.

Setup:
  Prior p(x) = N(0, I_d)
  Likelihood p(y|x) = N(Hx, Sigma_lik)
  Source x0 ~ N(0, I_d)
  Target x1 ~ p(x|y), conditioning c = y
  VF: BasisExpansion input_dim = 1+d+m, nqoi = d
  Path: LinearPath
  Loss: CFMLoss

Paired linear coupling: for each quadrature point (t, z, y) with
z ~ N(0,I), y ~ N(0,I), we set x0 = z and x1 = L_post @ z + mu_post(y).
Since x1 is an affine function of x0, the map x0 -> x_t is invertible
for all t in (0,1), so the CFM loss converges to zero with increasing
polynomial degree.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import slow_test

from pyapprox.typing.probability import UniformMarginal, GaussianMarginal
from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion

from pyapprox.typing.surrogates.flowmatching.linear_path import LinearPath
from pyapprox.typing.surrogates.flowmatching.cfm_loss import CFMLoss
from pyapprox.typing.surrogates.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.typing.surrogates.flowmatching.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.typing.surrogates.flowmatching.fitters.optimizer import (
    OptimizerFitter,
)
from pyapprox.typing.surrogates.flowmatching.ode_adapter import (
    integrate_flow,
)
from pyapprox.typing.inverse.conjugate.gaussian import (
    DenseGaussianConjugatePosterior,
)
from pyapprox.typing.pde.time.explicit_steppers.heun import HeunResidual


def _conjugate_params(d, m):
    """Generate fixed conjugate problem parameters."""
    np.random.seed(42)
    H_np = np.random.randn(m, d) * 0.5
    Sigma_lik_np = 0.1 * np.eye(m)
    return H_np, Sigma_lik_np


def _build_conjugate_setup(bkd, d, m, degree, n_per_dim=6):
    """Build paired conditional flow matching for Gaussian conjugate.

    Uses quadrature over (t, z, y) where z ~ N(0,I) and y ~ N(0,I).
    Paired coupling: x0 = z, x1 = L_post @ z + mu_post(y).

    Returns vf, path, loss, quad_data, conjugate_solver, test_y.
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
    # Use a dummy observation to compute Sigma_post
    dummy_y = bkd.array(np.zeros((m, 1)).tolist())
    conjugate.compute(dummy_y)
    Sigma_post_np = bkd.to_numpy(conjugate.posterior_covariance())
    L_post_np = np.linalg.cholesky(Sigma_post_np)

    # Quadrature over (t, z_1..z_d, y_1..y_m)
    # t ~ Uniform(0,1), z ~ N(0,1)^d, y ~ N(0,1)^m
    quad_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    quad_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d   # z
    quad_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * m   # y
    quad_bases_1d = create_bases_1d(quad_marginals, bkd)
    quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
    nvars = 1 + d + m
    quad_pts, quad_wts = quad_basis.tensor_product_quadrature(
        [n_per_dim] * nvars
    )

    t_all = quad_pts[0:1, :]           # (1, n_quad)
    z_all = quad_pts[1:1 + d, :]       # (d, n_quad)
    y_all = quad_pts[1 + d:, :]        # (m, n_quad)

    # Paired coupling: x0 = z, x1 = L_post @ z + mu_post(y)
    # mu_post(y) is linear in y; compute it for each quad point
    n_quad = quad_pts.shape[1]
    y_all_np = bkd.to_numpy(y_all)
    z_all_np = bkd.to_numpy(z_all)

    x1_np = np.zeros((d, n_quad))
    for i in range(n_quad):
        y_i = bkd.array(y_all_np[:, i:i + 1].tolist())
        conjugate.compute(y_i)
        mu_post_np = bkd.to_numpy(conjugate.posterior_mean())
        x1_np[:, i:i + 1] = L_post_np @ z_all_np[:, i:i + 1] + mu_post_np

    x0_all = z_all
    x1_all = bkd.array(x1_np.tolist())
    c_all = y_all

    # VF basis: input = (t, x, y) in R^{1+d+m}, output = R^d
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
        t=t_all, x0=x0_all, x1=x1_all,
        weights=quad_wts, bkd=bkd, c=c_all,
    )

    # Test observation
    np.random.seed(999)
    test_y_np = H_np @ np.random.randn(d, 1) + np.random.randn(m, 1) * 0.1
    test_y = bkd.array(test_y_np.tolist())

    return vf, path, loss, quad_data, conjugate, test_y


class TestGaussianConjugate(Generic[Array], ParametrizedTestCase,
                            unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @parametrize("d,m", [(1, 1), (2, 1)])
    def test_loss_decreases_with_degree(self, d: int, m: int) -> None:
        """Training loss decreases monotonically; degree-4 loss < 1e-4."""
        bkd = self._bkd
        degrees = [1, 2, 3, 4]
        losses = []
        for deg in degrees:
            vf, path, loss, qd, _, _ = _build_conjugate_setup(
                bkd, d, m, deg,
            )
            result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
            losses.append(result.training_loss())

        for i in range(len(losses) - 1):
            self.assertGreater(
                losses[i], losses[i + 1],
                f"Loss did not decrease from degree {degrees[i]} "
                f"({losses[i]:.2e}) to {degrees[i+1]} "
                f"({losses[i+1]:.2e})",
            )
        self.assertLess(losses[-1], 1e-4)

    @parametrize("d,m", [(1, 1)])
    def test_fitters_agree(self, d: int, m: int) -> None:
        """Both fitters achieve similar loss."""
        bkd = self._bkd
        for deg in [2, 4]:
            vf, path, loss, qd, _, _ = _build_conjugate_setup(
                bkd, d, m, deg,
            )
            lstsq_loss = LeastSquaresFitter(bkd).fit(
                vf, path, loss, qd
            ).training_loss()
            opt_loss = OptimizerFitter(bkd).fit(
                vf, path, loss, qd
            ).training_loss()
            self.assertLess(opt_loss, max(lstsq_loss * 100, 1e-4))

    @parametrize("d,m", [(1, 1)])
    def test_posterior_mean(self, d: int, m: int) -> None:
        """ODE-integrated samples approximate posterior mean."""
        bkd = self._bkd
        deg = 4
        vf, path, loss, qd, conjugate, test_y = _build_conjugate_setup(
            bkd, d, m, deg,
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        fitted_vf = result.surrogate()

        conjugate.compute(test_y)
        mu_post = conjugate.posterior_mean()

        np.random.seed(456)
        nsamples = 5000
        x0_np = np.random.randn(d, nsamples)
        x0_samples = bkd.array(x0_np.tolist())

        test_y_np = bkd.to_numpy(test_y)
        c_np = np.tile(test_y_np, (1, nsamples))
        c_samples = bkd.array(c_np.tolist())

        x1_samples = integrate_flow(
            fitted_vf, x0_samples, 0.0, 1.0, n_steps=50, bkd=bkd,
            c=c_samples, stepper_cls=HeunResidual,
        )

        x1_np = bkd.to_numpy(x1_samples)
        sample_mean = np.mean(x1_np, axis=1, keepdims=True)
        bkd.assert_allclose(
            bkd.array(sample_mean.tolist()), mu_post, atol=0.05,
        )

    @parametrize("d,m", [(1, 1)])
    @slow_test
    def test_posterior_covariance(self, d: int, m: int) -> None:
        """ODE-integrated samples approximate posterior covariance."""
        bkd = self._bkd
        deg = 4
        vf, path, loss, qd, conjugate, test_y = _build_conjugate_setup(
            bkd, d, m, deg,
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        fitted_vf = result.surrogate()

        conjugate.compute(test_y)
        Sigma_post = conjugate.posterior_covariance()

        np.random.seed(456)
        nsamples = 5000
        x0_np = np.random.randn(d, nsamples)
        x0_samples = bkd.array(x0_np.tolist())

        test_y_np = bkd.to_numpy(test_y)
        c_np = np.tile(test_y_np, (1, nsamples))
        c_samples = bkd.array(c_np.tolist())

        x1_samples = integrate_flow(
            fitted_vf, x0_samples, 0.0, 1.0, n_steps=50, bkd=bkd,
            c=c_samples, stepper_cls=HeunResidual,
        )

        x1_np = bkd.to_numpy(x1_samples)
        sample_cov = np.cov(x1_np)
        if d == 1:
            sample_cov = np.array([[sample_cov]])
        Sigma_post_np = bkd.to_numpy(Sigma_post)
        bkd.assert_allclose(
            bkd.array(sample_cov.tolist()),
            bkd.array(Sigma_post_np.tolist()),
            atol=0.1,
        )


class TestGaussianConjugateNumpy(TestGaussianConjugate[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianConjugateTorch(TestGaussianConjugate[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

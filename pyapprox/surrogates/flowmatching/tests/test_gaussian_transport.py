"""Integration test: unconditional Gaussian transport.

Source N(0, I_d) -> Target N(mu, Sigma) via paired coupling x1 = L*x0 + mu.

Because the coupling is linear, the map x0 -> x_t is invertible for all
t in (0,1), so the optimal VF is a deterministic function of (t, x_t)
with zero conditional variance. The VF is rational in t; a polynomial
approximation converges rapidly to zero training loss.
"""

import numpy as np
import pytest

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
from pyapprox.surrogates.flowmatching.fitters.optimizer import (
    OptimizerFitter,
)
from pyapprox.surrogates.flowmatching.linear_path import LinearPath
from pyapprox.surrogates.flowmatching.ode_adapter import (
    integrate_flow,
)
from pyapprox.surrogates.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.util.test_utils import slow_test


def _gaussian_params(d):
    """Generate fixed Gaussian transport parameters for dimension d."""
    np.random.seed(42)
    mu_np = np.random.randn(d, 1) * 0.5
    A_np = np.random.randn(d, d) * 0.2
    Sigma_np = A_np @ A_np.T + 0.5 * np.eye(d)
    L_np = np.linalg.cholesky(Sigma_np)
    return mu_np, L_np


def _build_transport_setup(bkd, d, degree, n_per_dim=6):
    """Build VF, path, loss, quad_data for N(0,I) -> N(mu, Sigma).

    Uses paired coupling x1 = L @ x0 + mu with Gauss quadrature
    over (t, x0).

    Returns vf, path, loss, quad_data, mu, L.
    """
    mu_np, L_np = _gaussian_params(d)
    mu = bkd.array(mu_np.tolist())
    L = bkd.array(L_np.tolist())

    vf_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    vf_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    vf_bases_1d = create_bases_1d(vf_marginals, bkd)
    indices = compute_hyperbolic_indices(1 + d, degree, 1.0, bkd)
    vf_basis = OrthonormalPolynomialBasis(vf_bases_1d, bkd, indices)
    vf = BasisExpansion(vf_basis, bkd, nqoi=d)

    quad_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    quad_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    quad_bases_1d = create_bases_1d(quad_marginals, bkd)
    quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
    quad_pts, quad_wts = quad_basis.tensor_product_quadrature([n_per_dim] * (1 + d))

    t_all = quad_pts[0:1, :]
    z0_all = quad_pts[1 : 1 + d, :]
    x1_all = L @ z0_all + mu

    path = LinearPath(bkd)
    loss = CFMLoss(bkd)
    quad_data = FlowMatchingQuadData(
        t=t_all,
        x0=z0_all,
        x1=x1_all,
        weights=quad_wts,
        bkd=bkd,
    )
    return vf, path, loss, quad_data, mu, L


class TestGaussianTransport:
    @pytest.mark.parametrize("d", [1, 2])
    @slow_test
    def test_loss_decreases_with_degree(self, bkd, d: int) -> None:
        """Training loss decreases monotonically; degree-4 loss < 1e-4."""
        degrees = [1, 2, 3, 4]
        losses = []
        for deg in degrees:
            vf, path, loss, qd, _, _ = _build_transport_setup(bkd, d, deg)
            result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
            losses.append(result.training_loss())

        for i in range(len(losses) - 1):
            assert losses[i] > losses[i + 1], (
                f"Loss did not decrease from degree {degrees[i]} "
                f"({losses[i]:.2e}) to {degrees[i + 1]} "
                f"({losses[i + 1]:.2e})"
            )
        assert losses[-1] < 1e-4

    @pytest.mark.parametrize("d", [1])
    def test_fitters_agree(self, bkd, d: int) -> None:
        """Both fitters achieve similar loss at each degree."""
        for deg in [2, 4]:
            vf, path, loss, qd, _, _ = _build_transport_setup(bkd, d, deg)
            lstsq_loss = LeastSquaresFitter(bkd).fit(vf, path, loss, qd).training_loss()
            opt_loss = OptimizerFitter(bkd).fit(vf, path, loss, qd).training_loss()
            assert opt_loss < max(lstsq_loss * 100, 1e-4)

    @pytest.mark.parametrize("d", [1, 2])
    def test_target_moments(self, bkd, d: int) -> None:
        """ODE-integrated samples match target mean and cov."""
        vf, path, loss, qd, mu, L = _build_transport_setup(bkd, d, degree=4)
        fitted_vf = LeastSquaresFitter(bkd).fit(vf, path, loss, qd).surrogate()

        np.random.seed(123)
        nsamples = 5000
        x0_samples = bkd.array(np.random.randn(d, nsamples).tolist())
        x1_samples = integrate_flow(
            fitted_vf,
            x0_samples,
            0.0,
            1.0,
            n_steps=50,
            bkd=bkd,
            stepper_cls=HeunResidual,
        )

        x1_np = bkd.to_numpy(x1_samples)
        sample_mean = np.mean(x1_np, axis=1, keepdims=True)
        bkd.assert_allclose(
            bkd.array(sample_mean.tolist()),
            mu,
            atol=0.05,
        )

        L_np = bkd.to_numpy(L)
        Sigma_np = L_np @ L_np.T
        sample_cov = np.cov(x1_np)
        if d == 1:
            sample_cov = np.array([[sample_cov]])
        bkd.assert_allclose(
            bkd.array(sample_cov.tolist()),
            bkd.array(Sigma_np.tolist()),
            atol=0.1,
        )

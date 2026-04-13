"""Pipeline validation: exact recovery of a known polynomial VF.

Constructs synthetic quadrature data where the conditional target field
u_t IS a polynomial of known degree in (t, x_t), and verifies the fitter
recovers it exactly. This tests the least-squares solve, weight handling,
and objective gradient without depending on any flow matching theory.

Strategy: for a linear path, given any (t, x_t) and u_t, the unique
(x0, x1) pair satisfying x_t = (1-t)*x0 + t*x1 and u_t = x1 - x0 is:
    x0 = x_t - t * u_t
    x1 = x_t + (1 - t) * u_t

So we generate (t, x_t [, c]) on a quadrature grid, evaluate a known
polynomial true_vf to get u_t = true_vf(t, x_t [, c]), reverse-engineer
(x0, x1), and verify the fitter recovers true_vf exactly.
"""

import numpy as np
import pytest

from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.generative.flowmatching.cfm_loss import CFMLoss
from pyapprox.generative.flowmatching.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.generative.flowmatching.fitters.optimizer import (
    OptimizerFitter,
)
from pyapprox.generative.flowmatching.linear_path import LinearPath
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from tests._helpers.markers import slow_test


def _make_vf(bkd, d, degree, m=0):
    """Create a BasisExpansion VF with input_dim = 1+d+m, nqoi = d."""
    marginals = [UniformMarginal(0.0, 1.0, bkd)]
    marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    marginals += [GaussianMarginal(0.0, 1.0, bkd)] * m
    bases_1d = create_bases_1d(marginals, bkd)
    nvars = 1 + d + m
    indices = compute_hyperbolic_indices(nvars, degree, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=d)


def _build_polynomial_vf_setup(bkd, d, degree, n_per_dim=6, m=0):
    """Build data where u_t is a known polynomial in (t, x_t [, c]).

    Returns vf, path, loss, quad_data, true_coef.
    """
    np.random.seed(42)

    # VF to be fitted (starts with zero coefficients)
    vf = _make_vf(bkd, d, degree, m=m)

    # "True" VF with known random coefficients
    true_vf = _make_vf(bkd, d, degree, m=m)
    nterms = true_vf.nterms()
    true_coef_np = np.random.randn(nterms, d) * 0.5
    true_coef = bkd.array(true_coef_np.tolist())
    true_vf.set_coefficients(true_coef)

    # Quadrature directly over the VF input space (t, x_t [, c])
    nvars_total = 1 + d + m
    quad_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    quad_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    quad_marginals += [GaussianMarginal(0.0, 1.0, bkd)] * m
    quad_bases_1d = create_bases_1d(quad_marginals, bkd)
    quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
    npts = [n_per_dim] * nvars_total
    quad_pts, quad_wts = quad_basis.tensor_product_quadrature(npts)
    # quad_pts: (nvars_total, n_quad), quad_wts: (n_quad,)

    t_all = quad_pts[0:1, :]  # (1, n_quad)
    x_t_all = quad_pts[1 : 1 + d, :]  # (d, n_quad)
    c_all = quad_pts[1 + d :, :] if m > 0 else None  # (m, n_quad) or None

    # Evaluate true VF: u_t = true_vf(t, x_t [, c])
    u_t = true_vf(quad_pts)  # (d, n_quad)

    # Reverse-engineer (x0, x1) from (t, x_t, u_t)
    one_minus_t = bkd.ones_like(t_all) - t_all
    x0_all = x_t_all - t_all * u_t
    x1_all = x_t_all + one_minus_t * u_t

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

    return vf, path, loss, quad_data, true_coef


class TestPolynomialVFRecovery:
    @pytest.mark.parametrize("d,degree", [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)])
    def test_lstsq_exact_recovery(self, bkd, d: int, degree: int) -> None:
        """Lstsq should exactly recover a polynomial VF of matching degree."""
        vf, path, loss, qd, true_coef = _build_polynomial_vf_setup(
            bkd,
            d,
            degree,
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        assert result.training_loss() < 1e-10

        fitted_coef = result.surrogate().get_coefficients()  # type: ignore
        bkd.assert_allclose(fitted_coef, true_coef, atol=1e-8)

    @pytest.mark.parametrize("d,degree", [(1, 1), (1, 2)])
    def test_optimizer_exact_recovery(self, bkd, d: int, degree: int) -> None:
        """Optimizer should also achieve near-zero loss."""
        vf, path, loss, qd, _ = _build_polynomial_vf_setup(bkd, d, degree)
        result = OptimizerFitter(bkd).fit(vf, path, loss, qd)
        assert result.training_loss() < 1e-6

    @pytest.mark.parametrize("d,degree", [(1, 2), (2, 1)])
    def test_fitter_agreement(self, bkd, d: int, degree: int) -> None:
        """Both fitters should produce similar coefficients."""
        vf, path, loss, qd, true_coef = _build_polynomial_vf_setup(
            bkd,
            d,
            degree,
        )

        lstsq_result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        opt_result = OptimizerFitter(bkd).fit(vf, path, loss, qd)

        assert lstsq_result.training_loss() < 1e-10
        assert opt_result.training_loss() < 1e-6

        lstsq_coef = lstsq_result.surrogate().get_coefficients()  # type: ignore
        opt_coef = opt_result.surrogate().get_coefficients()  # type: ignore
        bkd.assert_allclose(lstsq_coef, opt_coef, atol=1e-4)

    @pytest.mark.parametrize("d", [1, 2])
    @slow_test
    def test_with_conditioning(self, bkd, d: int) -> None:
        """Recovery should work with conditioning variables present."""
        vf, path, loss, qd, true_coef = _build_polynomial_vf_setup(
            bkd,
            d,
            degree=1,
            m=1,
        )
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        assert result.training_loss() < 1e-10

        fitted_coef = result.surrogate().get_coefficients()  # type: ignore
        bkd.assert_allclose(fitted_coef, true_coef, atol=1e-8)

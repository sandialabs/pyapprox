"""Integration test: unconditional chi-squared transport.

Source x0 ~ N(0, 1), Target x1 = x0^2 ~ chi^2(1) via paired coupling.

Because the coupling x1 = x0^2 is nonlinear, the map x0 -> x_t is
2-to-1 (not invertible) for most t, creating irreducible conditional
variance in the CFM loss.  The VF polynomial approximation still
converges, but the loss does not reach machine precision.  Tests
verify monotone loss decrease with degree and accuracy of ODE-integrated
moments at moderate polynomial degree.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import slow_test

from pyapprox.probability import UniformMarginal, GaussianMarginal
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion

from pyapprox.surrogates.flowmatching.linear_path import LinearPath
from pyapprox.surrogates.flowmatching.cfm_loss import CFMLoss
from pyapprox.surrogates.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.surrogates.flowmatching.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.flowmatching.ode_adapter import (
    integrate_flow,
)
from pyapprox.pde.time.explicit_steppers.heun import HeunResidual


def _build_chi_squared_setup(bkd, degree, n_per_dim=8):
    """Build paired flow matching for N(0,1) -> chi^2(1).

    Paired coupling: x0 ~ N(0,1), x1 = x0^2.
    Quadrature over (t, x0) with Gauss rules.

    Returns vf, path, loss, quad_data.
    """
    d = 1

    # VF basis: input = (t, x) in R^2, output = 1
    vf_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    vf_marginals += [GaussianMarginal(0.0, 1.0, bkd)]
    vf_bases_1d = create_bases_1d(vf_marginals, bkd)
    indices = compute_hyperbolic_indices(2, degree, 1.0, bkd)
    vf_basis = OrthonormalPolynomialBasis(vf_bases_1d, bkd, indices)
    vf = BasisExpansion(vf_basis, bkd, nqoi=d)

    # Quadrature over (t, x0)
    quad_marginals = [UniformMarginal(0.0, 1.0, bkd)]
    quad_marginals += [GaussianMarginal(0.0, 1.0, bkd)]
    quad_bases_1d = create_bases_1d(quad_marginals, bkd)
    quad_basis = OrthonormalPolynomialBasis(quad_bases_1d, bkd)
    quad_pts, quad_wts = quad_basis.tensor_product_quadrature(
        [n_per_dim] * 2
    )

    t_all = quad_pts[0:1, :]    # (1, n_quad)
    z0_all = quad_pts[1:2, :]   # (1, n_quad)
    x1_all = z0_all * z0_all    # chi^2(1): x1 = x0^2

    path = LinearPath(bkd)
    loss = CFMLoss(bkd)
    quad_data = FlowMatchingQuadData(
        t=t_all, x0=z0_all, x1=x1_all,
        weights=quad_wts, bkd=bkd,
    )

    return vf, path, loss, quad_data


class TestChiSquared(Generic[Array], ParametrizedTestCase,
                     unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_loss_decreases_with_degree(self) -> None:
        """Training loss should decrease with polynomial degree."""
        bkd = self._bkd
        degrees = [2, 3, 4, 5, 6]
        losses = []
        for deg in degrees:
            vf, path, loss, qd = _build_chi_squared_setup(bkd, deg)
            result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
            losses.append(result.training_loss())

        for i in range(len(losses) - 1):
            self.assertGreaterEqual(
                losses[i], losses[i + 1],
                f"Loss did not decrease from degree {degrees[i]} "
                f"({losses[i]:.2e}) to degree {degrees[i+1]} "
                f"({losses[i+1]:.2e})",
            )

    @slow_test
    def test_chi_squared_moments(self) -> None:
        """ODE-integrated samples should match chi^2(1) moments."""
        bkd = self._bkd
        deg = 4
        vf, path, loss, qd = _build_chi_squared_setup(bkd, deg)
        result = LeastSquaresFitter(bkd).fit(vf, path, loss, qd)
        fitted_vf = result.surrogate()

        np.random.seed(789)
        nsamples = 5000
        x0_np = np.random.randn(1, nsamples)
        x0_samples = bkd.array(x0_np.tolist())

        x1_samples = integrate_flow(
            fitted_vf, x0_samples, 0.0, 1.0, n_steps=50, bkd=bkd,
            stepper_cls=HeunResidual,
        )

        x1_np = bkd.to_numpy(x1_samples)
        sample_mean = float(np.mean(x1_np))
        sample_var = float(np.var(x1_np))

        # chi^2(1): E[X] = 1, Var[X] = 2
        self.assertAlmostEqual(sample_mean, 1.0, delta=0.15)
        self.assertAlmostEqual(sample_var, 2.0, delta=0.5)


class TestChiSquaredNumpy(TestChiSquared[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestChiSquaredTorch(TestChiSquared[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.util.test_utils import load_tests  # noqa: F401

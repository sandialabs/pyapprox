"""Tests for distribution marginalization via quadrature.

Verifies that marginalizing copula and independent distributions
recovers the correct lower-dimensional marginals.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.marginalize import (
    FunctionMarginalizer,
)
from pyapprox.probability.copula.correlation.cholesky import (
    CholeskyCorrelationParameterization,
)
from pyapprox.probability.copula.distribution import (
    CopulaDistribution,
)
from pyapprox.probability.copula.gaussian import GaussianCopula
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.beta import BetaMarginal
from pyapprox.surrogates.quadrature.tensor_product_factory import (
    TensorProductQuadratureFactory,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import (
    load_tests,  # noqa: F401
    slow_test,
)


def _make_3d_copula_distribution(bkd):
    """Create a 3D Gaussian copula distribution with Beta marginals."""
    beta_0 = BetaMarginal(2.0, 6.0, bkd, lb=0.0, ub=1.0)
    beta_1 = BetaMarginal(6.0, 2.0, bkd, lb=0.0, ub=1.0)
    beta_2 = BetaMarginal(3.0, 5.0, bkd, lb=0.0, ub=1.0)
    marginals = [beta_0, beta_1, beta_2]

    Sigma = np.array(
        [
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]
    )
    L = np.linalg.cholesky(Sigma)
    chol_params = bkd.asarray(np.array([L[1, 0], L[2, 0], L[2, 1]]))
    corr_param = CholeskyCorrelationParameterization(chol_params, nvars=3, bkd=bkd)
    copula = GaussianCopula(corr_param, bkd)
    return CopulaDistribution(copula, marginals, bkd), marginals, Sigma


def _make_quadrature_factory(domain, bkd, npoints=20):
    """Create a Gauss-Legendre factory for the given domain."""
    nvars = domain.shape[0]
    return TensorProductQuadratureFactory([npoints] * nvars, domain, bkd)


class TestMarginalizeDistributions(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    @slow_test
    def test_copula_1d_marginal_recovery(self) -> None:
        """1D marginal of copula distribution = original marginal PDF.

        The copula does not change marginals, so integrating out the
        other variables should recover each original marginal exactly.
        Absolute tolerance is used because at tail points the PDF is
        tiny and relative error from 2D quadrature is large.
        """
        bkd = self._bkd
        dist, marginals, _ = _make_3d_copula_distribution(bkd)
        factory = _make_quadrature_factory(dist.domain(), bkd, npoints=40)
        marginalizer = FunctionMarginalizer(dist, factory, bkd)

        test_pts = bkd.asarray(np.linspace(0.1, 0.9, 10).reshape(1, -1))

        for var_idx in range(3):
            marg_fn = marginalizer.marginalize([var_idx])
            result = marg_fn(test_pts)
            expected = marginals[var_idx].pdf(test_pts)
            bkd.assert_allclose(result, expected, atol=1e-5)

    @slow_test
    def test_copula_2d_marginal_recovery(self) -> None:
        """2D marginal of 3D copula matches 2D copula with sub-correlation.

        For a Gaussian copula, the 2D marginal over variables (i, j)
        has a bivariate Gaussian copula with correlation rho_{ij}.
        """
        bkd = self._bkd
        dist, marginals, Sigma = _make_3d_copula_distribution(bkd)
        factory = _make_quadrature_factory(dist.domain(), bkd, npoints=40)
        marginalizer = FunctionMarginalizer(dist, factory, bkd)

        # Keep variables 0 and 1
        marg_2d = marginalizer.marginalize([0, 1])

        # Build reference 2D copula distribution
        sub_Sigma = Sigma[:2, :2]
        L_sub = np.linalg.cholesky(sub_Sigma)
        chol_sub = bkd.asarray(np.array([L_sub[1, 0]]))
        corr_sub = CholeskyCorrelationParameterization(chol_sub, nvars=2, bkd=bkd)
        copula_2d = GaussianCopula(corr_sub, bkd)
        ref_dist = CopulaDistribution(copula_2d, [marginals[0], marginals[1]], bkd)

        # Compare on a grid
        x0 = np.linspace(0.05, 0.95, 8)
        x1 = np.linspace(0.05, 0.95, 8)
        xx0, xx1 = np.meshgrid(x0, x1)
        test_pts = bkd.asarray(np.vstack([xx0.ravel(), xx1.ravel()]))

        result = marg_2d(test_pts)
        expected = ref_dist.pdf(test_pts)
        bkd.assert_allclose(result, expected, rtol=1e-4)

    def test_independent_joint_marginal_via_quadrature(self) -> None:
        """IndependentJoint marginalized via quadrature matches product."""
        bkd = self._bkd
        beta_0 = BetaMarginal(2.0, 6.0, bkd, lb=0.0, ub=1.0)
        beta_1 = BetaMarginal(6.0, 2.0, bkd, lb=0.0, ub=1.0)
        beta_2 = BetaMarginal(3.0, 5.0, bkd, lb=0.0, ub=1.0)
        joint = IndependentJoint([beta_0, beta_1, beta_2], bkd)

        factory = _make_quadrature_factory(joint.domain(), bkd, npoints=10)
        marginalizer = FunctionMarginalizer(joint, factory, bkd)

        # 1D marginal
        test_1d = bkd.asarray(np.linspace(0.05, 0.95, 10).reshape(1, -1))
        marg_0 = marginalizer.marginalize([0])
        result = marg_0(test_1d)
        expected = beta_0.pdf(test_1d)
        bkd.assert_allclose(result, expected, rtol=1e-10)

        # 2D marginal = product of independent marginals
        x0 = np.linspace(0.1, 0.9, 5)
        x1 = np.linspace(0.1, 0.9, 5)
        xx0, xx1 = np.meshgrid(x0, x1)
        test_2d = bkd.asarray(np.vstack([xx0.ravel(), xx1.ravel()]))
        marg_01 = marginalizer.marginalize([0, 1])
        result = marg_01(test_2d)
        expected = beta_0.pdf(test_2d[0:1]) * beta_1.pdf(test_2d[1:2])
        bkd.assert_allclose(result, expected, rtol=1e-10)

    @slow_test
    def test_accuracy_scaling(self) -> None:
        """Error decreases as quadrature points increase."""
        bkd = self._bkd
        dist, marginals, _ = _make_3d_copula_distribution(bkd)
        test_pts = bkd.asarray(np.linspace(0.1, 0.9, 10).reshape(1, -1))
        expected = marginals[0].pdf(test_pts)

        errors = []
        for npts in [10, 20, 40]:
            factory = _make_quadrature_factory(dist.domain(), bkd, npoints=npts)
            marginalizer = FunctionMarginalizer(dist, factory, bkd)
            marg_0 = marginalizer.marginalize([0])
            result = marg_0(test_pts)
            diff = bkd.to_numpy(result - expected)
            errors.append(float(np.max(np.abs(diff))))

        # Each refinement should reduce error
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])


class TestMarginalizeDistributionsNumpy(TestMarginalizeDistributions[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMarginalizeDistributionsTorch(TestMarginalizeDistributions[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

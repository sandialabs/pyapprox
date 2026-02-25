"""Tests for GPMeanDimensionReducer."""

import math
import unittest
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.marginalize import (
    DimensionReducerProtocol,
    FunctionMarginalizer,
)
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.statistics import (
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.gaussianprocess.statistics.gp_dimension_reducer import (
    GPMeanDimensionReducer,
)
from pyapprox.surrogates.gaussianprocess.statistics.marginalization import (
    MarginalizedGP,
)
from pyapprox.surrogates.kernels.composition import (
    SeparableProductKernel,
)
from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


def _create_quadrature_bases(
    marginals: List[Any], nquad_points: int, bkd: Backend[Array]
) -> List[Any]:
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


class TestGPMeanDimensionReducer(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

        bkd = self._bkd

        # Create 3D GP with separable product kernel
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.8], (0.1, 10.0), 1, bkd)
        k3 = SquaredExponentialKernel([0.6], (0.1, 10.0), 1, bkd)
        self._kernel = SeparableProductKernel([k1, k2, k3], bkd)

        self._gp = ExactGaussianProcess(self._kernel, nvars=3, bkd=bkd, nugget=1e-6)
        self._gp.hyp_list().set_all_inactive()

        # Training data in [-1, 1]^3
        n_train = 15
        X_train_np = np.random.rand(3, n_train) * 2 - 1
        self._X_train = bkd.array(X_train_np)
        y_train = bkd.reshape(
            bkd.sin(math.pi * self._X_train[0, :])
            + 0.5 * bkd.cos(math.pi * self._X_train[1, :])
            + 0.3 * self._X_train[2, :],
            (1, -1),
        )
        self._gp.fit(self._X_train, y_train)

        # Marginals
        self._marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]

        # Quadrature bases for integral calculator
        nquad = 30
        self._bases = _create_quadrature_bases(self._marginals, nquad, bkd)

        # Integral calculator
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, self._bases, self._marginals, bkd=bkd
        )

        # Build reducer
        self._reducer = GPMeanDimensionReducer(self._gp, self._calc, bkd)

    def test_matches_marginalized_gp_directly(self) -> None:
        """Reducer output matches MarginalizedGP.predict_mean directly."""
        bkd = self._bkd
        np.random.seed(99)
        test_pts = bkd.asarray(np.random.uniform(-1, 1, (1, 5)))

        # Via reducer
        reduced = self._reducer.reduce([0])
        result = reduced(test_pts)  # (1, 5)

        # Via MarginalizedGP directly
        marg_gp = MarginalizedGP(self._gp, self._calc, active_dims=[0])
        expected = bkd.reshape(marg_gp.predict_mean(test_pts), (1, -1))  # (1, 5)

        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_matches_marginalized_gp_2d(self) -> None:
        """Reducer output matches MarginalizedGP for 2D keep."""
        bkd = self._bkd
        np.random.seed(77)
        test_pts = bkd.asarray(np.random.uniform(-1, 1, (2, 5)))

        # Via reducer
        result = self._reducer.reduce([0, 2])(test_pts)

        # Via MarginalizedGP directly
        marg_gp = MarginalizedGP(self._gp, self._calc, active_dims=[0, 2])
        expected = bkd.reshape(marg_gp.predict_mean(test_pts), (1, -1))

        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_equivalence_with_quadrature(self) -> None:
        """Reducer matches quadrature-based marginalization."""
        from pyapprox.surrogates.quadrature.probability_measure_factory import (
            ProbabilityMeasureQuadratureFactory,
        )

        bkd = self._bkd
        gp = self._gp

        # Build quadrature-based marginalizer using GP __call__
        quad_factory = ProbabilityMeasureQuadratureFactory(
            self._marginals, [15, 15, 15], bkd
        )
        marginalizer = FunctionMarginalizer(gp, quad_factory, bkd)

        np.random.seed(55)
        test_pts = bkd.asarray(np.random.uniform(-1, 1, (1, 5)))

        # Analytical (via MarginalizedGP)
        analytical = self._reducer.reduce([0])(test_pts)
        # Quadrature-based
        numerical = marginalizer.reduce([0])(test_pts)

        bkd.assert_allclose(analytical, numerical, rtol=1e-4)

    def test_protocol_compliance(self) -> None:
        """GPMeanDimensionReducer satisfies DimensionReducerProtocol."""
        self.assertIsInstance(self._reducer, DimensionReducerProtocol)

    def test_output_shapes(self) -> None:
        """All reductions produce (1, nsamples) output."""
        bkd = self._bkd
        np.random.seed(33)

        # 1D reduction
        pts_1d = bkd.asarray(np.random.uniform(-1, 1, (1, 4)))
        result_1d = self._reducer.reduce([0])(pts_1d)
        self.assertEqual(result_1d.shape, (1, 4))

        # 2D reduction
        pts_2d = bkd.asarray(np.random.uniform(-1, 1, (2, 4)))
        result_2d = self._reducer.reduce([0, 1])(pts_2d)
        self.assertEqual(result_2d.shape, (1, 4))

    def test_nvars_nqoi(self) -> None:
        """nvars and nqoi return correct values."""
        self.assertEqual(self._reducer.nvars(), 3)
        self.assertEqual(self._reducer.nqoi(), 1)


class TestGPMeanDimensionReducerNumpy(TestGPMeanDimensionReducer[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGPMeanDimensionReducerTorch(TestGPMeanDimensionReducer[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

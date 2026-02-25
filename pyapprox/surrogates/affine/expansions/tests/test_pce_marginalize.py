"""Tests for PCEDimensionReducer."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.marginalize import (
    DimensionReducerProtocol,
    FunctionMarginalizer,
)
from pyapprox.surrogates.affine.expansions.pce_marginalize import (
    PCEDimensionReducer,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestPCEDimensionReducer(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _build_3d_pce(self, nqoi: int = 1):
        """Build a 3D PCE for f(x,y,z) = 1 + x + 2y + 3z + xy + xz + yz + xyz.

        Uses Uniform[-1,1]^3 marginals and projection fitting (exact
        for this polynomial with sufficient quadrature points).
        """
        from pyapprox.probability.univariate.uniform import (
            UniformMarginal,
        )
        from pyapprox.surrogates.affine.expansions.pce import (
            create_pce_from_marginals,
        )

        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]

        pce = create_pce_from_marginals(marginals, 3, bkd, nqoi=nqoi)

        # Build quadrature for projection
        from pyapprox.surrogates.quadrature import (
            TensorProductQuadratureRule,
            gauss_quadrature_rule,
        )

        nquad = 5
        rules = [lambda n, m=m: gauss_quadrature_rule(m, n, bkd) for m in marginals]
        quad = TensorProductQuadratureRule(bkd, rules, [nquad] * 3)
        pts, wts = quad()

        # f(x,y,z) = 1 + x + 2y + 3z + xy + xz + yz + xyz
        x, y, z = pts[0], pts[1], pts[2]
        f_vals = 1.0 + x + 2.0 * y + 3.0 * z + x * y + x * z + y * z + x * y * z

        if nqoi == 1:
            values = bkd.reshape(f_vals, (1, -1))
        else:
            # nqoi=2: second QoI is 2*f
            values = bkd.stack([f_vals, 2.0 * f_vals], axis=0)

        pce.fit_via_projection(pts, values, wts)
        return pce, marginals

    def test_reduce_pce_keep_01(self) -> None:
        """reduce_pce([0,1]) should give 1 + x + 2y + xy."""
        bkd = self._bkd
        pce, _ = self._build_3d_pce()
        reducer = PCEDimensionReducer(pce, bkd)

        sub_pce = reducer.reduce_pce([0, 1])
        self.assertEqual(sub_pce.nvars(), 2)

        # Evaluate at test points
        test_pts = bkd.asarray([[0.5, -0.3], [0.2, 0.7]])
        result = sub_pce(test_pts)  # (1, 2)

        # Expected: 1 + x + 2y + xy
        x, y = 0.5, 0.2
        expected0 = 1.0 + x + 2.0 * y + x * y
        x, y = -0.3, 0.7
        expected1 = 1.0 + x + 2.0 * y + x * y
        expected = bkd.asarray([[expected0, expected1]])
        bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_reduce_pce_keep_0(self) -> None:
        """reduce_pce([0]) should give 1 + x."""
        bkd = self._bkd
        pce, _ = self._build_3d_pce()
        reducer = PCEDimensionReducer(pce, bkd)

        sub_pce = reducer.reduce_pce([0])
        self.assertEqual(sub_pce.nvars(), 1)

        test_pts = bkd.asarray([[0.5, -0.3, 0.8]])
        result = sub_pce(test_pts)  # (1, 3)

        expected = bkd.asarray([[1.0 + 0.5, 1.0 - 0.3, 1.0 + 0.8]])
        bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_reduce_pce_keep_02(self) -> None:
        """reduce_pce([0,2]) should give 1 + x + 3z + xz."""
        bkd = self._bkd
        pce, _ = self._build_3d_pce()
        reducer = PCEDimensionReducer(pce, bkd)

        sub_pce = reducer.reduce_pce([0, 2])
        self.assertEqual(sub_pce.nvars(), 2)

        test_pts = bkd.asarray([[0.4], [-0.6]])
        result = sub_pce(test_pts)

        x, z = 0.4, -0.6
        expected = bkd.asarray([[1.0 + x + 3.0 * z + x * z]])
        bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_reduce_pce_keep_all(self) -> None:
        """reduce_pce([0,1,2]) should match original PCE."""
        bkd = self._bkd
        pce, _ = self._build_3d_pce()
        reducer = PCEDimensionReducer(pce, bkd)

        sub_pce = reducer.reduce_pce([0, 1, 2])

        np.random.seed(42)
        test_pts = bkd.asarray(np.random.uniform(-1, 1, (3, 5)))
        bkd.assert_allclose(sub_pce(test_pts), pce(test_pts), rtol=1e-12)

    def test_mean_variance_of_marginalized_pce(self) -> None:
        """Mean and variance of marginalized PCE match analytical values."""
        bkd = self._bkd
        pce, _ = self._build_3d_pce()
        reducer = PCEDimensionReducer(pce, bkd)

        # reduce_pce([0]) gives g(x) = 1 + x
        sub_pce_0 = reducer.reduce_pce([0])
        # E[g] = E[1 + x] = 1 (for Uniform[-1,1])
        bkd.assert_allclose(sub_pce_0.mean(), bkd.asarray([1.0]), rtol=1e-10)
        # Var[g] = Var[x] = 1/3 for Uniform[-1,1]
        bkd.assert_allclose(sub_pce_0.variance(), bkd.asarray([1.0 / 3.0]), rtol=1e-10)

        # reduce_pce([0,1]) gives h(x,y) = 1 + x + 2y + xy
        sub_pce_01 = reducer.reduce_pce([0, 1])
        # E[h] = 1
        bkd.assert_allclose(sub_pce_01.mean(), bkd.asarray([1.0]), rtol=1e-10)
        # Var[h] = Var[x] + 4*Var[y] + Var[xy]
        # = 1/3 + 4/3 + 1/9 = 16/9
        bkd.assert_allclose(
            sub_pce_01.variance(), bkd.asarray([16.0 / 9.0]), rtol=1e-10
        )

    def test_equivalence_with_quadrature(self) -> None:
        """PCE analytical marginalization matches quadrature-based."""
        from pyapprox.surrogates.quadrature.probability_measure_factory import (
            ProbabilityMeasureQuadratureFactory,
        )

        bkd = self._bkd
        pce, marginals = self._build_3d_pce()
        reducer = PCEDimensionReducer(pce, bkd)

        # Build quadrature-based marginalizer
        quad_factory = ProbabilityMeasureQuadratureFactory(marginals, [10, 10, 10], bkd)
        marginalizer = FunctionMarginalizer(pce, quad_factory, bkd)

        np.random.seed(123)
        test_pts_1d = bkd.asarray(np.random.uniform(-1, 1, (1, 7)))
        test_pts_2d = bkd.asarray(np.random.uniform(-1, 1, (2, 7)))

        # Compare keep=[0]
        analytical = reducer.reduce([0])(test_pts_1d)
        numerical = marginalizer.reduce([0])(test_pts_1d)
        bkd.assert_allclose(analytical, numerical, rtol=1e-10)

        # Compare keep=[0,1]
        analytical = reducer.reduce([0, 1])(test_pts_2d)
        numerical = marginalizer.reduce([0, 1])(test_pts_2d)
        bkd.assert_allclose(analytical, numerical, rtol=1e-10)

        # Compare keep=[0,2]
        analytical = reducer.reduce([0, 2])(test_pts_2d)
        numerical = marginalizer.reduce([0, 2])(test_pts_2d)
        bkd.assert_allclose(analytical, numerical, rtol=1e-10)

    def test_multi_qoi(self) -> None:
        """PCE with nqoi=2 produces correct shapes and per-QoI values."""
        bkd = self._bkd
        pce, _ = self._build_3d_pce(nqoi=2)
        reducer = PCEDimensionReducer(pce, bkd)

        self.assertEqual(reducer.nqoi(), 2)

        sub_pce = reducer.reduce_pce([0])
        self.assertEqual(sub_pce.nqoi(), 2)
        self.assertEqual(sub_pce.nvars(), 1)

        test_pts = bkd.asarray([[0.5, -0.3]])
        result = sub_pce(test_pts)  # (2, 2)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 2)

        # QoI 0: 1 + x, QoI 1: 2*(1 + x)
        expected_q0 = bkd.asarray([[1.5, 0.7]])
        expected_q1 = bkd.asarray([[3.0, 1.4]])
        bkd.assert_allclose(result[0:1, :], expected_q0, rtol=1e-10)
        bkd.assert_allclose(result[1:2, :], expected_q1, rtol=1e-10)

    def test_protocol_compliance(self) -> None:
        """PCEDimensionReducer satisfies DimensionReducerProtocol."""
        bkd = self._bkd
        pce, _ = self._build_3d_pce()
        reducer = PCEDimensionReducer(pce, bkd)
        self.assertIsInstance(reducer, DimensionReducerProtocol)


class TestPCEDimensionReducerNumpy(TestPCEDimensionReducer[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPCEDimensionReducerTorch(TestPCEDimensionReducer[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

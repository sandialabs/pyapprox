"""Tests for PointEvaluationFunctional."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.typing.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)
from pyapprox.typing.optimization.implicitfunction.functionals.point_evaluation import (
    PointEvaluationFunctional,
)
from pyapprox.typing.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)


class TestPointEvaluationFunctional(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._npts = 15
        self._length = 2.0
        self._nparams = 3
        transform = AffineTransform1D((0.0, self._length), self._bkd)
        mesh = TransformedMesh1D(self._npts, self._bkd, transform)
        self._basis = ChebyshevBasis1D(mesh, self._bkd)
        self._phys_pts = mesh.points()[0, :]  # shape (npts,)

    def test_polynomial_exactness(self) -> None:
        """Evaluating a polynomial at x* via interpolation is exact."""
        bkd = self._bkd
        eval_point = 0.7
        func = PointEvaluationFunctional(
            self._basis, eval_point, self._nparams, bkd,
        )
        # u(x) = x^3 at collocation nodes
        state = bkd.reshape(self._phys_pts ** 3, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[eval_point ** 3]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_boundary_evaluation_left(self) -> None:
        """Evaluation at x=0 recovers state at left boundary node."""
        bkd = self._bkd
        func = PointEvaluationFunctional(
            self._basis, 0.0, self._nparams, bkd,
        )
        np.random.seed(42)
        state_vals = bkd.array(np.random.randn(self._npts))
        state = bkd.reshape(state_vals, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        # Left boundary is first node (CGL nodes ordered left to right)
        expected = bkd.reshape(state_vals[0:1], (1, 1))
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_tip_evaluation(self) -> None:
        """Evaluation at x=L recovers state at right boundary node."""
        bkd = self._bkd
        func = PointEvaluationFunctional(
            self._basis, self._length, self._nparams, bkd,
        )
        np.random.seed(42)
        state_vals = bkd.array(np.random.randn(self._npts))
        state = bkd.reshape(state_vals, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.reshape(state_vals[-1:], (1, 1))
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_constant_state_jacobian(self) -> None:
        """State Jacobian is independent of the state vector (linear)."""
        bkd = self._bkd
        func = PointEvaluationFunctional(
            self._basis, 0.7, self._nparams, bkd,
        )
        param = bkd.zeros((self._nparams, 1))
        state1 = bkd.zeros((self._npts, 1))
        state2 = bkd.ones((self._npts, 1))
        jac1 = func.state_jacobian(state1, param)
        jac2 = func.state_jacobian(state2, param)
        bkd.assert_allclose(jac1, jac2, atol=1e-15)

    def test_state_jacobian_shape(self) -> None:
        """State Jacobian has correct shape (1, nstates)."""
        bkd = self._bkd
        func = PointEvaluationFunctional(
            self._basis, 0.5, self._nparams, bkd,
        )
        param = bkd.zeros((self._nparams, 1))
        state = bkd.zeros((self._npts, 1))
        jac = func.state_jacobian(state, param)
        self.assertEqual(jac.shape, (1, self._npts))

    def test_nqoi_is_one(self) -> None:
        """Point evaluation returns a scalar QoI."""
        bkd = self._bkd
        func = PointEvaluationFunctional(
            self._basis, 0.5, self._nparams, bkd,
        )
        self.assertEqual(func.nqoi(), 1)

    def test_param_jacobian_is_zero(self) -> None:
        """Functional does not depend on parameters."""
        bkd = self._bkd
        func = PointEvaluationFunctional(
            self._basis, 0.5, self._nparams, bkd,
        )
        state = bkd.ones((self._npts, 1))
        param = bkd.ones((self._nparams, 1))
        jac = func.param_jacobian(state, param)
        expected = bkd.zeros((1, self._nparams))
        bkd.assert_allclose(jac, expected)

    def test_protocol_compliance(self) -> None:
        """Satisfies ParameterizedFunctionalWithJacobianProtocol."""
        bkd = self._bkd
        func = PointEvaluationFunctional(
            self._basis, 0.5, self._nparams, bkd,
        )
        self.assertIsInstance(func, ParameterizedFunctionalWithJacobianProtocol)

    def test_higher_degree_polynomial(self) -> None:
        """Exact for polynomial up to degree npts-1."""
        bkd = self._bkd
        eval_point = 1.3
        func = PointEvaluationFunctional(
            self._basis, eval_point, self._nparams, bkd,
        )
        # degree npts-2 should be exact (well within interpolation capacity)
        deg = self._npts - 2
        state = bkd.reshape(self._phys_pts ** deg, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[eval_point ** deg]])
        bkd.assert_allclose(result, expected, atol=1e-10)


class TestPointEvaluationFunctionalNumpy(
    TestPointEvaluationFunctional[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPointEvaluationFunctionalTorch(
    TestPointEvaluationFunctional[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()

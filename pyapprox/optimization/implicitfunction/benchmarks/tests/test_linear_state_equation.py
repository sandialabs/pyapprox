import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.optimization.implicitfunction.benchmarks.linear_state_equation import (
    LinearStateEquation,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestLinearStateEquation(Generic[Array], unittest.TestCase):
    """
    Base test class for LinearStateEquation.

    Derived classes must implement the bkd() method to provide the backend.
    """

    __test__ = False

    def setUp(self) -> None:
        """
        Set up the test environment for LinearStateEquation.
        """
        # Create a simple 2x3 system: state = Amat @ param + bvec
        # where Amat is 2x3 and bvec is 2x1
        self.Amat = self.bkd().array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.bvec = self.bkd().reshape(self.bkd().array([1.0, 2.0]), (2, 1))
        self.nstates = 2
        self.nparams = 3

    def bkd(self) -> Backend:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError("Derived classes must implement this method.")

    def test_initialization(self) -> None:
        """
        Test the initialization of LinearStateEquation.
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        self.assertEqual(eq.nstates(), self.nstates)
        self.assertEqual(eq.nparams(), self.nparams)
        self.assertEqual(eq.nqoi(), self.nstates)
        self.assertIsNotNone(eq.bkd())

    def test_invalid_bvec_shape(self) -> None:
        """
        Test that initialization fails with invalid bvec shape.
        """
        # bvec must be 2D with 1 column
        bad_bvec = self.bkd().array([1.0, 2.0])  # 1D array
        with self.assertRaises(ValueError):
            LinearStateEquation(self.Amat, bad_bvec, self.bkd())

    def test_inconsistent_dimensions(self) -> None:
        """
        Test that initialization fails with inconsistent dimensions.
        """
        # Amat has 2 rows but bvec has 3 rows
        bad_bvec = self.bkd().reshape(self.bkd().array([1.0, 2.0, 3.0]), (3, 1))
        with self.assertRaises(ValueError):
            LinearStateEquation(self.Amat, bad_bvec, self.bkd())

    def test_residual_computation(self) -> None:
        """
        Test the residual computation (__call__).

        Residual should be: r = state - Amat @ param - bvec
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        param = self.bkd().reshape(self.bkd().array([1.0, 0.0, -1.0]), (3, 1))
        state = self.bkd().reshape(self.bkd().array([2.0, 3.0]), (2, 1))

        residual = eq(state, param)

        # Expected: state - Amat @ param - bvec
        # Amat @ param = [[1, 2, 3], [4, 5, 6]] @ [[1], [0], [-1]]
        #               = [[1*1 + 2*0 + 3*(-1)], [4*1 + 5*0 + 6*(-1)]]
        #               = [[-2], [-2]]
        # residual = [[2], [3]] - [[-2], [-2]] - [[1], [2]]
        #          = [[2 + 2 - 1], [3 + 2 - 2]] = [[3], [3]]
        expected = self.bkd().reshape(self.bkd().array([3.0, 3.0]), (2, 1))
        self.bkd().assert_allclose(residual, expected)

    def test_solve(self) -> None:
        """
        Test the solve method.

        Solution should be: state = Amat @ param + bvec
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        param = self.bkd().reshape(self.bkd().array([1.0, 2.0, 3.0]), (3, 1))
        init_state = self.bkd().zeros((2, 1))  # Ignored for linear problem

        solution = eq.solve(init_state, param)

        # Expected: Amat @ param + bvec
        # = [[1, 2, 3], [4, 5, 6]] @ [[1], [2], [3]] + [[1], [2]]
        # = [[1*1 + 2*2 + 3*3], [4*1 + 5*2 + 6*3]] + [[1], [2]]
        # = [[14], [32]] + [[1], [2]] = [[15], [34]]
        expected = self.bkd().reshape(self.bkd().array([15.0, 34.0]), (2, 1))
        self.bkd().assert_allclose(solution, expected)

    def test_param_jacobian(self) -> None:
        """
        Test the parameter Jacobian computation.

        For linear system: dr/dparam = -Amat
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        param = self.bkd().zeros((3, 1))
        state = self.bkd().zeros((2, 1))

        jac = eq.param_jacobian(state, param)

        # Expected: -Amat
        expected = -self.Amat
        self.bkd().assert_allclose(jac, expected)

    def test_state_jacobian(self) -> None:
        """
        Test the state Jacobian computation.

        For linear system: dr/dstate = I (identity matrix)
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        param = self.bkd().zeros((3, 1))
        state = self.bkd().zeros((2, 1))

        jac = eq.state_jacobian(state, param)

        # Expected: Identity matrix
        expected = self.bkd().eye(self.nstates)
        self.bkd().assert_allclose(jac, expected)

    def test_param_param_hvp(self) -> None:
        """
        Test parameter-parameter Hessian-vector product.

        For linear system, all second derivatives are zero.
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        param = self.bkd().zeros((3, 1))
        state = self.bkd().zeros((2, 1))
        adj_state = self.bkd().ones((2, 1))
        vvec = self.bkd().ones((3, 1))

        hvp = eq.param_param_hvp(state, param, adj_state, vvec)

        # Expected: zeros
        expected = self.bkd().zeros((self.nparams, 1))
        self.bkd().assert_allclose(hvp, expected)

    def test_state_state_hvp(self) -> None:
        """
        Test state-state Hessian-vector product.

        For linear system, all second derivatives are zero.
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        param = self.bkd().zeros((3, 1))
        state = self.bkd().zeros((2, 1))
        adj_state = self.bkd().ones((2, 1))
        wvec = self.bkd().ones((2, 1))

        hvp = eq.state_state_hvp(state, param, adj_state, wvec)

        # Expected: zeros
        expected = self.bkd().zeros((self.nstates, 1))
        self.bkd().assert_allclose(hvp, expected)

    def test_param_state_hvp(self) -> None:
        """
        Test parameter-state Hessian-vector product.

        For linear system, all second derivatives are zero.
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        param = self.bkd().zeros((3, 1))
        state = self.bkd().zeros((2, 1))
        adj_state = self.bkd().ones((2, 1))
        wvec = self.bkd().ones((2, 1))

        hvp = eq.param_state_hvp(state, param, adj_state, wvec)

        # Expected: zeros
        expected = self.bkd().zeros((self.nparams, 1))
        self.bkd().assert_allclose(hvp, expected)

    def test_state_param_hvp(self) -> None:
        """
        Test state-parameter Hessian-vector product.

        For linear system, all second derivatives are zero.
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        param = self.bkd().zeros((3, 1))
        state = self.bkd().zeros((2, 1))
        adj_state = self.bkd().ones((2, 1))
        vvec = self.bkd().ones((3, 1))

        hvp = eq.state_param_hvp(state, param, adj_state, vvec)

        # Expected: zeros
        expected = self.bkd().zeros((self.nstates, 1))
        self.bkd().assert_allclose(hvp, expected)

    def test_invalid_state_dimension(self) -> None:
        """
        Test that methods raise ValueError for invalid state dimensions.
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        param = self.bkd().zeros((3, 1))
        bad_state = self.bkd().zeros((3, 1))  # Should be 2x1

        with self.assertRaises(ValueError):
            eq(bad_state, param)

    def test_invalid_param_dimension(self) -> None:
        """
        Test that methods raise ValueError for invalid param dimensions.
        """
        eq = LinearStateEquation(self.Amat, self.bvec, self.bkd())
        state = self.bkd().zeros((2, 1))
        bad_param = self.bkd().zeros((2, 1))  # Should be 3x1

        with self.assertRaises(ValueError):
            eq(state, bad_param)


class TestLinearStateEquationNumpy(TestLinearStateEquation[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestLinearStateEquationTorch(TestLinearStateEquation[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


if __name__ == "__main__":
    unittest.main()

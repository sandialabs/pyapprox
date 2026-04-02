import pytest

from pyapprox.benchmarks.functions.algebraic.linear_state_equation import (
    LinearStateEquation,
)


class TestLinearStateEquation:
    """
    Base test class for LinearStateEquation.
    """

    def _make_system(self, bkd):
        """
        Set up the test environment for LinearStateEquation.
        """
        # Create a simple 2x3 system: state = Amat @ param + bvec
        # where Amat is 2x3 and bvec is 2x1
        Amat = bkd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bvec = bkd.reshape(bkd.array([1.0, 2.0]), (2, 1))
        nstates = 2
        nparams = 3
        return Amat, bvec, nstates, nparams

    def test_initialization(self, bkd) -> None:
        """
        Test the initialization of LinearStateEquation.
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        assert eq.nstates() == nstates
        assert eq.nparams() == nparams
        assert eq.nqoi() == nstates
        assert eq.bkd() is not None

    def test_invalid_bvec_shape(self, bkd) -> None:
        """
        Test that initialization fails with invalid bvec shape.
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        # bvec must be 2D with 1 column
        bad_bvec = bkd.array([1.0, 2.0])  # 1D array
        with pytest.raises(ValueError):
            LinearStateEquation(Amat, bad_bvec, bkd)

    def test_inconsistent_dimensions(self, bkd) -> None:
        """
        Test that initialization fails with inconsistent dimensions.
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        # Amat has 2 rows but bvec has 3 rows
        bad_bvec = bkd.reshape(bkd.array([1.0, 2.0, 3.0]), (3, 1))
        with pytest.raises(ValueError):
            LinearStateEquation(Amat, bad_bvec, bkd)

    def test_residual_computation(self, bkd) -> None:
        """
        Test the residual computation (__call__).

        Residual should be: r = state - Amat @ param - bvec
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        param = bkd.reshape(bkd.array([1.0, 0.0, -1.0]), (3, 1))
        state = bkd.reshape(bkd.array([2.0, 3.0]), (2, 1))

        residual = eq(state, param)

        # Expected: state - Amat @ param - bvec
        # Amat @ param = [[1, 2, 3], [4, 5, 6]] @ [[1], [0], [-1]]
        #               = [[1*1 + 2*0 + 3*(-1)], [4*1 + 5*0 + 6*(-1)]]
        #               = [[-2], [-2]]
        # residual = [[2], [3]] - [[-2], [-2]] - [[1], [2]]
        #          = [[2 + 2 - 1], [3 + 2 - 2]] = [[3], [3]]
        expected = bkd.reshape(bkd.array([3.0, 3.0]), (2, 1))
        bkd.assert_allclose(residual, expected)

    def test_solve(self, bkd) -> None:
        """
        Test the solve method.

        Solution should be: state = Amat @ param + bvec
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        param = bkd.reshape(bkd.array([1.0, 2.0, 3.0]), (3, 1))
        init_state = bkd.zeros((2, 1))  # Ignored for linear problem

        solution = eq.solve(init_state, param)

        # Expected: Amat @ param + bvec
        # = [[1, 2, 3], [4, 5, 6]] @ [[1], [2], [3]] + [[1], [2]]
        # = [[14], [32]] + [[1], [2]] = [[15], [34]]
        expected = bkd.reshape(bkd.array([15.0, 34.0]), (2, 1))
        bkd.assert_allclose(solution, expected)

    def test_param_jacobian(self, bkd) -> None:
        """
        Test the parameter Jacobian computation.

        For linear system: dr/dparam = -Amat
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        param = bkd.zeros((3, 1))
        state = bkd.zeros((2, 1))

        jac = eq.param_jacobian(state, param)

        # Expected: -Amat
        expected = -Amat
        bkd.assert_allclose(jac, expected)

    def test_state_jacobian(self, bkd) -> None:
        """
        Test the state Jacobian computation.

        For linear system: dr/dstate = I (identity matrix)
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        param = bkd.zeros((3, 1))
        state = bkd.zeros((2, 1))

        jac = eq.state_jacobian(state, param)

        # Expected: Identity matrix
        expected = bkd.eye(nstates)
        bkd.assert_allclose(jac, expected)

    def test_param_param_hvp(self, bkd) -> None:
        """
        Test parameter-parameter Hessian-vector product.

        For linear system, all second derivatives are zero.
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        param = bkd.zeros((3, 1))
        state = bkd.zeros((2, 1))
        adj_state = bkd.ones((2, 1))
        vvec = bkd.ones((3, 1))

        hvp = eq.param_param_hvp(state, param, adj_state, vvec)

        # Expected: zeros
        expected = bkd.zeros((nparams, 1))
        bkd.assert_allclose(hvp, expected)

    def test_state_state_hvp(self, bkd) -> None:
        """
        Test state-state Hessian-vector product.

        For linear system, all second derivatives are zero.
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        param = bkd.zeros((3, 1))
        state = bkd.zeros((2, 1))
        adj_state = bkd.ones((2, 1))
        wvec = bkd.ones((2, 1))

        hvp = eq.state_state_hvp(state, param, adj_state, wvec)

        # Expected: zeros
        expected = bkd.zeros((nstates, 1))
        bkd.assert_allclose(hvp, expected)

    def test_param_state_hvp(self, bkd) -> None:
        """
        Test parameter-state Hessian-vector product.

        For linear system, all second derivatives are zero.
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        param = bkd.zeros((3, 1))
        state = bkd.zeros((2, 1))
        adj_state = bkd.ones((2, 1))
        wvec = bkd.ones((2, 1))

        hvp = eq.param_state_hvp(state, param, adj_state, wvec)

        # Expected: zeros
        expected = bkd.zeros((nparams, 1))
        bkd.assert_allclose(hvp, expected)

    def test_state_param_hvp(self, bkd) -> None:
        """
        Test state-parameter Hessian-vector product.

        For linear system, all second derivatives are zero.
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        param = bkd.zeros((3, 1))
        state = bkd.zeros((2, 1))
        adj_state = bkd.ones((2, 1))
        vvec = bkd.ones((3, 1))

        hvp = eq.state_param_hvp(state, param, adj_state, vvec)

        # Expected: zeros
        expected = bkd.zeros((nstates, 1))
        bkd.assert_allclose(hvp, expected)

    def test_invalid_state_dimension(self, bkd) -> None:
        """
        Test that methods raise ValueError for invalid state dimensions.
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        param = bkd.zeros((3, 1))
        bad_state = bkd.zeros((3, 1))  # Should be 2x1

        with pytest.raises(ValueError):
            eq(bad_state, param)

    def test_invalid_param_dimension(self, bkd) -> None:
        """
        Test that methods raise ValueError for invalid param dimensions.
        """
        Amat, bvec, nstates, nparams = self._make_system(bkd)
        eq = LinearStateEquation(Amat, bvec, bkd)
        state = bkd.zeros((2, 1))
        bad_param = bkd.zeros((2, 1))  # Should be 3x1

        with pytest.raises(ValueError):
            eq(state, bad_param)

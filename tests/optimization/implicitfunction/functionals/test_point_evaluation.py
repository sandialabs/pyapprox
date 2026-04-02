"""Tests for PointEvaluationFunctional."""

import numpy as np

from pyapprox.optimization.implicitfunction.functionals.point_evaluation import (
    PointEvaluationFunctional,
)
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)

# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module



class TestPointEvaluationFunctional:

    def _make_basis(self, bkd):
        npts = 15
        length = 2.0
        nparams = 3
        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        phys_pts = mesh.points()[0, :]  # shape (npts,)
        return basis, phys_pts, npts, length, nparams

    def test_polynomial_exactness(self, bkd) -> None:
        """Evaluating a polynomial at x* via interpolation is exact."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        eval_point = 0.7
        func = PointEvaluationFunctional(
            basis,
            eval_point,
            nparams,
            bkd,
        )
        # u(x) = x^3 at collocation nodes
        state = bkd.reshape(phys_pts**3, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[eval_point**3]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_boundary_evaluation_left(self, bkd) -> None:
        """Evaluation at x=0 recovers state at left boundary node."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        func = PointEvaluationFunctional(
            basis,
            0.0,
            nparams,
            bkd,
        )
        np.random.seed(42)
        state_vals = bkd.array(np.random.randn(npts))
        state = bkd.reshape(state_vals, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        # Left boundary is first node (CGL nodes ordered left to right)
        expected = bkd.reshape(state_vals[0:1], (1, 1))
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_tip_evaluation(self, bkd) -> None:
        """Evaluation at x=L recovers state at right boundary node."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        func = PointEvaluationFunctional(
            basis,
            length,
            nparams,
            bkd,
        )
        np.random.seed(42)
        state_vals = bkd.array(np.random.randn(npts))
        state = bkd.reshape(state_vals, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.reshape(state_vals[-1:], (1, 1))
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_constant_state_jacobian(self, bkd) -> None:
        """State Jacobian is independent of the state vector (linear)."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        func = PointEvaluationFunctional(
            basis,
            0.7,
            nparams,
            bkd,
        )
        param = bkd.zeros((nparams, 1))
        state1 = bkd.zeros((npts, 1))
        state2 = bkd.ones((npts, 1))
        jac1 = func.state_jacobian(state1, param)
        jac2 = func.state_jacobian(state2, param)
        bkd.assert_allclose(jac1, jac2, atol=1e-15)

    def test_state_jacobian_shape(self, bkd) -> None:
        """State Jacobian has correct shape (1, nstates)."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        func = PointEvaluationFunctional(
            basis,
            0.5,
            nparams,
            bkd,
        )
        param = bkd.zeros((nparams, 1))
        state = bkd.zeros((npts, 1))
        jac = func.state_jacobian(state, param)
        assert jac.shape == (1, npts)

    def test_nqoi_is_one(self, bkd) -> None:
        """Point evaluation returns a scalar QoI."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        func = PointEvaluationFunctional(
            basis,
            0.5,
            nparams,
            bkd,
        )
        assert func.nqoi() == 1

    def test_param_jacobian_is_zero(self, bkd) -> None:
        """Functional does not depend on parameters."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        func = PointEvaluationFunctional(
            basis,
            0.5,
            nparams,
            bkd,
        )
        state = bkd.ones((npts, 1))
        param = bkd.ones((nparams, 1))
        jac = func.param_jacobian(state, param)
        expected = bkd.zeros((1, nparams))
        bkd.assert_allclose(jac, expected)

    def test_protocol_compliance(self, bkd) -> None:
        """Satisfies ParameterizedFunctionalWithJacobianProtocol."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        func = PointEvaluationFunctional(
            basis,
            0.5,
            nparams,
            bkd,
        )
        assert isinstance(func, ParameterizedFunctionalWithJacobianProtocol)

    def test_higher_degree_polynomial(self, bkd) -> None:
        """Exact for polynomial up to degree npts-1."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        eval_point = 1.3
        func = PointEvaluationFunctional(
            basis,
            eval_point,
            nparams,
            bkd,
        )
        # degree npts-2 should be exact (well within interpolation capacity)
        deg = npts - 2
        state = bkd.reshape(phys_pts**deg, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[eval_point**deg]])
        bkd.assert_allclose(result, expected, atol=1e-10)

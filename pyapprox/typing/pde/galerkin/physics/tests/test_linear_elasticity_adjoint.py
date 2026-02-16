"""Tests for LinearElasticity parameter sensitivity and adjoint gradient."""

import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.typing.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity as LinearElasticity,
)
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.typing.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.typing.pde.galerkin.protocols.physics import (
    GalerkinPhysicsWithParamJacobianProtocol,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


def _to_dense(mat):
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def _make_physics(
    bkd: Backend[Array],
    E: float = 1.0,
    nu: float = 0.3,
    with_bcs: bool = True,
) -> "LinearElasticity[Array]":
    """Create a 2D LinearElasticity with constant body force and all-Dirichlet BCs."""
    mesh = StructuredMesh2D(
        nx=5, ny=5,
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        bkd=bkd,
    )
    basis = VectorLagrangeBasis(mesh, degree=1)

    def body_force(x, time):
        f = np.zeros_like(x)
        f[0, :] = 1.0
        f[1, :] = -2.0
        return f

    if with_bcs:
        bc_list = [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
            DirichletBC(basis, "bottom", 0.0, bkd),
            DirichletBC(basis, "top", 0.0, bkd),
        ]
    else:
        bc_list = []

    return LinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=E,
        poisson_ratio=nu,
        body_force=body_force,
        boundary_conditions=bc_list,
        bkd=bkd,
    )


class TestLinearElasticityAdjointBase(Generic[Array], unittest.TestCase):
    """Base test class for LinearElasticity parameter sensitivity."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_nparams(self) -> None:
        """nparams() returns 2 (E, nu)."""
        physics = _make_physics(self.bkd_inst)
        self.assertEqual(physics.nparams(), 2)

    def test_param_jacobian_shape(self) -> None:
        """param_jacobian returns shape (nstates, 2)."""
        physics = _make_physics(self.bkd_inst)
        n = physics.nstates()
        u = self.bkd_inst.asarray(np.ones(n) * 0.01)
        pj = physics.param_jacobian(u, 0.0)
        self.assertEqual(pj.shape, (n, 2))

    def test_initial_param_jacobian_is_zero(self) -> None:
        """initial_param_jacobian returns all zeros."""
        physics = _make_physics(self.bkd_inst)
        ipj = physics.initial_param_jacobian()
        ipj_np = self.bkd_inst.to_numpy(ipj)
        self.bkd_inst.assert_allclose(
            self.bkd_inst.asarray(ipj_np),
            self.bkd_inst.asarray(np.zeros_like(ipj_np)),
        )

    def test_set_param_changes_stiffness(self) -> None:
        """Stiffness matrix changes after set_param with new (E, nu)."""
        physics = _make_physics(self.bkd_inst, E=1.0, nu=0.3)
        K1 = _to_dense(physics.stiffness_matrix()).copy()

        physics.set_param(self.bkd_inst.asarray(np.array([2.0, 0.25])))
        K2 = _to_dense(physics.stiffness_matrix())

        diff = np.linalg.norm(K2 - K1)
        self.assertGreater(diff, 1e-10)

    def test_param_jacobian_fd_validation(self) -> None:
        """DerivativeChecker validation of param_jacobian.

        Wraps residual(p) and param_jacobian as a FunctionWithJacobian,
        then verifies error_ratio <= 1e-6.
        """
        bkd = self.bkd_inst
        E0, nu0 = 1.0, 0.3
        physics = _make_physics(bkd, E=E0, nu=nu0)

        # Solve for the state at base parameters
        solver = SteadyStateSolver(
            physics, tol=1e-12, max_iter=5, line_search=False,
        )
        result = solver.solve_linear()
        self.assertTrue(result.converged)
        u = result.solution
        nstates = physics.nstates()

        # Wrap residual(p) as a function of params for DerivativeChecker
        def residual_of_params(params: Array) -> Array:
            # params shape: (nparams, nsamples)
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                physics.set_param(p)
                res = physics.residual(u, 0.0)
                results.append(bkd.reshape(res, (nstates, 1)))
            physics.set_param(bkd.asarray(np.array([E0, nu0])))
            return bkd.hstack(results)

        def jacobian_of_params(params: Array) -> Array:
            # params shape: (nparams, 1); return (nstates, nparams)
            p = params[:, 0]
            physics.set_param(p)
            pj = physics.param_jacobian(u, 0.0)
            physics.set_param(bkd.asarray(np.array([E0, nu0])))
            return pj

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=2,
            fun=residual_of_params,
            jacobian=jacobian_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray(np.array([[E0], [nu0]]))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-6)

    def test_adjoint_gradient_steady(self) -> None:
        """DerivativeChecker validation of adjoint gradient.

        QoI: Q(u(p)) = c^T u(p) where u(p) solves K(p)*u = b.
        Adjoint gradient: dQ/dp = (dF/dp)^T lambda, with J^T lambda = -c.
        Wraps Q(p) and adjoint gradient as a FunctionWithJacobian,
        then verifies error_ratio <= 1e-6.
        """
        bkd = self.bkd_inst
        E0, nu0 = 1.0, 0.3
        physics = _make_physics(bkd, E=E0, nu=nu0)

        # Solve forward problem at base params
        solver = SteadyStateSolver(
            physics, tol=1e-12, max_iter=5, line_search=False,
        )
        result = solver.solve_linear()
        self.assertTrue(result.converged)
        u = result.solution

        # Random QoI direction
        np.random.seed(42)
        n = physics.nstates()
        c_np = np.random.randn(n)

        # QoI as function of params: Q(p) = c^T u(p)
        def qoi_of_params(params: Array) -> Array:
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                physics.set_param(p)
                r = SteadyStateSolver(
                    physics, tol=1e-12, max_iter=5, line_search=False,
                ).solve_linear()
                u_np = bkd.to_numpy(r.solution)
                Q = c_np @ u_np
                results.append(Q)
            physics.set_param(bkd.asarray(np.array([E0, nu0])))
            return bkd.reshape(bkd.asarray(np.array(results)), (1, nsamples))

        # Adjoint gradient at a given param
        def adjoint_gradient(params: Array) -> Array:
            p = params[:, 0]
            physics.set_param(p)
            r = SteadyStateSolver(
                physics, tol=1e-12, max_iter=5, line_search=False,
            ).solve_linear()
            u_sol = r.solution

            # Solve adjoint: J^T lambda = -c
            J_np = _to_dense(physics.jacobian(u_sol, 0.0))
            lam_np = np.linalg.solve(J_np.T, -c_np)

            # Adjoint gradient: dQ/dp = (dF/dp)^T lambda
            dF_dp_np = bkd.to_numpy(physics.param_jacobian(u_sol, 0.0))
            grad = dF_dp_np.T @ lam_np

            physics.set_param(bkd.asarray(np.array([E0, nu0])))
            return bkd.reshape(bkd.asarray(grad), (1, 2))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=qoi_of_params,
            jacobian=adjoint_gradient,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray(np.array([[E0], [nu0]]))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-6)

    def test_protocol_conformance(self) -> None:
        """LinearElasticity satisfies GalerkinPhysicsWithParamJacobianProtocol."""
        physics = _make_physics(self.bkd_inst)
        self.assertIsInstance(physics, GalerkinPhysicsWithParamJacobianProtocol)


class TestLinearElasticityAdjointNumpy(
    TestLinearElasticityAdjointBase[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


try:
    import torch
    from pyapprox.typing.util.backends.torch import TorchBkd

    class TestLinearElasticityAdjointTorch(
        TestLinearElasticityAdjointBase[torch.Tensor]
    ):
        """PyTorch backend tests."""

        __test__ = True

        def setUp(self) -> None:
            torch.set_default_dtype(torch.float64)
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_param_jacobian_shape(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_param_jacobian_fd_validation(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_set_param_changes_stiffness(self) -> None:
            pass

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_adjoint_gradient_steady(self) -> None:
            pass

except ImportError:
    pass


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    unittest.main()

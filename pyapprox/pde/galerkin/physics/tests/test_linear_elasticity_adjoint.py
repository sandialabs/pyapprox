"""Tests for LinearElasticity parameter sensitivity via parameterization.

Tests parameter sensitivity through GalerkinLameParameterization, which
maps (E, nu) to Lame parameters and computes chain-rule Jacobians via
physics.residual_lam_sensitivity() and residual_mu_sensitivity().
"""


import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from scipy.sparse import issparse

from pyapprox.util.backends.protocols import Array
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity as LinearElasticity,
)
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)


def _to_dense(mat):
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def _make_physics(
    bkd,
    E: float = 1.0,
    nu: float = 0.3,
    with_bcs: bool = True,
):
    """Create a 2D LinearElasticity with constant body force and all-Dirichlet BCs."""
    mesh = StructuredMesh2D(
        nx=5,
        ny=5,
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


class TestLinearElasticityAdjoint:
    """Test class for LinearElasticity parameter sensitivity
    via GalerkinLameParameterization."""

    def test_nparams(self, numpy_bkd) -> None:
        """Parameterization nparams() returns 2 (E, nu) for single material."""
        bkd = numpy_bkd
        physics = _make_physics(numpy_bkd)
        param = create_galerkin_lame_parameterization(physics, numpy_bkd)
        assert param.nparams() == 2

    def test_param_jacobian_shape(self, numpy_bkd) -> None:
        """param_jacobian returns shape (nstates, 2)."""
        bkd = numpy_bkd
        physics = _make_physics(numpy_bkd)
        param = create_galerkin_lame_parameterization(physics, numpy_bkd)
        n = physics.nstates()
        u = numpy_bkd.asarray(np.ones(n) * 0.01)
        params_1d = numpy_bkd.asarray(np.array([1.0, 0.3]))
        param.apply(physics, params_1d)
        pj = param.param_jacobian(physics, u, 0.0, params_1d)
        assert pj.shape == (n, 2)

    def test_initial_param_jacobian_is_zero(self, numpy_bkd) -> None:
        """initial_param_jacobian returns all zeros."""
        bkd = numpy_bkd
        physics = _make_physics(numpy_bkd)
        param = create_galerkin_lame_parameterization(physics, numpy_bkd)
        params_1d = numpy_bkd.asarray(np.array([1.0, 0.3]))
        ipj = param.initial_param_jacobian(physics, params_1d)
        ipj_np = numpy_bkd.to_numpy(ipj)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(ipj_np),
            numpy_bkd.asarray(np.zeros_like(ipj_np)),
        )

    def test_apply_changes_stiffness(self, numpy_bkd) -> None:
        """Stiffness matrix changes after parameterization.apply() with new (E, nu)."""
        bkd = numpy_bkd
        physics = _make_physics(numpy_bkd, E=1.0, nu=0.3)
        param = create_galerkin_lame_parameterization(physics, numpy_bkd)
        K1 = _to_dense(physics.stiffness_matrix()).copy()

        param.apply(physics, numpy_bkd.asarray(np.array([2.0, 0.25])))
        K2 = _to_dense(physics.stiffness_matrix())

        diff = np.linalg.norm(K2 - K1)
        assert diff > 1e-10

    def test_param_jacobian_fd_validation(self, numpy_bkd) -> None:
        """DerivativeChecker validation of parameterization param_jacobian.

        Wraps residual(p) and param_jacobian as a FunctionWithJacobian,
        then verifies error_ratio <= 1e-6.
        """
        bkd = numpy_bkd
        E0, nu0 = 1.0, 0.3
        physics = _make_physics(bkd, E=E0, nu=nu0)
        param = create_galerkin_lame_parameterization(physics, bkd)

        # Solve for the state at base parameters
        solver = SteadyStateSolver(
            physics,
            tol=1e-12,
            max_iter=5,
            line_search=False,
        )
        result = solver.solve_linear()
        assert result.converged
        u = result.solution
        nstates = physics.nstates()

        def residual_of_params(params: Array) -> Array:
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                param.apply(physics, p)
                res = physics.residual(u, 0.0)
                results.append(bkd.reshape(res, (nstates, 1)))
            param.apply(physics, bkd.asarray(np.array([E0, nu0])))
            return bkd.hstack(results)

        def jacobian_of_params(params: Array) -> Array:
            p = params[:, 0]
            param.apply(physics, p)
            # Raw param_jacobian (no BC enforcement)
            pj_raw = param.param_jacobian(physics, u, 0.0, p)
            # Apply BC enforcement
            pj = physics._apply_dirichlet_to_param_jacobian(pj_raw, u, 0.0)
            param.apply(physics, bkd.asarray(np.array([E0, nu0])))
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
        assert ratio <= 1e-6

    def test_adjoint_gradient_steady(self, numpy_bkd) -> None:
        """DerivativeChecker validation of adjoint gradient via parameterization.

        QoI: Q(u(p)) = c^T u(p) where u(p) solves K(p)*u = b.
        Adjoint gradient: dQ/dp = (dF/dp)^T lambda, with J^T lambda = -c.
        """
        bkd = numpy_bkd
        E0, nu0 = 1.0, 0.3
        physics = _make_physics(bkd, E=E0, nu=nu0)
        param = create_galerkin_lame_parameterization(physics, bkd)

        # Solve forward problem at base params
        solver = SteadyStateSolver(
            physics,
            tol=1e-12,
            max_iter=5,
            line_search=False,
        )
        result = solver.solve_linear()
        assert result.converged

        # Random QoI direction
        np.random.seed(42)
        n = physics.nstates()
        c_np = np.random.randn(n)

        def qoi_of_params(params: Array) -> Array:
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                param.apply(physics, p)
                r = SteadyStateSolver(
                    physics,
                    tol=1e-12,
                    max_iter=5,
                    line_search=False,
                ).solve_linear()
                u_np = bkd.to_numpy(r.solution)
                Q = c_np @ u_np
                results.append(Q)
            param.apply(physics, bkd.asarray(np.array([E0, nu0])))
            return bkd.reshape(bkd.asarray(np.array(results)), (1, nsamples))

        def adjoint_gradient(params: Array) -> Array:
            p = params[:, 0]
            param.apply(physics, p)
            r = SteadyStateSolver(
                physics,
                tol=1e-12,
                max_iter=5,
                line_search=False,
            ).solve_linear()
            u_sol = r.solution

            # Solve adjoint: J^T lambda = -c
            J_np = _to_dense(physics.jacobian(u_sol, 0.0))
            lam_np = np.linalg.solve(J_np.T, -c_np)

            # Parameterization param_jacobian (raw) + BC enforcement
            dF_dp_raw = bkd.to_numpy(param.param_jacobian(physics, u_sol, 0.0, p))
            dF_dp = bkd.to_numpy(
                physics._apply_dirichlet_to_param_jacobian(
                    bkd.asarray(dF_dp_raw), u_sol, 0.0
                )
            )

            grad = dF_dp.T @ lam_np
            param.apply(physics, bkd.asarray(np.array([E0, nu0])))
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
        assert ratio <= 1e-6

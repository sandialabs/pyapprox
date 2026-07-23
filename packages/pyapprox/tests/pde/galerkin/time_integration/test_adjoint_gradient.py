"""Transient adjoint gradient and HVP validation for the galerkin pipeline.

DerivativeChecker-validated dQ/dp and d^2Q/dp^2 v for Q = y_k(T)
through adapter -> stepper -> BC-enforcing wrapper -> TimeIntegrator,
with parameters entering via GalerkinLameParameterization. Backward
Euler and Crank-Nicolson (CN exercises the off-diagonal adjoint
coupling and the cross-step HVP zeroing that BE's zero cross-step
blocks cannot).
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity as LinearElasticity,
)
from pyapprox.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    GalerkinBCEnforcingAdjointResidual,
    GalerkinBCEnforcingHVPResidual,
    create_galerkin_bc_enforcing_residual,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.rootfinding.newton import NewtonSolver

from tests._helpers.adjoint_checks import HVPOperatorFunction

_NumpyArray = NDArray[Any]

_E0, _NU0 = 1.0, 0.3
_FINAL_TIME, _DELTAT = 0.4, 0.1


def _make_physics(bkd: NumpyBkd) -> LinearElasticity[_NumpyArray]:
    """Small 2D elasticity with body force and all-Dirichlet BCs."""
    mesh = StructuredMesh2D(
        nx=3, ny=3, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd
    )
    basis = VectorLagrangeBasis(mesh, degree=1)

    def body_force(x: _NumpyArray, time: float) -> _NumpyArray:
        f = np.zeros_like(x)
        f[0, :] = 1.0
        f[1, :] = -2.0
        return f

    bc_list: list[BoundaryConditionProtocol[_NumpyArray]] = [
        DirichletBC(basis, name, 0.0, bkd)
        for name in ("left", "right", "bottom", "top")
    ]
    return LinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=_E0,
        poisson_ratio=_NU0,
        body_force=body_force,
        boundary_conditions=bc_list,
        bkd=bkd,
    )


def _build_pipeline(
    bkd: NumpyBkd, method: str
) -> Tuple[TimeIntegrator[_NumpyArray], Any, LinearElasticity[_NumpyArray]]:
    physics = _make_physics(bkd)
    param = create_galerkin_lame_parameterization(physics, bkd)
    adapter = create_galerkin_physics_ode_residual(physics, param)
    stepper = create_stepper(method, adapter)
    wrapper = create_galerkin_bc_enforcing_residual(stepper, physics, bkd)
    assert isinstance(wrapper, GalerkinBCEnforcingAdjointResidual)
    newton = NewtonSolver(wrapper)
    newton.set_options(maxiters=20, atol=1e-12, rtol=0.0)
    integrator = TimeIntegrator(0.0, _FINAL_TIME, _DELTAT, newton)
    return integrator, adapter, physics


class TestTransientAdjointGradient:
    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_endpoint_gradient_matches_fd(
        self, numpy_bkd: NumpyBkd, method: str
    ) -> None:
        bkd = numpy_bkd
        integrator, adapter, physics = _build_pipeline(bkd, method)
        nstates = physics.nstates()
        # An interior DOF: constrained endpoint QoIs have zero gradient.
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        functional = EndpointFunctional(state_idx, nstates, 2, bkd)
        integrator.set_functional(functional)
        y0 = bkd.asarray(np.zeros(nstates))

        def solve(params_1d: _NumpyArray) -> Tuple[_NumpyArray, _NumpyArray]:
            adapter.set_param(params_1d)
            return integrator.solve(y0)

        def qoi_of_params(params: _NumpyArray) -> _NumpyArray:
            results = []
            for ii in range(params.shape[1]):
                sols, _ = solve(params[:, ii])
                results.append(float(bkd.to_numpy(sols)[state_idx, -1]))
            return bkd.reshape(
                bkd.asarray(np.array(results)), (1, params.shape[1])
            )

        def adjoint_gradient(params: _NumpyArray) -> _NumpyArray:
            sols, times = solve(params[:, 0])
            return integrator.gradient(sols, times, params)

        wrapper_fn = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=qoi_of_params,
            jacobian=adjoint_gradient,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper_fn)
        sample = bkd.asarray(np.array([[_E0], [_NU0]]))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6


class TestTransientAdjointHVP:
    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_endpoint_hvp_matches_fd(
        self, numpy_bkd: NumpyBkd, method: str
    ) -> None:
        """DerivativeChecker validation of dQ/dp and d^2Q/dp^2 v.

        The physics is linear in state, so the HVP is driven entirely
        by the galerkin_lame parameterization's second-order bundle
        (Lame-map Hessian + linearity/symmetry contractions) through
        the HVP wrapper tier.
        """
        bkd = numpy_bkd
        integrator, adapter, physics = _build_pipeline(bkd, method)
        wrapper = integrator.time_residual()
        assert isinstance(wrapper, GalerkinBCEnforcingHVPResidual)
        nstates = physics.nstates()
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        functional = EndpointFunctional(state_idx, nstates, 2, bkd)
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        y0 = bkd.asarray(np.zeros(nstates))
        fn = HVPOperatorFunction(operator, adapter, y0, bkd)
        checker = DerivativeChecker(fn)

        param = bkd.asarray(np.array([[_E0], [_NU0]]))
        direction = bkd.asarray(np.array([[0.7], [-0.4]]))
        errors = checker.check_derivatives(
            param, direction=direction, relative=True
        )

        jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert jac_ratio <= 1e-6
        hvp_ratio = float(bkd.to_numpy(checker.error_ratio(errors[1])))
        assert hvp_ratio <= 1e-6

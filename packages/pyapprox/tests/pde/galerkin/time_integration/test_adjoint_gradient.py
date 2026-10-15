"""Transient adjoint gradient and HVP validation for the galerkin pipeline.

DerivativeChecker-validated dQ/dp and d^2Q/dp^2 v for Q = y_k(T)
through adapter -> stepper -> BC-enforcing wrapper -> TimeIntegrator,
with parameters entering a scalar diffusion through a lognormal KLE
diffusivity. Backward Euler and Crank-Nicolson (CN exercises the off-diagonal adjoint
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
from pyapprox.pde.boundary import BoundaryConditionRole
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    NodalFieldForcing,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.kle_factory import (
    create_spde_lognormal_kle_field_map,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    GalerkinBCEnforcingAdjointResidual,
    GalerkinBCEnforcingHVPResidual,
    create_galerkin_bc_enforcing_residual,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.rootfinding.newton import NewtonSolver
from tests._helpers.adjoint_checks import HVPOperatorFunction

_NumpyArray = NDArray[Any]

_NMODES = 2
# A point away from the KLE mean, so the log-diffusivity is not uniform.
_PARAM0 = (0.3, -0.2)
_FINAL_TIME, _DELTAT = 0.4, 0.1


def _make_physics(
    bkd: NumpyBkd,
) -> Tuple[
    AdvectionDiffusionReaction[_NumpyArray],
    LagrangeBasis[_NumpyArray],
    GalerkinSystem[_NumpyArray],
]:
    """Small 2D diffusion with unit forcing and all-Dirichlet BCs."""
    mesh = StructuredMesh2D(nx=4, ny=4, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    bc_list: list[BoundaryConditionRole[_NumpyArray]] = [
        DirichletBC(basis, name, 0.0, bkd)
        for name in ("left", "right", "bottom", "top")
    ]
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis, dofs=0.1 * np.ones(basis.ndofs())),
        bkd=bkd,
        forcing=NodalFieldForcing(basis, dofs=np.ones(basis.ndofs())),
    )
    return physics, basis, compose_galerkin_system(physics, bc_list)


def _build_pipeline(
    bkd: NumpyBkd, method: str
) -> Tuple[TimeIntegrator[_NumpyArray], Any, GalerkinSystem[_NumpyArray]]:
    physics, basis, system = _make_physics(bkd)
    diffusivity_map = create_spde_lognormal_kle_field_map(
        basis,
        bkd.asarray(np.log(0.1) * np.ones(basis.ndofs())),
        bkd,
        n_modes=_NMODES,
        gamma=1.0,
        delta=4.0,
        sigma=0.3,
    )
    param = AdvectionDiffusionParameterization(
        physics, bkd=bkd, diffusivity_map=diffusivity_map
    )
    adapter = create_galerkin_physics_ode_residual(system, param)
    stepper = create_stepper(method, adapter)
    wrapper = create_galerkin_bc_enforcing_residual(
        stepper, system.constraint_set(), bkd
    )
    assert isinstance(wrapper, GalerkinBCEnforcingAdjointResidual)
    newton = NewtonSolver(wrapper)
    newton.set_options(maxiters=20, atol=1e-12, rtol=0.0)
    integrator = TimeIntegrator(0.0, _FINAL_TIME, _DELTAT, newton)
    return integrator, adapter, system


class TestTransientAdjointGradient:
    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_endpoint_gradient_matches_fd(
        self, numpy_bkd: NumpyBkd, method: str
    ) -> None:
        bkd = numpy_bkd
        integrator, adapter, system = _build_pipeline(bkd, method)
        nstates = system.nstates()
        # An interior DOF: constrained endpoint QoIs have zero gradient.
        constrained = set(int(d) for d in bkd.to_numpy(system.constraint_set().dofs()))
        state_idx = next(ii for ii in range(nstates) if ii not in constrained)
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
            return bkd.reshape(bkd.asarray(np.array(results)), (1, params.shape[1]))

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
        sample = bkd.asarray(np.array([[_PARAM0[0]], [_PARAM0[1]]]))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-6


class TestTransientAdjointHVP:
    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_endpoint_hvp_matches_fd(self, numpy_bkd: NumpyBkd, method: str) -> None:
        """DerivativeChecker validation of dQ/dp and d^2Q/dp^2 v.

        The physics is linear in state, so the HVP is driven entirely
        by the diffusivity parameterization's second-order bundle
        (lognormal-map Hessian + linearity contractions) through the
        HVP wrapper tier.
        """
        bkd = numpy_bkd
        integrator, adapter, system = _build_pipeline(bkd, method)
        wrapper = integrator.time_residual()
        assert isinstance(wrapper, GalerkinBCEnforcingHVPResidual)
        nstates = system.nstates()
        constrained = set(int(d) for d in bkd.to_numpy(system.constraint_set().dofs()))
        state_idx = next(ii for ii in range(nstates) if ii not in constrained)
        functional = EndpointFunctional(state_idx, nstates, 2, bkd)
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        y0 = bkd.asarray(np.zeros(nstates))
        fn = HVPOperatorFunction(operator, adapter, y0, bkd)
        checker = DerivativeChecker(fn)

        param = bkd.asarray(np.array([[_PARAM0[0]], [_PARAM0[1]]]))
        direction = bkd.asarray(np.array([[0.7], [-0.4]]))
        errors = checker.check_derivatives(param, direction=direction, relative=True)

        jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert jac_ratio <= 1e-6
        hvp_ratio = float(bkd.to_numpy(checker.error_ratio(errors[1])))
        assert hvp_ratio <= 1e-6

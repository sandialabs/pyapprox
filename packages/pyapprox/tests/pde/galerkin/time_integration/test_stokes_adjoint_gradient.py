"""Transient Stokes/Navier-Stokes adjoint gradient (DAE mass).

DerivativeChecker-validated dQ/dp through the DAE pipeline: the
BC-neutralized Stokes mass is singular (zero pressure rows), so the
backward sweep's lambda_0 solve exercises the wrapper's algebraic-DOF
handling (lambda_0 = 0 on algebraic DOFs, valid because the IC
parameterization has no support there). Parameters are [viscosity,
velocity-forcing amplitude] (viscosity gives a state-dependent
parameter Jacobian along the trajectory); Navier-Stokes exercises
genuinely multi-iteration Newton.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Dict, Tuple

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    create_galerkin_bc_enforcing_residual,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.rootfinding.newton import NewtonSolver

from tests._helpers.adjoint_checks import NumpyArray

_FINAL_TIME, _DELTAT = 0.3, 0.1


class _ViscosityForcingParameterization:
    """Stokes parameters p = [viscosity, forcing amplitude].

    F is affine in each parameter with no cross term (viscous term
    -nu*A1*u_vel; forcing term amp*b0; the NS convective term depends
    on neither), so each Jacobian column is the difference of two
    spatial residuals holding the other parameter fixed — exact. The
    viscosity column is STATE-DEPENDENT (-A1*u), exercising the
    per-step param_jacobian evaluation along the trajectory. The IC
    does not depend on the parameters (initial_param_jacobian = 0 —
    required by the DAE lambda_0 handling).
    """

    def __init__(
        self,
        physics: StokesPhysics[NumpyArray],
        amplitude: Dict[str, float],
        bkd: NumpyBkd,
    ) -> None:
        self._physics = physics
        self._amplitude = amplitude
        self._bkd = bkd
        self._derivs: ParamDerivatives[NumpyArray] = (
            ParamDerivatives.first_order(
                self._param_jacobian, self._initial_param_jacobian
            )
        )

    def nparams(self) -> int:
        return 2

    def physics(self) -> StokesPhysics[NumpyArray]:
        return self._physics

    def param_derivatives(self) -> ParamDerivatives[NumpyArray]:
        return self._derivs

    def apply(self, params_1d: NumpyArray) -> None:
        self._physics.set_viscosity(float(params_1d[0]))
        self._amplitude["value"] = float(params_1d[1])

    def _param_jacobian(
        self, state: NumpyArray, time: float, params_1d: NumpyArray
    ) -> NumpyArray:
        self.apply(params_1d)
        # Viscosity column (affine; amplitude terms cancel in the
        # difference).
        self._physics.set_viscosity(1.0)
        r_nu_one = self._physics.spatial_residual(state, time)
        self._physics.set_viscosity(0.0)
        r_nu_zero = self._physics.spatial_residual(state, time)
        self._physics.set_viscosity(float(params_1d[0]))
        # Amplitude column (affine; viscous terms cancel).
        old_amp = self._amplitude["value"]
        self._amplitude["value"] = 1.0
        r_amp_one = self._physics.spatial_residual(state, time)
        self._amplitude["value"] = 0.0
        r_amp_zero = self._physics.spatial_residual(state, time)
        self._amplitude["value"] = old_amp
        return np.stack(
            [r_nu_one - r_nu_zero, r_amp_one - r_amp_zero], axis=1
        )

    def _initial_param_jacobian(self, params_1d: NumpyArray) -> NumpyArray:
        return self._bkd.zeros((self._physics.nstates(), self.nparams()))


class _BadICParameterization(_ViscosityForcingParameterization):
    """IC Jacobian with (unphysical) support on every DOF, including
    the algebraic pressure DOFs — must be rejected."""

    def _initial_param_jacobian(self, params_1d: NumpyArray) -> NumpyArray:
        return self._bkd.asarray(
            np.ones((self._physics.nstates(), self.nparams()))
        )


def _build_pipeline(
    bkd: NumpyBkd,
    method: str,
    navier_stokes: bool,
    param_cls: type = _ViscosityForcingParameterization,
) -> Tuple[TimeIntegrator[NumpyArray], Any, StokesPhysics[NumpyArray]]:
    mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
    vel_basis = VectorLagrangeBasis(mesh, degree=2)
    pres_basis = LagrangeBasis(mesh, degree=1)
    amplitude = {"value": 1.0}

    def vel_forcing(x: NumpyArray) -> NumpyArray:
        return amplitude["value"] * np.sin(np.pi * x)[:, None]

    def zero_vel(x: NumpyArray) -> NumpyArray:
        return np.zeros((x.shape[1], 1))

    def zero_pres(x: NumpyArray) -> NumpyArray:
        return np.zeros(x.shape[1])

    physics = StokesPhysics(
        vel_basis,
        pres_basis,
        bkd,
        navier_stokes=navier_stokes,
        viscosity=1.0,
        vel_forcing=vel_forcing,
        vel_dirichlet_bcs=[("left", zero_vel), ("right", zero_vel)],
        pres_dirichlet_bcs=[("left", zero_pres)],
    )
    param_obj = param_cls(physics, amplitude, bkd)
    adapter = create_galerkin_physics_ode_residual(physics, param_obj)
    stepper = create_stepper(method, adapter)
    wrapper = create_galerkin_bc_enforcing_residual(stepper, physics, bkd)
    newton = NewtonSolver(wrapper)
    newton.set_options(maxiters=50, atol=1e-12, rtol=0.0)
    integrator = TimeIntegrator(0.0, _FINAL_TIME, _DELTAT, newton)
    return integrator, adapter, physics


class TestStokesAdjointGradient:
    @pytest.mark.parametrize("navier_stokes", [False, True])
    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_endpoint_gradient_matches_fd(
        self, numpy_bkd: NumpyBkd, method: str, navier_stokes: bool
    ) -> None:
        bkd = numpy_bkd
        integrator, adapter, physics = _build_pipeline(
            bkd, method, navier_stokes
        )
        nstates = physics.nstates()
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(physics.vel_ndofs()) if ii not in constrained
        )
        functional = EndpointFunctional(state_idx, nstates, 2, bkd)
        integrator.set_functional(functional)
        y0 = bkd.asarray(np.zeros(nstates))

        def solve(params_1d: NumpyArray) -> Tuple[NumpyArray, NumpyArray]:
            adapter.set_param(params_1d)
            return integrator.solve(y0)

        def qoi_of_params(params: NumpyArray) -> NumpyArray:
            results = []
            for ii in range(params.shape[1]):
                sols, _ = solve(params[:, ii])
                results.append(float(bkd.to_numpy(sols)[state_idx, -1]))
            return bkd.reshape(
                bkd.asarray(np.array(results)), (1, params.shape[1])
            )

        def adjoint_gradient(params: NumpyArray) -> NumpyArray:
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
        sample = bkd.asarray(np.array([[1.2], [0.7]]))
        errors = checker.check_derivatives(sample, relative=True)[0]
        # One-sided FD of iteratively solved quantities floors the
        # error ratio near 1e-6 (the V-bottom depth is
        # direction-dependent); assert the V-bottom (a genuine bug
        # plateaus at >= 1e-2 here) and a looser ratio.
        err_min = float(bkd.to_numpy(bkd.min(errors)))
        assert err_min <= 1e-6
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5

    def test_ic_param_on_algebraic_dofs_raises(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """An IC parameterization with support on algebraic (pressure)
        DOFs is rejected by the DAE lambda_0 handling."""
        bkd = numpy_bkd
        integrator, adapter, physics = _build_pipeline(
            bkd, "backward_euler", False, param_cls=_BadICParameterization
        )
        nstates = physics.nstates()
        functional = EndpointFunctional(0, nstates, 2, bkd)
        integrator.set_functional(functional)
        y0 = bkd.asarray(np.zeros(nstates))
        adapter.set_param(bkd.asarray(np.array([1.2, 0.7])))
        sols, times = integrator.solve(y0)
        with pytest.raises(NotImplementedError, match="algebraic"):
            integrator.gradient(
                sols, times, bkd.asarray(np.array([[1.2], [0.7]]))
            )

"""Tests for the Galerkin BC-enforcing time residual wrapper."""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse

from pyapprox.ode.protocols.ode_residual import ODEResidualProtocol
from pyapprox.ode.protocols.time_stepping import (
    SensitivityStepperProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.ode.time_quadrature import left_rectangle_quadrature
from pyapprox.pde.boundary import DofSignal
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolutionAdapter,
    create_adr_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    GalerkinBCEnforcingForwardResidual,
    create_galerkin_bc_enforcing_residual,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.util.backends.numpy import NumpyBkd


def _setup_adr(
    bkd: NumpyBkd, nx: int = 8
) -> Tuple[GalerkinSystem[Any], Any]:
    """1D ADR system with time-dependent Dirichlet-friendly solution."""
    bounds = [0.0, 1.0]
    functions, _ = create_adr_manufactured_test(
        bounds=bounds,
        sol_str="(1-x)*x*(1+T)",
        diff_str="4+1e-16*x",
        react_str="0*u",
        vel_strs=["0+1e-16*x"],
        bkd=bkd,
        time_dependent=True,
    )
    mesh = StructuredMesh1D(nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)
    adapter = GalerkinManufacturedSolutionAdapter(
        basis, functions, bkd, time_dependent=True
    )
    bc_set = adapter.create_boundary_conditions(["D", "D"], robin_alpha=1.0)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=4.0,
        bkd=bkd,
        forcing=adapter.forcing_for_galerkin(),
    )
    return compose_galerkin_system(physics, bc_set.all_conditions()), adapter


def _wrapped_backward_euler(
    bkd: NumpyBkd, nx: int = 8
) -> Tuple[
    GalerkinBCEnforcingForwardResidual[Any],
    GalerkinSystem[Any],
]:
    system, _ = _setup_adr(bkd, nx)
    ode_adapter = GalerkinPhysicsToODEResidualAdapter(system)
    stepper = create_stepper("backward_euler", ode_adapter)
    wrapper = create_galerkin_bc_enforcing_residual(
        stepper, system.constraint_set(), bkd
    )
    return wrapper, system


class _FakeExplicitStepper:
    """Minimal SensitivityStepperProtocol with a constant Jacobian."""

    def __init__(self, bkd: NumpyBkd, n: int) -> None:
        self._bkd = bkd
        self._n = n
        self.jacobian_calls = 0

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def bind(self, ctx: StepContext[Any]) -> None:
        pass

    def __call__(self, state: Any) -> Any:
        return self._bkd.asarray(np.zeros(self._n))

    def jacobian(self, state: Any) -> Any:
        self.jacobian_calls += 1
        return self._bkd.asarray(2.0 * np.eye(self._n))

    def linsolve(self, state: Any, residual: Any) -> Any:
        return residual

    def is_one_step_solvable(self) -> bool:
        return True

    def is_multistage(self) -> bool:
        return False

    def trajectory_quadrature(self, times: Any) -> Any:
        # Explicit one-step fake: left-rectangle is its honest rule.
        return left_rectangle_quadrature(times, self._bkd)

    def native_residual(self) -> ODEResidualProtocol[Any]:
        # A method (as the protocol declares), NOT a property: python
        # 3.11's runtime_checkable isinstance uses hasattr, which
        # invokes properties, so a raising property breaks the
        # adapter's protocol check on 3.11 (3.12+ uses getattr_static)
        raise NotImplementedError

    def is_explicit(self) -> bool:
        return True

    def has_prev_state_hessian(self) -> bool:
        return False

    def sensitivity_off_diag_jacobian(
        self, ctx: StepContext[Any], y_curr: Any
    ) -> Any:
        return self._bkd.asarray(np.full((self._n, self._n), 3.0))


class TestForwardWrapper:
    def test_factory_returns_forward_wrapper(self, numpy_bkd: NumpyBkd) -> None:
        wrapper, _ = _wrapped_backward_euler(numpy_bkd)
        assert isinstance(wrapper, GalerkinBCEnforcingForwardResidual)
        assert isinstance(wrapper, SensitivityStepperProtocol)

    def test_rejects_non_stepper(self, numpy_bkd: NumpyBkd) -> None:
        system, _ = _setup_adr(numpy_bkd)
        not_a_stepper: Any = object()
        with pytest.raises(TypeError, match="SensitivityStepperProtocol"):
            GalerkinBCEnforcingForwardResidual(
                not_a_stepper, system.constraint_set(), numpy_bkd
            )

    def test_residual_constraint_rows(self, numpy_bkd: NumpyBkd) -> None:
        """Constrained rows become y[d] - g(t_{n+1})."""
        bkd = numpy_bkd
        wrapper, system = _wrapped_backward_euler(bkd)
        n = system.nstates()
        t_n, dt = 0.0, 0.5
        y_prev = bkd.asarray(np.zeros(n))
        wrapper.bind(StepContext(t_prev=t_n, deltat=dt, y_prev=y_prev))

        state = bkd.asarray(np.linspace(0.3, 0.7, n))
        residual = wrapper(state)

        constraint_set = system.constraint_set()
        dofs = bkd.to_numpy(constraint_set.dofs())
        values = bkd.to_numpy(constraint_set.values(t_n + dt))
        residual_np = bkd.to_numpy(residual)
        state_np = bkd.to_numpy(state)
        bkd.assert_allclose(
            bkd.asarray(residual_np[dofs]),
            bkd.asarray(state_np[dofs] - values),
        )

    def test_jacobian_identity_rows_stay_sparse(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        wrapper, system = _wrapped_backward_euler(bkd)
        n = system.nstates()
        y_prev = bkd.asarray(np.zeros(n))
        wrapper.bind(StepContext(t_prev=0.0, deltat=0.5, y_prev=y_prev))

        state = bkd.asarray(np.linspace(0.3, 0.7, n))
        jacobian = wrapper.jacobian(state)
        assert issparse(jacobian)

        dofs = bkd.to_numpy(system.constraint_set().dofs())
        dense = jacobian.toarray()
        expected = np.eye(n)[dofs]
        bkd.assert_allclose(bkd.asarray(dense[dofs]), bkd.asarray(expected))

    def test_linsolve_matches_dense_solve(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        wrapper, system = _wrapped_backward_euler(bkd)
        n = system.nstates()
        y_prev = bkd.asarray(np.zeros(n))
        wrapper.bind(StepContext(t_prev=0.0, deltat=0.5, y_prev=y_prev))

        state = bkd.asarray(np.linspace(0.3, 0.7, n))
        rhs = bkd.asarray(np.random.RandomState(5).randn(n))
        delta = wrapper.linsolve(state, rhs)

        dense = wrapper.jacobian(state).toarray()
        expected: NDArray[np.floating[Any]] = np.linalg.solve(
            dense, bkd.to_numpy(rhs)
        )
        bkd.assert_allclose(delta, bkd.asarray(expected), rtol=1e-10)

    def test_sensitivity_off_diag_rows_zeroed(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        wrapper, system = _wrapped_backward_euler(bkd)
        n = system.nstates()
        y_prev = bkd.asarray(np.zeros(n))
        ctx = StepContext(t_prev=0.0, deltat=0.5, y_prev=y_prev)
        wrapper.bind(ctx)

        result = wrapper.sensitivity_off_diag_jacobian(ctx, y_prev)
        dofs = bkd.to_numpy(system.constraint_set().dofs())
        result_np = (
            result.toarray() if issparse(result) else bkd.to_numpy(result)
        )
        bkd.assert_allclose(
            bkd.asarray(result_np[dofs]),
            bkd.asarray(np.zeros((len(dofs), n))),
        )

    def test_zero_adjoint_rhs(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        wrapper, system = _wrapped_backward_euler(bkd)
        n = system.nstates()
        dqdu = bkd.asarray(np.full(n, 2.0))
        out = wrapper.zero_adjoint_rhs(dqdu)
        dofs = bkd.to_numpy(system.constraint_set().dofs())
        out_np = bkd.to_numpy(out)
        bkd.assert_allclose(
            bkd.asarray(out_np[dofs]), bkd.asarray(np.zeros(len(dofs)))
        )
        # flag disables zeroing (BC-parameter gradients)
        assert wrapper.zero_adjoint_rhs(dqdu, zero_essential=False) is dqdu

    def test_delegation(self, numpy_bkd: NumpyBkd) -> None:
        wrapper, _ = _wrapped_backward_euler(numpy_bkd)
        assert not wrapper.is_explicit()
        assert not wrapper.is_one_step_solvable()
        assert wrapper.native_residual is not None

    def test_constant_jacobian_cached_for_one_step_solvable(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """One-step-solvable steppers factorize the Jacobian once."""
        bkd = numpy_bkd
        system, _ = _setup_adr(bkd)
        n = system.nstates()
        fake = _FakeExplicitStepper(bkd, n)
        wrapper = GalerkinBCEnforcingForwardResidual(fake, system.constraint_set(), bkd)
        wrapper.bind(
            StepContext(
                t_prev=0.0, deltat=0.5, y_prev=bkd.asarray(np.zeros(n))
            )
        )
        rhs = bkd.asarray(np.random.RandomState(6).randn(n))
        out1 = wrapper.linsolve(bkd.asarray(np.zeros(n)), rhs)
        out2 = wrapper.linsolve(bkd.asarray(np.zeros(n)), rhs)
        assert fake.jacobian_calls == 1
        bkd.assert_allclose(out1, out2)


def _zero_forcing(x: NDArray[Any]) -> NDArray[Any]:
    return np.zeros(x.shape[1])


def _left_value(x: NDArray[Any]) -> NDArray[Any]:
    return np.full(x.shape[1], 2.0)


def _left_ramp(x: NDArray[Any], time: float) -> NDArray[Any]:
    return np.full(x.shape[1], 2.0 * time)


def _dof_ramp(time: float) -> NDArray[Any]:
    return np.array([2.0 * time])


def _ramp_rate(time: float) -> NDArray[Any]:
    return np.array([2.0])


class TestStageBCRequirement:
    """Analytic g_dot is required for multistage + consistent mass;
    exempt for one-step steppers, lumped mass, and static BCs."""

    def _system_with_callable_bc(
        self, bkd: NumpyBkd, with_derivative: bool
    ) -> GalerkinSystem[Any]:
        from pyapprox.pde.galerkin.boundary import CallableDirichletBC
        from pyapprox.pde.galerkin.mesh import StructuredMesh1D

        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        derivatives = [_ramp_rate] if with_derivative else []
        bc = CallableDirichletBC(
            [0], DofSignal(_dof_ramp, derivatives), bkd
        )
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            bkd=bkd,
            forcing=TimeIndependent(_zero_forcing),
        )
        return compose_galerkin_system(physics, [bc])

    def test_multistage_consistent_mass_missing_gdot_raises(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        system = self._system_with_callable_bc(
            numpy_bkd, with_derivative=False
        )
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        stepper = create_stepper("heun", adapter)
        with pytest.raises(TypeError, match="analytic boundary velocity"):
            create_galerkin_bc_enforcing_residual(
                stepper, system.constraint_set(), numpy_bkd
            )

    def test_one_step_stepper_exempt(self, numpy_bkd: NumpyBkd) -> None:
        system = self._system_with_callable_bc(
            numpy_bkd, with_derivative=False
        )
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        stepper = create_stepper("backward_euler", adapter)
        create_galerkin_bc_enforcing_residual(
            stepper, system.constraint_set(), numpy_bkd
        )

    def test_lumped_mass_exempt(self, numpy_bkd: NumpyBkd) -> None:
        system = self._system_with_callable_bc(
            numpy_bkd, with_derivative=False
        )
        adapter = GalerkinPhysicsToODEResidualAdapter(
            system, lumped_mass=True
        )
        stepper = create_stepper("heun", adapter)
        create_galerkin_bc_enforcing_residual(
            stepper, system.constraint_set(), numpy_bkd
        )

    def test_static_bcs_pass_with_multistage(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        system, _ = _setup_adr(numpy_bkd)  # static + manufactured g_dot
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        stepper = create_stepper("heun", adapter)
        create_galerkin_bc_enforcing_residual(
            stepper, system.constraint_set(), numpy_bkd
        )

    def test_supplied_gdot_passes(self, numpy_bkd: NumpyBkd) -> None:
        system = self._system_with_callable_bc(
            numpy_bkd, with_derivative=True
        )
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        stepper = create_stepper("heun", adapter)
        create_galerkin_bc_enforcing_residual(
            stepper, system.constraint_set(), numpy_bkd
        )

    def _system_with_dirichlet(
        self, bkd: NumpyBkd, value_func: Any
    ) -> GalerkinSystem[Any]:
        mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bc = DirichletBC(basis, "left", value_func, bkd)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            bkd=bkd,
            forcing=TimeIndependent(_zero_forcing),
        )
        return compose_galerkin_system(physics, [bc])

    def test_declared_steady_callable_passes(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A callable declared TimeIndependent needs no g_dot: its
        derivative is exactly zero."""
        system = self._system_with_dirichlet(
            numpy_bkd, TimeIndependent(_left_value)
        )
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        stepper = create_stepper("heun", adapter)
        create_galerkin_bc_enforcing_residual(
            stepper, system.constraint_set(), numpy_bkd
        )

    def test_time_dependent_dirichlet_missing_gdot_names_bc(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        system = self._system_with_dirichlet(
            numpy_bkd, TimeDependent(_left_ramp)
        )
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        stepper = create_stepper("heun", adapter)
        with pytest.raises(TypeError, match="DirichletBC\\(boundary='left'"):
            create_galerkin_bc_enforcing_residual(
                stepper, system.constraint_set(), numpy_bkd
            )

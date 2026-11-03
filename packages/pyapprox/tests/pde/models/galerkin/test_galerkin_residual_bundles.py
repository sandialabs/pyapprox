"""The Galerkin adapter tiers' derivatives bundles, by finite differences.

1D heat equation on [0, 1], P1 on 10 cells, diffusivity exp(W p) from a
three-term log-KLE, forcing (1 + x)(1 + sin 5t), u(0) = 0 and a moving
right end 0.5 sin 4t with its rate supplied. Each tier's bundle matches
its capability: without a parameterization the stack Jacobian alone; a
diffusivity map without second derivatives (no transform_deriv2) gives
the first-order tier; with them, the second-order tier. Every declared
term passes the bundle checker at state 0.3 sin(pi x) + 0.1 x, adjoint
cos(3x), parameters (0.4, -0.3, 0.2), t = 0.2, under both row treatments:
forward-difference errors below 1e-6, dot tests below 1e-12. The initial
state does not depend on the parameters, so the initial-state terms are
checked against a constant y0(p). The physics is linear and its
diffusivity time-independent, so every tier declares a constant
Jacobian and passes the checker's affinity and time-invariance tests,
though the forcing and the right end move; a coordinate diffusivity
declared time-dependent leaves the Jacobian state-free. An assembled
Jacobian with its essential rows left in fails both the Jacobian and the
affinity checks.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.residual_derivatives import Jacobian, ODEResidualDerivatives
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    CoordinateDiffusion,
    NodalFieldDiffusion,
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    NeutralizedRows,
    RowTreatmentFactory,
    SymmetricNeutralizedRows,
    skfem_discretization,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithHVPAdapter,
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.operators import LinearizationPoint, LinearizedOperator
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU

_Np = NDArray[Any]
_NPARAMS = 3
_PARAM = np.array([0.4, -0.3, 0.2])
_TIME = 0.2
_TREATMENTS = [NeutralizedRows, SymmetricNeutralizedRows]


def _forcing(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * (1.0 + np.sin(5.0 * t))
    return value


def _right_value(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 0.5 * np.sin(4.0 * t))
    return value


def _right_rate(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 2.0 * np.cos(4.0 * t))
    return value


class _Heat:
    """The heat system, its diffusivity parameterization and a state and
    adjoint to check at."""

    def __init__(self, bkd: NumpyBkd, second_order: bool) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            forcing=TimeDependent(_forcing),
        )
        right = BoundarySignal(
            TimeDependent(_right_value), [TimeDependent(_right_rate)]
        )
        self.system: GalerkinSystem[_Np] = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", right, bkd),
            ],
        )
        x = np.asarray(basis.dof_coordinates())[0]
        modes = np.stack(
            [0.4 * np.sin((k + 1) * np.pi * x) / (k + 1) for k in range(_NPARAMS)],
            axis=1,
        )
        exp = _ExpTransform(bkd)
        self.parameterization = AdvectionDiffusionParameterization(
            physics,
            diffusivity_map=TransformedFieldMap(
                MeshKLEFieldMap(bkd, bkd.zeros((x.shape[0],)), bkd.asarray(modes)),
                exp,
                exp,
                bkd,
                transform_deriv2=exp if second_order else None,
            ),
            bkd=bkd,
        )
        self.state: _Np = 0.3 * np.sin(np.pi * x) + 0.1 * x
        self.adj: _Np = np.cos(3.0 * x)
        self.nstates = x.shape[0]


def _kappa(x: _Np) -> _Np:
    value: _Np = 1.0 + x[0]
    return value


def _growing_kappa(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + t) * (1.0 + x[0])
    return value


def _coordinate_heat(
    bkd: NumpyBkd, diffusivity: CoordinateDiffusion
) -> GalerkinSystem[_Np]:
    """The heat problem's forcing and moving right end with a coordinate
    diffusivity."""
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis, diffusivity=diffusivity, bkd=bkd, forcing=TimeDependent(_forcing)
    )
    right = BoundarySignal(TimeDependent(_right_value), [TimeDependent(_right_rate)])
    system: GalerkinSystem[_Np] = compose_galerkin_system(
        physics,
        [DirichletBC(basis, "left", 0.0, bkd), DirichletBC(basis, "right", right, bkd)],
    )
    return system


class _ConstantInitialState:
    """y0(p) independent of p, as the problem's initial state is."""

    def __init__(self, nstates: int) -> None:
        self._value: _Np = np.linspace(0.0, 1.0, nstates)

    def __call__(self, param: _Np) -> _Np:
        return self._value


class _RawEssentialRows:
    """Wrong-model control: the adapter's residual with a Jacobian field
    that leaves the assembled essential rows in place, though the
    residual's essential rows do not depend on the state."""

    def __init__(
        self,
        adapter: GalerkinPhysicsToODEResidualAdapter[_Np, CSRFormat],
        discretization: GalerkinStackDiscretization[_Np, _Np, CSRFormat],
    ) -> None:
        self._adapter, self._discretization = adapter, discretization

    def bkd(self) -> Backend[_Np]:
        return self._adapter.bkd()

    def __call__(self, state: _Np, time: float) -> _Np:
        return self._adapter(state, time)

    def _raw(
        self, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[_Np, CSRFormat]:
        # Declared at the adapter's own level (the physics is linear), so
        # the affinity check sees the rows left in.
        state = np.zeros(self._discretization.system().nstates())
        return LinearizedOperator(
            self._discretization.assemble_jacobian(state, time), point
        )

    def derivatives(self) -> ODEResidualDerivatives[_Np, CSRFormat]:
        return replace(
            self._adapter.derivatives(), jacobian=Jacobian.state_free(self._raw)
        )


class TestGalerkinBundles:
    def test_unzeroed_essential_rows_fail_the_check(self, numpy_bkd: NumpyBkd) -> None:
        """The check can fail at this problem's scale: the assembled
        Jacobian, essential rows not zeroed, is refused."""
        heat = _Heat(numpy_bkd, second_order=True)
        discretization = skfem_discretization(heat.system)
        adapter = create_galerkin_physics_ode_residual(discretization, ScipySparseLU())
        report = ODEResidualBundleChecker(
            _RawEssentialRows(adapter, discretization)
        ).check(heat.state, _TIME)
        # The unzeroed rows also break the declared affinity: f's essential
        # rows do not move with the state, those of this Jacobian do.
        assert report.failures() == ["jacobian", "affine"]

    @pytest.mark.parametrize("treatment", _TREATMENTS)
    def test_base_tier_states_the_stack_jacobian(
        self, numpy_bkd: NumpyBkd, treatment: RowTreatmentFactory[_Np]
    ) -> None:
        heat = _Heat(numpy_bkd, second_order=True)
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(heat.system),
            ScipySparseLU(),
            row_treatment=treatment,
        )
        bundle = adapter.derivatives()
        assert bundle.param is None and bundle.state_state is None
        assert bundle.jacobian is not None
        report = ODEResidualBundleChecker(adapter).check(heat.state, _TIME)
        assert set(report.fd_checks) == {"jacobian"}
        assert set(report.structure_errors) == {"affine", "time_invariant"}
        assert report.failures() == []
        assert adapter.derivatives() is bundle  # built once

    @pytest.mark.parametrize("treatment", _TREATMENTS)
    def test_the_jacobian_level_follows_the_diffusivity(
        self, numpy_bkd: NumpyBkd, treatment: RowTreatmentFactory[_Np]
    ) -> None:
        """A diffusivity declared time-independent makes the Jacobian
        constant, though the forcing and the right end move; one declared
        time-dependent leaves it state-free. Either passes the checks its
        level brings."""
        state = 0.3 * np.sin(np.pi * np.linspace(0.0, 1.0, 11))
        for diffusivity, constant in (
            (CoordinateDiffusion(TimeIndependent(_kappa)), True),
            (CoordinateDiffusion(TimeDependent(_growing_kappa)), False),
        ):
            adapter = create_galerkin_physics_ode_residual(
                skfem_discretization(_coordinate_heat(numpy_bkd, diffusivity)),
                ScipySparseLU(),
                row_treatment=treatment,
            )
            jacobian = adapter.derivatives().jacobian
            assert jacobian is not None
            assert jacobian.affine()
            assert jacobian.time_invariant() == constant
            report = ODEResidualBundleChecker(adapter).check(state, _TIME)
            assert ("time_invariant" in report.structure_errors) == constant
            assert report.failures() == []

    def test_the_jacobian_is_the_stack_operator(self, numpy_bkd: NumpyBkd) -> None:
        """The bundle's Jacobian is a sparse stack operator, not a dense
        matrix, and acts as the adapter's matrix Jacobian."""
        heat = _Heat(numpy_bkd, second_order=True)
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(heat.system), ScipySparseLU()
        )
        field = adapter.derivatives().jacobian
        assert field is not None
        op = field(heat.state, _TIME, LinearizationPoint(0, "check"))
        assert isinstance(op.format(), CSRFormat)
        block = np.eye(heat.nstates)[:, :3]
        numpy_bkd.assert_allclose(
            op.apply(block),
            np.asarray(adapter.jacobian(heat.state, _TIME) @ block),
            rtol=1e-12,
            atol=1e-14,
        )

    @pytest.mark.parametrize("treatment", _TREATMENTS)
    def test_first_order_tier(
        self, numpy_bkd: NumpyBkd, treatment: RowTreatmentFactory[_Np]
    ) -> None:
        heat = _Heat(numpy_bkd, second_order=False)
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(heat.system),
            ScipySparseLU(),
            heat.parameterization,
            row_treatment=treatment,
        )
        assert isinstance(adapter, GalerkinPhysicsToODEResidualWithParamJacobianAdapter)
        assert not isinstance(adapter, GalerkinPhysicsToODEResidualWithHVPAdapter)
        param = numpy_bkd.asarray(_PARAM)
        adapter.set_param(param)
        bundle = adapter.derivatives()
        assert bundle.param is not None and bundle.state_state is None
        report = ODEResidualBundleChecker(
            adapter, _ConstantInitialState(heat.nstates)
        ).check(heat.state, _TIME, param=param)
        assert set(report.fd_checks) == {"jacobian", "param", "initial_state"}
        assert set(report.structure_errors) == {"affine", "time_invariant"}
        assert report.failures() == []

    @pytest.mark.parametrize("treatment", _TREATMENTS)
    def test_second_order_tier(
        self, numpy_bkd: NumpyBkd, treatment: RowTreatmentFactory[_Np]
    ) -> None:
        heat = _Heat(numpy_bkd, second_order=True)
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(heat.system),
            ScipySparseLU(),
            heat.parameterization,
            row_treatment=treatment,
        )
        assert isinstance(adapter, GalerkinPhysicsToODEResidualWithHVPAdapter)
        param = numpy_bkd.asarray(_PARAM)
        adapter.set_param(param)
        report = ODEResidualBundleChecker(
            adapter, _ConstantInitialState(heat.nstates)
        ).check(heat.state, _TIME, heat.adj, param)
        assert set(report.fd_checks) == {
            "jacobian",
            "param",
            "initial_state",
            "state_state",
            "state_param",
            "param_state",
            "param_param",
            "initial_state_curvature",
        }
        assert set(report.structure_errors) == {"affine", "time_invariant"}
        assert report.failures() == []

    def test_parameters_live_in_the_states_stack(self, numpy_bkd: NumpyBkd) -> None:
        """The parameter space is a space of the discretization's stack,
        with the state space's dtype."""
        heat = _Heat(numpy_bkd, second_order=True)
        discretization = skfem_discretization(heat.system)
        adapter = create_galerkin_physics_ode_residual(
            discretization, ScipySparseLU(), heat.parameterization
        )
        adapter.set_param(numpy_bkd.asarray(_PARAM))
        space = adapter.derivatives().param_space
        assert space is not None
        assert type(space) is type(discretization.space())
        descriptor = space.descriptor()
        assert (descriptor.name, descriptor.size) == ("param", _NPARAMS)
        assert descriptor.dtype == discretization.descriptor().dtype

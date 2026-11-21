"""Tests for the collocation system and its closed right-hand side.

Setup throughout: advection-diffusion-reaction on a 5 x 5 Chebyshev mesh
of [-1, 1]^2 (diffusion 1.5 or a KLE field, velocity (0.5, -0.2),
reaction -0.4), sides numbered left, right, bottom, top.
"""

from typing import Generic, List, Optional, Tuple

import numpy as np
import pytest

from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsDerivatives,
    ClosureRowsProtocol,
    NormalTermParamFn,
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
    MappedBoundaryData,
)
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    HyperelasticTractionNormalOperator,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic_rows import PeriodicPart, PeriodicRows
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.physics.hyperelasticity import HyperelasticityPhysics
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.models.collocation.closed_system import (
    ClosedRightHandSide,
    CollocationSystem,
)
from pyapprox.pde.models.collocation.right_hand_side import CollocationRightHandSide
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.collocation_hyperelasticity import (
    CollocationHyperelasticityParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet

_N1D = 5
_NPTS = _N1D * _N1D
_TIME = 0.3
_NMODES = 3


def _mesh(bkd: Backend[Array]) -> TransformedMesh2D[Array]:
    return TransformedMesh2D(_N1D, _N1D, bkd)


def _basis(bkd: Backend[Array]) -> ChebyshevBasis2D[Array]:
    return ChebyshevBasis2D(_mesh(bkd), bkd)


def _adr(bkd: Backend[Array]) -> AdvectionDiffusionReaction[Array]:
    velocity = [bkd.full((_NPTS,), 0.5), bkd.full((_NPTS,), -0.2)]
    return AdvectionDiffusionReaction(
        _basis(bkd), bkd, diffusion=1.5, velocity=velocity, reaction=-0.4
    )


def _state(bkd: Backend[Array], seed: int = 3) -> Array:
    return bkd.asarray(np.random.default_rng(seed).normal(0.0, 0.5, _NPTS))


def _indices(bkd: Backend[Array], side: int) -> List[int]:
    return [bkd.to_int(i) for i in _mesh(bkd).boundary_indices(side)]


def _dirichlet(bkd: Backend[Array], sides: Tuple[int, ...]) -> DirichletRows[Array]:
    mesh = _mesh(bkd)
    x = mesh.points()[0]
    parts = [
        DirichletPart(
            mesh.boundary_indices(side),
            NodalField(x[mesh.boundary_indices(side)] ** 2),
        )
        for side in sides
    ]
    return DirichletRows(bkd, StateLayout.uniform(_NPTS, 1), parts)


def _gradient_robin(
    bkd: Backend[Array], sides: Tuple[int, ...], alpha: float = 2.0
) -> RobinRows[Array]:
    mesh, basis = _mesh(bkd), _basis(bkd)
    derivatives = [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)]
    parts = [
        RobinPart(
            mesh.boundary_indices(side),
            GradientNormalOperator(
                bkd,
                mesh.boundary_indices(side),
                mesh.boundary_normals(side),
                derivatives,
            ),
            alpha,
            0.5,
            0.1 * (side + 1),
        )
        for side in sides
    ]
    return RobinRows(bkd, StateLayout.uniform(_NPTS, 1), parts)


def _flux_robin(
    bkd: Backend[Array],
    physics: AdvectionDiffusionReaction[Array],
    sides: Tuple[int, ...],
) -> RobinRows[Array]:
    mesh = _mesh(bkd)
    parts = [
        RobinPart(
            mesh.boundary_indices(side),
            FluxNormalOperator(
                bkd,
                mesh.boundary_indices(side),
                mesh.boundary_normals(side),
                physics,
            ),
            1.0,
            0.7,
            0.2,
        )
        for side in sides
    ]
    return RobinRows(bkd, StateLayout.uniform(_NPTS, 1), parts)


def _periodic_in_x(bkd: Backend[Array]) -> PeriodicRows[Array]:
    mesh = _mesh(bkd)
    part = PeriodicPart(
        mesh.boundary_indices(0),
        mesh.boundary_indices(1),
        _basis(bkd).derivative_matrix(1, 0),
    )
    return PeriodicRows(
        bkd,
        StateLayout.uniform(mesh.reference_points().shape[1], 1),
        mesh.reference_points(),
        [part],
    )


def _parameterized_adr(
    bkd: Backend[Array],
) -> Tuple[AdvectionDiffusionReaction[Array], ParameterState[Array]]:
    physics = _adr(bkd)
    rng = np.random.default_rng(5)
    field_map = MeshKLEFieldMap(
        bkd,
        bkd.full((_NPTS,), 2.0),
        bkd.asarray(rng.uniform(-0.3, 0.3, (_NPTS, _NMODES))),
    )
    parameterization = CollocationAdvectionDiffusionParameterization(
        physics, diffusion_map=field_map, bkd=bkd
    )
    return physics, ParameterState(parameterization)


def _param(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.random.default_rng(7).normal(0.0, 0.3, _NMODES))


class TestSharedNodes:
    def test_essential_beats_closure(self, bkd: Backend[Array]) -> None:
        """Dirichlet on left/right, Robin on all four sides: the Robin rows
        keep only bottom and top without their corners."""
        essential = _dirichlet(bkd, (0, 1))
        system = CollocationSystem(
            _adr(bkd), essential, [_gradient_robin(bkd, (0, 1, 2, 3))]
        )
        (robin,) = system.closure_masks()
        dirichlet = set(_indices(bkd, 0) + _indices(bkd, 1))
        expected = (set(_indices(bkd, 2)) | set(_indices(bkd, 3))) - dirichlet
        assert {i for i in range(_NPTS) if bkd.to_float(robin[i]) == 1.0} == expected
        bkd.assert_allclose(
            system.interior_mask() + robin, bkd.ones((_NPTS,)), atol=0.0
        )

    def test_later_closure_wins(self, bkd: Backend[Array]) -> None:
        """Two Robin declarations sharing the bottom-left corner: the
        later one supplies it."""
        first, second = _gradient_robin(bkd, (0,)), _gradient_robin(bkd, (2,))
        system = CollocationSystem(_adr(bkd), None, [first, second])
        masks = system.closure_masks()
        corner = (set(_indices(bkd, 0)) & set(_indices(bkd, 2))).pop()
        assert bkd.to_float(masks[0][corner]) == 0.0
        assert bkd.to_float(masks[1][corner]) == 1.0
        total = system.interior_mask() + masks[0] + masks[1]
        bkd.assert_allclose(total, bkd.ones((_NPTS,)), atol=0.0)

    def test_rows_take_their_source(self, bkd: Backend[Array]) -> None:
        """f~ is the physics residual off the closure rows (essential rows
        included) and the closure residual on them."""
        physics, closure = _adr(bkd), _gradient_robin(bkd, (2, 3))
        essential = _dirichlet(bkd, (0, 1))
        system = CollocationSystem(physics, essential, [closure])
        rhs = ClosedRightHandSide(system)
        state = _state(bkd)
        value = rhs(state, _TIME)
        (mask,) = system.closure_masks()
        bkd.assert_allclose(
            value,
            (1.0 - mask) * physics.residual(state, _TIME)
            + mask * closure.residual(state, _TIME),
            rtol=1e-14,
        )


class TestPeriodicEssentialConsistency:
    """Periodic in x, Dirichlet on bottom and top: the Dirichlet corners
    are periodically paired, so their data must agree."""

    def _system(self, bkd: Backend[Array], data_power: int) -> CollocationSystem[Array]:
        mesh = _mesh(bkd)
        x = mesh.points()[0]
        parts = [
            DirichletPart(
                mesh.boundary_indices(side),
                NodalField(x[mesh.boundary_indices(side)] ** data_power),
            )
            for side in (2, 3)
        ]
        essential = DirichletRows(bkd, StateLayout.uniform(_NPTS, 1), parts)
        return CollocationSystem(_adr(bkd), essential, [_periodic_in_x(bkd)])

    def test_periodic_data_accepted(self, bkd: Backend[Array]) -> None:
        """x^2 takes the same value at x = -1 and x = 1."""
        essential = self._system(bkd, 2).essential()
        assert essential is not None
        assert essential.values(_TIME).shape == (_NPTS, 1)

    def test_nonperiodic_data_refused(self, bkd: Backend[Array]) -> None:
        """x takes -1 and 1 at paired corners."""
        essential = self._system(bkd, 1).essential()
        assert essential is not None
        with pytest.raises(ValueError, match="periodic declaration pairs"):
            essential.values(_TIME)

    def test_unpaired_essential_rows_unchecked(self, bkd: Backend[Array]) -> None:
        """Dirichlet on the left side only: no pair has both rows pinned,
        so the declaration is passed through unwrapped."""
        essential = _dirichlet(bkd, (0,))
        system = CollocationSystem(_adr(bkd), essential, [_periodic_in_x(bkd)])
        assert system.essential() is essential


class TestBundle:
    def test_affine_time_invariant(self, bkd: Backend[Array]) -> None:
        """ADR with linear Robin rows: a constant Jacobian, checked
        against finite differences."""
        essential = _dirichlet(bkd, (0, 1))
        system = CollocationSystem(_adr(bkd), essential, [_gradient_robin(bkd, (2, 3))])
        rhs = ClosedRightHandSide(system)
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None
        assert jacobian.affine() and jacobian.time_invariant()
        report = ODEResidualBundleChecker(rhs).check(_state(bkd), _TIME)
        assert report.failures() == []

    def test_periodic_closure(self, bkd: Backend[Array]) -> None:
        essential = _dirichlet(bkd, (2, 3))
        system = CollocationSystem(_adr(bkd), essential, [_periodic_in_x(bkd)])
        rhs = ClosedRightHandSide(system)
        report = ODEResidualBundleChecker(rhs).check(_state(bkd), _TIME)
        assert report.failures() == []

    def test_flux_rows_param_jacobian(self, bkd: Backend[Array]) -> None:
        """Flux Robin rows read the parameterized diffusion: f~_p includes
        beta dN/dp, checked against finite differences."""
        physics, parameters = _parameterized_adr(bkd)
        essential = _dirichlet(bkd, (0, 1))
        system = CollocationSystem(
            physics, essential, [_flux_robin(bkd, physics, (2, 3))]
        )
        rhs = ClosedRightHandSide(system, parameters, initial_state=STATED_ZERO)
        assert rhs.derivatives().param is not None
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd), _TIME, param=_param(bkd)
        )
        assert report.failures() == []

    def test_dropping_flux_sensitivity_is_caught(self, bkd: Backend[Array]) -> None:
        """Wrong-model control: the same rows with dN/dp replaced by zero
        fail the parameter check."""
        physics, parameters = _parameterized_adr(bkd)
        rows = _ZeroSensitivity(_flux_robin(bkd, physics, (2, 3)))
        essential = _dirichlet(bkd, (0, 1))
        system = CollocationSystem(physics, essential, [rows])
        rhs = ClosedRightHandSide(system, parameters, initial_state=STATED_ZERO)
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd), _TIME, param=_param(bkd)
        )
        assert "param" in report.failures()

    def test_no_param_jacobian_without_closure_one(self, bkd: Backend[Array]) -> None:
        """A closure without its parameter Jacobian leaves f~ without one,
        rather than a wrong one."""
        physics, parameters = _parameterized_adr(bkd)
        rows = _NoParamJacobian(_flux_robin(bkd, physics, (2, 3)))
        system = CollocationSystem(physics, None, [rows])
        rhs = ClosedRightHandSide(system, parameters, initial_state=STATED_ZERO)
        assert rhs.derivatives().param is None


class TestHyperelasticTraction:
    """Neo-Hookean physics on the 5 x 5 mesh, Young's modulus a KLE field
    about 3 (3 modes, amplitude 0.3), Poisson ratio 0.3; traction Robin
    rows (alpha 0.5, beta 1) on left and bottom for both components;
    displacements of scale 0.01."""

    def _rhs(
        self, bkd: Backend[Array], zero_sensitivity: bool
    ) -> ClosedRightHandSide[Array]:
        mesh = _mesh(bkd)
        physics = HyperelasticityPhysics(
            _basis(bkd), bkd, NeoHookeanLaw(bkd), mu=1.0, lamda=1.5
        )
        rng = np.random.default_rng(11)
        youngs = MeshKLEFieldMap(
            bkd,
            bkd.full((_NPTS,), 3.0),
            bkd.asarray(rng.uniform(-0.3, 0.3, (_NPTS, _NMODES))),
        )
        parameterization = CollocationHyperelasticityParameterization(
            physics, youngs_modulus_map=youngs, poisson_ratio=0.3, bkd=bkd
        )
        parts = [
            RobinPart(
                mesh.boundary_indices(side),
                HyperelasticTractionNormalOperator(
                    bkd,
                    mesh.boundary_indices(side),
                    mesh.boundary_normals(side),
                    physics,
                    component,
                ),
                0.5,
                1.0,
                0.1,
                component,
            )
            for side in (0, 2)
            for component in (0, 1)
        ]
        rows = RobinRows(bkd, StateLayout.uniform(_NPTS, 2), parts)
        closure: ClosureRowsProtocol[Array] = (
            _ZeroSensitivity(rows) if zero_sensitivity else rows
        )
        system = CollocationSystem(physics, None, [closure])
        return ClosedRightHandSide(
            system, ParameterState(parameterization), initial_state=STATED_ZERO
        )

    def _report(self, bkd: Backend[Array], zero_sensitivity: bool) -> List[str]:
        rhs = self._rhs(bkd, zero_sensitivity)
        rng = np.random.default_rng(42)
        state = bkd.asarray(0.01 * rng.standard_normal(2 * _NPTS))
        report = ODEResidualBundleChecker(rhs).check(state, _TIME, param=_param(bkd))
        return report.failures()

    def test_param_jacobian(self, bkd: Backend[Array]) -> None:
        """f~_p includes beta dT/d(mu, lambda) through the Lame chain."""
        assert self._report(bkd, False) == []

    def test_dropping_traction_sensitivity_is_caught(self, bkd: Backend[Array]) -> None:
        assert "param" in self._report(bkd, True)


class TestSecondOrder:
    """Parameterized ADR (KLE diffusion) with Dirichlet left/right and
    linear Robin rows bottom/top."""

    def _rhs(
        self, bkd: Backend[Array]
    ) -> Tuple[ClosedRightHandSide[Array], CollocationRightHandSide[Array]]:
        """The closed right-hand side and the bare interior one, on the same
        physics and parameter state."""
        physics, parameters = _parameterized_adr(bkd)
        essential = _dirichlet(bkd, (0, 1))
        system = CollocationSystem(physics, essential, [_gradient_robin(bkd, (2, 3))])
        return (
            ClosedRightHandSide(system, parameters, initial_state=STATED_ZERO),
            CollocationRightHandSide(physics, parameters, initial_state=STATED_ZERO),
        )

    def test_bundle_checks(self, bkd: Backend[Array]) -> None:
        """Every second-order term against finite differences."""
        rhs, _ = self._rhs(bkd)
        bundle = rhs.derivatives()
        assert bundle.state_param is not None and bundle.param_param is not None
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd), _TIME, adj=_state(bkd, seed=6), param=_param(bkd)
        )
        assert report.failures() == []

    def test_interior_weighted_on_its_rows_only(self, bkd: Backend[Array]) -> None:
        """f~_up[w] is the interior's at w masked to its rows (the Robin
        terms are stated zero), and differs from the interior's at w:
        the masking is what the bundle check above exercises. (f_pp is
        identically zero here: f is linear in the diffusion and the KLE
        map linear in p.)"""
        rhs, bare = self._rhs(bkd)
        rhs.set_param(_param(bkd))
        state, adj = _state(bkd), _state(bkd, seed=6)
        closed = rhs.derivatives().state_param
        interior = bare.derivatives().state_param
        assert closed is not None and interior is not None
        eye = bkd.eye(_NMODES)
        masked = rhs.system().interior_mask() * adj
        bkd.assert_allclose(
            closed(state, _TIME, adj).apply(eye),
            interior(state, _TIME, masked).apply(eye),
            rtol=1e-12,
            atol=1e-14,
        )
        gap = closed(state, _TIME, adj).apply(eye) - interior(state, _TIME, adj).apply(
            eye
        )
        assert bkd.to_float(bkd.max(bkd.abs(gap))) > 1e-3

    def test_first_order_when_a_closure_lacks_curvature(
        self, bkd: Backend[Array]
    ) -> None:
        """Flux Robin rows supply no second parameter derivatives: f~ is
        first order, not second order with zeros."""
        physics, parameters = _parameterized_adr(bkd)
        system = CollocationSystem(physics, None, [_flux_robin(bkd, physics, (2, 3))])
        bundle = ClosedRightHandSide(
            system, parameters, initial_state=STATED_ZERO
        ).derivatives()
        assert bundle.param is not None and bundle.state_param is None


class _QuadraticInterior(Generic[Array]):
    """ADR plus 0.3 u^2: a parameterized interior with nonzero state
    curvature, exact second derivative 0.6 lambda * w. The added term does
    not depend on the parameters, so the ADR parameterization's
    derivatives hold unchanged; ownership is the ADR physics'."""

    def __init__(self, physics: AdvectionDiffusionReaction[Array]) -> None:
        self._physics = physics

    def bkd(self) -> Backend[Array]:
        return self._physics.bkd()

    def nstates(self) -> int:
        return self._physics.nstates()

    def interior_residual(self, state: Array, time: float) -> Array:
        return self._physics.interior_residual(state, time) + 0.3 * state**2

    def interior_jacobian(self, state: Array, time: float) -> Array:
        return self._physics.interior_jacobian(state, time) + self.bkd().diag(
            0.6 * state
        )

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        return StateDerivatives.second_order(self._state_state_hvp)

    def _state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        return 0.6 * adj_state * wvec

    def interior_is_time_invariant(self) -> bool:
        return self._physics.interior_is_time_invariant()

    def interior_jacobian_is_time_invariant(self) -> bool:
        return self._physics.interior_jacobian_is_time_invariant()

    def owns(self, target: object) -> bool:
        return target is self._physics


class TestNonlinearSecondOrder:
    """The quadratic interior with Dirichlet left/right and linear Robin
    rows bottom/top: f~_uu[w] keeps the interior curvature off the Robin
    rows only."""

    def _rhs(self, bkd: Backend[Array]) -> ClosedRightHandSide[Array]:
        physics, parameters = _parameterized_adr(bkd)
        essential = _dirichlet(bkd, (0, 1))
        system = CollocationSystem(
            _QuadraticInterior(physics), essential, [_gradient_robin(bkd, (2, 3))]
        )
        return ClosedRightHandSide(system, parameters, initial_state=STATED_ZERO)

    def test_bundle_checks(self, bkd: Backend[Array]) -> None:
        rhs = self._rhs(bkd)
        bundle = rhs.derivatives()
        assert bundle.jacobian is not None and not bundle.jacobian.affine()
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd), _TIME, adj=_state(bkd, seed=6), param=_param(bkd)
        )
        assert report.failures() == []

    def test_state_curvature_masked(self, bkd: Backend[Array]) -> None:
        """f~_uu[w] = diag(0.6 P w), exactly."""
        rhs = self._rhs(bkd)
        state_state = rhs.derivatives().state_state
        assert state_state is not None
        adj = _state(bkd, seed=6)
        expected = bkd.diag(0.6 * rhs.system().interior_mask() * adj)
        bkd.assert_allclose(
            state_state(_state(bkd), _TIME, adj).apply(bkd.eye(_NPTS)),
            expected,
            rtol=1e-12,
            atol=1e-14,
        )


class TestOwnParametersRefused:
    """Boundary data holding parameters of their own cannot join a system
    with a parameter state: the state must be the parameters' only
    writer."""

    def _mapped(self, bkd: Backend[Array], side: int) -> MappedBoundaryData[Array]:
        mesh = _mesh(bkd)
        x = mesh.points()[0, mesh.boundary_indices(side)]
        field_map = BasisExpansion(bkd, 1.0, [x, x**2])
        return MappedBoundaryData(field_map, bkd.asarray([0.1, 0.2]), bkd)

    def test_dirichlet(self, bkd: Backend[Array]) -> None:
        physics, parameters = _parameterized_adr(bkd)
        mesh = _mesh(bkd)
        essential = DirichletRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [DirichletPart(mesh.boundary_indices(2), self._mapped(bkd, 2))],
        )
        system = CollocationSystem(physics, essential)
        with pytest.raises(ValueError, match="essential data depend"):
            ClosedRightHandSide(system, parameters, initial_state=STATED_ZERO)

    def _robin(
        self,
        bkd: Backend[Array],
        physics: AdvectionDiffusionReaction[Array],
        flux: bool,
    ) -> RobinRows[Array]:
        mesh, basis = _mesh(bkd), _basis(bkd)
        indices, normals = mesh.boundary_indices(2), mesh.boundary_normals(2)
        derivatives = [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)]
        normal_term: "FluxNormalOperator[Array] | GradientNormalOperator[Array]" = (
            FluxNormalOperator(bkd, indices, normals, physics)
            if flux
            else GradientNormalOperator(bkd, indices, normals, derivatives)
        )
        part = RobinPart(indices, normal_term, 1.0, 0.5, self._mapped(bkd, 2))
        return RobinRows(bkd, StateLayout.uniform(_NPTS, 1), [part])

    def test_robin(self, bkd: Backend[Array]) -> None:
        physics, parameters = _parameterized_adr(bkd)
        system = CollocationSystem(physics, None, [self._robin(bkd, physics, False)])
        with pytest.raises(ValueError, match="closure 0 holds parameters"):
            ClosedRightHandSide(system, parameters, initial_state=STATED_ZERO)

    def test_flux_robin(self, bkd: Backend[Array]) -> None:
        """Flux rows: refused by the rows when handed the sensitivity."""
        physics, parameters = _parameterized_adr(bkd)
        system = CollocationSystem(physics, None, [self._robin(bkd, physics, True)])
        with pytest.raises(ValueError, match="parameters of their own"):
            ClosedRightHandSide(system, parameters, initial_state=STATED_ZERO)

    def test_accepted_without_a_parameter_state(self, bkd: Backend[Array]) -> None:
        """Without a parameter state there is no second writer."""
        physics = _adr(bkd)
        system = CollocationSystem(physics, None, [self._robin(bkd, physics, False)])
        rhs = ClosedRightHandSide(system)
        assert rhs.derivatives().param is None


class TestRefusals:
    def test_physics_type(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="CollocationInteriorOperatorProtocol"):
            CollocationSystem("physics")  # type: ignore[arg-type]

    def test_closure_type(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="ClosureRowsProtocol"):
            CollocationSystem(_adr(bkd), None, ["rows"])  # type: ignore[list-item]

    def test_rows_of_another_space(self, bkd: Backend[Array]) -> None:
        """Robin rows declared for two components on a scalar physics."""
        mesh, basis = _mesh(bkd), _basis(bkd)
        derivatives = [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)]
        part = RobinPart(
            mesh.boundary_indices(2),
            GradientNormalOperator(
                bkd, mesh.boundary_indices(2), mesh.boundary_normals(2), derivatives
            ),
            1.0,
            1.0,
            0.0,
        )
        with pytest.raises(ValueError, match="space of size"):
            CollocationSystem(
                _adr(bkd), None, [RobinRows(bkd, StateLayout.uniform(_NPTS, 2), [part])]
            )


class _ZeroSensitivity(Generic[Array]):
    """Robin rows whose dN/dp is replaced by zero."""

    def __init__(self, rows: RobinRows[Array]) -> None:
        self._rows = rows

    def rows(self) -> BackendRowSet[Array]:
        return self._rows.rows()

    def residual(self, state: Array, time: float) -> Array:
        return self._rows.residual(state, time)

    def jacobian(self, state: Array, time: float) -> Array:
        return self._rows.jacobian(state, time)

    def derivatives(
        self, normal_term_param: Optional[NormalTermParamFn[Array]] = None
    ) -> ClosureRowsDerivatives[Array]:
        return self._rows.derivatives(
            None if normal_term_param is None else _zero_sensitivity
        )

    def is_time_invariant(self) -> bool:
        return self._rows.is_time_invariant()


class _NoParamJacobian(_ZeroSensitivity[Array]):
    """Robin rows that supply no parameter Jacobian."""

    def derivatives(
        self, normal_term_param: Optional[NormalTermParamFn[Array]] = None
    ) -> ClosureRowsDerivatives[Array]:
        return self._rows.derivatives(None)


def _zero_sensitivity(state: Array, time: float, rows: Array, normals: Array) -> Array:
    return state[rows][:, None] * 0.0 + state[:_NMODES][None, :] * 0.0

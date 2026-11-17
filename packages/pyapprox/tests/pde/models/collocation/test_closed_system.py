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
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.models.collocation.closed_system import (
    ClosedRightHandSide,
    CollocationSystem,
)
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
    return DirichletRows(bkd, _NPTS, 1, parts)


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
    return RobinRows(bkd, _NPTS, 1, parts)


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
    return RobinRows(bkd, _NPTS, 1, parts)


def _periodic_in_x(bkd: Backend[Array]) -> PeriodicRows[Array]:
    mesh = _mesh(bkd)
    part = PeriodicPart(
        mesh.boundary_indices(0),
        mesh.boundary_indices(1),
        _basis(bkd).derivative_matrix(1, 0),
    )
    return PeriodicRows(bkd, mesh.reference_points(), 1, [part])


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
        essential = DirichletRows(bkd, _NPTS, 1, parts)
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
        rows = RobinRows(bkd, _NPTS, 2, parts)
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
            CollocationSystem(_adr(bkd), None, [RobinRows(bkd, _NPTS, 2, [part])])


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
        return self._rows.derivatives(_zero_sensitivity)

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

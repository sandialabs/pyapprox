r"""The step-end closure of a collocation system.

ADR on a 5 x 5 Chebyshev mesh, Dirichlet on the left and right sides and
Robin (gradient normal term) on the bottom and top, the Dirichlet rows
winning the corners:

- a selected block's residual and own block equal the old physics'
  ``apply_boundary_conditions`` with the same conditions (Robin listed
  first, so Dirichlet wins the corners there too) to 1e-14;
- a linear solve with ``own`` and ``right_hand_side`` satisfies the
  pinned block's equations: the residual of the solution is zero;
- transpose dot test of ``own``;
- ``coupled`` and ``adjoint`` zero every replaced row;
- the placement: the step end by default, every block on request;
- a time-dependent Robin coefficient makes ``own`` follow the time, and
  the pin say so;
- refusals: closure rows not affine in the state, holding parameters of
  their own, or reading a physics coefficient when the system has
  parameters.
"""

from typing import Generic, List, Optional

import numpy as np
import pytest

from pyapprox.ode.pinned_rows import AtEveryBlock, StepBlock
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import constant_dirichlet_bc
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
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
)
from pyapprox.pde.collocation.boundary.robin import RobinBC
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField, NodalSupplier
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.closure_pinning import ClosurePinning
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator, BackendRowSet

_N1D = 5
_NPTS = _N1D * _N1D
_DIRICHLET = {0: 1.0, 1: -0.5}
_ROBIN_SIDES = (2, 3)
_TIME = 0.3


def _mesh(bkd: Backend[Array]) -> TransformedMesh2D[Array]:
    return TransformedMesh2D(_N1D, _N1D, bkd)


def _physics(bkd: Backend[Array]) -> AdvectionDiffusionReaction[Array]:
    return AdvectionDiffusionReaction(
        ChebyshevBasis2D(_mesh(bkd), bkd), bkd, diffusion=1.5
    )


def _gradient(bkd: Backend[Array], side: int) -> GradientNormalOperator[Array]:
    mesh = _mesh(bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    return GradientNormalOperator(
        bkd,
        mesh.boundary_indices(side),
        mesh.boundary_normals(side),
        [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)],
    )


def _robin_data(bkd: Backend[Array], side: int) -> Array:
    mesh = _mesh(bkd)
    return mesh.points()[0, mesh.boundary_indices(side)] ** 2 - 0.3 * side


def _ends(bkd: Backend[Array]) -> DirichletRows[Array]:
    mesh = _mesh(bkd)
    return DirichletRows(
        bkd,
        StateLayout.uniform(_NPTS, 1),
        [DirichletPart(mesh.boundary_indices(s), g) for s, g in _DIRICHLET.items()],
    )


def _robin(bkd: Backend[Array]) -> RobinRows[Array]:
    mesh = _mesh(bkd)
    return RobinRows(
        bkd,
        StateLayout.uniform(_NPTS, 1),
        [
            RobinPart(
                mesh.boundary_indices(side),
                _gradient(bkd, side),
                2.0 + side,
                0.5,
                NodalField(_robin_data(bkd, side)),
            )
            for side in _ROBIN_SIDES
        ],
    )


def _system(
    bkd: Backend[Array], closures: Optional[List[ClosureRowsProtocol[Array]]] = None
) -> CollocationSystem[Array]:
    return CollocationSystem(
        _physics(bkd), _ends(bkd), [_robin(bkd)] if closures is None else closures
    )


def _random(bkd: Backend[Array], *shape: int, seed: int = 3) -> Array:
    return bkd.asarray(np.random.default_rng(seed).standard_normal(shape))


def _operator(bkd: Backend[Array], matrix: Array) -> BackendDenseOperator[Array]:
    return BackendDenseOperator(
        matrix, backend_state_space(bkd, _NPTS).descriptor(), bkd
    )


class TestAgainstTheOldConditions:
    def test_residual_and_own(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        old = _physics(bkd)
        old.set_boundary_conditions(
            [
                RobinBC(
                    bkd,
                    mesh.boundary_indices(side),
                    _gradient(bkd, side),
                    2.0 + side,
                    0.5,
                    _robin_data(bkd, side),
                )
                for side in _ROBIN_SIDES
            ]
            + [
                constant_dirichlet_bc(bkd, mesh.boundary_indices(s), g)
                for s, g in _DIRICHLET.items()
            ]
        )
        block, state = _random(bkd, _NPTS), _random(bkd, _NPTS, seed=4)
        matrix = _random(bkd, _NPTS, _NPTS, seed=5)
        old_residual, old_jacobian = old.apply_boundary_conditions(
            block, matrix, state, _TIME
        )
        pin = ClosurePinning(_system(bkd)).at(StepBlock(0, 1))
        assert pin is not None
        space = backend_state_space(bkd, _NPTS)
        residual = pin.residual(block[:, None], state[:, None], _TIME, space)
        bkd.assert_allclose(residual[:, 0], old_residual, rtol=1e-14, atol=1e-14)
        own = pin.own(_operator(bkd, matrix), _TIME)
        bkd.assert_allclose(own.apply(bkd.eye(_NPTS)), old_jacobian, atol=1e-14)


class TestTheBlockPin:
    def test_a_linear_solve_satisfies_the_block(self, bkd: Backend[Array]) -> None:
        """Y = own(A)^{-1} right_hand_side(b): the pinned residual of the
        block A Y - b vanishes."""
        pin = ClosurePinning(_system(bkd)).at(StepBlock(0, 1))
        assert pin is not None
        matrix = 5.0 * bkd.eye(_NPTS) + _random(bkd, _NPTS, _NPTS, seed=5)
        rhs = _random(bkd, _NPTS, 1, seed=6)
        own = pin.own(_operator(bkd, matrix), _TIME)
        solution = bkd.solve(own.apply(bkd.eye(_NPTS)), pin.right_hand_side(rhs, _TIME))
        space = backend_state_space(bkd, _NPTS)
        residual = pin.residual(matrix @ solution - rhs, solution, _TIME, space)
        bkd.assert_allclose(residual, bkd.zeros((_NPTS, 1)), atol=1e-12)

    def test_own_transpose(self, bkd: Backend[Array]) -> None:
        pin = ClosurePinning(_system(bkd)).at(StepBlock(0, 1))
        assert pin is not None
        own = pin.own(_operator(bkd, _random(bkd, _NPTS, _NPTS, seed=5)), _TIME)
        x, y = _random(bkd, _NPTS, 1, seed=7), _random(bkd, _NPTS, 1, seed=8)
        lhs = bkd.sum(own.apply(x) * y)
        rhs = bkd.sum(x * own.apply_transpose(y))
        bkd.assert_allclose(lhs, rhs, rtol=1e-12)

    def test_coupled_and_adjoint_zero_the_replaced_rows(
        self, bkd: Backend[Array]
    ) -> None:
        system = _system(bkd)
        pin = ClosurePinning(system).at(StepBlock(0, 1))
        assert pin is not None
        replaced = 1.0 - system.interior_mask()
        essential = system.essential()
        assert essential is not None
        replaced = (
            replaced
            + essential.rows().replace_rows(
                bkd.zeros((_NPTS, 1)), bkd.ones((_NPTS, 1))
            )[:, 0]
        )
        kept = (1.0 - replaced)[:, None]
        matrix = _random(bkd, _NPTS, _NPTS, seed=5)
        coupled = pin.coupled(_operator(bkd, matrix))
        bkd.assert_allclose(coupled.apply(bkd.eye(_NPTS)), kept * matrix)
        block = _random(bkd, _NPTS, 1, seed=6)
        space = backend_state_space(bkd, _NPTS)
        bkd.assert_allclose(pin.adjoint(block, space), kept * block)

    def test_placement(self, bkd: Backend[Array]) -> None:
        at_end = ClosurePinning(_system(bkd))
        assert at_end.at(StepBlock(0, 2)) is None
        assert at_end.at(StepBlock(1, 2)) is not None
        every = ClosurePinning(_system(bkd), AtEveryBlock())
        assert every.at(StepBlock(0, 2)) is not None

    def test_time_dependent_coefficient(self, bkd: Backend[Array]) -> None:
        """alpha = 2 + t on the bottom side: own's closure rows are c_u at
        the block's time, and the pin declares own time-dependent."""
        mesh = _mesh(bkd)
        side = 2
        indices = mesh.boundary_indices(side)

        def growing(coords: Array, time: float) -> Array:
            return (2.0 + time) + 0.0 * coords[0]

        alpha = NodalSupplier(TimeDependent(growing), mesh.points()[:, indices])
        rows = RobinRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [RobinPart(indices, _gradient(bkd, side), alpha, 0.5, 0.0)],
        )
        constant = ClosurePinning(_system(bkd)).at(StepBlock(0, 1))
        assert constant is not None and constant.own_is_time_invariant()
        pin = ClosurePinning(_system(bkd, [rows])).at(StepBlock(0, 1))
        assert pin is not None and not pin.own_is_time_invariant()
        matrix = _random(bkd, _NPTS, _NPTS, seed=5)
        zero = bkd.zeros((_NPTS,))
        for time in (0.0, 0.7):
            own = pin.own(_operator(bkd, matrix), time).apply(bkd.eye(_NPTS))
            expected = rows.jacobian(zero, time)
            row = bkd.to_int(indices[2])
            bkd.assert_allclose(own[row], expected[row], rtol=1e-14)


class _Nonlinear(Generic[Array]):
    """Robin rows declaring no state curvature."""

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
        return ClosureRowsDerivatives()

    def is_time_invariant(self) -> bool:
        return True


def _parameters(
    bkd: Backend[Array], physics: AdvectionDiffusionReaction[Array]
) -> ParameterState[Array]:
    field_map = MeshKLEFieldMap(
        bkd,
        bkd.full((_NPTS,), 2.0),
        bkd.asarray(np.random.default_rng(5).uniform(-0.3, 0.3, (_NPTS, 2))),
    )
    return ParameterState(
        CollocationAdvectionDiffusionParameterization(
            physics, diffusion_map=field_map, bkd=bkd
        )
    )


class TestRefusals:
    def test_not_affine(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="affine"):
            ClosurePinning(_system(bkd, [_Nonlinear(_robin(bkd))]))

    def test_own_parameters(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        side = 2
        indices = mesh.boundary_indices(side)
        y = mesh.points()[1, indices]
        data = MappedBoundaryData(
            BasisExpansion(bkd, 0.2, [0.5 * y]), bkd.asarray([0.5]), bkd
        )
        rows = RobinRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [RobinPart(indices, _gradient(bkd, side), 1.0, 0.5, data)],
        )
        with pytest.raises(ValueError, match="parameters of its own"):
            ClosurePinning(_system(bkd, [rows]))

    def test_coefficient_reading_rows_with_parameters(
        self, bkd: Backend[Array]
    ) -> None:
        """A flux normal term reads the diffusion: accepted without
        parameters, refused with them."""
        mesh = _mesh(bkd)
        side = 2
        indices = mesh.boundary_indices(side)
        physics = _physics(bkd)
        flux = FluxNormalOperator(bkd, indices, mesh.boundary_normals(side), physics)
        rows = RobinRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [RobinPart(indices, flux, 1.0, 0.5, 0.0)],
        )
        closures: List[ClosureRowsProtocol[Array]] = [rows]
        system: CollocationSystem[Array] = CollocationSystem(
            physics, _ends(bkd), closures
        )
        ClosurePinning(system)
        with pytest.raises(ValueError, match="physics coefficient"):
            ClosurePinning(system, parameters=_parameters(bkd, physics))

    def test_not_a_system(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="CollocationSystem"):
            ClosurePinning(_physics(bkd))  # type: ignore[arg-type]

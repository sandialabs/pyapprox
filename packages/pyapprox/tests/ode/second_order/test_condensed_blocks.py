"""Tests for the condensed second-order state storage.

The condensed storage must solve exactly what the assembled one solves:
every comparison runs the same operator, rhs or trajectory through both
and asserts agreement, on the dense backend stack (both backends) and,
for the operator solves, the SciPy CSR stack. Then the refusals of what
the condensed solve cannot represent.
"""

from typing import Generic, Optional, Protocol, TypeVar

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.integration import Integrator
from pyapprox.ode.second_order.assembled_blocks import backend_dense_state_blocks
from pyapprox.ode.second_order.condensed_blocks import (
    CondensedFormat,
    backend_dense_condensed_state_blocks,
)
from pyapprox.ode.second_order.reduction import Blocks, StateBlocksProtocol
from pyapprox.ode.second_order.scipy_blocks import (
    scipy_csr_condensed_state_blocks,
    scipy_csr_state_blocks,
)
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    backward_euler,
    crank_nicolson,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    InnerProductSpaceProtocol,
    LinearSolverProtocol,
    OperatorProperties,
    RowReplacedOperator,
    RowsZeroedOperator,
    ScaledSum,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import CSRFormat, NumpyRowSet, ScipyCSROperator
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.ode.second_order_cases import (
    N,
    Spring,
    endpoint_function,
    reduced_system_on,
)

_SPD = OperatorProperties(symmetric=True, spd=True)
_NumpySpace = InnerProductSpaceProtocol[NDArray[np.float64], NDArray[np.float64]]
# The format of a storage's assembled operators.
G = TypeVar("G", bound=FormatProtocol)


def _matrices() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """A mass, a restoring Jacobian K = f_u and a damping Jacobian C = g_v."""
    rng = np.random.default_rng(3)
    root = rng.normal(size=(N, N))
    mass = np.diag([1.0, 2.0, 1.5])
    stiffness = -(root @ root.T + np.eye(N))
    damping = -(0.4 * np.eye(N) + 0.1 * rng.normal(size=(N, N)))
    return mass, stiffness, damping


_Assembled = ActionOperatorProtocol[Array, DenseFormat[Array]]
_Condensed = ActionOperatorProtocol[Array, CondensedFormat[Array, DenseFormat[Array]]]


class _Pair(Generic[Array]):
    """The same operators built in both storages on the dense stack."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self.bkd = bkd
        self.assembled = backend_dense_state_blocks(bkd, N)
        self.condensed = backend_dense_condensed_state_blocks(bkd, N)
        mass, stiffness, damping = _matrices()
        half = self.assembled.half_space().descriptor()

        def op(
            matrix: NDArray[np.float64], properties: Optional[OperatorProperties] = None
        ) -> BackendDenseOperator[Array]:
            return BackendDenseOperator(
                bkd.asarray(matrix), half, bkd, properties=properties
            )

        self.mass, self.stiffness, self.damping = (
            op(mass, _SPD),
            op(stiffness),
            op(damping),
        )

    def _stage(
        self,
        storage: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
        a: float,
        s: float,
    ) -> ScaledSum[Array, G]:
        """a blockdiag(I, M) - s [0, I; K, C] in ``storage``."""
        block_mass = storage.assemble(((storage.identity(), None), (None, self.mass)))
        jacobian = storage.assemble(
            ((None, storage.identity()), (self.stiffness, self.damping))
        )
        return ScaledSum([(a, block_mass), (-s, jacobian)], storage.space())

    def stage_operators(
        self, a: float, s: float
    ) -> tuple[_Assembled[Array], _Condensed[Array]]:
        """The stage operator in the assembled and the condensed storage."""
        return self._stage(self.assembled, a, s), self._stage(self.condensed, a, s)

    def rows(self, indices: list[int]) -> BackendRowSet[Array]:
        return BackendRowSet(self.assembled.space().descriptor(), indices, self.bkd)

    def rhs(self, ncols: int = 2) -> Array:
        return self.bkd.asarray(np.random.default_rng(4).normal(size=(2 * N, ncols)))


class _CSRStorage(Protocol[G]):
    """What the CSR test uses of a storage. ``StateBlocksProtocol``
    bounds its reduction type by the backend's array protocol, as
    ``MassFormSystem`` does, which NumPy arrays do not satisfy."""

    def half_space(self) -> _NumpySpace: ...

    def space(self) -> _NumpySpace: ...

    def identity(self) -> ActionOperatorProtocol[NDArray[np.float64], CSRFormat]: ...

    def assemble(
        self, blocks: Blocks[NDArray[np.float64], CSRFormat]
    ) -> ActionOperatorProtocol[NDArray[np.float64], G]: ...

    def solver(self) -> LinearSolverProtocol[NDArray[np.float64], G]: ...


def _csr_stage_solve(storage: _CSRStorage[G]) -> NDArray[np.float64]:
    """The stage operator with displacement row 0 pinned, solved in
    ``storage`` on the CSR stack."""
    half = storage.half_space().descriptor()
    mass, stiffness, damping = _matrices()

    def op(matrix: NDArray[np.float64]) -> ScipyCSROperator[np.float64]:
        return ScipyCSROperator(csr_matrix(matrix), half, scalar=np.float64)

    block_mass = storage.assemble(((storage.identity(), None), (None, op(mass))))
    jacobian = storage.assemble(
        ((None, storage.identity()), (op(stiffness), op(damping)))
    )
    stage = RowReplacedOperator(
        ScaledSum([(1.0, block_mass), (-0.05, jacobian)], storage.space()),
        NumpyRowSet(storage.space().descriptor(), [0]),
    )
    rhs = np.random.default_rng(5).normal(size=(2 * N, 2))
    solution, info = storage.solver().setup(stage).solve(rhs)
    assert info.converged
    return solution


class TestOperatorSolves:
    @pytest.mark.parametrize("pinned", [[], [0], [0, N], [1, N + 2], [N + 1]], ids=str)
    def test_stage_solve_matches_the_assembled_one(
        self, bkd: Backend[Array], pinned: list[int]
    ) -> None:
        """Pinned displacement rows, velocity rows, or both, solved with
        A and A^T."""
        pair = _Pair(bkd)
        assembled, condensed = pair.stage_operators(1.0, 0.05)
        if pinned:
            rows = pair.rows(pinned)
            assembled = RowReplacedOperator(assembled, rows)
            condensed = RowReplacedOperator(condensed, rows)
        rhs = pair.rhs()
        reference_handle = pair.assembled.solver().setup(assembled)
        handle = pair.condensed.solver().setup(condensed)
        reference, _ = reference_handle.solve(rhs)
        solution, info = handle.solve(rhs)
        assert info.converged
        bkd.assert_allclose(solution, reference, rtol=1e-12, atol=1e-14)
        bkd.assert_allclose(condensed.apply(solution), rhs, rtol=1e-12, atol=1e-13)
        transposed_reference = reference_handle.transpose()
        transposed = handle.transpose()
        assert transposed is not None and transposed_reference is not None
        reference, _ = transposed_reference.solve(rhs)
        solution, info = transposed.solve(rhs)
        assert info.converged
        bkd.assert_allclose(solution, reference, rtol=1e-12, atol=1e-14)

    def test_zeroed_velocity_rows_then_pins(self, bkd: Backend[Array]) -> None:
        """A velocity row zeroed, then replaced: an identity row in both."""
        pair = _Pair(bkd)
        assembled, condensed = pair.stage_operators(2.0, 0.1)
        rows = pair.rows([N + 1])
        assembled = RowReplacedOperator(RowsZeroedOperator(assembled, rows), rows)
        condensed = RowReplacedOperator(RowsZeroedOperator(condensed, rows), rows)
        rhs = pair.rhs(1)
        reference, _ = pair.assembled.solver().setup(assembled).solve(rhs)
        solution, _ = pair.condensed.solver().setup(condensed).solve(rhs)
        bkd.assert_allclose(solution, reference, rtol=1e-12, atol=1e-14)

    def test_block_mass_solve(self, bkd: Backend[Array]) -> None:
        pair = _Pair(bkd)
        mass = pair.condensed.assemble(
            ((pair.condensed.identity(), None), (None, pair.mass))
        )
        assert mass.properties().spd
        rhs = pair.rhs()
        solution, _ = pair.condensed.solver().setup(mass).solve(rhs)
        bkd.assert_allclose(mass.apply(solution), rhs, rtol=1e-13)

    def test_stage_format_holds_the_eliminated_coefficients(
        self, bkd: Backend[Array]
    ) -> None:
        pair = _Pair(bkd)
        _, condensed = pair.stage_operators(2.0, 0.1)
        fmt = condensed.format()
        assert isinstance(fmt, CondensedFormat)
        assert fmt.coefficients() == (2.0, -0.1)

    def test_csr_stack(self) -> None:
        """The same stage solve on the CSR stack, with a pinned
        displacement row, condensed against assembled."""
        half = SpaceDescriptor("u", N, dtype="float64")
        space = SpaceDescriptor("w", 2 * N, dtype="float64")
        assembled = _csr_stage_solve(scipy_csr_state_blocks(half, space))
        condensed = _csr_stage_solve(scipy_csr_condensed_state_blocks(half, space))
        np.testing.assert_allclose(condensed, assembled, rtol=1e-12, atol=1e-14)


class TestRefusals:
    def test_zeroing_a_displacement_row(self, bkd: Backend[Array]) -> None:
        pair = _Pair(bkd)
        _, condensed = pair.stage_operators(1.0, 0.05)
        with pytest.raises(ValueError, match="singular"):
            RowsZeroedOperator(condensed, pair.rows([0])).format()

    def test_a_first_row_block_that_is_not_the_identity(
        self, bkd: Backend[Array]
    ) -> None:
        pair = _Pair(bkd)
        with pytest.raises(ValueError, match="declared the identity"):
            pair.condensed.assemble(((pair.mass, None), (None, pair.mass)))

    def test_no_displacement_coefficient(self, bkd: Backend[Array]) -> None:
        """The Jacobian alone (c1 = 0) has no displacement to eliminate."""
        pair = _Pair(bkd)
        jacobian = pair.condensed.assemble(
            ((None, pair.condensed.identity()), (pair.stiffness, pair.damping))
        )
        with pytest.raises(ValueError, match="c1 = 0"):
            pair.condensed.solver().setup(jacobian)

    def test_sum_after_a_pin_and_edits_once_transposed(
        self, bkd: Backend[Array]
    ) -> None:
        pair = _Pair(bkd)
        _, condensed = pair.stage_operators(1.0, 0.05)
        fmt = condensed.format()
        pinned = fmt.rows_replaced(pair.rows([0]))
        with pytest.raises(TypeError, match="pinned"):
            pinned.scaled_sum(1.0, [(1.0, fmt)])
        with pytest.raises(TypeError, match="pinned"):
            pinned.rows_replaced(pair.rows([1]))
        with pytest.raises(TypeError, match="transposed"):
            fmt.transposed().rows_zeroed(pair.rows([N]))
        with pytest.raises(TypeError, match="not nested"):
            fmt.block_matrix([[fmt]], [2 * N])


def _trajectory(
    blocks: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
    bkd: Backend[Array],
    spring: Spring[Array],
    damper: Optional[str],
    method: str,
    final_time: float,
) -> InMemoryTrajectoryStore[Array]:
    """The reduced spring integrated in ``blocks`` by ``method``."""
    system = reduced_system_on(blocks, bkd, spring, damper)
    tableau = {
        "backward_euler": backward_euler(),
        "crank_nicolson": crank_nicolson(),
        "implicit_midpoint": implicit_midpoint(),
    }[method]
    newton: Newton[Array, Array] = Newton(atol=1e-14, rtol=0.0, maxiters=20)
    stepper = RungeKutta(tableau, newton).stepper(
        system, RecomputeAlways(system.solver)
    )
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    w0 = bkd.reshape(bkd.asarray([0.5, -0.2, 0.1, 0.1, 0.0, -0.2]), (-1, 1))
    Integrator(stepper).solve(w0, FixedStep(0.05, (0.0, final_time)), store)
    return store


def _spring(bkd: Backend[Array], cubic: float = 0.4) -> Spring[Array]:
    spring = Spring(bkd, cubic=cubic)
    spring.set_param(bkd.asarray([0.2, 0.7]))
    return spring


@pytest.mark.parametrize(
    "method", ["backward_euler", "crank_nicolson", "implicit_midpoint"]
)
@pytest.mark.parametrize("damper", [None, "linear", "cubic"])
def test_trajectories_match_the_assembled_storage(
    bkd: Backend[Array], method: str, damper: Optional[str]
) -> None:
    """The nonlinear spring, undamped, linearly and cubically damped."""
    assembled = _trajectory(
        backend_dense_state_blocks(bkd, N), bkd, _spring(bkd), damper, method, 1.0
    )
    condensed = _trajectory(
        backend_dense_condensed_state_blocks(bkd, N),
        bkd,
        _spring(bkd),
        damper,
        method,
        1.0,
    )
    assert assembled.ntimes() == condensed.ntimes() == 21
    for k in range(assembled.ntimes()):
        bkd.assert_allclose(
            condensed.state(k), assembled.state(k), rtol=1e-11, atol=1e-13
        )


def test_midpoint_conserves_energy_in_the_condensed_storage(
    bkd: Backend[Array],
) -> None:
    """A linear undamped spring: the implicit midpoint rule conserves
    its energy whichever storage solves the stages."""
    spring = Spring(bkd, cubic=0.0)
    spring.set_param(bkd.asarray([0.2, 0.0]))
    store = _trajectory(
        backend_dense_condensed_state_blocks(bkd, N),
        bkd,
        spring,
        None,
        "implicit_midpoint",
        10.0,
    )
    mass, stiffness = bkd.asarray([1.0, 2.0, 1.5]), spring.stiffness()

    def energy(w: Array) -> float:
        u, v = w[:N, 0], w[N:, 0]
        return bkd.to_float(
            0.5 * bkd.sum(mass * v**2) + 0.5 * bkd.sum(u * bkd.dot(stiffness, u))
        )

    start = energy(store.state(0))
    drift = max(abs(energy(store.state(k)) - start) for k in range(store.ntimes()))
    assert store.ntimes() == 201
    assert drift <= 1e-12 * start


def _derivatives(
    blocks: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
    bkd: Backend[Array],
    damper: Optional[str],
    sample: Array,
    vec: Array,
) -> tuple[Array, Array]:
    """The endpoint's adjoint Jacobian and HVP through ``blocks``."""
    derivatives = endpoint_function(bkd, damper, sample, blocks).derivatives()
    if derivatives.jacobian is None or derivatives.hvp is None:
        raise AssertionError("the endpoint function has no Jacobian or HVP")
    return derivatives.jacobian(sample), derivatives.hvp(sample, vec)


@pytest.mark.parametrize("damper", [None, "linear"])
def test_gradient_and_hvp_match_the_assembled_storage(
    numpy_bkd: Backend[Array], damper: Optional[str]
) -> None:
    """The adjoint gradient and second-order adjoint HVP through the
    transient map, which solve with the transposed stage operators."""
    bkd = numpy_bkd
    sample, vec = bkd.asarray([[0.2], [0.7]]), bkd.asarray([[0.6], [-0.8]])
    jacobian, hvp = _derivatives(
        backend_dense_state_blocks(bkd, N), bkd, damper, sample, vec
    )
    condensed_jacobian, condensed_hvp = _derivatives(
        backend_dense_condensed_state_blocks(bkd, N), bkd, damper, sample, vec
    )
    bkd.assert_allclose(condensed_jacobian, jacobian, rtol=1e-10)
    bkd.assert_allclose(condensed_hvp, hvp, rtol=1e-10)

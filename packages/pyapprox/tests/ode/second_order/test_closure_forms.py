"""Tests for a relation on the displacement imposed in a second-order
system, with essential rows pinned in both blocks.

The three-mass damped spring of ``second_order_cases``: node 0 prescribed,
g(t) = 0.3 sin(2t) + 0.1 t, and node 2 held by the Robin-like relation
c(u, t) = alpha u_2 + beta (u_2 - u_1) - h(t), h(t) = 0.2 sin t, imposed
directly on the velocity row of node 2. Pinned in every block, each
stored state after the first satisfies the relation and carries g and g'
on node 0, in both storages, which agree.
"""

import math
from typing import Generic

import numpy as np
import pytest

from pyapprox.implicitfunction.implicit_map import ArrayParameters, ImplicitMapFunction
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.affine_rows import AffineRowPinning, AffineRowsProtocol
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import AtEveryBlock, StepBlock
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.ode.second_order.assembled_blocks import backend_dense_state_blocks
from pyapprox.ode.second_order.closure_forms import (
    ClosureBlocksProtocol,
    DirectClosure,
    DisplacementConstraintProtocol,
)
from pyapprox.ode.second_order.condensed_blocks import (
    backend_dense_condensed_state_blocks,
)
from pyapprox.ode.second_order.doubled_rows import DoubledPinnedRows
from pyapprox.ode.second_order.reduction import StateBlocksProtocol
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import backward_euler, crank_nicolson
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TransientImplicitMap,
    TransientPoint,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.ode.second_order_cases import (
    G,
    N,
    Prescribed,
    Spring,
    prescribed_g,
    reduced_system_on,
)

_ALPHA, _BETA = 1.0, 0.5


def _h(time: float) -> float:
    return 0.2 * math.sin(time)


class _Relation(Generic[Array]):
    """c(u, t) = alpha u_2 + beta (u_2 - u_1) - h(t) on node 2."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._half = backend_state_space(bkd, N).descriptor()
        self._rows = BackendRowSet(self._half, [2], bkd)
        matrix = np.zeros((N, N))
        matrix[2, 1], matrix[2, 2] = -_BETA, _ALPHA + _BETA
        self._matrix = bkd.asarray(matrix)

    def rows(self) -> BackendRowSet[Array]:
        return self._rows

    def residual(self, displacement: Array, time: float) -> Array:
        data = self._bkd.asarray([[0.0], [0.0], [_h(time)]])
        return self._bkd.dot(self._matrix, displacement) - data

    def jacobian(self, time: float) -> BackendDenseOperator[Array]:
        return BackendDenseOperator(self._matrix, self._half, self._bkd)

    def is_time_invariant(self) -> bool:
        return True


def _relation_value(bkd: Backend[Array], state: Array, time: float) -> float:
    u1, u2 = state[1, 0], state[2, 0]
    return bkd.to_float(_ALPHA * u2 + _BETA * (u2 - u1)) - _h(time)


class TestDirectClosure:
    def test_rows_residual_and_jacobian(self, bkd: Backend[Array]) -> None:
        """The velocity row of node 2; c(u, t) there from w's displacement;
        the Jacobian [0, 0; C, 0]."""
        blocks = backend_dense_state_blocks(bkd, N)
        relation = _Relation(bkd)
        assert isinstance(relation, DisplacementConstraintProtocol)
        closure = DirectClosure(relation, blocks)
        assert isinstance(closure, AffineRowsProtocol)
        rows = closure.rows()
        assert isinstance(rows, BackendRowSet)
        assert rows.indices() == (N + 2,)
        state = bkd.reshape(bkd.asarray([0.4, -0.3, 0.2, 0.1, 0.5, -0.6]), (-1, 1))
        residual = closure.residual(state, 0.7)
        assert residual.shape == (2 * N, 1)
        bkd.assert_allclose(
            residual[N + 2, 0],
            bkd.asarray(_relation_value(bkd, state, 0.7)),
            rtol=1e-14,
        )
        expected = np.zeros((2 * N, 2 * N))
        expected[N + 2, 1], expected[N + 2, 2] = -_BETA, _ALPHA + _BETA
        bkd.assert_allclose(
            closure.jacobian(0.0).apply(bkd.eye(2 * N)),
            bkd.asarray(expected),
            rtol=1e-14,
        )

    def test_refusals(self, bkd: Backend[Array]) -> None:
        blocks = backend_dense_state_blocks(bkd, N)
        with pytest.raises(TypeError, match="DisplacementConstraintProtocol"):
            DirectClosure[Array, DenseFormat[Array], DenseFormat[Array]](
                None,  # type: ignore[arg-type]
                blocks,
            )
        with pytest.raises(TypeError, match="ClosureBlocksProtocol"):
            DirectClosure[Array, DenseFormat[Array], DenseFormat[Array]](
                _Relation(bkd),
                None,  # type: ignore[arg-type]
            )


def _pinning(
    bkd: Backend[Array],
    blocks: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
    lifting: ClosureBlocksProtocol[Array, DenseFormat[Array], G],
) -> tuple[AffineRowPinning[Array, G], DoubledPinnedRows[Array]]:
    """Node 0 pinned in both blocks, node 2's relation on its velocity row,
    at every block; ``blocks`` and ``lifting`` are the same storage."""
    doubled: DoubledPinnedRows[Array] = DoubledPinnedRows(Prescribed(bkd, 1), lifting)
    closure: DirectClosure[Array, DenseFormat[Array], G] = DirectClosure(
        _Relation(bkd), lifting
    )
    return (
        AffineRowPinning(closure, doubled, blocks.space(), AtEveryBlock()),
        doubled,
    )


class TestAffineRowPinning:
    def test_block_edits(self, bkd: Backend[Array]) -> None:
        """In the assembled storage, a pinned block's own operator has an
        identity row on each essential row and C on node 2's velocity row,
        the rest of op unchanged; its residual and right-hand side read the
        relation there and the data on the essential rows; coupling and
        adjoint rows are zero on both."""
        blocks = backend_dense_state_blocks(bkd, N)
        pinning, doubled = _pinning(bkd, blocks, blocks)
        pin = pinning.at(StepBlock(0, 1))
        assert pin is not None
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(2 * N, 2 * N))
        op = BackendDenseOperator(bkd.asarray(matrix), blocks.space().descriptor(), bkd)
        expected = matrix.copy()
        expected[[0, N], :] = 0.0
        expected[0, 0] = expected[N, N] = 1.0
        expected[N + 2, :] = 0.0
        expected[N + 2, 1], expected[N + 2, 2] = -_BETA, _ALPHA + _BETA
        own = pin.own(op, 0.4)
        bkd.assert_allclose(own.apply(bkd.eye(2 * N)), bkd.asarray(expected))
        bkd.assert_allclose(own.format().matrix(), bkd.asarray(expected), rtol=1e-14)

        time = 0.4
        state = bkd.reshape(bkd.asarray(rng.normal(size=2 * N)), (-1, 1))
        block = bkd.reshape(bkd.asarray(rng.normal(size=2 * N)), (-1, 1))
        space = blocks.space()
        residual = pin.residual(block, state, time, space)
        values = doubled.values(time)
        bkd.assert_allclose(residual[[0, N], 0], (state - values)[[0, N], 0])
        bkd.assert_allclose(
            residual[N + 2, 0], bkd.asarray(_relation_value(bkd, state, time))
        )
        bkd.assert_allclose(residual[[1, 2, N + 1], 0], block[[1, 2, N + 1], 0])
        rhs = pin.right_hand_side(block, time)
        bkd.assert_allclose(rhs[[0, N], 0], values[[0, N], 0])
        bkd.assert_allclose(rhs[N + 2, 0], bkd.asarray(_h(time)))
        zeroed = pin.adjoint(block, space)
        bkd.assert_allclose(zeroed[[0, N, N + 2], 0], bkd.zeros((3,)))
        coupled = pin.coupled(op).apply(bkd.eye(2 * N))
        bkd.assert_allclose(coupled[[0, N, N + 2], :], bkd.zeros((3, 2 * N)))
        assert pin.own_is_time_invariant()


def _trajectory(
    bkd: Backend[Array],
    blocks: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
    lifting: ClosureBlocksProtocol[Array, DenseFormat[Array], G],
    method: str,
) -> InMemoryTrajectoryStore[Array]:
    spring = Spring(bkd)
    spring.set_param(bkd.asarray([0.2, 0.7]))
    system = reduced_system_on(blocks, bkd, spring, "linear")
    pinning, doubled = _pinning(bkd, blocks, lifting)
    tableau = backward_euler() if method == "backward_euler" else crank_nicolson()
    newton: Newton[Array, Array] = Newton(atol=1e-14, rtol=0.0, maxiters=20)
    stepper = RungeKutta(tableau, newton).stepper(
        system, RecomputeAlways(system.solver), pinning
    )
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    w0 = bkd.reshape(bkd.asarray([0.5, -0.2, 0.1, 0.1, 0.0, -0.2]), (-1, 1))
    start = doubled.rows().replace_rows(w0, doubled.values(0.0))
    Integrator(stepper).solve(start, FixedStep(0.05, (0.0, 1.0)), store)
    return store


@pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
def test_trajectory_satisfies_the_relation(bkd: Backend[Array], method: str) -> None:
    """After the first step every stored state satisfies c(u, t) = 0 to
    1e-12 and carries g, g' on node 0, in both storages, which agree to
    1e-11."""
    dense = backend_dense_state_blocks(bkd, N)
    condensed_blocks = backend_dense_condensed_state_blocks(bkd, N)
    assembled = _trajectory(bkd, dense, dense, method)
    condensed = _trajectory(bkd, condensed_blocks, condensed_blocks, method)
    assert assembled.ntimes() == condensed.ntimes() == 21
    for k in range(1, assembled.ntimes()):
        time = assembled.time(k)
        for store in (assembled, condensed):
            state = store.state(k)
            assert abs(_relation_value(bkd, state, time)) <= 1e-12
            bkd.assert_allclose(
                state[[0, N], 0],
                bkd.asarray([prescribed_g(time, 0), prescribed_g(time, 1)]),
                rtol=1e-12,
                atol=1e-14,
            )
        bkd.assert_allclose(
            condensed.state(k), assembled.state(k), rtol=1e-11, atol=1e-13
        )


class _Discretization(Generic[Array, G]):
    def __init__(
        self,
        system: MassFormSystem[Array, Array, G],
        spring: Spring[Array],
        pinning: AffineRowPinning[Array, G],
        doubled: DoubledPinnedRows[Array],
    ) -> None:
        self._system, self._spring = system, spring
        self._pinning, self._doubled = pinning, doubled

    def system(self) -> MassFormSystem[Array, Array, G]:
        return self._system

    def pinning(self) -> AffineRowPinning[Array, G]:
        return self._pinning

    def initial_state(self, state: Array, time: float) -> Array:
        return self._doubled.rows().replace_rows(state, self._doubled.values(time))

    def nparams(self) -> int:
        return 2

    def set_param(self, param: Array) -> None:
        self._spring.set_param(param[:, 0])


def _endpoint(
    bkd: Backend[Array],
    blocks: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
    lifting: ClosureBlocksProtocol[Array, DenseFormat[Array], G],
) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
    """u_1(T) under CN with both pins."""
    spring = Spring(bkd)
    spring.set_param(bkd.asarray([0.2, 0.7]))
    system = reduced_system_on(blocks, bkd, spring, "linear")
    pinning, doubled = _pinning(bkd, blocks, lifting)
    discretization = _Discretization(system, spring, pinning, doubled)
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(EndpointTerm(StateEntries(bkd, 2 * N, [1])),)
        )
    )
    w0 = bkd.reshape(bkd.asarray([0.5, -0.2, 0.1, 0.1, 0.0, -0.2]), (-1, 1))
    implicit = TransientImplicitMap(
        discretization,
        discretization,
        RungeKutta(crank_nicolson(), newton),
        RecomputeAlways(system.solver),
        FixedStep(0.05, (0.0, 0.4)),
        w0,
        functional,
    )
    return ImplicitMapFunction(implicit, ArrayParameters(bkd))


def test_gradient_through_the_relation(numpy_bkd: Backend[Array]) -> None:
    """The adjoint gradient of u_1(T) with both pins equals the condensed
    storage's to 1e-10, and its p1 entry a central difference to 1e-7 (p1
    only: the initial state is fixed while the spring declares du0/dp0)."""
    bkd = numpy_bkd
    sample = bkd.asarray([[0.2], [0.7]])
    dense = backend_dense_state_blocks(bkd, N)
    condensed_blocks = backend_dense_condensed_state_blocks(bkd, N)
    assembled = _endpoint(bkd, dense, dense)
    condensed = _endpoint(bkd, condensed_blocks, condensed_blocks)
    jacobian = assembled.derivatives().jacobian
    condensed_jacobian = condensed.derivatives().jacobian
    assert jacobian is not None and condensed_jacobian is not None
    gradient = jacobian(sample)
    bkd.assert_allclose(condensed_jacobian(sample), gradient, rtol=1e-10)
    step = 1e-5
    shift = bkd.asarray([[0.0], [step]])
    difference = (assembled(sample + shift) - assembled(sample - shift)) / (2 * step)
    bkd.assert_allclose(gradient[0, 1], difference[0, 0], rtol=1e-7)

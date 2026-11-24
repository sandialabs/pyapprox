"""Tests for essential rows pinned in both blocks of a second-order state.

The prescribed displacement is g(t) = 0.3 sin(2t) + 0.1 t on node 0 of the
three-mass spring of ``second_order_cases``, its velocity g'(t) on the same
node of the second block. Pinned in every block, the stored states carry
g and g' there exactly, in either storage, and the adjoint gradient
through the pinned map agrees between the storages and with finite
differences.
"""

from typing import Callable, Generic, Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.implicitfunction.implicit_map import ArrayParameters, ImplicitMapFunction
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import (
    AtEveryBlock,
    PinnedRowsProtocol,
    RowPinning,
)
from pyapprox.ode.residual_derivatives import STATED_ZERO, backend_state_space
from pyapprox.ode.second_order.assembled_blocks import backend_dense_state_blocks
from pyapprox.ode.second_order.condensed_blocks import (
    backend_dense_condensed_state_blocks,
)
from pyapprox.ode.second_order.doubled_rows import (
    DoubledPinnedRows,
    RowLiftingProtocol,
    TimeDerivativeRowsProtocol,
)
from pyapprox.ode.second_order.reduction import StateBlocksProtocol
from pyapprox.ode.second_order.scipy_blocks import (
    scipy_csr_condensed_state_blocks,
    scipy_csr_state_blocks,
)
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
    BackendRowPartition,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.linalg.operators import RowSetProtocol, SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import (
    NumpyRowPartition,
    NumpyRowSet,
    NumpyVectorSpace,
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


class TestJoinRows:
    def test_dense_inverts_split_rows(self, bkd: Backend[Array]) -> None:
        half = backend_state_space(bkd, 4)
        whole = backend_state_space(bkd, 8)
        partition = BackendRowPartition(whole, [half, half], bkd)
        parts = [
            BackendRowSet(half.descriptor(), [1, 3], bkd),
            BackendRowSet(half.descriptor(), [0, 2], bkd),
        ]
        joined = partition.join_rows(parts)
        assert joined.indices() == (1, 3, 4, 6)
        split = partition.split_rows(joined)
        assert [rows.indices() for rows in split] == [(1, 3), (0, 2)]

    def test_numpy_inverts_split_rows(self) -> None:
        half = NumpyVectorSpace(SpaceDescriptor("u", 4, "float64"), np.float64)
        whole = NumpyVectorSpace(SpaceDescriptor("w", 8, "float64"), np.float64)
        partition = NumpyRowPartition(whole, [half, half])
        parts: list[RowSetProtocol[NDArray[np.float64]]] = [
            NumpyRowSet(half.descriptor(), [1, 3]),
            NumpyRowSet(half.descriptor(), [0, 2]),
        ]
        joined = partition.join_rows(parts)
        np.testing.assert_array_equal(joined.indices(), [1, 3, 4, 6])
        whole_rows: RowSetProtocol[NDArray[np.float64]] = joined
        split = partition.split_rows(whole_rows)
        np.testing.assert_array_equal(split[1].indices(), [0, 2])

    def test_refusals(self, bkd: Backend[Array]) -> None:
        """One row set per part, of the stack's type, of its part's space."""
        half = backend_state_space(bkd, 4)
        whole = backend_state_space(bkd, 8)
        partition = BackendRowPartition(whole, [half, half], bkd)
        rows = BackendRowSet(half.descriptor(), [0], bkd)
        with pytest.raises(ValueError, match="2 row sets"):
            partition.join_rows([rows])
        foreign: RowSetProtocol[Array] = NumpyRowSet(  # type: ignore[assignment]
            half.descriptor(), [0]
        )
        with pytest.raises(TypeError, match="BackendRowSets"):
            partition.join_rows([rows, foreign])
        with pytest.raises(ValueError):
            partition.join_rows([rows, BackendRowSet(whole.descriptor(), [0], bkd)])


@pytest.mark.parametrize("storage", ["assembled", "condensed"])
def test_lift_rows(bkd: Backend[Array], storage: str) -> None:
    """Rows of each block offset to their place in w, in both storages."""
    lifting: RowLiftingProtocol[Array] = (
        backend_dense_state_blocks(bkd, N)
        if storage == "assembled"
        else backend_dense_condensed_state_blocks(bkd, N)
    )
    half = backend_state_space(bkd, N).descriptor()
    lifted = lifting.lift_rows(
        BackendRowSet(half, [0, 2], bkd), BackendRowSet(half, [1], bkd)
    )
    assert isinstance(lifted, BackendRowSet)
    assert lifted.indices() == (0, 2, N + 1)


@pytest.mark.parametrize(
    "factory", [scipy_csr_state_blocks, scipy_csr_condensed_state_blocks]
)
def test_lift_rows_on_the_csr_stack(
    factory: Callable[
        [SpaceDescriptor, SpaceDescriptor], RowLiftingProtocol[NDArray[np.float64]]
    ],
) -> None:
    half = SpaceDescriptor("u", N, dtype="float64")
    blocks = factory(half, SpaceDescriptor("w", 2 * N, "float64"))
    lifted = blocks.lift_rows(NumpyRowSet(half, [0]), NumpyRowSet(half, [0]))
    assert isinstance(lifted, NumpyRowSet)
    np.testing.assert_array_equal(lifted.indices(), [0, N])


class TestDoubledPinnedRows:
    def _doubled(self, bkd: Backend[Array], norders: int) -> DoubledPinnedRows[Array]:
        return DoubledPinnedRows(
            Prescribed(bkd, norders), backend_dense_state_blocks(bkd, N)
        )

    def test_values_and_rates(self, bkd: Backend[Array]) -> None:
        """Values [g; g'], rates [g'; g''], order 2 [g''; g''']; a shared
        pin reads it as any declaration."""
        doubled = self._doubled(bkd, 3)
        assert isinstance(doubled, PinnedRowsProtocol)
        assert isinstance(doubled, TimeDerivativeRowsProtocol)
        rows = doubled.rows()
        assert isinstance(rows, BackendRowSet)
        assert rows.indices() == (0, N)
        time = 0.7
        for order, column in [
            (0, doubled.values(time)),
            (1, _call(doubled.rates(), time)),
            (2, _call(doubled.time_derivative(2), time)),
        ]:
            assert column.shape == (2 * N, 1)
            bkd.assert_allclose(
                column[[0, N], 0],
                bkd.asarray([prescribed_g(time, order), prescribed_g(time, order + 1)]),
                rtol=1e-14,
            )
        assert doubled.time_derivative(3) is None
        bundle = doubled.parameter_derivatives()
        assert bundle.values is STATED_ZERO and bundle.rates is STATED_ZERO

    def test_no_rates_without_the_acceleration(self, bkd: Backend[Array]) -> None:
        """With g' alone the velocity rows have their data, but the rates
        [g'; g''] are not supplied."""
        doubled = self._doubled(bkd, 1)
        assert doubled.rates() is None
        bkd.assert_allclose(
            doubled.values(0.2)[N, 0], bkd.asarray(prescribed_g(0.2, 1)), rtol=1e-14
        )

    def test_refusals(self, bkd: Backend[Array]) -> None:
        blocks = backend_dense_state_blocks(bkd, N)
        with pytest.raises(ValueError, match="no rate"):
            DoubledPinnedRows[Array](Prescribed(bkd, 0), blocks)
        with pytest.raises(ValueError, match="independent of the parameters"):
            DoubledPinnedRows[Array](Prescribed(bkd, 1, parameterized=True), blocks)
        with pytest.raises(TypeError, match="TimeDerivativeRowsProtocol"):
            DoubledPinnedRows[Array](None, blocks)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="RowLiftingProtocol"):
            DoubledPinnedRows[Array](Prescribed(bkd, 1), None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least 1"):
            self._doubled(bkd, 1).time_derivative(0)


def _call(fn: Optional[Callable[[float], Array]], time: float) -> Array:
    if fn is None:
        raise AssertionError("the derivative is not supplied")
    return fn(time)


def _spring(bkd: Backend[Array]) -> Spring[Array]:
    spring = Spring(bkd)
    spring.set_param(bkd.asarray([0.2, 0.7]))
    return spring


def _initial_state(bkd: Backend[Array], doubled: DoubledPinnedRows[Array]) -> Array:
    w0 = bkd.reshape(bkd.asarray([0.5, -0.2, 0.1, 0.1, 0.0, -0.2]), (-1, 1))
    return doubled.rows().replace_rows(w0, doubled.values(0.0))


def _pinned_trajectory(
    bkd: Backend[Array],
    blocks: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
    lifting: RowLiftingProtocol[Array],
    method: str,
) -> InMemoryTrajectoryStore[Array]:
    """The damped spring with node 0 pinned at every block, integrated by
    ``method``; ``blocks`` and ``lifting`` are the same storage."""
    system = reduced_system_on(blocks, bkd, _spring(bkd), "linear")
    doubled: DoubledPinnedRows[Array] = DoubledPinnedRows(Prescribed(bkd, 1), lifting)
    tableau = backward_euler() if method == "backward_euler" else crank_nicolson()
    newton: Newton[Array, Array] = Newton(atol=1e-14, rtol=0.0, maxiters=20)
    stepper = RungeKutta(tableau, newton).stepper(
        system, RecomputeAlways(system.solver), RowPinning(doubled, AtEveryBlock())
    )
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    Integrator(stepper).solve(
        _initial_state(bkd, doubled), FixedStep(0.05, (0.0, 1.0)), store
    )
    return store


@pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
def test_pinned_trajectory_carries_the_data(bkd: Backend[Array], method: str) -> None:
    """The damped nonlinear spring pinned in every block: each stored
    state's node 0 is g and g' at its time, in either storage, and the two
    storages give the same trajectory."""
    dense = backend_dense_state_blocks(bkd, N)
    condensed_blocks = backend_dense_condensed_state_blocks(bkd, N)
    assembled = _pinned_trajectory(bkd, dense, dense, method)
    condensed = _pinned_trajectory(bkd, condensed_blocks, condensed_blocks, method)
    assert assembled.ntimes() == condensed.ntimes() == 21
    for k in range(assembled.ntimes()):
        time = assembled.time(k)
        expected = bkd.asarray([prescribed_g(time, 0), prescribed_g(time, 1)])
        for store in (assembled, condensed):
            bkd.assert_allclose(
                store.state(k)[[0, N], 0], expected, rtol=1e-12, atol=1e-14
            )
        bkd.assert_allclose(
            condensed.state(k), assembled.state(k), rtol=1e-11, atol=1e-13
        )


class _PinnedDiscretization(Generic[Array, G]):
    """The reduced spring with node 0 pinned in both blocks at every
    block, for the transient map.

    The initial displacement is the spring's u0(p) = base + p0 e, rebuilt
    from the parameters at every solve, as the spring's bundle declares;
    the given state supplies only the initial velocity. The pins then set
    node 0 of both blocks."""

    def __init__(
        self,
        system: MassFormSystem[Array, Array, G],
        spring: Spring[Array],
        doubled: DoubledPinnedRows[Array],
        blocks: RowLiftingProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        self._system, self._spring, self._doubled = system, spring, doubled
        self._blocks, self._bkd = blocks, bkd
        self._pinning: RowPinning[Array] = RowPinning(doubled, AtEveryBlock())
        self._param = bkd.asarray([0.2, 0.7])

    def system(self) -> MassFormSystem[Array, Array, G]:
        return self._system

    def pinning(self) -> RowPinning[Array]:
        return self._pinning

    def initial_state(self, state: Array, time: float) -> Array:
        displacement = self._bkd.reshape(
            self._spring.initial_displacement(self._param), (N, 1)
        )
        start = self._blocks.join(displacement, state[N:])
        return self._doubled.rows().replace_rows(start, self._doubled.values(time))

    def nparams(self) -> int:
        return 2

    def set_param(self, param: Array) -> None:
        self._param = param[:, 0]
        self._spring.set_param(self._param)


def _pinned_endpoint(
    bkd: Backend[Array],
    blocks: StateBlocksProtocol[Array, Array, DenseFormat[Array], G],
    lifting: RowLiftingProtocol[Array],
) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
    """u_1(T) of the damped spring with node 0 pinned, under CN;
    ``blocks`` and ``lifting`` are the same storage."""
    spring = _spring(bkd)
    system = reduced_system_on(blocks, bkd, spring, "linear")
    doubled: DoubledPinnedRows[Array] = DoubledPinnedRows(Prescribed(bkd, 1), lifting)
    discretization = _PinnedDiscretization(system, spring, doubled, lifting, bkd)
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(EndpointTerm(StateEntries(bkd, 2 * N, [1])),)
        )
    )
    implicit = TransientImplicitMap(
        discretization,
        discretization,
        RungeKutta(crank_nicolson(), newton),
        RecomputeAlways(system.solver),
        FixedStep(0.05, (0.0, 0.4)),
        _initial_state(bkd, doubled),
        functional,
    )
    return ImplicitMapFunction(implicit, ArrayParameters(bkd))


def test_pinned_gradient(numpy_bkd: Backend[Array]) -> None:
    """The adjoint gradient of u_1(T) through the pinned map equals the
    condensed storage's to 1e-10, and its p1 entry (the forcing
    amplitude, on which u0 does not depend) a central difference to
    1e-7."""
    bkd = numpy_bkd
    sample = bkd.asarray([[0.2], [0.7]])
    dense = backend_dense_state_blocks(bkd, N)
    condensed_blocks = backend_dense_condensed_state_blocks(bkd, N)
    assembled = _pinned_endpoint(bkd, dense, dense)
    condensed = _pinned_endpoint(bkd, condensed_blocks, condensed_blocks)
    jacobian = assembled.derivatives().jacobian
    condensed_jacobian = condensed.derivatives().jacobian
    assert jacobian is not None and condensed_jacobian is not None
    gradient = jacobian(sample)
    bkd.assert_allclose(condensed_jacobian(sample), gradient, rtol=1e-10)
    step = 1e-5
    shift = bkd.asarray([[0.0], [step]])
    difference = (assembled(sample + shift) - assembled(sample - shift)) / (2 * step)
    bkd.assert_allclose(gradient[0, 1], difference[0, 0], rtol=1e-7)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the transient map takes dy0/dp from the right-hand side's bundle "
        "unprojected, though initial_state pins rows of y0 to data "
        "independent of p: on pinned rows it should be dg/dp (here 0). "
        "Without pins the same map passes (ratio 3.5e-7)"
    ),
)
def test_pinned_gradient_through_a_parameterized_initial_state(
    numpy_bkd: Backend[Array],
) -> None:
    """With u0(p) = base + p0 e and node 0 pinned, the full gradient
    passes DerivativeChecker."""
    bkd = numpy_bkd
    dense = backend_dense_state_blocks(bkd, N)
    checker: DerivativeChecker[Array] = DerivativeChecker(
        _pinned_endpoint(bkd, dense, dense)
    )
    errors = checker.check_derivatives(bkd.asarray([[0.2], [0.7]]), verbosity=0)
    assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-6

"""The skeleton residual of 1D diffusion chains, its Schur builders, the
point stores, and coupled solves.

Values are checked against independent facts (two halves of [0, 1]
balance at u(1/2) = f/(8k); interface values from q' = f alone); the
Schur action against finite differences of the residual; the stores by
counting linearizations, with results equal under every store.
"""

from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJVPFromCallable,
)
from pyapprox.pde.coupling.analytic import KirchhoffResponse1D
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.responses import (
    PortQuantity,
    ResponseDerivatives,
    ResponsePointProtocol,
    TimeWindow,
)
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    DenseSchur,
    KeepCurrentPoints,
    MatrixFreeSchur,
    PointStoreProtocol,
    RecomputePoints,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition, local_subdomains
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    MatrixFree,
    SpaceDescriptor,
)
from pyapprox.util.linalg.operators.krylov import GMRES
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling import MockPortRows
from tests._helpers.linalg.mock_stack import MockVector
from tests._helpers.linalg.stacks import MockCase, mock_case

_LABEL = LinearizationPoint(0, "test", provisional=True)
T = TypeVar("T")


class _Counting(Generic[Array]):
    """A response that counts its linearizations."""

    def __init__(self, inner: KirchhoffResponse1D[Array]) -> None:
        self._inner = inner
        self.count = 0

    def bkd(self) -> Backend[Array]:
        return self._inner.bkd()

    def nport(self) -> int:
        return self._inner.nport()

    def receives(self) -> PortQuantity:
        return self._inner.receives()

    def returns(self) -> PortQuantity:
        return self._inner.returns()

    def nparams(self) -> int:
        return self._inner.nparams()

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def linearize(
        self, port_data: Array, params: Array, window: TimeWindow
    ) -> ResponsePointProtocol[Array]:
        self.count += 1
        return self._inner.linearize(port_data, params, window)


def _chain(
    bkd: Backend[Array],
    coefs: Sequence[tuple[float, ...]],
    source: float = 1.7,
    store: Optional[PointStoreProtocol[Array, Array]] = None,
) -> tuple[
    SkeletonResidual[Array, Array, Array, MatrixFree],
    BackendVectorSpace[Array],
    list[_Counting[Array]],
]:
    """Subdomains of [0, 1], subdomain i with conductivity ``coefs[i]``,
    u = 0 at both ends."""
    n = len(coefs)
    topo = TensorBoxPartition((n,))
    bases = UniformBases(lobatto_lagrange_basis(0, 1, bkd))
    numbering = VertexWeightNumbering(topo, bases)
    skeleton = ContinuousSkeletonSpace(topo, bases, numbering)
    space = BackendVectorSpace(skeleton.descriptor(), bkd)
    responses, subdomains = [], []
    for s in range(n):
        left = 0.0 if s == 0 else None
        right = 0.0 if s == n - 1 else None
        r = _Counting(KirchhoffResponse1D(1.0 / n, source, coefs[s], left, right, bkd))
        dofs = subdomain_port_dofs(topo, skeleton, s)
        rows = BackendPortRows(skeleton.descriptor(), dofs, bkd)
        responses.append(r)
        subdomains.append(CoupledSubdomain(r, MatchingNodalTransfer(rows)))
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    residual = SkeletonResidual(space, subdomains, schur, store)
    return residual, space, responses


def test_two_halves_balance_at_the_exact_value(bkd: Backend[Array]) -> None:
    k, f = 2.0, 1.7
    residual, _, _ = _chain(bkd, [(k,), (k,)], f)
    assert isinstance(residual, ResidualProtocol)
    exact = bkd.asarray([[f / (8.0 * k)]])
    bkd.assert_allclose(residual(exact), bkd.zeros((1, 1)), atol=1e-14)
    bkd.assert_allclose(
        residual(exact + 0.1), bkd.asarray([[0.1 * 2 * k / 0.5]]), rtol=1e-12
    )


@pytest.mark.parametrize(
    "coefs", [[(2.0,), (0.5,), (3.0,)], [(1.5, 0.0, 1.5), (1.0, 0.4), (2.0,)]]
)
def test_schur_action_matches_finite_differences(
    bkd: Backend[Array], coefs: list[tuple[float, ...]]
) -> None:
    residual, space, _ = _chain(bkd, coefs)
    jacobian = residual.derivatives().jacobian
    assert jacobian is not None
    n = space.descriptor().size

    def fun(samples: Array) -> Array:
        cols = [residual(samples[:, j : j + 1]) for j in range(samples.shape[1])]
        return bkd.hstack(cols)

    def jvp(sample: Array, vec: Array) -> Array:
        return jacobian(sample, _LABEL).apply(vec)

    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=n, nvars=n, fun=fun, jvp=jvp, bkd=bkd)
    ).check_derivatives(bkd.asarray([[0.3], [-0.4]]), relative=True, central=True)[0]
    assert bkd.to_float(bkd.min(errors)) <= 1e-6


def test_schur_acts_on_blocks(bkd: Backend[Array]) -> None:
    residual, _, _ = _chain(bkd, [(1.5, 0.0, 1.5), (1.0,), (2.0, 1.0)])
    jacobian = residual.derivatives().jacobian
    assert jacobian is not None
    op = jacobian(bkd.asarray([[0.3], [-0.4]]), _LABEL)
    block = bkd.asarray([[1.0, 0.2, -0.5], [0.4, -1.0, 2.0]])
    columns = bkd.hstack([op.apply(block[:, j : j + 1]) for j in range(3)])
    bkd.assert_allclose(op.apply(block), columns, rtol=1e-12)


def test_dense_schur_matches_the_matrix_free_action(bkd: Backend[Array]) -> None:
    coefs = [(1.5, 0.0, 1.5), (1.0, 0.4), (2.0,), (0.5,)]
    residual, space, _ = _chain(bkd, coefs)
    state = bkd.asarray([[0.3], [-0.4], [0.7]])
    free = residual.derivatives().jacobian
    assert free is not None
    subdomains = residual.subdomains()
    dense = SkeletonResidual(space, subdomains, DenseSchur(bkd))
    dense_jacobian = dense.derivatives().jacobian
    assert dense_jacobian is not None
    op = dense_jacobian(state, _LABEL)
    assert isinstance(op.format(), DenseFormat)
    identity = bkd.eye(3)
    bkd.assert_allclose(
        op.format().matrix(), free(state, _LABEL).apply(identity), rtol=1e-12
    )


def _value_and_jacobian(
    residual: SkeletonResidual[Array, Array, Array, MatrixFree],
    state: Array,
    bkd: Backend[Array],
) -> Array:
    jacobian = residual.derivatives().jacobian
    assert jacobian is not None
    value = residual(state)
    return bkd.hstack([value, jacobian(state, _LABEL).apply(bkd.ones((2, 1)))])


def test_stores_change_only_the_number_of_linearizations(
    bkd: Backend[Array],
) -> None:
    coefs = [(1.5, 0.0, 1.5), (1.0,), (2.0,)]
    state = bkd.asarray([[0.3], [-0.4]])
    kept, _, kept_responses = _chain(bkd, coefs, store=KeepCurrentPoints())
    fresh, _, fresh_responses = _chain(bkd, coefs, store=RecomputePoints())
    bkd.assert_allclose(
        _value_and_jacobian(kept, state, bkd),
        _value_and_jacobian(fresh, state, bkd),
        rtol=1e-14,
    )
    # A value and a Jacobian at one state: one linearization per subdomain
    # when kept, two when recomputed.
    assert [r.count for r in kept_responses] == [1, 1, 1]
    assert [r.count for r in fresh_responses] == [2, 2, 2]
    # A state equal by value, but a different object, reuses the points.
    kept(bkd.copy(state))
    assert [r.count for r in kept_responses] == [1, 1, 1]
    kept(state + 1e-3)
    assert [r.count for r in kept_responses] == [2, 2, 2]


def test_keep_current_survives_overwriting_and_clears(bkd: Backend[Array]) -> None:
    store: KeepCurrentPoints[Array, Array] = KeepCurrentPoints()
    residual, _, responses = _chain(bkd, [(1.0,), (2.0,)], store=store)
    state = bkd.asarray([[0.2]])
    first = residual(state)
    state[0, 0] = 0.9  # a solver overwriting its iterate in place
    second = residual(state)
    assert [r.count for r in responses] == [2, 2]
    assert bkd.to_float(bkd.max(bkd.abs(first - second))) > 0.0
    store.clear()
    residual(state)
    assert [r.count for r in responses] == [3, 3]


class _NoTangent(Generic[Array]):
    def __init__(self, inner: ResponsePointProtocol[Array]) -> None:
        self._inner = inner

    def output(self) -> Array:
        return self._inner.output()

    def qoi(self) -> Array:
        return self._inner.qoi()

    def derivatives(self) -> ResponseDerivatives[Array]:
        return ResponseDerivatives.none()


class _Untangent(_Counting[Array]):
    def linearize(
        self, port_data: Array, params: Array, window: TimeWindow
    ) -> ResponsePointProtocol[Array]:
        return _NoTangent(super().linearize(port_data, params, window))


def test_invalid_input(bkd: Backend[Array]) -> None:
    residual, space, responses = _chain(bkd, [(1.0,), (2.0,)])
    jacobian = residual.derivatives().jacobian
    assert jacobian is not None
    op = jacobian(bkd.zeros((1, 1)), _LABEL)
    with pytest.raises(TypeError, match="vjp_trace"):
        op.apply_transpose(bkd.zeros((1, 1)))
    transfer = MatchingNodalTransfer(BackendPortRows(space.descriptor(), [0], bkd))
    two_ports = KirchhoffResponse1D(1.0, 1.0, (1.0,), None, None, bkd)
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    with pytest.raises(ValueError, match="port rows"):
        SkeletonResidual(space, [CoupledSubdomain(two_ports, transfer)], schur)
    with pytest.raises(TypeError):
        SkeletonResidual(object(), [], schur)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="schur"):
        SkeletonResidual(space, [], object())  # type: ignore[arg-type]
    untangent = _Untangent(KirchhoffResponse1D(0.5, 1.0, (1.0,), 0.0, None, bkd))
    other = KirchhoffResponse1D(0.5, 1.0, (1.0,), None, 0.0, bkd)
    no_jvp = SkeletonResidual(
        space,
        [CoupledSubdomain(untangent, transfer), CoupledSubdomain(other, transfer)],
        schur,
    )
    no_jvp_jacobian = no_jvp.derivatives().jacobian
    assert no_jvp_jacobian is not None
    with pytest.raises(TypeError, match="subdomain 0"):
        no_jvp_jacobian(bkd.zeros((1, 1)), _LABEL)


def _exact_piecewise_constant(conductivities: Sequence[float], f: float) -> list[float]:
    """Interface values of -(k u')' = f on [0, 1], u(0) = u(1) = 0, with
    constant k_i on [i/n, (i+1)/n], from q' = f alone (q = -k u')."""
    n = len(conductivities)
    ends = [(i / n, (i + 1) / n) for i in range(n)]
    # u(1) = -sum_i int (q0 + f x)/k_i dx = 0 fixes q0.
    q0 = -sum(f * (b * b - a * a) / (2 * k) for (a, b), k in zip(ends, conductivities))
    q0 /= sum((b - a) / k for (a, b), k in zip(ends, conductivities))
    values, u = [], 0.0
    for (a, b), k in zip(ends[:-1], conductivities[:-1]):
        u -= (q0 * (b - a) + f * (b * b - a * a) / 2) / k
        values.append(u)
    return values


def _exact_one_nonlinear(coefs: Sequence[float], f: float, n: int) -> list[float]:
    """Interface values for one k(u) = sum_j c_j u^j throughout: K(u(x)) =
    f x (1 - x) / 2, inverted by a scalar Newton iteration."""
    values = []
    for j in range(1, n):
        x, u = j / n, 0.0
        target = f * x * (1 - x) / 2
        for _ in range(50):
            big_k = sum(c * u ** (m + 1) / (m + 1) for m, c in enumerate(coefs))
            small_k = sum(c * u**m for m, c in enumerate(coefs))
            u -= (big_k - target) / small_k
        values.append(u)
    return values


def _gmres_solve(
    residual: SkeletonResidual[Array, Array, Array, MatrixFree],
    space: BackendVectorSpace[Array],
    bkd: Backend[Array],
) -> SteadyPoint[Array, Array]:
    setups: RecomputeAlways[Array, MatrixFree] = RecomputeAlways(
        GMRES(space, bkd, tol=1e-13)
    )
    system = SteadySystem(residual, space, setups)
    return system.solve(Newton(atol=1e-11, rtol=1e-11, maxiters=20), space.zeros(1))


_PIECEWISE = [(2.0,), (0.5,), (3.0,), (1.0,)]


def test_linear_chain_solves_in_one_newton_step(bkd: Backend[Array]) -> None:
    f = 1.7
    residual, space, _ = _chain(bkd, _PIECEWISE, f)
    point = _gmres_solve(residual, space, bkd)
    assert point.converged() and point.info().iterations == 1
    exact = _exact_piecewise_constant([k[0] for k in _PIECEWISE], f)
    bkd.assert_allclose(point.solution(), bkd.asarray([[v] for v in exact]), rtol=1e-10)


def test_nonlinear_chain_converges_to_the_exact_values(bkd: Backend[Array]) -> None:
    coefs, f, n = (1.5, 0.0, 1.5), 6.0, 5
    residual, space, responses = _chain(bkd, [coefs] * n, f)
    point = _gmres_solve(residual, space, bkd)
    assert point.converged() and point.info().iterations > 1
    exact = _exact_one_nonlinear(coefs, f, n)
    bkd.assert_allclose(point.solution(), bkd.asarray([[v] for v in exact]), rtol=1e-10)
    # Kept points: one linearization per distinct iterate, never one per
    # Krylov product.
    iterations = point.info().iterations
    assert all(r.count == iterations + 1 for r in responses)
    fresh, _, fresh_responses = _chain(bkd, [coefs] * n, f, store=RecomputePoints())
    again = _gmres_solve(fresh, space, bkd)
    bkd.assert_allclose(again.solution(), point.solution(), rtol=1e-12)
    assert all(r.count > iterations + 1 for r in fresh_responses)


def test_dense_schur_solves_with_a_direct_method(bkd: Backend[Array]) -> None:
    f = 1.7
    free, space, _ = _chain(bkd, _PIECEWISE, f)
    dense: SkeletonResidual[Array, Array, Array, DenseFormat[Array]] = SkeletonResidual(
        space, free.subdomains(), DenseSchur(bkd)
    )
    setups: RecomputeAlways[Array, DenseFormat[Array]] = RecomputeAlways(
        BackendDenseLU(bkd)
    )
    point = SteadySystem(dense, space, setups).solve(
        Newton(atol=1e-11, rtol=1e-11), space.zeros(1)
    )
    assert point.converged() and point.info().iterations == 1
    exact = _exact_piecewise_constant([k[0] for k in _PIECEWISE], f)
    bkd.assert_allclose(point.solution(), bkd.asarray([[v] for v in exact]), rtol=1e-12)


def _mock_chain(
    coefs: Sequence[tuple[float, ...]], f: float
) -> tuple[
    SkeletonResidual[MockVector, NDArray[Any], NDArray[Any], MatrixFree],
    MockCase,
    SpaceDescriptor,
    list[_Counting[NDArray[Any]]],
]:
    """:func:`_chain` on the opaque mock stack: skeleton vectors are mock
    handles, port blocks and responses NumPy."""
    bkd, case = NumpyBkd(), mock_case()
    n = len(coefs)
    topo = TensorBoxPartition((n,))
    bases = UniformBases(lobatto_lagrange_basis(0, 1, bkd))
    numbering = VertexWeightNumbering(topo, bases)
    skeleton = ContinuousSkeletonSpace(topo, bases, numbering)
    descriptor = skeleton.descriptor()
    responses, subdomains = [], []
    for s in range(n):
        left = 0.0 if s == 0 else None
        right = 0.0 if s == n - 1 else None
        r = _Counting(KirchhoffResponse1D(1.0 / n, f, coefs[s], left, right, bkd))
        rows = MockPortRows(descriptor, subdomain_port_dofs(topo, skeleton, s))
        responses.append(r)
        subdomains.append(CoupledSubdomain(r, MatchingNodalTransfer(rows)))
    schur: MatrixFreeSchur[MockVector, NDArray[Any]] = MatrixFreeSchur()
    residual = SkeletonResidual(case.space(descriptor), subdomains, schur)
    return residual, case, descriptor, responses


@pytest.mark.parametrize(
    "coefs,f,exact",
    [
        (_PIECEWISE, 1.7, _exact_piecewise_constant([k[0] for k in _PIECEWISE], 1.7)),
        ([(1.5, 0.0, 1.5)] * 5, 6.0, _exact_one_nonlinear((1.5, 0.0, 1.5), 6.0, 5)),
    ],
)
def test_mock_stack_solves_to_the_exact_values(
    coefs: list[tuple[float, ...]], f: float, exact: list[float]
) -> None:
    residual, case, descriptor, responses = _mock_chain(coefs, f)
    space = case.space(descriptor)
    setups: RecomputeAlways[MockVector, MatrixFree] = RecomputeAlways(
        GMRES(space, case.bkd, tol=1e-13)
    )
    point = SteadySystem(residual, space, setups).solve(
        Newton(atol=1e-11, rtol=1e-11, maxiters=20), space.zeros(1)
    )
    assert point.converged()
    iterations = point.info().iterations
    assert iterations == 1 if len(coefs[0]) == 1 else iterations > 1
    case.assert_vector(
        point.solution(), case.dense(np.asarray(exact)[:, None]), rtol=1e-10
    )
    # Kept points are recognized by value on an opaque stack too.
    assert all(r.count == iterations + 1 for r in responses)


class _ReversedExecutor:
    """Runs tasks last to first and records them, returning results in
    the order given, as the protocol requires of any scheduling."""

    def __init__(self) -> None:
        self.ntasks = 0

    def map(self, task: Callable[[int], T], subdomains: Sequence[int]) -> list[T]:
        results = {i: task(i) for i in reversed(list(subdomains))}
        self.ntasks += len(results)
        return [results[i] for i in subdomains]


def _jacobian_action(
    residual: SkeletonResidual[Array, Array, Array, MatrixFree],
    state: Array,
    direction: Array,
) -> Array:
    jacobian = residual.derivatives().jacobian
    assert jacobian is not None
    return jacobian(state, _LABEL).apply(direction)


def test_rank_residuals_sum_to_the_whole(bkd: Backend[Array]) -> None:
    coefs = [(1.5, 0.0, 1.5), (1.0,), (2.0, 0.4), (0.5,)]
    whole, space, _ = _chain(bkd, coefs)
    # Ranks own subdomains 1, 3 and 0, 2: each rank's residual over its
    # own subdomains, summed as the skeleton stack's scatter-add would.
    topo = TensorBoxPartition((4,), owners=[1, 0, 1, 0])
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    ranks = [
        SkeletonResidual(
            space, [whole.subdomains()[s] for s in local_subdomains(topo, r)], schur
        )
        for r in (0, 1)
    ]
    state, direction = bkd.asarray([[0.3], [-0.4], [0.2]]), bkd.ones((3, 1))
    bkd.assert_allclose(ranks[0](state) + ranks[1](state), whole(state), rtol=1e-14)
    bkd.assert_allclose(
        _jacobian_action(ranks[0], state, direction)
        + _jacobian_action(ranks[1], state, direction),
        _jacobian_action(whole, state, direction),
        rtol=1e-14,
    )


def test_order_and_scheduling_leave_the_solve_unchanged(bkd: Backend[Array]) -> None:
    coefs = [(1.5, 0.0, 1.5)] * 3 + [(2.0,), (0.5, 0.0, 1.0)]
    serial, space, _ = _chain(bkd, coefs, 4.0)
    reference = _gmres_solve(serial, space, bkd)
    assert reference.converged()
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    permuted = SkeletonResidual(
        space, [serial.subdomains()[s] for s in (3, 0, 4, 2, 1)], schur
    )
    again = _gmres_solve(permuted, space, bkd)
    bkd.assert_allclose(again.solution(), reference.solution(), rtol=1e-12)
    executor = _ReversedExecutor()
    reversed_run = SkeletonResidual(
        space, serial.subdomains(), schur, executor=executor
    )
    scheduled = _gmres_solve(reversed_run, space, bkd)
    # Results come back in subdomain order, so the sums are the same
    # floating-point operations as the serial run's.
    bkd.assert_allclose(scheduled.solution(), reference.solution(), rtol=0.0)
    assert executor.ntasks > 2 * len(coefs) * scheduled.info().iterations
    with pytest.raises(TypeError, match="executor"):
        SkeletonResidual(space, [], schur, executor=object())  # type: ignore[arg-type]

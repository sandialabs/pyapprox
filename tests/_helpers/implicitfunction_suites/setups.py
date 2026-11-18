r"""Stack-generic suites: setup strategies on steady requests.

The quadratic of ``tests._helpers.implicitfunction_stack_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i` on 3 states; :math:`p_0 =
0` makes it affine, its Jacobian constant), the stack's direct solver
counting its setups; positions are tokens standing for solved points:

- Keys: an affine residual's are equal at any state; a general one's are
  equal at equal positions whatever the state objects (a recomputed
  state), and an iterate's (no position) equals no other; keys of
  distinct, even equal-valued, residuals never are.
- Strategies: ``RecomputeAlways`` sets up every request;
  ``KeepStateFree`` keeps the affine operator only; ``KeepAll`` also keeps
  a solution's, found again from a new state object at its position, but
  not an iterate's; ``clear`` drops what was kept; a kept handle solves as
  a fresh one does; a kept setup keeps no state alive; a strategy pickles
  after use, kept handles dropped.
- A strategy written only here serves a consumer typed on the protocol.
- The strategy checker: the shipped strategies pass a script of affine,
  positioned and iterate requests, every handle and its transpose over
  three passes; a strategy keyed on the residual alone fails at the
  second position.
"""

import gc
import pickle
import weakref
from typing import Dict, Generic, Hashable, List, Optional, Tuple

import pytest

from pyapprox.implicitfunction.residual_derivatives import ResidualDerivatives
from pyapprox.implicitfunction.setups import (
    KeepAll,
    KeepStateFree,
    OperatorBuilderProtocol,
    RecomputeAlways,
    ResidualJacobianBuilder,
    SetupKey,
    SetupStrategyProtocol,
    SteadyRequest,
)
from pyapprox.implicitfunction.setups_check import check_strategy
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    LinearSolverProtocol,
    SolverHandleProtocol,
)
from tests._helpers.implicitfunction_stack_cases import STATES, QuadraticResidual
from tests._helpers.linalg.counting_solver import CountingSolver
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V


class _Position:
    """A position: names one solved state, equal only to itself."""


def _vector(case: TypedStackCase[V, F, A], column: List[float]) -> V:
    return case.vector(case.bkd.reshape(case.bkd.asarray(column), (3, 1)), STATES)


def _zeros(case: TypedStackCase[V, F, A]) -> V:
    return _vector(case, [0.0, 0.0, 0.0])


def _ones(case: TypedStackCase[V, F, A]) -> V:
    return _vector(case, [1.0, 1.0, 1.0])


def _request_three_times(
    strategy: SetupStrategyProtocol[SteadyRequest[V, F], V, F],
    residual: QuadraticResidual[V, F, A],
    state: V,
    position: Optional[Hashable],
) -> List[SolverHandleProtocol[V]]:
    """A consumer typed on the protocol: the same request, three times."""
    builder = ResidualJacobianBuilder(residual)
    return [
        strategy.handle(SteadyRequest(residual, state, position), builder)
        for _ in range(3)
    ]


class KeysSuite:
    def test_an_affine_residuals_keys_are_equal_at_any_state(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case, affine=True)
        first = SteadyRequest(f, _zeros(case), None)
        second = SteadyRequest(f, _ones(case), _Position())
        assert not first.sees_state()
        assert first.key() == second.key()
        assert hash(first.key()) == hash(second.key())

    def test_a_general_residuals_keys_compare_positions_not_states(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        position = _Position()
        first = SteadyRequest(f, _zeros(case), position)
        recomputed = SteadyRequest(f, _zeros(case), position)
        assert first.sees_state() and first.at_solution()
        assert first.key() == recomputed.key()
        assert hash(first.key()) == hash(recomputed.key())
        assert first.key() != SteadyRequest(f, _zeros(case), _Position()).key()

    def test_an_iterates_key_equals_no_other(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        state = _zeros(case)
        iterate = SteadyRequest(f, state, None)
        assert not iterate.at_solution()
        assert iterate.key() == iterate.key()
        assert iterate.key() != SteadyRequest(f, state, None).key()

    def test_distinct_residuals_never_share_a_key(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case, affine=True)
        g: QuadraticResidual[V, F, A] = QuadraticResidual(case, affine=True)
        state = _zeros(case)
        key = SteadyRequest(f, state, None).key()
        assert key != SteadyRequest(g, state, None).key()
        assert key != SetupKey((f,), (None, None))
        assert key != "not a key"

    def test_a_residual_without_a_jacobian_has_no_request(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        class _NoJacobian(QuadraticResidual[V, F, A]):
            def derivatives(self) -> ResidualDerivatives[V, F]:
                return ResidualDerivatives.none()

        with pytest.raises(TypeError, match="no jacobian"):
            SteadyRequest(_NoJacobian(case), _zeros(case), None)


class TheBuilderSuite:
    def test_it_forms_the_jacobian_at_the_requests_state(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        state = _vector(case, [0.3, -0.2, 0.5])
        built = ResidualJacobianBuilder(f).build(SteadyRequest(f, state, None))
        jacobian = f.derivatives().jacobian
        assert jacobian is not None
        expected = jacobian(state, LinearizationPoint(0, ("check",)))
        x = _vector(case, [1.0, 2.0, -1.0])
        case.assert_vector(built.apply(x), case.array(expected.apply(x)), rtol=1e-14)

    def test_it_refuses_another_residuals_request(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        g: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        with pytest.raises(ValueError, match="builder of"):
            ResidualJacobianBuilder(f).build(SteadyRequest(g, _zeros(case), None))


class StrategiesSuite:
    def test_recompute_always_sets_up_every_request(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case, affine=True)
        lu = CountingSolver(case.direct_solver())
        _request_three_times(RecomputeAlways(lu), f, _zeros(case), None)
        assert lu.setups == 3

    def test_keep_state_free_keeps_only_a_constant_jacobian(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        affine: QuadraticResidual[V, F, A] = QuadraticResidual(case, affine=True)
        general: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        lu = CountingSolver(case.direct_solver())
        strategy: KeepStateFree[V, F] = KeepStateFree(lu)
        handles = _request_three_times(strategy, affine, _zeros(case), None)
        assert lu.setups == 1 and handles[0] is handles[2]
        _request_three_times(strategy, general, _zeros(case), _Position())
        assert lu.setups == 4 and strategy.nkept() == 1
        strategy.clear()
        _request_three_times(strategy, affine, _zeros(case), None)
        assert lu.setups == 5

    def test_keep_all_keeps_a_solutions_setup_not_an_iterates(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        lu = CountingSolver(case.direct_solver())
        strategy: KeepAll[V, F] = KeepAll(lu)
        _request_three_times(strategy, f, _zeros(case), None)
        assert lu.setups == 3 and strategy.nkept() == 0
        position = _Position()
        handles = _request_three_times(strategy, f, _ones(case), position)
        assert lu.setups == 4 and handles[0] is handles[2]
        # A new object at the same position, as a store recomputing from a
        # checkpoint gives: the kept setup.
        builder = ResidualJacobianBuilder(f)
        recomputed = SteadyRequest(f, _ones(case), position)
        assert strategy.handle(recomputed, builder) is handles[0]
        strategy.clear()
        _request_three_times(strategy, f, _ones(case), position)
        assert lu.setups == 5

    def test_a_kept_handle_solves_as_a_fresh_one(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        solution = _vector(case, [0.3, -0.2, 0.5])
        position = _Position()
        rhs = _vector(case, [1.0, -2.0, 0.5])
        kept = _request_three_times(
            KeepAll(case.direct_solver()), f, solution, position
        )[2]
        fresh = _request_three_times(
            RecomputeAlways(case.direct_solver()), f, solution, position
        )[0]
        case.assert_vector(
            kept.solve(rhs)[0], case.array(fresh.solve(rhs)[0]), rtol=1e-14
        )

    def test_a_kept_setup_keeps_no_state_alive(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        strategy: KeepAll[V, F] = KeepAll(case.direct_solver())
        state = _vector(case, [0.3, -0.2, 0.5])
        alive = weakref.ref(state)
        strategy.handle(
            SteadyRequest(f, state, _Position()), ResidualJacobianBuilder(f)
        )
        del state
        gc.collect()
        assert strategy.nkept() == 1 and alive() is None

    def test_a_strategy_pickles_after_use_without_its_handles(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case, affine=True)
        strategies: Tuple[SetupStrategyProtocol[SteadyRequest[V, F], V, F], ...] = (
            KeepStateFree(case.direct_solver()),
            KeepAll(case.direct_solver()),
        )
        for strategy in strategies:
            assert isinstance(strategy, (KeepStateFree, KeepAll))
            _request_three_times(strategy, f, _zeros(case), None)
            assert strategy.nkept() == 1
            restored = pickle.loads(pickle.dumps(strategy))
            assert restored.nkept() == 0
            _request_three_times(restored, f, _zeros(case), None)
            assert restored.nkept() == 1


class _KeepFirst(Generic[V, F]):
    """A strategy defined only here, for steady requests alone: keeps each
    key's first setup."""

    def __init__(self, solver: LinearSolverProtocol[V, F]) -> None:
        self._solver = solver
        self._kept: Dict[SetupKey, SolverHandleProtocol[V]] = {}

    def handle(
        self,
        request: SteadyRequest[V, F],
        builder: OperatorBuilderProtocol[SteadyRequest[V, F], V, F],
    ) -> SolverHandleProtocol[V]:
        key = request.key()
        if key not in self._kept:
            self._kept[key] = self._solver.setup(builder.build(request))
        return self._kept[key]

    def clear(self) -> None:
        self._kept = {}


class ExtensibilitySuite:
    def test_a_strategy_written_here_serves_a_protocol_consumer(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case, affine=True)
        handles = _request_three_times(
            _KeepFirst(case.direct_solver()), f, _zeros(case), None
        )
        assert handles[0] is handles[1] is handles[2]


_Script = List[
    Tuple[
        SteadyRequest[V, F], OperatorBuilderProtocol[SteadyRequest[V, F], V, F]
    ]
]


def _script(
    case: TypedStackCase[V, F, A],
) -> Tuple[QuadraticResidual[V, F, A], "_Script[V, F]"]:
    affine: QuadraticResidual[V, F, A] = QuadraticResidual(case, affine=True)
    general: QuadraticResidual[V, F, A] = QuadraticResidual(case)
    first, second = _Position(), _Position()
    iterate = _vector(case, [0.1, 0.0, -0.2])
    a, b = ResidualJacobianBuilder(affine), ResidualJacobianBuilder(general)
    script: _Script[V, F] = [
        (SteadyRequest(affine, _zeros(case), None), a),
        (SteadyRequest(affine, _ones(case), None), a),
        (SteadyRequest(general, _vector(case, [0.3, -0.2, 0.5]), first), b),
        (SteadyRequest(general, _vector(case, [0.3, -0.2, 0.5]), first), b),
        (SteadyRequest(general, _vector(case, [1.5, 0.7, -1.1]), second), b),
        (SteadyRequest(general, iterate, None), b),
        (SteadyRequest(general, iterate, None), b),
    ]
    return general, script


class _KeyedOnTheResidual(Generic[V, F]):
    """Wrong: one setup per residual, wherever its Jacobian is taken."""

    def __init__(self, solver: LinearSolverProtocol[V, F]) -> None:
        self._solver = solver
        self._kept: Dict[int, SolverHandleProtocol[V]] = {}

    def handle(
        self,
        request: SteadyRequest[V, F],
        builder: OperatorBuilderProtocol[SteadyRequest[V, F], V, F],
    ) -> SolverHandleProtocol[V]:
        key = id(request.source())
        if key not in self._kept:
            self._kept[key] = self._solver.setup(builder.build(request))
        return self._kept[key]

    def clear(self) -> None:
        self._kept = {}


class TheCheckerSuite:
    def test_the_shipped_strategies_pass(self, case: TypedStackCase[V, F, A]) -> None:
        general, script = _script(case)
        rhs = _vector(case, [1.0, -2.0, 0.5])
        strategies: Tuple[SetupStrategyProtocol[SteadyRequest[V, F], V, F], ...] = (
            RecomputeAlways(case.direct_solver()),
            KeepStateFree(case.direct_solver()),
            KeepAll(case.direct_solver()),
        )
        for strategy in strategies:
            report = check_strategy(
                strategy, case.direct_solver(), script, general.states, rhs
            )
            assert report.passed()
            assert report.nchecked == 3 * len(script) * 2

    def test_a_strategy_keyed_on_less_fails(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        general, script = _script(case)
        report = check_strategy(
            _KeyedOnTheResidual(case.direct_solver()),
            case.direct_solver(),
            script,
            general.states,
            _vector(case, [1.0, -2.0, 0.5]),
        )
        # The residual's first setup is at position one; every later request
        # elsewhere gets it, in every pass, forward and transposed.
        assert {m.request for m in report.mismatches} == {4, 5, 6}
        assert {m.sweep for m in report.mismatches} == {0, 1, 2}
        assert {m.transposed for m in report.mismatches} == {False, True}

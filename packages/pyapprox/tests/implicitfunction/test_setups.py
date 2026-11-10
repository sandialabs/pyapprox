r"""Setup strategies on steady requests.

The quadratic of ``tests._helpers.implicitfunction_cases`` (:math:`F_i =
(A u)_i + p_0 u_i^2 + p_1^2 c_i` on 3 states; :math:`p_0 = 0` makes it
affine, its Jacobian constant), dense LU through the backend counting its
setups; positions are tokens standing for solved points:

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
"""

import gc
import pickle
import weakref
from typing import Dict, Generic, Hashable, List, Optional

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
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import LinearizationPoint, SolverHandleProtocol
from tests._helpers.implicitfunction_cases import QuadraticResidual
from tests._helpers.linalg.counting_solver import CountingSolver

_Request = SteadyRequest[Array, DenseFormat[Array]]
_Strategy = SetupStrategyProtocol[_Request[Array], Array, DenseFormat[Array]]


class _Position:
    """A position: names one solved state, equal only to itself."""


def _lu(bkd: Backend[Array]) -> CountingSolver[Array, DenseFormat[Array]]:
    return CountingSolver(BackendDenseLU(bkd))


def _request_three_times(
    strategy: _Strategy[Array],
    residual: QuadraticResidual[Array],
    state: Array,
    position: Optional[Hashable],
) -> List[SolverHandleProtocol[Array]]:
    """A consumer typed on the protocol: the same request, three times."""
    builder = ResidualJacobianBuilder(residual)
    return [
        strategy.handle(SteadyRequest(residual, state, position), builder)
        for _ in range(3)
    ]


class TestKeys:
    def test_an_affine_residuals_keys_are_equal_at_any_state(
        self, bkd: Backend[Array]
    ) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd, affine=True)
        first = SteadyRequest(f, bkd.zeros((3, 1)), None)
        second = SteadyRequest(f, bkd.ones((3, 1)), _Position())
        assert not first.sees_state()
        assert first.key() == second.key()
        assert hash(first.key()) == hash(second.key())

    def test_a_general_residuals_keys_compare_positions_not_states(
        self, bkd: Backend[Array]
    ) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        position = _Position()
        first = SteadyRequest(f, bkd.zeros((3, 1)), position)
        recomputed = SteadyRequest(f, bkd.zeros((3, 1)), position)
        assert first.sees_state() and first.at_solution()
        assert first.key() == recomputed.key()
        assert hash(first.key()) == hash(recomputed.key())
        assert first.key() != SteadyRequest(f, bkd.zeros((3, 1)), _Position()).key()

    def test_an_iterates_key_equals_no_other(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        state = bkd.zeros((3, 1))
        iterate = SteadyRequest(f, state, None)
        assert not iterate.at_solution()
        assert iterate.key() == iterate.key()
        assert iterate.key() != SteadyRequest(f, state, None).key()

    def test_distinct_residuals_never_share_a_key(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd, affine=True)
        g: QuadraticResidual[Array] = QuadraticResidual(bkd, affine=True)
        state = bkd.zeros((3, 1))
        key = SteadyRequest(f, state, None).key()
        assert key != SteadyRequest(g, state, None).key()
        assert key != SetupKey((f,), (None, None))
        assert key != "not a key"

    def test_a_residual_without_a_jacobian_has_no_request(
        self, bkd: Backend[Array]
    ) -> None:
        class _NoJacobian(QuadraticResidual[Array]):
            def derivatives(self) -> ResidualDerivatives[Array, DenseFormat[Array]]:
                return ResidualDerivatives.none()

        with pytest.raises(TypeError, match="no jacobian"):
            SteadyRequest(_NoJacobian(bkd), bkd.zeros((3, 1)), None)


class TestTheBuilder:
    def test_it_forms_the_jacobian_at_the_requests_state(
        self, bkd: Backend[Array]
    ) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        state = bkd.asarray([[0.3], [-0.2], [0.5]])
        built = ResidualJacobianBuilder(f).build(SteadyRequest(f, state, None))
        jacobian = f.derivatives().jacobian
        assert jacobian is not None
        expected = jacobian(state, LinearizationPoint(0, ("check",)))
        x = bkd.asarray([[1.0], [2.0], [-1.0]])
        bkd.assert_allclose(built.apply(x), expected.apply(x), rtol=1e-14)

    def test_it_refuses_another_residuals_request(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        g: QuadraticResidual[Array] = QuadraticResidual(bkd)
        with pytest.raises(ValueError, match="builder of"):
            ResidualJacobianBuilder(f).build(SteadyRequest(g, bkd.zeros((3, 1)), None))


class TestStrategies:
    def test_recompute_always_sets_up_every_request(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd, affine=True)
        lu = _lu(bkd)
        _request_three_times(RecomputeAlways(lu), f, bkd.zeros((3, 1)), None)
        assert lu.setups == 3

    def test_keep_state_free_keeps_only_a_constant_jacobian(
        self, bkd: Backend[Array]
    ) -> None:
        affine: QuadraticResidual[Array] = QuadraticResidual(bkd, affine=True)
        general: QuadraticResidual[Array] = QuadraticResidual(bkd)
        lu = _lu(bkd)
        strategy: KeepStateFree[Array, DenseFormat[Array]] = KeepStateFree(lu)
        handles = _request_three_times(strategy, affine, bkd.zeros((3, 1)), None)
        assert lu.setups == 1 and handles[0] is handles[2]
        _request_three_times(strategy, general, bkd.zeros((3, 1)), _Position())
        assert lu.setups == 4 and strategy.nkept() == 1
        strategy.clear()
        _request_three_times(strategy, affine, bkd.zeros((3, 1)), None)
        assert lu.setups == 5

    def test_keep_all_keeps_a_solutions_setup_not_an_iterates(
        self, bkd: Backend[Array]
    ) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        lu = _lu(bkd)
        strategy: KeepAll[Array, DenseFormat[Array]] = KeepAll(lu)
        _request_three_times(strategy, f, bkd.zeros((3, 1)), None)
        assert lu.setups == 3 and strategy.nkept() == 0
        position = _Position()
        handles = _request_three_times(strategy, f, bkd.ones((3, 1)), position)
        assert lu.setups == 4 and handles[0] is handles[2]
        # A new object at the same position, as a store recomputing from a
        # checkpoint gives: the kept setup.
        builder = ResidualJacobianBuilder(f)
        recomputed = SteadyRequest(f, bkd.ones((3, 1)), position)
        assert strategy.handle(recomputed, builder) is handles[0]
        strategy.clear()
        _request_three_times(strategy, f, bkd.ones((3, 1)), position)
        assert lu.setups == 5

    def test_a_kept_handle_solves_as_a_fresh_one(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        solution = bkd.asarray([[0.3], [-0.2], [0.5]])
        position = _Position()
        rhs = bkd.asarray([[1.0], [-2.0], [0.5]])
        kept = _request_three_times(KeepAll(_lu(bkd)), f, solution, position)[2]
        fresh = _request_three_times(RecomputeAlways(_lu(bkd)), f, solution, position)[
            0
        ]
        bkd.assert_allclose(kept.solve(rhs)[0], fresh.solve(rhs)[0], rtol=1e-14)

    def test_a_kept_setup_keeps_no_state_alive(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        strategy: KeepAll[Array, DenseFormat[Array]] = KeepAll(BackendDenseLU(bkd))
        state = bkd.asarray([[0.3], [-0.2], [0.5]])
        alive = weakref.ref(state)
        strategy.handle(
            SteadyRequest(f, state, _Position()), ResidualJacobianBuilder(f)
        )
        del state
        gc.collect()
        assert strategy.nkept() == 1 and alive() is None

    def test_a_strategy_pickles_after_use_without_its_handles(
        self, bkd: Backend[Array]
    ) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd, affine=True)
        for strategy in (
            KeepStateFree(BackendDenseLU(bkd)),
            KeepAll(BackendDenseLU(bkd)),
        ):
            _request_three_times(strategy, f, bkd.zeros((3, 1)), None)
            assert strategy.nkept() == 1
            restored = pickle.loads(pickle.dumps(strategy))
            assert restored.nkept() == 0
            _request_three_times(restored, f, bkd.zeros((3, 1)), None)
            assert restored.nkept() == 1


class _KeepFirst(Generic[Array]):
    """A strategy defined only here, for steady requests alone: keeps each
    key's first setup."""

    def __init__(self, solver: BackendDenseLU[Array]) -> None:
        self._solver = solver
        self._kept: Dict[SetupKey, SolverHandleProtocol[Array]] = {}

    def handle(
        self,
        request: _Request[Array],
        builder: OperatorBuilderProtocol[_Request[Array], Array, DenseFormat[Array]],
    ) -> SolverHandleProtocol[Array]:
        key = request.key()
        if key not in self._kept:
            self._kept[key] = self._solver.setup(builder.build(request))
        return self._kept[key]

    def clear(self) -> None:
        self._kept = {}


class TestExtensibility:
    def test_a_strategy_written_here_serves_a_protocol_consumer(
        self, bkd: Backend[Array]
    ) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd, affine=True)
        handles = _request_three_times(
            _KeepFirst(BackendDenseLU(bkd)), f, bkd.zeros((3, 1)), None
        )
        assert handles[0] is handles[1] is handles[2]

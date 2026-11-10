r"""The strategy checker on steady requests.

The quadratic of ``tests._helpers.implicitfunction_cases``, affine
(:math:`p_0 = 0`) and general, dense LU: a script of an affine request at
two states, a general request at a position, at the same position from a
recomputed state, at a second position, and an iterate twice.

- ``RecomputeAlways``, ``KeepStateFree`` and ``KeepAll`` pass, every
  handle (and its transpose) checked over three passes.
- A strategy keyed on the residual alone, ignoring where :math:`R_u` is
  evaluated, fails at the second position.
"""

from typing import Dict, Generic, List, Tuple

from pyapprox.implicitfunction.setups import (
    KeepAll,
    KeepStateFree,
    OperatorBuilderProtocol,
    RecomputeAlways,
    ResidualJacobianBuilder,
    SteadyRequest,
)
from pyapprox.implicitfunction.setups_check import check_strategy
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import SolverHandleProtocol
from tests._helpers.implicitfunction_cases import QuadraticResidual

_Request = SteadyRequest[Array, DenseFormat[Array]]
_Builder = OperatorBuilderProtocol[_Request[Array], Array, DenseFormat[Array]]


class _Position:
    """A position: names one solved state, equal only to itself."""


def _script(
    bkd: Backend[Array],
) -> Tuple[QuadraticResidual[Array], List[Tuple[_Request[Array], _Builder[Array]]]]:
    affine: QuadraticResidual[Array] = QuadraticResidual(bkd, affine=True)
    general: QuadraticResidual[Array] = QuadraticResidual(bkd)
    first, second = _Position(), _Position()
    iterate = bkd.asarray([[0.1], [0.0], [-0.2]])
    a, b = ResidualJacobianBuilder(affine), ResidualJacobianBuilder(general)
    script: List[Tuple[_Request[Array], _Builder[Array]]] = [
        (SteadyRequest(affine, bkd.zeros((3, 1)), None), a),
        (SteadyRequest(affine, bkd.ones((3, 1)), None), a),
        (SteadyRequest(general, bkd.asarray([[0.3], [-0.2], [0.5]]), first), b),
        (SteadyRequest(general, bkd.asarray([[0.3], [-0.2], [0.5]]), first), b),
        (SteadyRequest(general, bkd.asarray([[1.5], [0.7], [-1.1]]), second), b),
        (SteadyRequest(general, iterate, None), b),
        (SteadyRequest(general, iterate, None), b),
    ]
    return general, script


class _KeyedOnTheResidual(Generic[Array]):
    """Wrong: one setup per residual, wherever its Jacobian is taken."""

    def __init__(self, solver: BackendDenseLU[Array]) -> None:
        self._solver = solver
        self._kept: Dict[int, SolverHandleProtocol[Array]] = {}

    def handle(
        self, request: _Request[Array], builder: _Builder[Array]
    ) -> SolverHandleProtocol[Array]:
        key = id(request.source())
        if key not in self._kept:
            self._kept[key] = self._solver.setup(builder.build(request))
        return self._kept[key]

    def clear(self) -> None:
        self._kept = {}


class TestTheChecker:
    def test_the_shipped_strategies_pass(self, bkd: Backend[Array]) -> None:
        general, script = _script(bkd)
        rhs = bkd.asarray([[1.0], [-2.0], [0.5]])
        for strategy in (
            RecomputeAlways(BackendDenseLU(bkd)),
            KeepStateFree(BackendDenseLU(bkd)),
            KeepAll(BackendDenseLU(bkd)),
        ):
            report = check_strategy(
                strategy, BackendDenseLU(bkd), script, general.states, rhs
            )
            assert report.passed()
            assert report.nchecked == 3 * len(script) * 2

    def test_a_strategy_keyed_on_less_fails(self, bkd: Backend[Array]) -> None:
        general, script = _script(bkd)
        report = check_strategy(
            _KeyedOnTheResidual(BackendDenseLU(bkd)),
            BackendDenseLU(bkd),
            script,
            general.states,
            bkd.asarray([[1.0], [-2.0], [0.5]]),
        )
        # The residual's first setup is at position one; every later request
        # elsewhere gets it, in every pass, forward and transposed.
        assert {m.request for m in report.mismatches} == {4, 5, 6}
        assert {m.sweep for m in report.mismatches} == {0, 1, 2}
        assert {m.transposed for m in report.mismatches} == {False, True}

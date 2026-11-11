r"""Setup requests of a mass-form stepper and the operators they describe.

Models of :math:`M \dot y = f(y, t)` on two states, :math:`M = [[2, 0.5],
[0.5, 1]]`, :math:`K = [[-1, 0.3], [0.2, -0.8]]`, at each declared
Jacobian level: constant (:math:`f = K y`), of the time (:math:`f = (1 +
t) K y`) and general (:math:`f = K y + y \circ y`, :math:`J = K + 2
\operatorname{diag}(y)`), states in the residual's form (1-D) as the
bundle reads them. Pinned blocks pin the first row
(``PinnedPrey``).

- Keys: which changes to a request leave its key equal, per level, for
  :math:`s \ne 0` and for :math:`s = 0`.
- Positions: equal for one trajectory, step and block; refused when
  negative or given the wrong type.
- The builder forms :math:`a M - s J(y, t)` with the pinned rows identity
  rows (dense, to 1e-15 relative) and refuses another system's request.
- The checker: every shipped strategy returns only right handles over a
  script of iterates, positions, mass-only and pinned requests, and
  ``KeepAll`` keeps a positioned setup for a new state object at that
  position but no iterate's.
- Refusals: :math:`a = s = 0`; :math:`s \ne 0` without a Jacobian
  (:math:`s = 0` is fine).
"""

from dataclasses import dataclass, replace
from typing import Generic, List, Optional, Tuple, Type, Union

import pytest

from pyapprox.implicitfunction.setups import (
    KeepAll,
    KeepStateFree,
    OperatorBuilderProtocol,
    RecomputeAlways,
)
from pyapprox.implicitfunction.setups_check import check_strategy
from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.mass_matrix import ConstantDenseMassMatrix
from pyapprox.ode.pinned_rows import BlockPinProtocol, RowPin
from pyapprox.ode.residual_derivatives import (
    ODEResidualDerivatives,
    backend_state_space,
    constant_jacobian,
    dense_jacobian,
    state_free_jacobian,
)
from pyapprox.ode.setups import MassShiftBuilder, MassShiftRequest, StatePosition
from pyapprox.ode.stepper_derivatives import StepOrigin, TrajectoryId
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    DenseFormat,
)
from pyapprox.util.linalg.operators import ActionOperatorProtocol
from tests._helpers.linalg.counting_solver import CountingSolver
from tests._helpers.ode.trajectory_cases import PinnedPrey

_MASS = [[2.0, 0.5], [0.5, 1.0]]
_K = [[-1.0, 0.3], [0.2, -0.8]]

_System = MassFormSystem[Array, Array, DenseFormat[Array]]
_Request = MassShiftRequest[Array, Array, DenseFormat[Array]]


class _Model(Generic[Array]):
    """The mass, :math:`K`, and a bundle the subclass declares."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._k = bkd.asarray(_K)
        self._mass = ConstantDenseMassMatrix(bkd.asarray(_MASS), bkd)
        self._states = backend_state_space(bkd, 2)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def mass_matrix(self) -> ConstantDenseMassMatrix[Array]:
        return self._mass

    def operator(self, matrix: Array) -> BackendDenseOperator[Array]:
        return BackendDenseOperator(matrix, self._states.descriptor(), self._bkd)


class _Constant(_Model[Array]):
    def __call__(self, state: Array, time: float) -> Array:
        return self._k @ state

    def matrix(self, state: Array, time: float) -> Array:
        return self._k

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return ODEResidualDerivatives.state_only(
            constant_jacobian(self.operator(self._k))
        )


class _OfTime(_Model[Array]):
    def __call__(self, state: Array, time: float) -> Array:
        return (1.0 + time) * (self._k @ state)

    def matrix(self, state: Array, time: float) -> Array:
        return (1.0 + time) * self._k

    def _of_time(self, time: float) -> BackendDenseOperator[Array]:
        return self.operator((1.0 + time) * self._k)

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return ODEResidualDerivatives.state_only(state_free_jacobian(self._of_time))


class _General(_Model[Array]):
    def __call__(self, state: Array, time: float) -> Array:
        return self._k @ state + state * state

    def matrix(self, state: Array, time: float) -> Array:
        return self._k + 2.0 * self._bkd.diag(state)

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return ODEResidualDerivatives.state_only(
            dense_jacobian(self.matrix, self._states, self._bkd)
        )


class _NoJacobian(_Constant[Array]):
    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return ODEResidualDerivatives.none()


_Level = Union[Type[_Constant[Array]], Type[_OfTime[Array]], Type[_General[Array]]]
_LEVELS = [_Constant, _OfTime, _General]
_LEVEL_IDS = ["constant", "of time", "general"]


@dataclass(frozen=True)
class _Args(Generic[Array]):
    """A request's arguments, so a test can change one at a time."""

    source: _System[Array]
    a: float
    s: float
    pin: Optional[BlockPinProtocol[Array]]
    time: float
    state: Array
    position: Optional[StatePosition]

    def request(self) -> _Request[Array]:
        return MassShiftRequest(
            self.source, self.a, self.s, self.pin, self.time, self.state, self.position
        )


class _Case(Generic[Array]):
    def __init__(self, bkd: Backend[Array], level: _Level[Array]) -> None:
        self.bkd = bkd
        self.model = level(bkd)
        self.system: _System[Array] = dense_mass_form_system(self.model)
        self.trajectory = TrajectoryId()
        self.pin: RowPin[Array] = RowPin(PinnedPrey(bkd))
        self.state = bkd.asarray([[0.4], [-0.7]])

    def position(self, step: int, block: int) -> StatePosition:
        return StatePosition(StepOrigin(self.trajectory, step), block)

    def args(self, s: float = 0.25) -> _Args[Array]:
        return _Args(self.system, 1.0, s, None, 0.1, self.state, self.position(3, 1))

    def dense(self, op: ActionOperatorProtocol[Array, DenseFormat[Array]]) -> Array:
        eye = self.bkd.eye(2)
        return self.bkd.hstack([op.apply(eye[:, j : j + 1]) for j in range(2)])


def _variants(
    case: _Case[Array],
) -> List[Tuple[str, _Args[Array], Tuple[bool, bool, bool]]]:
    """Changes to the base request, each with whether the key stays equal
    for the (constant, of time, general) levels when s != 0."""
    base, bkd = case.args(), case.bkd
    other = dense_mass_form_system(type(case.model)(bkd))
    yes, no = True, False
    return [
        ("a new state object", replace(base, state=bkd.copy(base.state)), (yes,) * 3),
        ("another time", replace(base, time=0.3), (yes, no, no)),
        (
            "another position",
            replace(base, position=case.position(4, 1)),
            (yes, yes, no),
        ),
        ("an iterate", replace(base, position=None), (yes, yes, no)),
        ("another a", replace(base, a=1.5), (no,) * 3),
        ("another s", replace(base, s=0.5), (no,) * 3),
        ("a pin", replace(base, pin=case.pin), (no,) * 3),
        ("another system", replace(base, source=other), (no,) * 3),
    ]


class TestKeys:
    @pytest.mark.parametrize("level", _LEVELS, ids=_LEVEL_IDS)
    def test_a_shifted_request(self, bkd: Backend[Array], level: _Level[Array]) -> None:
        case = _Case(bkd, level)
        column = 0 if level is _Constant else 1 if level is _OfTime else 2
        base = case.args().request()
        assert base.key() == base.key() and base.at_solution()
        assert base.sees_state() == (level is _General)
        for name, args, equal in _variants(case):
            key = args.request().key()
            assert (key == base.key()) == equal[column], name
            if equal[column]:
                assert hash(key) == hash(base.key()), name

    @pytest.mark.parametrize("level", _LEVELS, ids=_LEVEL_IDS)
    def test_a_mass_only_request(
        self, bkd: Backend[Array], level: _Level[Array]
    ) -> None:
        """With s = 0 only the system, the pin and a are compared."""
        case = _Case(bkd, level)
        base = replace(case.args(), s=0.0)
        assert not base.request().sees_state()
        compared = ("another a", "another s", "a pin", "another system")
        for name, args, _ in _variants(case):
            if name != "another s":
                args = replace(args, s=0.0)
            equal = args.request().key() == base.request().key()
            assert equal == (name not in compared), name

    def test_the_same_pin_object(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, _General)
        pinned = replace(case.args(), pin=case.pin)
        assert pinned.request().key() == pinned.request().key()
        copy = replace(pinned, pin=RowPin(PinnedPrey(bkd)))
        assert copy.request().key() != pinned.request().key()


class TestPositions:
    def test_equality(self) -> None:
        trajectory = TrajectoryId()
        origin = StepOrigin(trajectory, 2)
        position = StatePosition(origin, 0)
        assert position == StatePosition(StepOrigin(trajectory, 2), 0)
        assert hash(position) == hash(StatePosition(StepOrigin(trajectory, 2), 0))
        assert position != StatePosition(origin, 1)
        assert position != StatePosition(StepOrigin(trajectory, 1), 0)
        assert position != StatePosition(StepOrigin(TrajectoryId(), 2), 0)
        assert position.origin() == origin and position.block() == 0
        assert origin.step() == 2

    def test_negative_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            StepOrigin(TrajectoryId(), -1)
        with pytest.raises(ValueError, match="nonnegative"):
            StatePosition(StepOrigin(TrajectoryId(), 0), -1)

    def test_a_wrong_type_is_refused(self) -> None:
        with pytest.raises(TypeError, match="TrajectoryId"):
            StepOrigin(object(), 0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="StepOrigin"):
            StatePosition(TrajectoryId(), 0)  # type: ignore[arg-type]


class TestTheBuilder:
    @pytest.mark.parametrize("level", _LEVELS, ids=_LEVEL_IDS)
    @pytest.mark.parametrize("pinned", [False, True])
    @pytest.mark.parametrize("a, s", [(1.0, 0.25), (1.5, 0.4), (1.0, 0.0), (0.7, 0.0)])
    def test_it_forms_the_shifted_operator(
        self,
        bkd: Backend[Array],
        level: _Level[Array],
        pinned: bool,
        a: float,
        s: float,
    ) -> None:
        case = _Case(bkd, level)
        args = replace(
            case.args(), a=a, s=s, pin=case.pin if pinned else None, time=0.3
        )
        operator = MassShiftBuilder(case.system).build(args.request())
        state = case.system.form.to_residual(case.state)
        expected = a * bkd.asarray(_MASS) - s * case.model.matrix(state, 0.3)
        if pinned:
            expected = bkd.vstack([bkd.eye(2)[0:1], expected[1:]])
        bkd.assert_allclose(case.dense(operator), expected, rtol=1e-15)

    def test_it_refuses_another_systems_request(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, _Constant)
        other = dense_mass_form_system(_Constant(bkd))
        with pytest.raises(ValueError, match="builder of"):
            MassShiftBuilder(other).build(case.args().request())


class TestWithTheStrategies:
    def _script(
        self, case: _Case[Array]
    ) -> List[
        Tuple[
            _Request[Array],
            OperatorBuilderProtocol[_Request[Array], Array, DenseFormat[Array]],
        ]
    ]:
        bkd, base = case.bkd, case.args()
        builder: MassShiftBuilder[Array, Array, DenseFormat[Array]] = MassShiftBuilder(
            case.system
        )
        moved = bkd.asarray([[0.9], [0.2]])
        args = [
            replace(base, position=None),
            replace(base, position=None, state=moved),
            base,
            replace(base, state=bkd.copy(base.state)),
            replace(base, position=case.position(4, 0), state=moved),
            replace(base, s=0.0),
            replace(base, s=0.0, pin=case.pin),
            replace(base, pin=case.pin),
            replace(base, time=0.6),
        ]
        return [(arg.request(), builder) for arg in args]

    @pytest.mark.parametrize("level", _LEVELS, ids=_LEVEL_IDS)
    def test_every_strategy_returns_right_handles(
        self, bkd: Backend[Array], level: _Level[Array]
    ) -> None:
        case = _Case(bkd, level)
        rhs = bkd.asarray([[1.0], [-2.0]])
        for strategy in (
            RecomputeAlways(BackendDenseLU(bkd)),
            KeepStateFree(BackendDenseLU(bkd)),
            KeepAll(BackendDenseLU(bkd)),
        ):
            report = check_strategy(
                strategy,
                BackendDenseLU(bkd),
                self._script(case),
                case.system.space,
                rhs,
            )
            assert report.passed() and report.nchecked == 9 * 3 * 2

    def test_keep_all_finds_a_position_not_an_iterate(
        self, bkd: Backend[Array]
    ) -> None:
        case = _Case(bkd, _General)
        lu = CountingSolver(BackendDenseLU(bkd))
        keep: KeepAll[Array, DenseFormat[Array]] = KeepAll(lu)
        builder: MassShiftBuilder[Array, Array, DenseFormat[Array]] = MassShiftBuilder(
            case.system
        )
        base = case.args()
        first = keep.handle(base.request(), builder)
        again = replace(base, state=bkd.copy(base.state)).request()
        assert keep.handle(again, builder) is first and lu.setups == 1
        iterate = replace(base, position=None).request()
        keep.handle(iterate, builder)
        keep.handle(iterate, builder)
        assert lu.setups == 3 and keep.nkept() == 1


class TestRefusals:
    def test_a_zero_operator(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, _Constant)
        with pytest.raises(ValueError, match="both zero"):
            replace(case.args(), a=0.0, s=0.0).request()

    def test_a_shift_without_a_jacobian(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, _NoJacobian)
        with pytest.raises(TypeError, match="_NoJacobian.*no jacobian"):
            case.args().request()
        mass_only = replace(case.args(), s=0.0).request()
        operator = MassShiftBuilder(case.system).build(mass_only)
        bkd.assert_allclose(case.dense(operator), bkd.asarray(_MASS), rtol=1e-15)

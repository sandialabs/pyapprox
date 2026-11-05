r"""A state term read at the method's value at given times.

A state term :math:`\sum_k g_k(y_k;\, p)` reads stored states. Read
instead at :math:`y(t_k)`, the stepping method's continuous extension at
times :math:`t_k` that need not be stored times (observations between
steps), it becomes a step term: time :math:`t_k` belongs to the step whose
span :math:`(t_n, t_{n+1}]` contains it, and :math:`y(t_k)` is a function
of that step's window (:mod:`pyapprox.ode.continuous_extension`). The
term's index :math:`k` then names the :math:`k`-th time. A time at the
initial state belongs to no step: read it with the state term itself.

Its derivatives are the chain rule through the extension. With
:math:`G_k = \partial g_k/\partial y` and :math:`D_k = \partial
y(t_k)/\partial X` for an argument :math:`X` of the step, the first
derivative is :math:`\sum_k G_k D_k` (plus :math:`\partial g_k/\partial
p` in the parameters). Its transpose weighted by :math:`\omega` changes
along a tangent by the extension's curvature weighted by :math:`\lambda_k
= G_k^\top \omega`, plus :math:`D_k^\top` applied to the term's own
second derivatives along :math:`\delta y_k = D_k\, dX + D_{p,k}\, v`. So
any native state term (a misfit, an observation) is read between steps
with no code of its own.
"""

from typing import Generic, List, Optional, Sequence, TypeVar

from pyapprox.ode.continuous_extension import (
    ContinuousExtensionDerivatives,
    ContinuousExtensionProtocol,
)
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.trajectory_terms import (
    ChainedLinearization,
    ChainPart,
    LinearizationProtocol,
    StateTermDerivatives,
    StateTermProtocol,
    StepTermDerivatives,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import BlockVector

V = TypeVar("V")
X = TypeVar("X")
A = TypeVar("A", bound=ArrayProtocol)


def _one(vector: V) -> BlockVector[V]:
    return BlockVector((vector,))


def _first(vector: BlockVector[V]) -> V:
    return vector.blocks[0]


def _same(vector: BlockVector[V]) -> BlockVector[V]:
    return vector


class _OnFirstBlock(Generic[V, A]):
    r"""A state term's derivative :math:`G_k` read on the extension's
    output, a one-block vector."""

    def __init__(self, state: LinearizationProtocol[V, A]) -> None:
        self._state = state

    def apply(self, x: BlockVector[V]) -> A:
        return self._state.apply(_first(x))

    def apply_transpose(self, weights: A) -> BlockVector[V]:
        return _one(self._state.apply_transpose(weights))


class TermAtTimes(Generic[V, A]):
    r"""A state term read at the method's value :math:`y(t_k)` at
    ``times``, as a step term (see the module docstring).

    Parameters
    ----------
    term : StateTermProtocol
        The term; its index :math:`k` names ``times[k]``, and it is read
        for each :math:`k` its ``reads(k, False)`` accepts.
    extension : ContinuousExtensionProtocol
        The stepping method's continuous extension.
    times : sequence of float
        The times :math:`t_k`, after the initial time.
    bkd : Backend
        The backend of the values, for the zero value of a step with no
        time.

    Raises
    ------
    TypeError
        If ``term`` or ``extension`` does not satisfy its protocol.
    ValueError
        If ``times`` is empty.
    """

    def __init__(
        self,
        term: StateTermProtocol[V, A],
        extension: ContinuousExtensionProtocol[V, A],
        times: Sequence[float],
        bkd: Backend[A],
    ) -> None:
        if not isinstance(term, StateTermProtocol):
            raise TypeError(
                f"term must satisfy StateTermProtocol, got {type(term).__name__}"
            )
        if not isinstance(extension, ContinuousExtensionProtocol):
            raise TypeError(
                "extension must satisfy ContinuousExtensionProtocol, got "
                f"{type(extension).__name__}"
            )
        if not times:
            raise ValueError("TermAtTimes needs at least one time")
        self._term, self._extension, self._bkd = term, extension, bkd
        self._times = tuple(times)
        self._read = tuple(k for k in range(len(times)) if term.reads(k, False))
        own, extended = term.derivatives(), extension.derivatives()
        first = own.wrt_state is not None and extended.wrt_unknown is not None
        second = (
            first
            and own.state_state is not None
            and extended.curvature_wrt_unknown is not None
        )
        self._derivatives: StepTermDerivatives[V, A] = (
            StepTermDerivatives.second_order(
                self._wrt_unknown,
                self._wrt_lag,
                self._wrt_param,
                self._curvature_wrt_unknown,
                self._curvature_wrt_lag,
                self._curvature_wrt_param,
            )
            if second
            else StepTermDerivatives.first_order(
                self._wrt_unknown, self._wrt_lag, self._wrt_param
            )
            if first
            else StepTermDerivatives.none()
        )

    def nqoi(self) -> int:
        return self._term.nqoi()

    def _in(self, window: StepWindow[V]) -> List[int]:
        """The times read in the window's step, :math:`(t_n, t_{n+1}]`."""
        start, end = window.time(), window.lag_time(0)
        return [k for k in self._read if start < self._times[k] <= end]

    def value(self, step: int, window: StepWindow[V]) -> A:
        total = self._bkd.zeros((self.nqoi(), 1))
        for k in self._in(window):
            time = self._times[k]
            total = total + self._term.value(
                k, time, self._extension.value(window, time)
            )
        return total

    def derivatives(self) -> StepTermDerivatives[V, A]:
        return self._derivatives

    # ------------------------------------------------------------------
    # The pieces of the chain rule
    # ------------------------------------------------------------------

    def _own(self) -> StateTermDerivatives[V, A]:
        return self._term.derivatives()

    def _extended(self) -> ContinuousExtensionDerivatives[V]:
        return self._extension.derivatives()

    def _state_derivative(
        self, k: int, window: StepWindow[V]
    ) -> tuple[V, LinearizationProtocol[V, A]]:
        """:math:`y(t_k)` and :math:`G_k` there."""
        time = self._times[k]
        state = self._extension.value(window, time)
        field = self._own().wrt_state
        if field is None:
            raise ValueError("the term has no state derivative")
        return state, field(k, time, state)

    # Each returns a new vector: x + y, or a copy of y for x None, since
    # a term's results may be its own storage, valid until its next call.

    def _add_state(self, x: Optional[V], y: V) -> V:
        space = self._extension.state_space().component(0)
        return space.copy(y) if x is None else space.axpby(1.0, x, 1.0, y)

    def _add_param(self, x: Optional[V], y: V) -> V:
        params = self._extension.param_space()
        if params is None:
            raise ValueError("the extension has no parameters")
        space = params.component(0)
        return space.copy(y) if x is None else space.axpby(1.0, x, 1.0, y)

    def _add_unknown(
        self, x: Optional[BlockVector[V]], y: BlockVector[V]
    ) -> BlockVector[V]:
        space = self._extension.unknown_space()
        return space.copy(y) if x is None else space.axpby(1.0, x, 1.0, y)

    # ------------------------------------------------------------------
    # First derivatives
    # ------------------------------------------------------------------

    def _wrt_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[BlockVector[V], A]]:
        field = self._extended().wrt_unknown
        if field is None:
            raise ValueError("the extension has no derivatives")
        parts: List[ChainPart[BlockVector[V], V, A]] = []
        for k in self._in(window):
            _, state = self._state_derivative(k, window)
            extension = field(window, self._times[k])
            parts.append(ChainPart(_OnFirstBlock(state), extension, None))
        if not parts:
            return None
        return ChainedLinearization(parts, _same, _same, self._add_unknown)

    def _wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[LinearizationProtocol[V, A]]:
        field = self._extended().wrt_lag
        if field is None:
            raise ValueError("the extension has no derivatives")
        parts: List[ChainPart[V, V, A]] = []
        for k in self._in(window):
            _, state = self._state_derivative(k, window)
            extension = field(window, self._times[k], lag)
            parts.append(ChainPart(_OnFirstBlock(state), extension, None))
        if not parts:
            return None
        return ChainedLinearization(parts, _one, _first, self._add_state)

    def _wrt_param(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[V, A]]:
        extended, own = self._extended().wrt_param, self._own().wrt_param
        parts: List[ChainPart[V, V, A]] = []
        for k in self._in(window):
            time = self._times[k]
            y, state = self._state_derivative(k, window)
            extension = None if extended is None else extended(window, time)
            direct = None if own is None else own(k, time, y)
            if extension is not None or direct is not None:
                parts.append(ChainPart(_OnFirstBlock(state), extension, direct))
        if not parts:
            return None
        return ChainedLinearization(parts, _one, _first, self._add_param)

    # ------------------------------------------------------------------
    # Curvatures
    # ------------------------------------------------------------------

    def _tangent(
        self, k: int, window: StepWindow[V], tangents: StepWindow[V], direction: V
    ) -> V:
        r""":math:`\delta y_k = D_k\, dX + D_{p,k}\, v`: the extension's
        tangent at :math:`t_k` along the step's tangents and the parameter
        direction."""
        extended, time = self._extended(), self._times[k]
        if extended.wrt_unknown is None or extended.wrt_lag is None:
            raise ValueError("the extension has no derivatives")
        total = _first(extended.wrt_unknown(window, time).apply(tangents.unknown))
        for lag in range(1, len(window.past_states) + 1):
            moved = extended.wrt_lag(window, time, lag).apply(
                _one(tangents.past_state(lag))
            )
            total = self._add_state(total, _first(moved))
        if extended.wrt_param is not None:
            moved = extended.wrt_param(window, time).apply(_one(direction))
            total = self._add_state(total, _first(moved))
        return total

    def _state_curvature(
        self, k: int, y: V, weights: A, tangent: V, direction: V
    ) -> Optional[V]:
        r""":math:`(\omega^\top g_k)_{yy}\, \delta y_k + (\omega^\top
        g_k)_{yp}\, v`, a state; None for zero."""
        own, time = self._own(), self._times[k]
        if own.state_state is None or own.state_param is None:
            raise ValueError("the term has no second derivatives")
        total: Optional[V] = None
        for value in (
            own.state_state(k, time, y, weights, tangent),
            own.state_param(k, time, y, weights, direction)
            if self._extension.param_space() is not None
            else None,
        ):
            if value is not None:
                total = self._add_state(total, value)
        return total

    def _per_time(
        self,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> List[tuple[int, V, V, V, Optional[V]]]:
        """For each time read: :math:`k`, :math:`y(t_k)`, :math:`\\lambda_k
        = G_k^\\top \\omega`, :math:`\\delta y_k` and the term's state
        curvature."""
        result = []
        for k in self._in(window):
            y, state = self._state_derivative(k, window)
            tangent = self._tangent(k, window, tangents, direction)
            # Copied: held past the term's next call.
            adjoint = self._add_state(None, state.apply_transpose(weights))
            result.append(
                (
                    k,
                    y,
                    adjoint,
                    tangent,
                    self._state_curvature(k, y, weights, tangent, direction),
                )
            )
        return result

    def _curvature_wrt_unknown(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[BlockVector[V]]:
        extended = self._extended()
        if extended.curvature_wrt_unknown is None or extended.wrt_unknown is None:
            raise ValueError("the extension has no curvature")
        total: Optional[BlockVector[V]] = None
        for k, _, adjoint, _, curvature in self._per_time(
            window, tangents, direction, weights
        ):
            time = self._times[k]
            pieces = [
                extended.curvature_wrt_unknown(
                    window, time, tangents, direction, adjoint
                ),
                None
                if curvature is None
                else extended.wrt_unknown(window, time).apply_transpose(
                    _one(curvature)
                ),
            ]
            for piece in pieces:
                if piece is not None:
                    total = self._add_unknown(total, piece)
        return total

    def _curvature_wrt_lag(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        lag: int,
        direction: V,
        weights: A,
    ) -> Optional[V]:
        extended = self._extended()
        if extended.curvature_wrt_lag is None or extended.wrt_lag is None:
            raise ValueError("the extension has no curvature")
        total: Optional[V] = None
        for k, _, adjoint, _, curvature in self._per_time(
            window, tangents, direction, weights
        ):
            time = self._times[k]
            pieces = [
                extended.curvature_wrt_lag(
                    window, time, tangents, lag, direction, adjoint
                ),
                None
                if curvature is None
                else _first(
                    extended.wrt_lag(window, time, lag).apply_transpose(
                        _one(curvature)
                    )
                ),
            ]
            for piece in pieces:
                if piece is not None:
                    total = self._add_state(total, piece)
        return total

    def _curvature_wrt_param(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[V]:
        extended, own = self._extended(), self._own()
        if extended.curvature_wrt_param is None:
            raise ValueError("the extension has no curvature")
        if own.param_state is None or own.param_param is None:
            raise ValueError("the term has no second derivatives")
        total: Optional[V] = None
        for k, y, adjoint, tangent, curvature in self._per_time(
            window, tangents, direction, weights
        ):
            time = self._times[k]
            pieces: List[Optional[V]] = [
                extended.curvature_wrt_param(
                    window, time, tangents, direction, adjoint
                ),
                own.param_state(k, time, y, weights, tangent),
                own.param_param(k, time, y, weights, direction),
            ]
            if curvature is not None and extended.wrt_param is not None:
                pieces.append(
                    _first(
                        extended.wrt_param(window, time).apply_transpose(
                            _one(curvature)
                        )
                    )
                )
            for piece in pieces:
                if piece is not None:
                    total = self._add_param(total, piece)
        return total

r"""The terms of a functional of a recorded trajectory.

A functional :math:`Q` of a trajectory, with values of ``nqoi`` entries,
is a sum of terms of three kinds, each read where it lives:

- a parameter term :math:`r(p)` (a regularization), read from the
  parameters alone (:class:`ParameterTermProtocol`);
- a state term :math:`\sum_{m} g_m(y_m;\, p)` over the stored states it
  names (an observation, an endpoint value), read from one stored state
  at a time (:class:`StateTermProtocol`);
- a step term :math:`\sum_n c_n(u_n;\, y_n, \dots, y_{n+1-k};\, p)`,
  read from each step's window (its unknown, so its stages, and the
  stored states it read): a time integral, or a residual of the step
  (:class:`StepTermProtocol`).

A term gives its value and its first derivatives at the same place, the
derivatives as linearizations: one object that applies forward
(:math:`\partial c\, x`, the tangent sweep, a column of ``nqoi``
entries) and transposed (:math:`(\partial c)^\top \omega` for weights
:math:`\omega` of the values, the adjoint sweep). Its second derivatives,
weighted by :math:`\omega` as an adjoint weighs them, are optional and
come in a bundle. A term holds no trajectory: it is handed the state or
window it reads, so the same term serves any store and the value sink.

:class:`TrajectoryFunctional` gathers the terms; the sweeps of
:mod:`pyapprox.ode.trajectory_derivatives` add their derivatives, so a
term never knows of another, and ``a + b`` is the functional of both.
:class:`RecursionWeighted` weighs a step term's contributions by the
stepper's recursion (:func:`~pyapprox.ode.step_quadrature.
recursion_sensitivities`), which makes a sum of per-step contributions
the method's own integral over time.

Library code scales the stack vectors a term returns by plain numbers
only; products of a stack vector with value-shaped arrays are the term's
own, inside its linearizations.
"""

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import BlockVector

V = TypeVar("V")
X = TypeVar("X")
A = TypeVar("A", bound=ArrayProtocol)


@runtime_checkable
class LinearizationProtocol(Protocol[X, A]):
    r"""A term's first derivative in one argument, :math:`D = \partial
    c/\partial x`, as a linear map from that argument's vectors to value
    columns."""

    def apply(self, x: X) -> A:
        r""":math:`D x`, a column of ``nqoi`` entries."""
        ...

    def apply_transpose(self, weights: A) -> X:
        r""":math:`D^\top \omega` for a column of value weights; may be
        the term's own storage, valid until its next call."""
        ...


class BackendScalarGradient(Generic[A]):
    r"""The linearization of a scalar by its gradient column :math:`g`, on
    the backend stack (vectors and values both backend arrays): forward
    :math:`g^\top x`, shape ``(1, 1)``; transposed :math:`g\, \omega`. Its
    products keep an autograd graph through :math:`g`.

    Parameters
    ----------
    gradient : A
        :math:`g`, a column.
    bkd : Backend
        The backend.
    """

    def __init__(self, gradient: A, bkd: Backend[A]) -> None:
        self._gradient, self._bkd = gradient, bkd

    def apply(self, x: A) -> A:
        return self._bkd.dot(self._gradient.T, x)

    def apply_transpose(self, weights: A) -> A:
        return self._gradient * weights


StateCurvatureFn = Callable[[int, float, V, A, V], Optional[V]]
"""``(index, time, state, weights, x) ->`` a state term's
:math:`\\omega`-weighted second derivative at stored state ``index``
applied to the column ``x``; None for a zero block."""

StepUnknownCurvatureFn = Callable[
    [int, StepWindow[V], StepWindow[V], V, A], Optional[BlockVector[V]]
]
"""``(step, window, tangents, direction, weights) ->`` the derivative of
:math:`(\\partial c_n/\\partial u_n)^\\top \\omega` along the tangent."""

StepLagCurvatureFn = Callable[
    [int, StepWindow[V], StepWindow[V], int, V, A], Optional[V]
]
"""``(step, window, tangents, lag, direction, weights) ->`` the
derivative of :math:`(\\partial c_n/\\partial y_{n+1-\\mathrm{lag}})^\\top
\\omega` along the tangent."""

StepParamCurvatureFn = Callable[[int, StepWindow[V], StepWindow[V], V, A], Optional[V]]
"""``(step, window, tangents, direction, weights) ->`` the derivative of
:math:`(\\partial c_n/\\partial p)^\\top \\omega` along the tangent."""


@dataclass(frozen=True)
class ParameterTermDerivatives(Generic[V, A]):
    r"""A parameter term's optional second derivative,
    :math:`(\omega^\top r)_{pp}\, v`, as ``(weights, v) ->`` a parameter
    column, None when zero. Build with :meth:`first_order` or
    :meth:`second_order`."""

    param_param: Optional[Callable[[A, V], Optional[V]]] = None

    @staticmethod
    def first_order() -> "ParameterTermDerivatives[V, A]":
        """No second derivative."""
        return ParameterTermDerivatives()

    @staticmethod
    def second_order(
        param_param: Callable[[A, V], Optional[V]],
    ) -> "ParameterTermDerivatives[V, A]":
        """The second derivative."""
        return ParameterTermDerivatives(param_param)


@dataclass(frozen=True)
class StateTermDerivatives(Generic[V, A]):
    r"""A state term's optional second derivatives at one stored state,
    weighted by :math:`\omega`.

    The second derivative in the states is block diagonal over the stored
    states (:math:`\partial^2 g/\partial y_m \partial y_k = 0` for
    :math:`m \ne k`): a term coupling states of one step is a step term.
    Build with :meth:`first_order` or :meth:`second_order`.

    Parameters
    ----------
    state_state : callable, optional
        :math:`(\omega^\top g_m)_{yy}\, w`, state to state.
    state_param : callable, optional
        :math:`(\omega^\top g_m)_{yp}\, v`, parameters to state.
    param_state : callable, optional
        :math:`(\omega^\top g_m)_{py}\, w`, state to parameters.
    param_param : callable, optional
        :math:`(\omega^\top g_m)_{pp}\, v`, parameters to parameters.

    Raises
    ------
    ValueError
        If some but not all four are given: the second-order sweep needs
        every one.
    """

    state_state: Optional[StateCurvatureFn[V, A]] = None
    state_param: Optional[StateCurvatureFn[V, A]] = None
    param_state: Optional[StateCurvatureFn[V, A]] = None
    param_param: Optional[StateCurvatureFn[V, A]] = None

    def __post_init__(self) -> None:
        given = [
            field is not None
            for field in (
                self.state_state,
                self.state_param,
                self.param_state,
                self.param_param,
            )
        ]
        if any(given) and not all(given):
            raise ValueError("a state term's second derivatives come all four or none")

    @staticmethod
    def first_order() -> "StateTermDerivatives[V, A]":
        """No second derivatives."""
        return StateTermDerivatives()

    @staticmethod
    def second_order(
        state_state: StateCurvatureFn[V, A],
        state_param: StateCurvatureFn[V, A],
        param_state: StateCurvatureFn[V, A],
        param_param: StateCurvatureFn[V, A],
    ) -> "StateTermDerivatives[V, A]":
        """All four second derivatives."""
        return StateTermDerivatives(state_state, state_param, param_state, param_param)


@dataclass(frozen=True)
class StepTermDerivatives(Generic[V, A]):
    r"""A step term's optional derivatives along a tangent, weighted by
    :math:`\omega`: each is the derivative of the matching
    :math:`(\partial c_n/\partial\,\cdot)^\top \omega` along :math:`(dX,
    v)`, with :math:`dX` the tangents of the step's arguments read from
    ``tangents`` (the tangent trajectory's window of the step:
    :math:`\delta u_n` as its unknown, :math:`w_{n+1-l}` as its past
    states); a call returns None for a zero term. Build with
    :meth:`first_order` or :meth:`second_order`.

    Raises
    ------
    ValueError
        If some but not all three are given.
    """

    curvature_wrt_unknown: Optional[StepUnknownCurvatureFn[V, A]] = None
    curvature_wrt_lag: Optional[StepLagCurvatureFn[V, A]] = None
    curvature_wrt_param: Optional[StepParamCurvatureFn[V, A]] = None

    def __post_init__(self) -> None:
        given = [
            field is not None
            for field in (
                self.curvature_wrt_unknown,
                self.curvature_wrt_lag,
                self.curvature_wrt_param,
            )
        ]
        if any(given) and not all(given):
            raise ValueError("a step term's curvatures come all three or none")

    @staticmethod
    def first_order() -> "StepTermDerivatives[V, A]":
        """No derivatives along a tangent."""
        return StepTermDerivatives()

    @staticmethod
    def second_order(
        curvature_wrt_unknown: StepUnknownCurvatureFn[V, A],
        curvature_wrt_lag: StepLagCurvatureFn[V, A],
        curvature_wrt_param: StepParamCurvatureFn[V, A],
    ) -> "StepTermDerivatives[V, A]":
        """All three derivatives along a tangent."""
        return StepTermDerivatives(
            curvature_wrt_unknown, curvature_wrt_lag, curvature_wrt_param
        )


@runtime_checkable
class ParameterTermProtocol(Protocol[V, A]):
    r"""A term :math:`r(p)` read from the parameters alone, at the
    parameters currently set."""

    def nqoi(self) -> int:
        """The number of entries of the term's value."""
        ...

    def value(self) -> A:
        """:math:`r(p)`, a column of ``nqoi`` entries."""
        ...

    def wrt_param(self) -> LinearizationProtocol[V, A]:
        r""":math:`\partial r/\partial p`."""
        ...

    def derivatives(self) -> ParameterTermDerivatives[V, A]:
        """The optional second derivative."""
        ...


@runtime_checkable
class StateTermProtocol(Protocol[V, A]):
    r"""A term :math:`\sum_m g_m(y_m;\, p)` over the stored states it
    names, each read alone.

    It holds no trajectory: it names its states for a trajectory of a
    given length, and is handed each one with its index and time.
    """

    def nqoi(self) -> int:
        """The number of entries of the term's value."""
        ...

    def indices(self, ntimes: int) -> Sequence[int]:
        """The stored states the term reads, for a trajectory of
        ``ntimes`` states; nonnegative, each once."""
        ...

    def value(self, index: int, time: float, state: V) -> A:
        """:math:`g_m` at stored state ``index``, a column of ``nqoi``
        entries."""
        ...

    def wrt_state(
        self, index: int, time: float, state: V
    ) -> LinearizationProtocol[V, A]:
        r""":math:`\partial g_m/\partial y_m`."""
        ...

    def wrt_param(
        self, index: int, time: float, state: V
    ) -> Optional[LinearizationProtocol[V, A]]:
        r""":math:`\partial g_m/\partial p`; None when :math:`g_m` does not
        read the parameters."""
        ...

    def derivatives(self) -> StateTermDerivatives[V, A]:
        """The optional second derivatives."""
        ...


@runtime_checkable
class StepTermProtocol(Protocol[V, A]):
    r"""A term :math:`\sum_n c_n(u_n;\, y_n, \dots, y_{n+1-k};\, p)` read
    from each step's window.

    It holds no trajectory: it is handed each step's window, the step's
    unknown and the stored states it read.
    """

    def nqoi(self) -> int:
        """The number of entries of the term's value."""
        ...

    def value(self, step: int, window: StepWindow[V]) -> A:
        """:math:`c_n`, a column of ``nqoi`` entries."""
        ...

    def wrt_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[BlockVector[V], A]]:
        r""":math:`\partial c_n/\partial u_n`; None when zero."""
        ...

    def wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[LinearizationProtocol[V, A]]:
        r""":math:`\partial c_n/\partial y_{n+1-\mathrm{lag}}`; None when
        zero."""
        ...

    def wrt_param(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[V, A]]:
        r""":math:`\partial c_n/\partial p`; None when zero."""
        ...

    def derivatives(self) -> StepTermDerivatives[V, A]:
        """The optional derivatives along a tangent."""
        ...


@dataclass(frozen=True)
class TrajectoryFunctional(Generic[V, A]):
    r"""A functional of one recorded trajectory as the sweeps read it: a
    sum of parameter, state and step terms, with values of ``nqoi``
    entries.

    A functional made of several (a misfit and a regularization, a time
    integral and an endpoint value) is the sum of theirs, ``a + b``.

    Parameters
    ----------
    zero : A
        A zero value, shape ``(nqoi, 1)``: the functional's value type and
        size, from which its unit weights are made for ``nqoi = 1``.
    parameter_terms, state_terms, step_terms : tuple
        The terms of each kind.

    Raises
    ------
    TypeError
        If a term does not satisfy its protocol.
    ValueError
        If ``zero`` is not a column, or a term's ``nqoi`` differs.
    """

    zero: A
    parameter_terms: Tuple[ParameterTermProtocol[V, A], ...] = ()
    state_terms: Tuple[StateTermProtocol[V, A], ...] = ()
    step_terms: Tuple[StepTermProtocol[V, A], ...] = ()

    def __post_init__(self) -> None:
        if self.zero.ndim != 2 or self.zero.shape[1] != 1:
            raise ValueError(
                f"zero must be a column (nqoi, 1), got shape {tuple(self.zero.shape)}"
            )
        checks: Tuple[tuple[Sequence[object], type, str], ...] = (
            (self.parameter_terms, ParameterTermProtocol, "ParameterTermProtocol"),
            (self.state_terms, StateTermProtocol, "StateTermProtocol"),
            (self.step_terms, StepTermProtocol, "StepTermProtocol"),
        )
        for terms, protocol, name in checks:
            for term in terms:
                if not isinstance(term, protocol):
                    raise TypeError(
                        f"each term must satisfy {name}, got {type(term).__name__}"
                    )
        sizes = [
            *((type(t).__name__, t.nqoi()) for t in self.parameter_terms),
            *((type(t).__name__, t.nqoi()) for t in self.state_terms),
            *((type(t).__name__, t.nqoi()) for t in self.step_terms),
        ]
        for name, size in sizes:
            if size != self.nqoi():
                raise ValueError(
                    f"{name} has nqoi {size}, the functional {self.nqoi()}"
                )

    @staticmethod
    def from_terms(
        bkd: Backend[A],
        nqoi: int,
        parameter_terms: Tuple[ParameterTermProtocol[V, A], ...] = (),
        state_terms: Tuple[StateTermProtocol[V, A], ...] = (),
        step_terms: Tuple[StepTermProtocol[V, A], ...] = (),
    ) -> "TrajectoryFunctional[V, A]":
        """The functional of these terms, with values of ``nqoi`` entries
        on ``bkd`` (its ``zero`` made here)."""
        return TrajectoryFunctional(
            bkd.zeros((nqoi, 1)), parameter_terms, state_terms, step_terms
        )

    def nqoi(self) -> int:
        """The number of entries of the functional's value."""
        return int(self.zero.shape[0])

    def weights(self, given: Optional[A] = None) -> A:
        r"""The weights :math:`\omega` of the functional's entries for an
        adjoint sweep: ``given``, or :math:`\omega = 1` for a scalar
        functional.

        Raises
        ------
        ValueError
            If ``given`` is not of shape ``(nqoi, 1)``, or none is given
            for a functional of several entries.
        """
        if given is not None:
            if tuple(given.shape) != (self.nqoi(), 1):
                raise ValueError(
                    f"weights must have shape ({self.nqoi()}, 1), got "
                    f"{tuple(given.shape)}"
                )
            return given
        if self.nqoi() != 1:
            raise ValueError(
                f"a functional of {self.nqoi()} entries needs its weights given"
            )
        return self.zero + 1.0

    def __add__(
        self, other: "TrajectoryFunctional[V, A]"
    ) -> "TrajectoryFunctional[V, A]":
        """The sum of the two functionals: their terms together.

        Raises
        ------
        ValueError
            If their ``nqoi`` differ.
        """
        if other.nqoi() != self.nqoi():
            raise ValueError(
                f"cannot add functionals of {self.nqoi()} and {other.nqoi()} entries"
            )
        return TrajectoryFunctional(
            self.zero,
            self.parameter_terms + other.parameter_terms,
            self.state_terms + other.state_terms,
            self.step_terms + other.step_terms,
        )

    @staticmethod
    def stack(
        bkd: Backend[A], functionals: Sequence["TrajectoryFunctional[V, A]"]
    ) -> "TrajectoryFunctional[V, A]":
        """The functionals' values one above another: each functional's
        terms placed in its own rows of the stacked value, so its value
        and forward derivatives land there and its transposed derivatives
        and curvatures read only its rows of the weights.

        Raises
        ------
        ValueError
            If no functional is given.
        """
        if not functionals:
            raise ValueError("stack needs at least one functional")
        total = sum(functional.nqoi() for functional in functionals)
        parameter_terms: list[ParameterTermProtocol[V, A]] = []
        state_terms: list[StateTermProtocol[V, A]] = []
        step_terms: list[StepTermProtocol[V, A]] = []
        start = 0
        for functional in functionals:
            rows = _Rows(bkd, start, functional.nqoi(), total)
            parameter_terms += [
                _PlacedParameterTerm(term, rows) for term in functional.parameter_terms
            ]
            state_terms += [
                _PlacedStateTerm(term, rows) for term in functional.state_terms
            ]
            step_terms += [
                _PlacedStepTerm(term, rows) for term in functional.step_terms
            ]
            start += functional.nqoi()
        return TrajectoryFunctional.from_terms(
            bkd, total, tuple(parameter_terms), tuple(state_terms), tuple(step_terms)
        )


class _Rows(Generic[A]):
    """Rows ``start`` to ``start + size`` of a stacked value of ``total``
    entries: values are placed there, weights read from there."""

    def __init__(self, bkd: Backend[A], start: int, size: int, total: int) -> None:
        self._bkd, self._start, self._size, self._total = bkd, start, size, total

    def place(self, value: A) -> A:
        """``value`` in these rows, zero elsewhere (by stacking, so an
        autograd graph through ``value`` is kept)."""
        bkd, end = self._bkd, self._start + self._size
        blocks = [value]
        if self._start > 0:
            blocks.insert(0, bkd.zeros((self._start, 1)))
        if end < self._total:
            blocks.append(bkd.zeros((self._total - end, 1)))
        return blocks[0] if len(blocks) == 1 else bkd.vstack(blocks)

    def read(self, weights: A) -> A:
        """These rows of ``weights``."""
        return weights[self._start : self._start + self._size]

    def nqoi(self) -> int:
        return self._total


class _PlacedLinearization(Generic[X, A]):
    """A linearization whose values sit in some rows of a stacked value."""

    def __init__(self, inner: LinearizationProtocol[X, A], rows: _Rows[A]) -> None:
        self._inner, self._rows = inner, rows

    def apply(self, x: X) -> A:
        return self._rows.place(self._inner.apply(x))

    def apply_transpose(self, weights: A) -> X:
        return self._inner.apply_transpose(self._rows.read(weights))


def _placed(
    inner: Optional[LinearizationProtocol[X, A]], rows: _Rows[A]
) -> Optional[LinearizationProtocol[X, A]]:
    return None if inner is None else _PlacedLinearization(inner, rows)


class _PlacedParameterTerm(Generic[V, A]):
    """A parameter term in some rows of a stacked value."""

    def __init__(self, term: ParameterTermProtocol[V, A], rows: _Rows[A]) -> None:
        self._term, self._rows = term, rows
        inner = term.derivatives().param_param
        self._param_param = inner
        self._derivatives: ParameterTermDerivatives[V, A] = (
            ParameterTermDerivatives.first_order()
            if inner is None
            else ParameterTermDerivatives.second_order(self._curvature)
        )

    def nqoi(self) -> int:
        return self._rows.nqoi()

    def value(self) -> A:
        return self._rows.place(self._term.value())

    def wrt_param(self) -> LinearizationProtocol[V, A]:
        return _PlacedLinearization(self._term.wrt_param(), self._rows)

    def derivatives(self) -> ParameterTermDerivatives[V, A]:
        return self._derivatives

    def _curvature(self, weights: A, v: V) -> Optional[V]:
        if self._param_param is None:
            raise RuntimeError("the term has no second derivative")
        return self._param_param(self._rows.read(weights), v)


class _PlacedStateTerm(Generic[V, A]):
    """A state term in some rows of a stacked value."""

    def __init__(self, term: StateTermProtocol[V, A], rows: _Rows[A]) -> None:
        self._term, self._rows = term, rows
        inner = term.derivatives()
        self._inner = inner
        self._derivatives: StateTermDerivatives[V, A] = (
            StateTermDerivatives.first_order()
            if inner.state_state is None
            else StateTermDerivatives.second_order(
                self._state_state,
                self._state_param,
                self._param_state,
                self._param_param,
            )
        )

    def nqoi(self) -> int:
        return self._rows.nqoi()

    def indices(self, ntimes: int) -> Sequence[int]:
        return self._term.indices(ntimes)

    def value(self, index: int, time: float, state: V) -> A:
        return self._rows.place(self._term.value(index, time, state))

    def wrt_state(
        self, index: int, time: float, state: V
    ) -> LinearizationProtocol[V, A]:
        return _PlacedLinearization(
            self._term.wrt_state(index, time, state), self._rows
        )

    def wrt_param(
        self, index: int, time: float, state: V
    ) -> Optional[LinearizationProtocol[V, A]]:
        return _placed(self._term.wrt_param(index, time, state), self._rows)

    def derivatives(self) -> StateTermDerivatives[V, A]:
        return self._derivatives

    def _call(
        self,
        field: Optional[StateCurvatureFn[V, A]],
        index: int,
        time: float,
        state: V,
        weights: A,
        x: V,
    ) -> Optional[V]:
        if field is None:
            raise RuntimeError("the term has no second derivatives")
        return field(index, time, state, self._rows.read(weights), x)

    def _state_state(
        self, index: int, time: float, state: V, weights: A, x: V
    ) -> Optional[V]:
        return self._call(self._inner.state_state, index, time, state, weights, x)

    def _state_param(
        self, index: int, time: float, state: V, weights: A, x: V
    ) -> Optional[V]:
        return self._call(self._inner.state_param, index, time, state, weights, x)

    def _param_state(
        self, index: int, time: float, state: V, weights: A, x: V
    ) -> Optional[V]:
        return self._call(self._inner.param_state, index, time, state, weights, x)

    def _param_param(
        self, index: int, time: float, state: V, weights: A, x: V
    ) -> Optional[V]:
        return self._call(self._inner.param_param, index, time, state, weights, x)


class _PlacedStepTerm(Generic[V, A]):
    """A step term in some rows of a stacked value."""

    def __init__(self, term: StepTermProtocol[V, A], rows: _Rows[A]) -> None:
        self._term, self._rows = term, rows
        inner = term.derivatives()
        self._inner = inner
        self._derivatives: StepTermDerivatives[V, A] = (
            StepTermDerivatives.first_order()
            if inner.curvature_wrt_unknown is None
            else StepTermDerivatives.second_order(
                self._curvature_wrt_unknown,
                self._curvature_wrt_lag,
                self._curvature_wrt_param,
            )
        )

    def nqoi(self) -> int:
        return self._rows.nqoi()

    def value(self, step: int, window: StepWindow[V]) -> A:
        return self._rows.place(self._term.value(step, window))

    def wrt_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[BlockVector[V], A]]:
        return _placed(self._term.wrt_unknown(step, window), self._rows)

    def wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[LinearizationProtocol[V, A]]:
        return _placed(self._term.wrt_lag(step, window, lag), self._rows)

    def wrt_param(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[V, A]]:
        return _placed(self._term.wrt_param(step, window), self._rows)

    def derivatives(self) -> StepTermDerivatives[V, A]:
        return self._derivatives

    def _curvature_wrt_unknown(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[BlockVector[V]]:
        field = self._inner.curvature_wrt_unknown
        if field is None:
            raise RuntimeError("the term has no curvature")
        return field(step, window, tangents, direction, self._rows.read(weights))

    def _curvature_wrt_lag(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        lag: int,
        direction: V,
        weights: A,
    ) -> Optional[V]:
        field = self._inner.curvature_wrt_lag
        if field is None:
            raise RuntimeError("the term has no curvature")
        return field(step, window, tangents, lag, direction, self._rows.read(weights))

    def _curvature_wrt_param(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[V]:
        field = self._inner.curvature_wrt_param
        if field is None:
            raise RuntimeError("the term has no curvature")
        return field(step, window, tangents, direction, self._rows.read(weights))


class _Scaled(Generic[X, A]):
    r""":math:`\theta D` for a linearization :math:`D` and a number
    :math:`\theta`: the transpose scales the weights, so no stack vector
    is scaled here."""

    def __init__(self, inner: LinearizationProtocol[X, A], theta: float) -> None:
        self._inner, self._theta = inner, theta

    def apply(self, x: X) -> A:
        return self._theta * self._inner.apply(x)

    def apply_transpose(self, weights: A) -> X:
        return self._inner.apply_transpose(self._theta * weights)


class RecursionWeighted(Generic[V, A]):
    r"""A step term weighed by the stepper's recursion: its step
    contributions :math:`c_n` combined as :math:`\sum_n \theta_n c_n`,
    :math:`\theta` the recursion's sensitivities on the trajectory (see
    :func:`~pyapprox.ode.step_quadrature.recursion_sensitivities`), so
    the sum is the method's own integral over time; 1 throughout for a
    Runge-Kutta or Adams method.

    Every derivative is the term's scaled by :math:`\theta_n`, through the
    weights, so the term's own code serves both.

    Parameters
    ----------
    term : StepTermProtocol
        The step contributions.
    theta : sequence of float
        :math:`\theta_n` for every step of the trajectory.
    """

    def __init__(self, term: StepTermProtocol[V, A], theta: Sequence[float]) -> None:
        self._term, self._theta = term, tuple(theta)
        derivatives = term.derivatives()
        self._unknown = derivatives.curvature_wrt_unknown
        self._lag = derivatives.curvature_wrt_lag
        self._param = derivatives.curvature_wrt_param
        self._derivatives: StepTermDerivatives[V, A] = (
            StepTermDerivatives.first_order()
            if self._unknown is None
            else StepTermDerivatives.second_order(
                self._curvature_wrt_unknown,
                self._curvature_wrt_lag,
                self._curvature_wrt_param,
            )
        )

    def nqoi(self) -> int:
        return self._term.nqoi()

    def value(self, step: int, window: StepWindow[V]) -> A:
        return self._theta[step] * self._term.value(step, window)

    def _scaled(
        self, inner: Optional[LinearizationProtocol[X, A]], step: int
    ) -> Optional[LinearizationProtocol[X, A]]:
        return None if inner is None else _Scaled(inner, self._theta[step])

    def wrt_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[BlockVector[V], A]]:
        return self._scaled(self._term.wrt_unknown(step, window), step)

    def wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[LinearizationProtocol[V, A]]:
        return self._scaled(self._term.wrt_lag(step, window, lag), step)

    def wrt_param(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[V, A]]:
        return self._scaled(self._term.wrt_param(step, window), step)

    def derivatives(self) -> StepTermDerivatives[V, A]:
        return self._derivatives

    def _curvature_wrt_unknown(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[BlockVector[V]]:
        if self._unknown is None:
            raise RuntimeError("the term has no curvature")
        return self._unknown(
            step, window, tangents, direction, self._theta[step] * weights
        )

    def _curvature_wrt_lag(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        lag: int,
        direction: V,
        weights: A,
    ) -> Optional[V]:
        if self._lag is None:
            raise RuntimeError("the term has no curvature")
        return self._lag(
            step, window, tangents, lag, direction, self._theta[step] * weights
        )

    def _curvature_wrt_param(
        self,
        step: int,
        window: StepWindow[V],
        tangents: StepWindow[V],
        direction: V,
        weights: A,
    ) -> Optional[V]:
        if self._param is None:
            raise RuntimeError("the term has no curvature")
        return self._param(
            step, window, tangents, direction, self._theta[step] * weights
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._term!r})"

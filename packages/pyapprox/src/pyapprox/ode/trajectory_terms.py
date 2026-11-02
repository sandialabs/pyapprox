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

A term must give only its value; every derivative is optional and comes
in its bundle (``derivatives()``), None when absent, so a quantity that
is only evaluated implements nothing for the sweeps. Its first
derivatives, at the same places as its value, are linearizations: one
object that applies forward (:math:`\partial c\, x`, the tangent sweep,
a column of ``nqoi`` entries) and transposed (:math:`(\partial c)^\top
\omega` for weights :math:`\omega` of the values, the adjoint sweep);
its second derivatives are weighted by :math:`\omega` as an adjoint
weighs them. Each sweep reads the bundles once and refuses, naming the
term, when one lacks what it needs. A term holds no trajectory: it is
handed the state or window it reads, so the same term serves any store
and the value sink.

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

StateLinearizationFn = Callable[[int, float, V], LinearizationProtocol[V, A]]
"""``(index, time, state) ->`` a state term's first derivative at stored
state ``index``."""

StateParamLinearizationFn = Callable[
    [int, float, V], Optional[LinearizationProtocol[V, A]]
]
"""``(index, time, state) ->`` a state term's parameter derivative at
stored state ``index``; None where it is zero."""

StepUnknownLinearizationFn = Callable[
    [int, StepWindow[V]], Optional[LinearizationProtocol[BlockVector[V], A]]
]
"""``(step, window) ->`` :math:`\\partial c_n/\\partial u_n`; None when
zero."""

StepLagLinearizationFn = Callable[
    [int, StepWindow[V], int], Optional[LinearizationProtocol[V, A]]
]
"""``(step, window, lag) ->`` :math:`\\partial c_n/\\partial
y_{n+1-\\mathrm{lag}}`; None when zero."""

StepParamLinearizationFn = Callable[
    [int, StepWindow[V]], Optional[LinearizationProtocol[V, A]]
]
"""``(step, window) ->`` :math:`\\partial c_n/\\partial p`; None when
zero."""


@dataclass(frozen=True)
class ParameterTermDerivatives(Generic[V, A]):
    r"""A parameter term's optional derivatives: :math:`\partial r/\partial
    p` as a linearization, and :math:`(\omega^\top r)_{pp}\, v` as
    ``(weights, v) ->`` a parameter column (None when zero). Build with
    :meth:`none`, :meth:`first_order` or :meth:`second_order`.

    Raises
    ------
    ValueError
        If the second derivative is given without the first.
    """

    wrt_param: Optional[Callable[[], LinearizationProtocol[V, A]]] = None
    param_param: Optional[Callable[[A, V], Optional[V]]] = None

    def __post_init__(self) -> None:
        if self.param_param is not None and self.wrt_param is None:
            raise ValueError("a second derivative needs the first")

    @staticmethod
    def none() -> "ParameterTermDerivatives[V, A]":
        """No derivatives: the term is only evaluated."""
        return ParameterTermDerivatives()

    @staticmethod
    def first_order(
        wrt_param: Callable[[], LinearizationProtocol[V, A]],
    ) -> "ParameterTermDerivatives[V, A]":
        """The first derivative."""
        return ParameterTermDerivatives(wrt_param)

    @staticmethod
    def second_order(
        wrt_param: Callable[[], LinearizationProtocol[V, A]],
        param_param: Callable[[A, V], Optional[V]],
    ) -> "ParameterTermDerivatives[V, A]":
        """The first and second derivatives."""
        return ParameterTermDerivatives(wrt_param, param_param)


@dataclass(frozen=True)
class StateTermDerivatives(Generic[V, A]):
    r"""A state term's optional derivatives at one stored state: the first
    as linearizations, the second weighted by :math:`\omega`.

    The second derivative in the states is block diagonal over the stored
    states (:math:`\partial^2 g/\partial y_m \partial y_k = 0` for
    :math:`m \ne k`): a term coupling states of one step is a step term.
    Build with :meth:`none`, :meth:`first_order` or :meth:`second_order`.

    Parameters
    ----------
    wrt_state : callable, optional
        :math:`\partial g_m/\partial y_m`; None without derivatives.
    wrt_param : callable, optional
        :math:`\partial g_m/\partial p`; None when :math:`g` does not read
        the parameters.
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
        If a parameter derivative or second derivatives are given without
        ``wrt_state``, or some but not all four second derivatives (the
        second-order sweep needs every one).
    """

    wrt_state: Optional[StateLinearizationFn[V, A]] = None
    wrt_param: Optional[StateParamLinearizationFn[V, A]] = None
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
        if self.wrt_state is None and (self.wrt_param is not None or any(given)):
            raise ValueError("a state term's other derivatives need wrt_state")

    @staticmethod
    def none() -> "StateTermDerivatives[V, A]":
        """No derivatives: the term is only evaluated."""
        return StateTermDerivatives()

    @staticmethod
    def first_order(
        wrt_state: StateLinearizationFn[V, A],
        wrt_param: Optional[StateParamLinearizationFn[V, A]] = None,
    ) -> "StateTermDerivatives[V, A]":
        """The first derivatives."""
        return StateTermDerivatives(wrt_state, wrt_param)

    @staticmethod
    def second_order(
        wrt_state: StateLinearizationFn[V, A],
        wrt_param: Optional[StateParamLinearizationFn[V, A]],
        state_state: StateCurvatureFn[V, A],
        state_param: StateCurvatureFn[V, A],
        param_state: StateCurvatureFn[V, A],
        param_param: StateCurvatureFn[V, A],
    ) -> "StateTermDerivatives[V, A]":
        """The first derivatives and all four second derivatives."""
        return StateTermDerivatives(
            wrt_state,
            wrt_param,
            state_state,
            state_param,
            param_state,
            param_param,
        )


@dataclass(frozen=True)
class StepTermDerivatives(Generic[V, A]):
    r"""A step term's optional derivatives: the first as linearizations at
    the step's window (each call returning None where the derivative is
    zero), and their derivatives along a tangent, weighted by
    :math:`\omega` (each the derivative of the matching :math:`(\partial
    c_n/\partial\,\cdot)^\top \omega` along :math:`(dX, v)`, with
    :math:`dX` the tangents of the step's arguments read from
    ``tangents``: the tangent trajectory's window of the step,
    :math:`\delta u_n` as its unknown, :math:`w_{n+1-l}` as its past
    states). Build with :meth:`none`, :meth:`first_order` or
    :meth:`second_order`.

    Raises
    ------
    ValueError
        If some but not all three first derivatives, or some but not all
        three curvatures, are given, or curvatures without the first
        derivatives.
    """

    wrt_unknown: Optional[StepUnknownLinearizationFn[V, A]] = None
    wrt_lag: Optional[StepLagLinearizationFn[V, A]] = None
    wrt_param: Optional[StepParamLinearizationFn[V, A]] = None
    curvature_wrt_unknown: Optional[StepUnknownCurvatureFn[V, A]] = None
    curvature_wrt_lag: Optional[StepLagCurvatureFn[V, A]] = None
    curvature_wrt_param: Optional[StepParamCurvatureFn[V, A]] = None

    def __post_init__(self) -> None:
        first = [
            field is not None
            for field in (self.wrt_unknown, self.wrt_lag, self.wrt_param)
        ]
        second = [
            field is not None
            for field in (
                self.curvature_wrt_unknown,
                self.curvature_wrt_lag,
                self.curvature_wrt_param,
            )
        ]
        if any(first) and not all(first):
            raise ValueError("a step term's first derivatives come all three or none")
        if any(second) and not all(second):
            raise ValueError("a step term's curvatures come all three or none")
        if any(second) and not any(first):
            raise ValueError("a step term's curvatures need its first derivatives")

    @staticmethod
    def none() -> "StepTermDerivatives[V, A]":
        """No derivatives: the term is only evaluated."""
        return StepTermDerivatives()

    @staticmethod
    def first_order(
        wrt_unknown: StepUnknownLinearizationFn[V, A],
        wrt_lag: StepLagLinearizationFn[V, A],
        wrt_param: StepParamLinearizationFn[V, A],
    ) -> "StepTermDerivatives[V, A]":
        """The first derivatives."""
        return StepTermDerivatives(wrt_unknown, wrt_lag, wrt_param)

    @staticmethod
    def second_order(
        wrt_unknown: StepUnknownLinearizationFn[V, A],
        wrt_lag: StepLagLinearizationFn[V, A],
        wrt_param: StepParamLinearizationFn[V, A],
        curvature_wrt_unknown: StepUnknownCurvatureFn[V, A],
        curvature_wrt_lag: StepLagCurvatureFn[V, A],
        curvature_wrt_param: StepParamCurvatureFn[V, A],
    ) -> "StepTermDerivatives[V, A]":
        """The first derivatives and all three curvatures."""
        return StepTermDerivatives(
            wrt_unknown,
            wrt_lag,
            wrt_param,
            curvature_wrt_unknown,
            curvature_wrt_lag,
            curvature_wrt_param,
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

    def derivatives(self) -> ParameterTermDerivatives[V, A]:
        """The optional derivatives."""
        ...


@runtime_checkable
class StateTermProtocol(Protocol[V, A]):
    r"""A term :math:`\sum_m g_m(y_m;\, p)` over the stored states it
    names, each read alone.

    It holds no trajectory: it says which stored states it reads, by
    index and whether a state is the last, and is handed each one it reads
    with its index and time. Naming the last state by ``last``, not by an
    index, lets a sink evaluate the term while the trajectory is still
    being computed, its length unknown (an adaptive plan's).
    """

    def nqoi(self) -> int:
        """The number of entries of the term's value."""
        ...

    def reads(self, index: int, last: bool) -> bool:
        """Whether the term reads stored state ``index``, the trajectory's
        last when ``last``."""
        ...

    def value(self, index: int, time: float, state: V) -> A:
        """:math:`g_m` at stored state ``index``, a column of ``nqoi``
        entries."""
        ...

    def derivatives(self) -> StateTermDerivatives[V, A]:
        """The optional derivatives."""
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

    def derivatives(self) -> StepTermDerivatives[V, A]:
        """The optional derivatives."""
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
                _MappedParameterTerm(term, rows) for term in functional.parameter_terms
            ]
            state_terms += [
                _MappedStateTerm(term, rows) for term in functional.state_terms
            ]
            step_terms += [
                _MappedStepTerm(term, rows) for term in functional.step_terms
            ]
            start += functional.nqoi()
        return TrajectoryFunctional.from_terms(
            bkd, total, tuple(parameter_terms), tuple(state_terms), tuple(step_terms)
        )


class _Rows(Generic[A]):
    """Rows ``start`` to ``start + size`` of a stacked value of ``total``
    entries, as a value map: values are placed there, weights read from
    there, at every step alike."""

    def __init__(self, bkd: Backend[A], start: int, size: int, total: int) -> None:
        self._bkd, self._start, self._size, self._total = bkd, start, size, total

    def out(self, step: int, value: A) -> A:
        """``value`` in these rows, zero elsewhere (by stacking, so an
        autograd graph through ``value`` is kept)."""
        bkd, end = self._bkd, self._start + self._size
        blocks = [value]
        if self._start > 0:
            blocks.insert(0, bkd.zeros((self._start, 1)))
        if end < self._total:
            blocks.append(bkd.zeros((self._total - end, 1)))
        return blocks[0] if len(blocks) == 1 else bkd.vstack(blocks)

    def back(self, step: int, weights: A) -> A:
        """These rows of ``weights``."""
        return weights[self._start : self._start + self._size]

    def nqoi(self, inner: int) -> int:
        return self._total


class _ValueMapProtocol(Protocol[A]):
    r"""A linear map of a term's values, ``out`` (to the functional's
    values) and its transpose ``back`` (functional weights to the term's),
    possibly depending on the step: a stack's placement in rows, or the
    recursion's :math:`\theta_n`."""

    def out(self, step: int, value: A) -> A: ...

    def back(self, step: int, weights: A) -> A: ...

    def nqoi(self, inner: int) -> int: ...


class _MappedLinearization(Generic[X, A]):
    r""":math:`P D` for a linearization :math:`D` and a value map
    :math:`P` at one step: forward through ``out``, transposed through
    ``back``, so no stack vector is touched here."""

    def __init__(
        self,
        inner: LinearizationProtocol[X, A],
        values: _ValueMapProtocol[A],
        step: int,
    ) -> None:
        self._inner, self._values, self._step = inner, values, step

    def apply(self, x: X) -> A:
        return self._values.out(self._step, self._inner.apply(x))

    def apply_transpose(self, weights: A) -> X:
        return self._inner.apply_transpose(self._values.back(self._step, weights))


def _mapped(
    inner: Optional[LinearizationProtocol[X, A]],
    values: _ValueMapProtocol[A],
    step: int,
) -> Optional[LinearizationProtocol[X, A]]:
    return None if inner is None else _MappedLinearization(inner, values, step)


class _MappedParameterTerm(Generic[V, A]):
    """A parameter term with its values mapped (step 0 for the map); its
    bundle mirrors the inner one's."""

    def __init__(
        self, term: ParameterTermProtocol[V, A], values: _ValueMapProtocol[A]
    ) -> None:
        self._term, self._values = term, values
        inner = term.derivatives()
        self._inner = inner
        self._derivatives: ParameterTermDerivatives[V, A] = ParameterTermDerivatives(
            None if inner.wrt_param is None else self._wrt_param,
            None if inner.param_param is None else self._param_param,
        )

    def nqoi(self) -> int:
        return self._values.nqoi(self._term.nqoi())

    def value(self) -> A:
        return self._values.out(0, self._term.value())

    def derivatives(self) -> ParameterTermDerivatives[V, A]:
        return self._derivatives

    def _wrt_param(self) -> LinearizationProtocol[V, A]:
        field = self._inner.wrt_param
        if field is None:
            raise RuntimeError("the term has no derivative")
        return _MappedLinearization(field(), self._values, 0)

    def _param_param(self, weights: A, v: V) -> Optional[V]:
        field = self._inner.param_param
        if field is None:
            raise RuntimeError("the term has no second derivative")
        return field(self._values.back(0, weights), v)


class _MappedStateTerm(Generic[V, A]):
    """A state term with its values mapped (step 0 for the map); its
    bundle mirrors the inner one's."""

    def __init__(
        self, term: StateTermProtocol[V, A], values: _ValueMapProtocol[A]
    ) -> None:
        self._term, self._values = term, values
        inner = term.derivatives()
        self._inner = inner
        second = inner.state_state is not None
        self._derivatives: StateTermDerivatives[V, A] = StateTermDerivatives(
            None if inner.wrt_state is None else self._wrt_state,
            None if inner.wrt_param is None else self._wrt_param,
            self._state_state if second else None,
            self._state_param if second else None,
            self._param_state if second else None,
            self._param_param if second else None,
        )

    def nqoi(self) -> int:
        return self._values.nqoi(self._term.nqoi())

    def reads(self, index: int, last: bool) -> bool:
        return self._term.reads(index, last)

    def value(self, index: int, time: float, state: V) -> A:
        return self._values.out(0, self._term.value(index, time, state))

    def derivatives(self) -> StateTermDerivatives[V, A]:
        return self._derivatives

    def _wrt_state(
        self, index: int, time: float, state: V
    ) -> LinearizationProtocol[V, A]:
        field = self._inner.wrt_state
        if field is None:
            raise RuntimeError("the term has no derivative")
        return _MappedLinearization(field(index, time, state), self._values, 0)

    def _wrt_param(
        self, index: int, time: float, state: V
    ) -> Optional[LinearizationProtocol[V, A]]:
        field = self._inner.wrt_param
        if field is None:
            raise RuntimeError("the term has no parameter derivative")
        return _mapped(field(index, time, state), self._values, 0)

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
        return field(index, time, state, self._values.back(0, weights), x)

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


class _MappedStepTerm(Generic[V, A]):
    """A step term with its values mapped, the map read at each step; its
    bundle mirrors the inner one's."""

    def __init__(
        self, term: StepTermProtocol[V, A], values: _ValueMapProtocol[A]
    ) -> None:
        self._term, self._values = term, values
        inner = term.derivatives()
        self._inner = inner
        first = inner.wrt_unknown is not None
        second = inner.curvature_wrt_unknown is not None
        self._derivatives: StepTermDerivatives[V, A] = StepTermDerivatives(
            self._wrt_unknown if first else None,
            self._wrt_lag if first else None,
            self._wrt_param if first else None,
            self._curvature_wrt_unknown if second else None,
            self._curvature_wrt_lag if second else None,
            self._curvature_wrt_param if second else None,
        )

    def nqoi(self) -> int:
        return self._values.nqoi(self._term.nqoi())

    def value(self, step: int, window: StepWindow[V]) -> A:
        return self._values.out(step, self._term.value(step, window))

    def derivatives(self) -> StepTermDerivatives[V, A]:
        return self._derivatives

    def _wrt_unknown(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[BlockVector[V], A]]:
        field = self._inner.wrt_unknown
        if field is None:
            raise RuntimeError("the term has no derivatives")
        return _mapped(field(step, window), self._values, step)

    def _wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[LinearizationProtocol[V, A]]:
        field = self._inner.wrt_lag
        if field is None:
            raise RuntimeError("the term has no derivatives")
        return _mapped(field(step, window, lag), self._values, step)

    def _wrt_param(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[V, A]]:
        field = self._inner.wrt_param
        if field is None:
            raise RuntimeError("the term has no derivatives")
        return _mapped(field(step, window), self._values, step)

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
        return field(
            step, window, tangents, direction, self._values.back(step, weights)
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
        field = self._inner.curvature_wrt_lag
        if field is None:
            raise RuntimeError("the term has no curvature")
        return field(
            step, window, tangents, lag, direction, self._values.back(step, weights)
        )

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
        return field(
            step, window, tangents, direction, self._values.back(step, weights)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._term!r})"


class _Recursion(Generic[A]):
    r"""The value map :math:`c_n \mapsto \theta_n c_n`, its own
    transpose."""

    def __init__(self, theta: Sequence[float]) -> None:
        self._theta = tuple(theta)

    def out(self, step: int, value: A) -> A:
        return self._theta[step] * value

    def back(self, step: int, weights: A) -> A:
        return self._theta[step] * weights

    def nqoi(self, inner: int) -> int:
        return inner


class RecursionWeighted(_MappedStepTerm[V, A]):
    r"""A step term weighed by the stepper's recursion: its step
    contributions :math:`c_n` combined as :math:`\sum_n \theta_n c_n`,
    :math:`\theta` the recursion's sensitivities on the trajectory (see
    :func:`~pyapprox.ode.step_quadrature.recursion_sensitivities`), so
    the sum is the method's own integral over time; 1 throughout for a
    Runge-Kutta or Adams method.

    Every derivative is the term's scaled by :math:`\theta_n`, through the
    weights, so the term's own code serves both; its bundle holds what the
    term's holds.

    Parameters
    ----------
    term : StepTermProtocol
        The step contributions.
    theta : sequence of float
        :math:`\theta_n` for every step of the trajectory.
    """

    def __init__(self, term: StepTermProtocol[V, A], theta: Sequence[float]) -> None:
        super().__init__(term, _Recursion(theta))

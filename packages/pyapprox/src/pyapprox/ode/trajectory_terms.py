r"""The terms of a functional of a recorded trajectory.

A functional :math:`Q` of a trajectory, with values of ``nqoi`` entries,
is a sum of terms of three kinds, each read where it lives:

- a parameter term :math:`r(p)` (a regularization), read from the
  parameters alone (:class:`ParameterTermProtocol`);
- a state term :math:`\sum_{m} g_m(y_m;\, p)` over the stored states it
  names (an observation, an endpoint value): a family of the core's
  one-state terms (:class:`~pyapprox.implicitfunction.functional_terms.
  StateTermProtocol`), member :math:`g_m` built for stored state
  :math:`m` at its time (:class:`StoredStateTermProtocol`);
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
term, when one lacks what it needs; a state term's family is as capable
as its weakest member read. A term holds no trajectory: it is handed the
state or window it reads, so the same term serves any store and the value
sink.

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
    FrozenSet,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.implicitfunction.functional_terms import (
    LinearizationProtocol,
    ParameterTermProtocol,
    StateTermProtocol,
    value_weights,
)
from pyapprox.implicitfunction.mapped_terms import (
    MappedParameterTerm,
    MappedStateTerm,
    ValueMapProtocol,
    mapped,
    row_blocks,
)
from pyapprox.ode.stepper_derivatives import StepOperator, StepWindow
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import BlockVector

V = TypeVar("V")
X = TypeVar("X")
A = TypeVar("A", bound=ArrayProtocol)


@dataclass(frozen=True)
class ChainPart(Generic[X, V, A]):
    r"""One piece of a chained first derivative (see
    :class:`ChainedLinearization`): an outer linear map :math:`G` of a
    step operator's output :math:`D x` (None for a zero :math:`D`), and a
    direct derivative in the same argument (None for zero)."""

    outer: LinearizationProtocol[BlockVector[V], A]
    inner: Optional[StepOperator[V]]
    direct: Optional[LinearizationProtocol[X, A]]


class ChainedLinearization(Generic[X, V, A]):
    r""":math:`\sum_k (G_k D_k + E_k)`: a term's first derivative through
    step operators :math:`D_k` (a continuous extension's, a step
    residual's) and outer maps :math:`G_k`, plus direct derivatives
    :math:`E_k`. Forward through :math:`D_k` then :math:`G_k`; transposed
    through :math:`G_k^\top` then :math:`D_k^\top`.

    Parameters
    ----------
    parts : sequence of ChainPart
        The pieces, at least one.
    wrap, unwrap : callable
        The argument as the step operators take it (a block vector), and
        back.
    add : callable
        ``(x, y) ->`` a new vector, :math:`x + y`, or a copy of :math:`y`
        for ``x`` None: a term's transposed derivative may be its own
        storage, overwritten by its next call.

    Raises
    ------
    ValueError
        If ``parts`` is empty.
    """

    def __init__(
        self,
        parts: Sequence[ChainPart[X, V, A]],
        wrap: Callable[[X], BlockVector[V]],
        unwrap: Callable[[BlockVector[V]], X],
        add: Callable[[Optional[X], X], X],
    ) -> None:
        if not parts:
            raise ValueError("a chained derivative needs at least one part")
        self._parts = tuple(parts)
        self._wrap, self._unwrap, self._add = wrap, unwrap, add

    def apply(self, x: X) -> A:
        total: Optional[A] = None
        for part in self._parts:
            values: List[A] = []
            if part.inner is not None:
                values.append(part.outer.apply(part.inner.apply(self._wrap(x))))
            if part.direct is not None:
                values.append(part.direct.apply(x))
            for value in values:
                total = value if total is None else total + value
        if total is None:
            raise ValueError("every part of the chained derivative is zero")
        return total

    def apply_transpose(self, weights: A) -> X:
        total: Optional[X] = None
        for part in self._parts:
            if part.inner is not None:
                pulled = part.inner.apply_transpose(part.outer.apply_transpose(weights))
                total = self._add(total, self._unwrap(pulled))
            if part.direct is not None:
                total = self._add(total, part.direct.apply_transpose(weights))
        if total is None:
            raise ValueError("every part of the chained derivative is zero")
        return total


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
class StoredStateTermProtocol(Protocol[V, A]):
    r"""A term :math:`\sum_m g_m(y_m;\, p)` over the stored states it
    names, each read alone: a family of one-state terms, member
    :math:`g_m` a :class:`~pyapprox.implicitfunction.functional_terms.
    StateTermProtocol` with its value and its derivatives.

    It holds no trajectory: it says which stored states it reads, by
    index and whether a state is the last, and builds the member for each
    one it reads from its index and time, so a member's time is bound
    once, never passed per call. Naming the last state by ``last``, not
    by an index, lets a sink evaluate the term while the trajectory is
    still being computed, its length unknown (an adaptive plan's).

    The second derivative in the states is block diagonal over the stored
    states (:math:`\partial^2 g/\partial y_m \partial y_k = 0` for
    :math:`m \ne k`): a term coupling states of one step is a step term.
    The family is as capable as its weakest member read: a consumer reads
    every member's bundle once, before any sweep, and refuses naming the
    member that lacks what it needs.
    """

    def nqoi(self) -> int:
        """The number of entries of every member's value."""
        ...

    def reads(self, index: int, last: bool) -> bool:
        """Whether the term reads stored state ``index``, the trajectory's
        last when ``last``."""
        ...

    def at(self, index: int, time: float) -> StateTermProtocol[V, A]:
        """:math:`g_m`, the member for stored state ``index`` at ``time``;
        called only for a state the term reads. A member serves the
        trajectory whose time it was built at: consumers build members
        anew for each trajectory, whose times an adaptive plan changes."""
        ...


class StoredStates:
    """Which stored states a term reads: those of ``indices``, and the
    last when ``last``.

    Parameters
    ----------
    indices : iterable of int, optional
        Stored-state indices, non-negative.
    last : bool, default False
        Whether the trajectory's last state is read, whatever its index.

    Raises
    ------
    ValueError
        If no state is named or an index is negative.
    """

    def __init__(self, indices: Iterable[int] = (), last: bool = False) -> None:
        self._indices: FrozenSet[int] = frozenset(indices)
        if not self._indices and not last:
            raise ValueError("a term reads at least one stored state")
        if any(index < 0 for index in self._indices):
            raise ValueError(
                "stored-state indices must be non-negative, got "
                f"{sorted(self._indices)}"
            )
        self._last = last

    def reads(self, index: int, last: bool) -> bool:
        """Whether stored state ``index`` is named, the trajectory's last
        when ``last``."""
        return index in self._indices or (self._last and last)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(self._indices)}, last={self._last})"


class AtStoredStates(Generic[V, A]):
    r"""One state term read at several stored states, the same member at
    each: :math:`\sum_m g(y_m;\, p)` over the states ``where`` names (the
    last alone, for an endpoint value).

    Parameters
    ----------
    term : StateTermProtocol
        :math:`g`, a term of one state that reads no time.
    where : StoredStates
        The stored states read.

    Raises
    ------
    TypeError
        If ``term`` does not satisfy :class:`~pyapprox.implicitfunction.
        functional_terms.StateTermProtocol`.
    """

    def __init__(self, term: StateTermProtocol[V, A], where: StoredStates) -> None:
        if not isinstance(term, StateTermProtocol):
            raise TypeError(
                f"term must satisfy StateTermProtocol, got {type(term).__name__}"
            )
        self._term, self._where = term, where

    def nqoi(self) -> int:
        return self._term.nqoi()

    def reads(self, index: int, last: bool) -> bool:
        return self._where.reads(index, last)

    def at(self, index: int, time: float) -> StateTermProtocol[V, A]:
        return self._term

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._term!r}, {self._where!r})"


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
    state_terms: Tuple[StoredStateTermProtocol[V, A], ...] = ()
    step_terms: Tuple[StepTermProtocol[V, A], ...] = ()

    def __post_init__(self) -> None:
        if self.zero.ndim != 2 or self.zero.shape[1] != 1:
            raise ValueError(
                f"zero must be a column (nqoi, 1), got shape {tuple(self.zero.shape)}"
            )
        checks: Tuple[tuple[Sequence[object], type, str], ...] = (
            (self.parameter_terms, ParameterTermProtocol, "ParameterTermProtocol"),
            (self.state_terms, StoredStateTermProtocol, "StoredStateTermProtocol"),
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
        state_terms: Tuple[StoredStateTermProtocol[V, A], ...] = (),
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
        return value_weights(self.zero, given)

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
        parameter_terms: list[ParameterTermProtocol[V, A]] = []
        state_terms: list[StoredStateTermProtocol[V, A]] = []
        step_terms: list[StepTermProtocol[V, A]] = []
        sizes = [functional.nqoi() for functional in functionals]
        for functional, rows in zip(functionals, row_blocks(bkd, sizes)):
            parameter_terms += [
                MappedParameterTerm(term, rows) for term in functional.parameter_terms
            ]
            state_terms += [
                _MappedStoredStateTerm(term, rows) for term in functional.state_terms
            ]
            step_terms += [
                _MappedStepTerm(term, _EveryStep(rows))
                for term in functional.step_terms
            ]
        return TrajectoryFunctional.from_terms(
            bkd,
            sum(sizes),
            tuple(parameter_terms),
            tuple(state_terms),
            tuple(step_terms),
        )


class _StepValueMapProtocol(Protocol[A]):
    r"""A linear map of a step term's values that may depend on the step:
    ``out`` (to the functional's values) and its transpose ``back``
    (functional weights to the term's); the recursion's :math:`\theta_n`,
    or a stack's rows at every step alike."""

    def out(self, step: int, value: A) -> A: ...

    def back(self, step: int, weights: A) -> A: ...

    def nqoi(self, inner: int) -> int: ...


class _EveryStep(Generic[A]):
    """A value map the same at every step."""

    def __init__(self, values: ValueMapProtocol[A]) -> None:
        self._values = values

    def out(self, step: int, value: A) -> A:
        return self._values.out(value)

    def back(self, step: int, weights: A) -> A:
        return self._values.back(weights)

    def nqoi(self, inner: int) -> int:
        return self._values.nqoi(inner)


class _AtStep(Generic[A]):
    """A step's value map at one step."""

    def __init__(self, values: _StepValueMapProtocol[A], step: int) -> None:
        self._values, self._step = values, step

    def out(self, value: A) -> A:
        return self._values.out(self._step, value)

    def back(self, weights: A) -> A:
        return self._values.back(self._step, weights)

    def nqoi(self, inner: int) -> int:
        return self._values.nqoi(inner)


class _MappedStoredStateTerm(Generic[V, A]):
    """A stored-state term with its values mapped: each member the core's
    mapped term of the inner member."""

    def __init__(
        self, term: StoredStateTermProtocol[V, A], values: ValueMapProtocol[A]
    ) -> None:
        self._term, self._values = term, values

    def nqoi(self) -> int:
        return self._values.nqoi(self._term.nqoi())

    def reads(self, index: int, last: bool) -> bool:
        return self._term.reads(index, last)

    def at(self, index: int, time: float) -> StateTermProtocol[V, A]:
        return MappedStateTerm(self._term.at(index, time), self._values)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._term!r})"


class _MappedStepTerm(Generic[V, A]):
    """A step term with its values mapped, the map read at each step; its
    bundle mirrors the inner one's."""

    def __init__(
        self, term: StepTermProtocol[V, A], values: _StepValueMapProtocol[A]
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
        return mapped(field(step, window), _AtStep(self._values, step))

    def _wrt_lag(
        self, step: int, window: StepWindow[V], lag: int
    ) -> Optional[LinearizationProtocol[V, A]]:
        field = self._inner.wrt_lag
        if field is None:
            raise RuntimeError("the term has no derivatives")
        return mapped(field(step, window, lag), _AtStep(self._values, step))

    def _wrt_param(
        self, step: int, window: StepWindow[V]
    ) -> Optional[LinearizationProtocol[V, A]]:
        field = self._inner.wrt_param
        if field is None:
            raise RuntimeError("the term has no derivatives")
        return mapped(field(step, window), _AtStep(self._values, step))

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

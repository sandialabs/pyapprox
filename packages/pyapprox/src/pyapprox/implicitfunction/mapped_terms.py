r"""Terms whose values pass through a linear map, and the rows a stack of
functionals places each one's values in.

A value map :math:`P` (:class:`ValueMapProtocol`) takes a term's values to
the functional's (``out``) and the functional's weights back to the
term's (``back``, :math:`P^\top`). A mapped term applies it to its value
and its first derivatives forward, and to the weights its transposed
derivatives and curvatures read, so the term's own code serves the mapped
one and no stack vector is touched here.

:class:`Rows` places a term's values in rows of a taller value, which
stacks functionals one above another (:func:`row_blocks`).
"""

from typing import Generic, List, Optional, Protocol, Sequence, TypeVar

from pyapprox.implicitfunction.functional_terms import (
    LinearizationProtocol,
    ParameterTermDerivatives,
    ParameterTermProtocol,
    StateCurvatureFn,
    StateTermDerivatives,
    StateTermProtocol,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend

V = TypeVar("V")
X = TypeVar("X")
A = TypeVar("A", bound=ArrayProtocol)


class ValueMapProtocol(Protocol[A]):
    r"""A linear map :math:`P` of a term's values: ``out`` to the
    functional's values, ``back`` (:math:`P^\top`) from the functional's
    weights to the term's."""

    def out(self, value: A) -> A: ...

    def back(self, weights: A) -> A: ...

    def nqoi(self, inner: int) -> int:
        """The number of the functional's entries, for a term of
        ``inner``."""
        ...


class Rows(Generic[A]):
    """Rows ``start`` to ``start + size`` of a value of ``total`` entries,
    as a value map: values are placed there, weights read from there."""

    def __init__(self, bkd: Backend[A], start: int, size: int, total: int) -> None:
        self._bkd, self._start, self._size, self._total = bkd, start, size, total

    def out(self, value: A) -> A:
        """``value`` in these rows, zero elsewhere (by stacking, so an
        autograd graph through ``value`` is kept)."""
        bkd, end = self._bkd, self._start + self._size
        blocks = [value]
        if self._start > 0:
            blocks.insert(0, bkd.zeros((self._start, 1)))
        if end < self._total:
            blocks.append(bkd.zeros((self._total - end, 1)))
        return blocks[0] if len(blocks) == 1 else bkd.vstack(blocks)

    def back(self, weights: A) -> A:
        """These rows of ``weights``."""
        return weights[self._start : self._start + self._size]

    def nqoi(self, inner: int) -> int:
        return self._total


def row_blocks(bkd: Backend[A], sizes: Sequence[int]) -> List[Rows[A]]:
    """The rows of each of values of ``sizes`` entries stacked in order.

    Raises
    ------
    ValueError
        If no size is given.
    """
    if not sizes:
        raise ValueError("a stack needs at least one functional")
    total, start, blocks = sum(sizes), 0, []
    for size in sizes:
        blocks.append(Rows(bkd, start, size, total))
        start += size
    return blocks


class MappedLinearization(Generic[X, A]):
    r""":math:`P D` for a linearization :math:`D` and a value map
    :math:`P`: forward through ``out``, transposed through ``back``."""

    def __init__(
        self, inner: LinearizationProtocol[X, A], values: ValueMapProtocol[A]
    ) -> None:
        self._inner, self._values = inner, values

    def apply(self, x: X) -> A:
        return self._values.out(self._inner.apply(x))

    def apply_transpose(self, weights: A) -> X:
        return self._inner.apply_transpose(self._values.back(weights))


def mapped(
    inner: Optional[LinearizationProtocol[X, A]], values: ValueMapProtocol[A]
) -> Optional[LinearizationProtocol[X, A]]:
    """``inner`` mapped by ``values``; None stays None (a zero
    derivative)."""
    return None if inner is None else MappedLinearization(inner, values)


class MappedParameterTerm(Generic[V, A]):
    """A parameter term with its values mapped; its bundle mirrors the
    inner one's."""

    def __init__(
        self, term: ParameterTermProtocol[V, A], values: ValueMapProtocol[A]
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
        return self._values.out(self._term.value())

    def derivatives(self) -> ParameterTermDerivatives[V, A]:
        return self._derivatives

    def _wrt_param(self) -> LinearizationProtocol[V, A]:
        field = self._inner.wrt_param
        if field is None:
            raise RuntimeError("the term has no derivative")
        return MappedLinearization(field(), self._values)

    def _param_param(self, weights: A, v: V) -> Optional[V]:
        field = self._inner.param_param
        if field is None:
            raise RuntimeError("the term has no second derivative")
        return field(self._values.back(weights), v)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._term!r})"


class MappedStateTerm(Generic[V, A]):
    """A state term with its values mapped; its bundle mirrors the inner
    one's."""

    def __init__(
        self, term: StateTermProtocol[V, A], values: ValueMapProtocol[A]
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

    def value(self, state: V) -> A:
        return self._values.out(self._term.value(state))

    def derivatives(self) -> StateTermDerivatives[V, A]:
        return self._derivatives

    def _wrt_state(self, state: V) -> LinearizationProtocol[V, A]:
        field = self._inner.wrt_state
        if field is None:
            raise RuntimeError("the term has no derivative")
        return MappedLinearization(field(state), self._values)

    def _wrt_param(self, state: V) -> Optional[LinearizationProtocol[V, A]]:
        field = self._inner.wrt_param
        if field is None:
            raise RuntimeError("the term has no parameter derivative")
        return mapped(field(state), self._values)

    def _call(
        self, field: Optional[StateCurvatureFn[V, A]], state: V, weights: A, x: V
    ) -> Optional[V]:
        if field is None:
            raise RuntimeError("the term has no second derivatives")
        return field(state, self._values.back(weights), x)

    def _state_state(self, state: V, weights: A, x: V) -> Optional[V]:
        return self._call(self._inner.state_state, state, weights, x)

    def _state_param(self, state: V, weights: A, x: V) -> Optional[V]:
        return self._call(self._inner.state_param, state, weights, x)

    def _param_state(self, state: V, weights: A, x: V) -> Optional[V]:
        return self._call(self._inner.param_state, state, weights, x)

    def _param_param(self, state: V, weights: A, x: V) -> Optional[V]:
        return self._call(self._inner.param_param, state, weights, x)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._term!r})"

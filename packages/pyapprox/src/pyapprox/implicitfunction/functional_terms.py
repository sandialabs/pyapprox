r"""What every functional's terms share: linearizations, parameter terms,
value weights, and the readers of their bundles.

A functional :math:`Q`, with values of ``nqoi`` entries, is a sum of
terms; each gives its value and, optionally, its derivatives in a bundle
(``derivatives()``), None when absent. Its first derivatives are
linearizations (:class:`LinearizationProtocol`): one object applying
forward (:math:`D x`, a column of ``nqoi`` entries, for a tangent) and
transposed (:math:`D^\top \omega` for weights :math:`\omega` of the
values, for an adjoint); its second derivatives are weighted by
:math:`\omega` as an adjoint weighs them.

A parameter term :math:`r(p)` (:class:`ParameterTermProtocol`) reads the
parameters alone, so it is the same in a steady functional
(:mod:`pyapprox.implicitfunction.steady_functional`) and a functional of a
trajectory (:mod:`pyapprox.ode.trajectory_terms`); the terms reading
states differ, by what they read. The readers here take what a
derivative computation needs from the bundles once and refuse, naming
the term, when one lacks it.
"""

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import VectorSpaceProtocol

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


def no_first_derivatives(term: object) -> TypeError:
    """The refusal of a term whose bundle has no first derivatives."""
    return TypeError(
        f"{type(term).__name__}'s derivatives() has no first derivatives, "
        "so the functional is only evaluated"
    )


def no_second_derivatives(term: object) -> TypeError:
    """The refusal of a term whose bundle has no second derivatives."""
    return TypeError(
        f"{type(term).__name__}'s derivatives() has no second derivatives, "
        "so the functional has no Hessian-vector products"
    )


def parameter_first_derivatives(
    terms: Sequence[ParameterTermProtocol[V, A]],
) -> List[Callable[[], LinearizationProtocol[V, A]]]:
    r"""Each parameter term's :math:`\partial r/\partial p`, read once.

    Raises
    ------
    TypeError
        If a term's bundle has none.
    """
    found = []
    for term in terms:
        wrt_param = term.derivatives().wrt_param
        if wrt_param is None:
            raise no_first_derivatives(term)
        found.append(wrt_param)
    return found


def parameter_second_derivatives(
    terms: Sequence[ParameterTermProtocol[V, A]],
) -> List[Callable[[A, V], Optional[V]]]:
    r"""Each parameter term's :math:`(\omega^\top r)_{pp}`, read once.

    Raises
    ------
    TypeError
        If a term's bundle has none.
    """
    found = []
    for term in terms:
        param_param = term.derivatives().param_param
        if param_param is None:
            raise no_second_derivatives(term)
        found.append(param_param)
    return found


def accumulate(
    total: Optional[V], term: Optional[V], space: VectorSpaceProtocol[V]
) -> Optional[V]:
    """``total + term``, into ``total`` when there is one (a new vector the
    caller owns); a term starting the total is copied, since it may be a
    term's own storage, overwritten by that term's next call. So each
    term is added as soon as it is returned, never kept for later."""
    if term is None:
        return total
    if total is None:
        return space.copy(term)
    return space.axpby(1.0, term, 1.0, total, out=total)


def value_weights(zero: A, given: Optional[A] = None) -> A:
    r"""The weights :math:`\omega` of a functional's entries, whose zero
    value is ``zero``, for an adjoint: ``given``, or :math:`\omega = 1`
    for a scalar functional.

    Raises
    ------
    ValueError
        If ``given`` is not of the shape of ``zero``, or none is given for
        a functional of several entries.
    """
    nqoi = int(zero.shape[0])
    if given is not None:
        if tuple(given.shape) != (nqoi, 1):
            raise ValueError(
                f"weights must have shape ({nqoi}, 1), got {tuple(given.shape)}"
            )
        return given
    if nqoi != 1:
        raise ValueError(f"a functional of {nqoi} entries needs its weights given")
    return zero + 1.0

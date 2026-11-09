r"""A functional of a steady state, :math:`Q(u;\, p)`, as a sum of terms.

Its terms are parameter terms :math:`r(p)` and state terms :math:`g(u;\,
p)` read from the one state (:class:`~pyapprox.implicitfunction.
functional_terms.ParameterTermProtocol`, :class:`~pyapprox.
implicitfunction.functional_terms.StateTermProtocol`).
:class:`SteadyFunctional` gathers them; ``a + b`` is the functional of
both, and :meth:`SteadyFunctional.stack` places several one above
another as the entries of one.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Sequence, Tuple, TypeVar

from pyapprox.implicitfunction.functional_terms import (
    ParameterTermProtocol,
    StateTermProtocol,
    value_weights,
)
from pyapprox.implicitfunction.mapped_terms import (
    MappedParameterTerm,
    MappedStateTerm,
    row_blocks,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)


@dataclass(frozen=True)
class SteadyFunctional(Generic[V, A]):
    r""":math:`Q(u;\, p)`: a sum of parameter and state terms, with values
    of ``nqoi`` entries.

    Parameters
    ----------
    zero : A
        A zero value, shape ``(nqoi, 1)``: the functional's value type and
        size, from which its unit weights are made for ``nqoi = 1``.
    parameter_terms, state_terms : tuple
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

    def __post_init__(self) -> None:
        if self.zero.ndim != 2 or self.zero.shape[1] != 1:
            raise ValueError(
                f"zero must be a column (nqoi, 1), got shape {tuple(self.zero.shape)}"
            )
        checks: Tuple[Tuple[Sequence[object], type, str], ...] = (
            (self.parameter_terms, ParameterTermProtocol, "ParameterTermProtocol"),
            (self.state_terms, StateTermProtocol, "StateTermProtocol"),
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
    ) -> "SteadyFunctional[V, A]":
        """The functional of these terms, with values of ``nqoi`` entries
        on ``bkd`` (its ``zero`` made here)."""
        return SteadyFunctional(bkd.zeros((nqoi, 1)), parameter_terms, state_terms)

    def nqoi(self) -> int:
        """The number of entries of the functional's value."""
        return int(self.zero.shape[0])

    def weights(self, given: Optional[A] = None) -> A:
        r"""The weights :math:`\omega` of the entries for an adjoint:
        ``given``, or :math:`\omega = 1` for a scalar functional.

        Raises
        ------
        ValueError
            If ``given`` is not of shape ``(nqoi, 1)``, or none is given
            for a functional of several entries.
        """
        return value_weights(self.zero, given)

    def value(self, state: V) -> A:
        """:math:`Q(u)`, a column of ``nqoi`` entries."""
        total = self.zero
        for parameter_term in self.parameter_terms:
            total = total + parameter_term.value()
        for state_term in self.state_terms:
            total = total + state_term.value(state)
        return total

    def __add__(self, other: "SteadyFunctional[V, A]") -> "SteadyFunctional[V, A]":
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
        return SteadyFunctional(
            self.zero,
            self.parameter_terms + other.parameter_terms,
            self.state_terms + other.state_terms,
        )

    @staticmethod
    def stack(
        bkd: Backend[A], functionals: Sequence["SteadyFunctional[V, A]"]
    ) -> "SteadyFunctional[V, A]":
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
        state_terms: list[StateTermProtocol[V, A]] = []
        sizes = [functional.nqoi() for functional in functionals]
        for functional, rows in zip(functionals, row_blocks(bkd, sizes)):
            parameter_terms += [
                MappedParameterTerm(term, rows) for term in functional.parameter_terms
            ]
            state_terms += [
                MappedStateTerm(term, rows) for term in functional.state_terms
            ]
        return SteadyFunctional.from_terms(
            bkd, sum(sizes), tuple(parameter_terms), tuple(state_terms)
        )

"""The natural-BC operator: the sum of weak-form boundary terms.

A natural boundary condition is a term ``c_k(u, t)`` of the spatial
operator in ``M du/dt = F(u, t)`` (see ``WeakFormBCProtocol`` for the sign
convention). This operator adds ``sum_k c_k`` to a residual and
``sum_k dc_k/du`` to a Jacobian, so a discretization composes the
natural-BC part of ``F`` in exactly one place:

    F = F_interior + F_Gamma,    F_Gamma = sum_k c_k.

With no terms the operator returns its inputs unchanged and allocates
nothing.
"""

from typing import Generic, List, Sequence, Union, overload

from scipy.sparse import spmatrix

from pyapprox.pde.boundary.protocols import WeakFormBCProtocol
from pyapprox.util.backends.protocols import Array


class NaturalBCOperator(Generic[Array]):
    """Sum of weak-form boundary terms, ``F_Gamma = sum_k c_k``.

    Parameters
    ----------
    terms : Sequence[WeakFormBCProtocol]
        The natural-BC terms, in the order they are added.
    """

    def __init__(self, terms: Sequence[WeakFormBCProtocol[Array]]) -> None:
        for term in terms:
            if not isinstance(term, WeakFormBCProtocol):
                raise TypeError(
                    "terms must satisfy WeakFormBCProtocol, got "
                    f"{type(term).__name__}"
                )
        self._terms: List[WeakFormBCProtocol[Array]] = list(terms)

    def terms(self) -> List[WeakFormBCProtocol[Array]]:
        """Return the terms, in the order they are added."""
        return list(self._terms)

    def is_empty(self) -> bool:
        """Whether there are no terms (the operator is then the identity)."""
        return not self._terms

    def add_to_residual(self, residual: Array, state: Array, time: float) -> Array:
        """Return ``residual + sum_k c_k(state, time)``.

        Parameters
        ----------
        residual : Array
            Residual to add to, typically the interior residual.
            Shape: (nstates,)
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Current time.
        """
        for term in self._terms:
            residual = term.apply_to_residual(residual, state, time)
        return residual

    @overload
    def add_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array: ...

    @overload
    def add_to_jacobian(
        self, jacobian: spmatrix, state: Array, time: float
    ) -> spmatrix: ...

    def add_to_jacobian(
        self, jacobian: Union[spmatrix, Array], state: Array, time: float
    ) -> Union[spmatrix, Array]:
        """Return ``jacobian + sum_k dc_k/du`` (same matrix type as input).

        Parameters
        ----------
        jacobian : sparse matrix or Array
            Jacobian to add to, typically the interior Jacobian.
            Shape: (nstates, nstates)
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Current time.
        """
        for term in self._terms:
            jacobian = term.apply_to_jacobian(jacobian, state, time)
        return jacobian

    def __repr__(self) -> str:
        names = ", ".join(type(term).__name__ for term in self._terms)
        return f"NaturalBCOperator([{names}])"

r"""The boundary between stack vectors and a residual's own state form.

Generic time-integration code holds states as single-column vectors of
the state space, the stack's vector type ``V``, and forms them with the
space's operations (a stage :math:`Y_i = y_n + \Delta t \sum_j a_{ij}
K_j`, a finite-difference step). A residual's callables take states in
the residual's own form: on the backend stack a 1-D array. A
:class:`ResidualFormProtocol` converts between the two at the one place
generic code calls the residual and its bundle.

On the backend stack the conversion is a reshape
(:class:`BackendColumnForm`); a residual whose callables already take
the stack's vectors uses :class:`IdentityForm`. The same conversion
serves parameters, which are vectors of a space of the same stack.
"""

from typing import Generic, Protocol, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend

V = TypeVar("V")
S = TypeVar("S")


@runtime_checkable
class ResidualFormProtocol(Protocol[V, S]):
    """Converts single-column stack vectors to a residual's state form
    ``S`` and back."""

    def to_residual(self, x: V) -> S:
        """The residual's form of the single-column vector ``x``."""
        ...

    def from_residual(self, state: S) -> V:
        """The single-column stack vector holding ``state``."""
        ...


class BackendColumnForm(Generic[Array]):
    """Single-column backend arrays of shape ``(n, 1)`` to the 1-D arrays
    a backend residual takes, and back.

    Parameters
    ----------
    bkd : Backend
        The backend of the arrays.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def to_residual(self, x: Array) -> Array:
        """``x[:, 0]``.

        Raises
        ------
        ValueError
            If ``x`` is not a single column.
        """
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(
                f"x must be a single column of shape (n, 1), got {tuple(x.shape)}"
            )
        return x[:, 0]

    def from_residual(self, state: Array) -> Array:
        """``state`` as a column of shape ``(n, 1)``.

        Raises
        ------
        ValueError
            If ``state`` is not 1-D.
        """
        if state.ndim != 1:
            raise ValueError(f"state must be 1-D, got shape {tuple(state.shape)}")
        return self._bkd.reshape(state, (state.shape[0], 1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class IdentityForm(Generic[V]):
    """For a residual whose callables take the stack's vectors: no
    conversion."""

    def to_residual(self, x: V) -> V:
        return x

    def from_residual(self, state: V) -> V:
        return state

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

"""Result classes for fitting operations.

All attributes accessed via methods per CLAUDE.md conventions.
"""

from typing import Generic, Protocol, TypeVar, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class CallableSurrogateProtocol(Protocol, Generic[Array]):
    """Minimal protocol for surrogates that can be evaluated."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate surrogate at samples."""
        ...


S = TypeVar("S", bound=CallableSurrogateProtocol)  # type: ignore[type-arg]


class DirectSolverResult(Generic[Array, S]):
    """Result from direct linear solver (LeastSquares, Ridge).

    Contains only essential fields - no expensive diagnostics like
    condition number or rank that would require additional SVD computation.

    All attributes are accessed via methods per CLAUDE.md convention.

    Parameters
    ----------
    surrogate : S
        The fitted surrogate (e.g., BasisExpansion with coefficients set).
    params : Array
        Fitted parameters. Shape: (nterms, nqoi)
    """

    def __init__(
        self,
        surrogate: S,
        params: Array,
    ):
        self._surrogate = surrogate
        self._params = params

    def surrogate(self) -> S:
        """Return the fitted surrogate."""
        return self._surrogate

    def params(self) -> Array:
        """Return fitted parameters. Shape: (nterms, nqoi)"""
        return self._params

    def bkd(self) -> Backend[Array]:
        """Return backend from surrogate."""
        return self._surrogate.bkd()

    def __call__(self, samples: Array) -> Array:
        """Evaluate fitted surrogate at samples.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Values at samples. Shape: (nqoi, nsamples)
        """
        return self._surrogate(samples)

    def __repr__(self) -> str:
        return f"DirectSolverResult(params_shape={self._params.shape})"

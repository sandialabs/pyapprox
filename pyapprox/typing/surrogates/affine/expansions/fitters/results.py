"""Result classes for fitting operations.

All attributes accessed via methods per CLAUDE.md conventions.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.optimization.linear.sparse import OMPTerminationFlag


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
        result: Array = self._surrogate(samples)
        return result

    def __repr__(self) -> str:
        return f"DirectSolverResult(params_shape={self._params.shape})"


class SparseResult(Generic[Array, S]):
    """Result from sparse estimation (LASSO, BPDN, etc.).

    Extends DirectSolverResult pattern with sparsity-specific fields.

    All attributes are accessed via methods per CLAUDE.md convention.

    Parameters
    ----------
    surrogate : S
        The fitted surrogate (e.g., BasisExpansion with coefficients set).
    params : Array
        Fitted parameters. Shape: (nterms, nqoi)
    n_nonzero : int
        Number of non-zero coefficients.
    support : Array
        Indices of non-zero coefficients. Shape: (n_nonzero,)
    regularization_strength : float
        The regularization parameter used (e.g., alpha for LASSO).
    """

    def __init__(
        self,
        surrogate: S,
        params: Array,
        n_nonzero: int,
        support: Array,
        regularization_strength: float,
    ):
        self._surrogate = surrogate
        self._params = params
        self._n_nonzero = n_nonzero
        self._support = support
        self._regularization_strength = regularization_strength

    def surrogate(self) -> S:
        """Return the fitted surrogate."""
        return self._surrogate

    def params(self) -> Array:
        """Return fitted parameters. Shape: (nterms, nqoi)"""
        return self._params

    def bkd(self) -> Backend[Array]:
        """Return backend from surrogate."""
        return self._surrogate.bkd()

    def n_nonzero(self) -> int:
        """Return number of non-zero coefficients."""
        return self._n_nonzero

    def support(self) -> Array:
        """Return indices of non-zero coefficients. Shape: (n_nonzero,)"""
        return self._support

    def regularization_strength(self) -> float:
        """Return the regularization parameter used."""
        return self._regularization_strength

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
        result: Array = self._surrogate(samples)
        return result

    def __repr__(self) -> str:
        return (
            f"SparseResult(params_shape={self._params.shape}, "
            f"n_nonzero={self._n_nonzero}, "
            f"regularization={self._regularization_strength})"
        )


class OMPResult(Generic[Array, S]):
    """Result from Orthogonal Matching Pursuit.

    Extends SparseResult pattern with OMP-specific fields:
    selection order and residual history.

    All attributes are accessed via methods per CLAUDE.md convention.

    Parameters
    ----------
    surrogate : S
        The fitted surrogate (e.g., BasisExpansion with coefficients set).
    params : Array
        Fitted parameters. Shape: (nterms, nqoi)
    n_nonzero : int
        Number of non-zero coefficients.
    support : Array
        Indices of non-zero coefficients. Shape: (n_nonzero,)
    selection_order : Array
        Order in which basis terms were selected. Shape: (n_nonzero,)
    residual_history : Array
        Residual norm at each iteration. Shape: (n_iterations,)
    termination_flag : OMPTerminationFlag
        Why the algorithm terminated.
    """

    def __init__(
        self,
        surrogate: S,
        params: Array,
        n_nonzero: int,
        support: Array,
        selection_order: Array,
        residual_history: Array,
        termination_flag: Optional[OMPTerminationFlag],
    ):
        self._surrogate = surrogate
        self._params = params
        self._n_nonzero = n_nonzero
        self._support = support
        self._selection_order = selection_order
        self._residual_history = residual_history
        self._termination_flag = termination_flag

    def surrogate(self) -> S:
        """Return the fitted surrogate."""
        return self._surrogate

    def params(self) -> Array:
        """Return fitted parameters. Shape: (nterms, nqoi)"""
        return self._params

    def bkd(self) -> Backend[Array]:
        """Return backend from surrogate."""
        return self._surrogate.bkd()

    def n_nonzero(self) -> int:
        """Return number of non-zero coefficients."""
        return self._n_nonzero

    def support(self) -> Array:
        """Return indices of non-zero coefficients. Shape: (n_nonzero,)"""
        return self._support

    def selection_order(self) -> Array:
        """Return order in which basis terms were selected. Shape: (n_nonzero,)"""
        return self._selection_order

    def residual_history(self) -> Array:
        """Return residual norm at each iteration. Shape: (n_iterations,)"""
        return self._residual_history

    def termination_flag(self) -> Optional[OMPTerminationFlag]:
        """Return termination flag indicating why algorithm stopped."""
        return self._termination_flag

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
        result: Array = self._surrogate(samples)
        return result

    def __repr__(self) -> str:
        flag_name = self._termination_flag.name if self._termination_flag else "None"
        return (
            f"OMPResult(params_shape={self._params.shape}, "
            f"n_nonzero={self._n_nonzero}, "
            f"termination={flag_name})"
        )

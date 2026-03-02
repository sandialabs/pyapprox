"""
Operator-based covariance.

Provides a callback-based covariance operator for cases where the
covariance is defined implicitly (e.g., infinite-dimensional fields,
kernel operators).
"""

from typing import Callable, Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend


class OperatorBasedCovarianceOperator(Generic[Array]):
    """
    Callback-based covariance operator.

    For infinite-dimensional fields or kernel-based covariances where
    explicit matrix storage is infeasible. Operations are defined via
    user-provided callbacks.

    Parameters
    ----------
    apply_sqrt : Callable[[Array], Array]
        Callback for L @ vectors.
    apply_sqrt_inv : Callable[[Array], Array]
        Callback for L^{-1} @ vectors.
    log_determinant : float
        Pre-computed log determinant of L.
        Required explicitly since it cannot be computed from callbacks.
    nvars : int
        Dimension of the operator.
    bkd : Backend[Array]
        Computational backend.
    apply_sqrt_transpose : Callable[[Array], Array], optional
        Callback for L.T @ vectors. If None, uses apply_sqrt
        (assumes symmetric operator).
    apply_sqrt_inv_transpose : Callable[[Array], Array], optional
        Callback for L^{-T} @ vectors. If None, uses apply_sqrt_inv
        (assumes symmetric operator).

    Notes
    -----
    The log_determinant must be provided explicitly because it cannot
    be computed efficiently from matrix-vector products alone. For
    infinite-dimensional operators, this often involves regularization
    or truncation strategies.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBackend
    >>> bkd = NumpyBackend()
    >>> # Define a simple scaling operator: L = 2*I
    >>> apply_sqrt = lambda x: 2.0 * x
    >>> apply_sqrt_inv = lambda x: 0.5 * x
    >>> log_det = 3 * np.log(2.0)  # log|L| for 3x3
    >>> op = OperatorBasedCovarianceOperator(
    ...     apply_sqrt, apply_sqrt_inv, log_det, nvars=3, bkd=bkd
    ... )
    """

    def __init__(
        self,
        apply_sqrt: Callable[[Array], Array],
        apply_sqrt_inv: Callable[[Array], Array],
        log_determinant: float,
        nvars: int,
        bkd: Backend[Array],
        apply_sqrt_transpose: Optional[Callable[[Array], Array]] = None,
        apply_sqrt_inv_transpose: Optional[Callable[[Array], Array]] = None,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._log_det = log_determinant

        # Store callbacks
        self._apply_sqrt = apply_sqrt
        self._apply_sqrt_inv = apply_sqrt_inv

        # Default to same as forward (symmetric operator)
        self._apply_sqrt_transpose = (
            apply_sqrt_transpose if apply_sqrt_transpose else apply_sqrt
        )
        self._apply_sqrt_inv_transpose = (
            apply_sqrt_inv_transpose if apply_sqrt_inv_transpose else apply_sqrt_inv
        )

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the dimension of the covariance operator."""
        return self._nvars

    def _check_vectors(self, vectors: Array) -> None:
        """Validate input vectors shape."""
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if vectors.shape[0] != self._nvars:
            raise ValueError(
                f"vectors had {vectors.shape[0]} rows, expected {self._nvars}"
            )

    def apply(self, vectors: Array) -> Array:
        """
        Apply sqrt covariance: L @ vectors.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            L @ vectors. Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._apply_sqrt(vectors)

    def apply_transpose(self, vectors: Array) -> Array:
        """
        Apply transpose of sqrt covariance: L.T @ vectors.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            L.T @ vectors. Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._apply_sqrt_transpose(vectors)

    def apply_inv(self, vectors: Array) -> Array:
        """
        Apply inverse of sqrt covariance: L^{-1} @ vectors.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            L^{-1} @ vectors. Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._apply_sqrt_inv(vectors)

    def apply_inv_transpose(self, vectors: Array) -> Array:
        """
        Apply inverse transpose: L^{-T} @ vectors.

        Parameters
        ----------
        vectors : Array
            Input vectors. Shape: (nvars, nvectors)

        Returns
        -------
        Array
            L^{-T} @ vectors. Shape: (nvars, nvectors)
        """
        self._check_vectors(vectors)
        return self._apply_sqrt_inv_transpose(vectors)

    def log_determinant(self) -> float:
        """
        Return the pre-computed log determinant of L.

        Returns
        -------
        float
            log|L|
        """
        return self._log_det

    def compute_covariance_diagonal(
        self, batch_size: Optional[int] = None, active_indices: Optional[Array] = None
    ) -> Array:
        """
        Compute diagonal of covariance via probe vectors.

        This method computes the diagonal of Cov = L @ L.T by applying
        the operator to unit vectors.

        Parameters
        ----------
        batch_size : int, optional
            Number of probe vectors per batch. If None, processes all
            at once.
        active_indices : Array, optional
            Indices of diagonal terms to compute. If None, computes all.

        Returns
        -------
        Array
            Diagonal terms. Shape: (n_active,)
        """
        if active_indices is None:
            active_indices = self._bkd.arange(self._nvars)

        n_active = active_indices.shape[0]
        if batch_size is None:
            batch_size = n_active

        diagonal = self._bkd.zeros((n_active,))
        cnt = 0

        while cnt < n_active:
            nvectors = min(batch_size, n_active - cnt)

            # Create unit vectors
            vectors = self._bkd.zeros((self._nvars, nvectors))
            for jj in range(nvectors):
                idx = self._bkd.to_int(active_indices[cnt + jj])
                vectors[idx, jj] = 1.0

            # Apply Cov = L @ L.T
            tmp = self._apply_sqrt_transpose(vectors)
            result = self._apply_sqrt(tmp)

            # Extract diagonal terms
            for jj in range(nvectors):
                idx = self._bkd.to_int(active_indices[cnt + jj])
                diagonal[cnt + jj] = result[idx, jj]

            cnt += nvectors

        return diagonal

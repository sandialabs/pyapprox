"""Matrix-free kernel matrix-vector products.

A kernel covariance matrix ``K`` on ``n`` points costs ``n^2`` storage,
which caps the problem size long before the arithmetic does. Randomized
eigensolvers never need ``K`` itself, only its action on a handful of
vectors, so this applies ``K @ V`` a block of rows at a time and keeps
peak memory at ``block_size * n``.
"""

from __future__ import annotations

from typing import Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.pivoted_cholesky import KernelLike
from pyapprox.util.linalg.randomized import SymmetricMatVecOperator


class KernelMatVecOperator(SymmetricMatVecOperator[Array]):
    r"""Applies a kernel covariance matrix to vectors without forming it.

    Computes :math:`K V` where :math:`K_{ij} = k(x_i, x_j)`, evaluating
    the kernel one block of rows at a time. Peak memory is
    ``block_size * n`` rather than ``n^2``.

    Optionally applies the symmetrized form
    :math:`W^{1/2} K W^{1/2}` for quadrature weights :math:`W`, which is
    the operator whose eigenvectors are orthonormal under the quadrature
    inner product. Folding the weights in here rather than symmetrizing
    an assembled matrix is what keeps the weighted case matrix-free.

    Inherits rather than satisfying a protocol because
    :class:`SymmetricMatVecOperator` supplies real implementations --
    ``apply_transpose`` returning ``apply`` (the content of "symmetric"),
    ``nrows``/``ncols`` returning ``nvars``, and ``bkd``/``right_apply``
    from ``MatVecOperator`` above it. Only ``apply`` remains, so a
    protocol would force six members to be rewritten by hand.

    Parameters
    ----------
    kernel : KernelLike[Array]
        Kernel with ``__call__(X1, X2) -> Array``.
    X : Array
        Points defining the matrix, shape ``(nvars_in, n)``.
    bkd : Backend[Array]
        Computational backend.
    sqrt_weights : Array, optional
        Square roots of the quadrature weights, shape ``(n,)``. When
        given, the operator applies :math:`W^{1/2} K W^{1/2}` instead of
        :math:`K`.
    block_size : int
        Rows of ``K`` evaluated per pass. Trades peak memory against
        kernel-call overhead; the default suits a few thousand points
        per block.

    Examples
    --------
    >>> op = KernelMatVecOperator(kernel, X, bkd)      # doctest: +SKIP
    >>> KV = op.apply(V)                               # doctest: +SKIP
    """

    def __init__(
        self,
        kernel: KernelLike[Array],
        X: Array,
        bkd: Backend[Array],
        sqrt_weights: Optional[Array] = None,
        block_size: int = 2048,
    ) -> None:
        n = int(X.shape[1])
        super().__init__(bkd, n)
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if sqrt_weights is not None:
            if sqrt_weights.ndim != 1:
                raise ValueError(
                    "sqrt_weights must be 1D with shape (n,), got ndim="
                    f"{sqrt_weights.ndim}"
                )
            if sqrt_weights.shape[0] != n:
                raise ValueError(
                    f"sqrt_weights has {sqrt_weights.shape[0]} entries but "
                    f"X has {n} points"
                )
        self._kernel = kernel
        self._X = X
        self._sqrt_weights = sqrt_weights
        self._block_size = int(block_size)

    def kernel(self) -> KernelLike[Array]:
        """Return the kernel being applied."""
        return self._kernel

    def points(self) -> Array:
        """Return the points defining the matrix, shape (nvars_in, n)."""
        return self._X

    def sqrt_weights(self) -> Optional[Array]:
        """Return the quadrature weight square roots, or None."""
        return self._sqrt_weights

    def block_size(self) -> int:
        """Return the number of rows evaluated per pass."""
        return self._block_size

    def apply(self, vecs: Array) -> Array:
        """Apply the operator to vectors: ``K @ vecs``.

        Parameters
        ----------
        vecs : Array
            Shape ``(n, nvecs)``.

        Returns
        -------
        Array
            Shape ``(n, nvecs)``.
        """
        if vecs.ndim != 2:
            raise ValueError(
                f"vecs must be 2D with shape (n, nvecs), got ndim={vecs.ndim}"
            )
        if vecs.shape[0] != self._nvars:
            raise ValueError(
                f"vecs has {vecs.shape[0]} rows but the operator has "
                f"{self._nvars}"
            )
        bkd = self._bkd
        scaled = vecs
        if self._sqrt_weights is not None:
            scaled = vecs * self._sqrt_weights[:, None]
        blocks = []
        for start in range(0, self._nvars, self._block_size):
            stop = min(start + self._block_size, self._nvars)
            # (stop - start, n) block of K, never the whole matrix
            kblock = self._kernel(self._X[:, start:stop], self._X)
            blocks.append(kblock @ scaled)
        out = bkd.vstack(blocks)
        if self._sqrt_weights is not None:
            out = out * self._sqrt_weights[:, None]
        return out

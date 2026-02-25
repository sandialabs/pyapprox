"""Jacobi preconditioner for iterative solvers.

Implements diagonal (Jacobi) preconditioning.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class JacobiPreconditioner(Generic[Array]):
    """Jacobi (diagonal) preconditioner.

    The Jacobi preconditioner uses the diagonal of A as an approximation:
    M = diag(A), so M^{-1} @ r = r / diag(A)

    This is a simple but effective preconditioner for diagonally dominant
    systems.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._inv_diag: Array = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def setup(self, A: Array) -> None:
        """Setup preconditioner for given matrix.

        Extracts and inverts the diagonal of A.

        Parameters
        ----------
        A : Array
            System matrix. Shape: (n, n)
        """
        diag = self._bkd.diag(A)
        # Avoid division by zero for near-zero diagonal entries
        # Use sign(diag) * max(|diag|, eps) to preserve sign and avoid division by zero
        abs_diag = self._bkd.abs(diag)
        eps = 1e-15
        # For entries with |diag| < eps, use 1.0 instead
        # sign(diag) * max(|diag|, eps) = diag + sign(diag) * max(0, eps - |diag|)
        # Simpler: just add eps to abs_diag and use sign
        sign_diag = diag / (abs_diag + eps)  # Approximate sign, safe from division
        safe_diag = sign_diag * (abs_diag + eps)
        self._inv_diag = 1.0 / safe_diag

    def apply(self, r: Array) -> Array:
        """Apply Jacobi preconditioner to vector.

        Computes M^{-1} @ r = r / diag(A).

        Parameters
        ----------
        r : Array
            Input vector. Shape: (n,)

        Returns
        -------
        Array
            Preconditioned vector. Shape: (n,)
        """
        if self._inv_diag is None:
            raise RuntimeError(
                "Preconditioner not set up. Call setup(A) first."
            )
        return self._inv_diag * r


class BlockJacobiPreconditioner(Generic[Array]):
    """Block Jacobi preconditioner.

    Uses block diagonal of A as the preconditioner. Each block is
    inverted independently.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    block_size : int
        Size of each diagonal block.
    """

    def __init__(self, bkd: Backend[Array], block_size: int):
        self._bkd = bkd
        self._block_size = block_size
        self._inv_blocks: Array = None
        self._nblocks: int = 0

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def block_size(self) -> int:
        """Return the block size."""
        return self._block_size

    def setup(self, A: Array) -> None:
        """Setup block Jacobi preconditioner.

        Extracts and inverts diagonal blocks of A.

        Parameters
        ----------
        A : Array
            System matrix. Shape: (n, n)
            n must be divisible by block_size.
        """
        bkd = self._bkd
        n = A.shape[0]
        bs = self._block_size

        if n % bs != 0:
            raise ValueError(
                f"Matrix size {n} not divisible by block size {bs}"
            )

        self._nblocks = n // bs
        # Store inverted blocks: (nblocks, bs, bs)
        self._inv_blocks = bkd.zeros((self._nblocks, bs, bs))

        for i in range(self._nblocks):
            start = i * bs
            end = start + bs
            block = A[start:end, start:end]
            # Invert each block
            inv_block = bkd.solve(
                block, bkd.eye(bs)
            )
            self._inv_blocks[i, :, :] = inv_block

    def apply(self, r: Array) -> Array:
        """Apply block Jacobi preconditioner.

        Parameters
        ----------
        r : Array
            Input vector. Shape: (n,)

        Returns
        -------
        Array
            Preconditioned vector. Shape: (n,)
        """
        if self._inv_blocks is None:
            raise RuntimeError(
                "Preconditioner not set up. Call setup(A) first."
            )

        bkd = self._bkd
        bs = self._block_size
        n = self._nblocks * bs
        result = bkd.zeros((n,))

        for i in range(self._nblocks):
            start = i * bs
            end = start + bs
            r_block = r[start:end]
            result[start:end] = self._inv_blocks[i, :, :] @ r_block

        return result


def jacobi_preconditioner(
    A: Array, bkd: Backend[Array]
) -> JacobiPreconditioner[Array]:
    """Create a Jacobi preconditioner for the given matrix.

    Parameters
    ----------
    A : Array
        System matrix. Shape: (n, n)
    bkd : Backend
        Computational backend.

    Returns
    -------
    JacobiPreconditioner
        Preconditioner ready to use.
    """
    precond = JacobiPreconditioner(bkd)
    precond.setup(A)
    return precond


def block_jacobi_preconditioner(
    A: Array, bkd: Backend[Array], block_size: int
) -> BlockJacobiPreconditioner[Array]:
    """Create a block Jacobi preconditioner for the given matrix.

    Parameters
    ----------
    A : Array
        System matrix. Shape: (n, n)
    bkd : Backend
        Computational backend.
    block_size : int
        Size of diagonal blocks.

    Returns
    -------
    BlockJacobiPreconditioner
        Preconditioner ready to use.
    """
    precond = BlockJacobiPreconditioner(bkd, block_size)
    precond.setup(A)
    return precond

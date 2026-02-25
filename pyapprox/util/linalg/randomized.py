"""
Randomized linear algebra algorithms.

This module provides randomized algorithms for matrix decompositions
that work efficiently with the backend abstraction.

Classes
-------
MatVecOperator
    Base protocol for matrix-vector product operators.
SymmetricMatVecOperator
    Matrix-vector operator for symmetric matrices.
DenseMatVecOperator
    Matrix-vector operator for dense matrices.
DenseSymmetricMatVecOperator
    Matrix-vector operator for dense symmetric matrices.
FunctionMatVecOperator
    Matrix-vector operator from a callable function.
FunctionSymmetricMatVecOperator
    Symmetric matrix-vector operator from a callable function.
RandomizedSVD
    Abstract base class for randomized SVD algorithms.
SinglePassRandomizedSVD
    Single-pass randomized SVD algorithm.
DoublePassRandomizedSVD
    Double-pass randomized SVD for symmetric matrices.

Functions
---------
randomized_symmetric_eigendecomposition
    Convenience function for low-rank eigendecomposition.
adjust_sign_svd
    Ensure uniqueness of SVD by sign adjustment.
"""

from abc import ABC, abstractmethod
from typing import Callable, Generic, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


class MatVecOperator(Generic[Array], ABC):
    """
    Base class for matrix-vector product operators.

    This provides an interface for matrix-free operations where
    the matrix is never explicitly formed. Useful for large-scale
    problems where storing the full matrix is impractical.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    @abstractmethod
    def apply(self, vecs: Array) -> Array:
        """
        Apply the operator to vectors (left multiplication): A @ vecs.

        Parameters
        ----------
        vecs : Array
            Input vectors. Shape: (ncols, nvecs)

        Returns
        -------
        Array
            Result of applying operator. Shape: (nrows, nvecs)
        """
        raise NotImplementedError

    @abstractmethod
    def apply_transpose(self, vecs: Array) -> Array:
        """
        Apply the transpose of the operator: A.T @ vecs.

        Parameters
        ----------
        vecs : Array
            Input vectors. Shape: (nrows, nvecs)

        Returns
        -------
        Array
            Result of applying transpose. Shape: (ncols, nvecs)
        """
        raise NotImplementedError

    def right_apply(self, vecs: Array) -> Array:
        """
        Apply the operator from the right: vecs @ A.

        Parameters
        ----------
        vecs : Array
            Input vectors. Shape: (nvecs, nrows)

        Returns
        -------
        Array
            Result of right application. Shape: (nvecs, ncols)
        """
        raise NotImplementedError("right_apply is not implemented")

    def right_apply_implemented(self) -> bool:
        """Return True if right_apply is implemented."""
        return False

    @abstractmethod
    def nrows(self) -> int:
        """Return the number of rows."""
        raise NotImplementedError

    @abstractmethod
    def ncols(self) -> int:
        """Return the number of columns."""
        raise NotImplementedError


class SymmetricMatVecOperator(MatVecOperator[Array]):
    """
    Base class for symmetric matrix-vector product operators.

    For symmetric matrices, apply_transpose is the same as apply,
    and nrows equals ncols.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Dimension of the square symmetric matrix.
    """

    def __init__(self, bkd: Backend[Array], nvars: int):
        super().__init__(bkd)
        self._nvars = nvars

    def nvars(self) -> int:
        """Return the number of variables (matrix dimension)."""
        return self._nvars

    def apply_transpose(self, vecs: Array) -> Array:
        """
        Apply the transpose of the operator.

        For symmetric operators, this is the same as apply().
        """
        return self.apply(vecs)

    def nrows(self) -> int:
        """Return the number of rows."""
        return self._nvars

    def ncols(self) -> int:
        """Return the number of columns."""
        return self._nvars


class DenseMatVecOperator(MatVecOperator[Array]):
    """
    Matrix-vector operator for dense matrices.

    Parameters
    ----------
    mat : Array
        Dense matrix. Shape: (nrows, ncols)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, mat: Array, bkd: Backend[Array]):
        super().__init__(bkd)
        self._mat = mat

    def apply(self, vecs: Array) -> Array:
        """Apply the matrix: A @ vecs."""
        return self._mat @ vecs

    def apply_transpose(self, vecs: Array) -> Array:
        """Apply the transpose: A.T @ vecs."""
        return self._mat.T @ vecs

    def right_apply(self, vecs: Array) -> Array:
        """Apply from right: vecs @ A."""
        return vecs @ self._mat

    def right_apply_implemented(self) -> bool:
        """Return True since right_apply is implemented."""
        return True

    def nrows(self) -> int:
        """Return the number of rows."""
        return self._mat.shape[0]

    def ncols(self) -> int:
        """Return the number of columns."""
        return self._mat.shape[1]


class DenseSymmetricMatVecOperator(SymmetricMatVecOperator[Array]):
    """
    Matrix-vector operator for dense symmetric matrices.

    Parameters
    ----------
    mat : Array
        Dense symmetric matrix. Shape: (nvars, nvars)
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If the matrix is not square or not symmetric.
    """

    def __init__(self, mat: Array, bkd: Backend[Array]):
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("Matrix must be square")
        mat_np = bkd.to_numpy(mat)
        if not np.allclose(mat_np, mat_np.T, atol=1e-12):
            raise ValueError("Matrix must be symmetric")
        super().__init__(bkd, mat.shape[0])
        self._mat = mat

    def apply(self, vecs: Array) -> Array:
        """Apply the matrix: A @ vecs."""
        return self._mat @ vecs

    def right_apply(self, vecs: Array) -> Array:
        """Apply from right: vecs @ A."""
        return vecs @ self._mat

    def right_apply_implemented(self) -> bool:
        """Return True since right_apply is implemented."""
        return True


class FunctionMatVecOperator(MatVecOperator[Array]):
    """
    Matrix-vector operator from callable functions.

    Parameters
    ----------
    apply_fn : Callable[[Array], Array]
        Function that applies the matrix: apply_fn(vecs) -> A @ vecs
    apply_transpose_fn : Callable[[Array], Array]
        Function that applies the transpose: apply_transpose_fn(vecs) -> A.T @ vecs
    nrows : int
        Number of rows in the matrix.
    ncols : int
        Number of columns in the matrix.
    bkd : Backend[Array]
        Computational backend.
    right_apply_fn : Callable[[Array], Array], optional
        Function for right application: right_apply_fn(vecs) -> vecs @ A
    """

    def __init__(
        self,
        apply_fn: Callable[[Array], Array],
        apply_transpose_fn: Callable[[Array], Array],
        nrows: int,
        ncols: int,
        bkd: Backend[Array],
        right_apply_fn: Callable[[Array], Array] = None,
    ):
        super().__init__(bkd)
        self._apply_fn = apply_fn
        self._apply_transpose_fn = apply_transpose_fn
        self._nrows = nrows
        self._ncols = ncols
        self._right_apply_fn = right_apply_fn

    def apply(self, vecs: Array) -> Array:
        """Apply the operator to vectors."""
        return self._apply_fn(vecs)

    def apply_transpose(self, vecs: Array) -> Array:
        """Apply the transpose of the operator."""
        return self._apply_transpose_fn(vecs)

    def right_apply(self, vecs: Array) -> Array:
        """Apply the operator from the right."""
        if self._right_apply_fn is None:
            raise NotImplementedError("right_apply is not implemented")
        return self._right_apply_fn(vecs)

    def right_apply_implemented(self) -> bool:
        """Return True if right_apply is implemented."""
        return self._right_apply_fn is not None

    def nrows(self) -> int:
        """Return the number of rows."""
        return self._nrows

    def ncols(self) -> int:
        """Return the number of columns."""
        return self._ncols


class FunctionSymmetricMatVecOperator(SymmetricMatVecOperator[Array]):
    """
    Symmetric matrix-vector operator from a callable function.

    Parameters
    ----------
    apply_fn : Callable[[Array], Array]
        Function that applies the matrix to vectors.
    nvars : int
        Dimension of the square matrix.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        apply_fn: Callable[[Array], Array],
        nvars: int,
        bkd: Backend[Array],
    ):
        super().__init__(bkd, nvars)
        self._apply_fn = apply_fn

    def apply(self, vecs: Array) -> Array:
        """Apply the operator to vectors."""
        return self._apply_fn(vecs)


def adjust_sign_svd(
    U: Array,
    Vh: Array,
    adjust_based_upon_U: bool = True,
    bkd: Backend[Array] = None,
) -> Tuple[Array, Array]:
    """
    Ensure uniqueness of SVD by ensuring consistent signs.

    The SVD is unique up to sign changes in the singular vectors.
    This function ensures the first entry of each left singular
    vector (or right, depending on adjust_based_upon_U) is positive.

    Parameters
    ----------
    U : Array
        Left singular vectors. Shape: (M, K)
    Vh : Array
        Right singular vectors (transposed). Shape: (K, N)
    adjust_based_upon_U : bool, default=True
        If True, make the first entry of each column of U positive.
        If False, make the first entry of each row of Vh positive.
    bkd : Backend[Array], optional
        Computational backend. If None, operations use numpy.

    Returns
    -------
    U : Array
        Sign-adjusted left singular vectors.
    Vh : Array
        Sign-adjusted right singular vectors.
    """
    if U.shape[1] != Vh.shape[0]:
        msg = "U.shape[1] must equal Vh.shape[0]. If using np.linalg.svd set "
        msg += "full_matrices=False"
        raise ValueError(msg)

    if bkd is not None:
        U_np = bkd.to_numpy(U)
        Vh_np = bkd.to_numpy(Vh)
    else:
        U_np = np.asarray(U)
        Vh_np = np.asarray(Vh)

    if adjust_based_upon_U:
        s = np.sign(U_np[0, :])
    else:
        s = np.sign(Vh_np[:, 0])

    # Handle zero entries
    s[s == 0] = 1.0

    U_np = U_np * s
    Vh_np = Vh_np * s[:, None]

    if bkd is not None:
        return bkd.asarray(U_np), bkd.asarray(Vh_np)
    return U_np, Vh_np


class RandomizedSVD(Generic[Array], ABC):
    """
    Abstract base class for randomized SVD algorithms.

    Parameters
    ----------
    matvec : MatVecOperator[Array]
        Matrix-vector product operator.
    noversampling : int, default=10
        Number of additional random samples beyond the rank.
    npower_iters : int, default=1
        Number of power iterations for improved accuracy.

    References
    ----------
    Halko, Martinsson, and Tropp (2011). "Finding Structure with
    Randomness: Probabilistic Algorithms for Constructing Approximate
    Matrix Decompositions."
    """

    def __init__(
        self,
        matvec: MatVecOperator[Array],
        noversampling: int = 10,
        npower_iters: int = 1,
    ):
        self._check_matvec(matvec)
        self._bkd = matvec.bkd()
        self._matvec = matvec
        self._noversampling = noversampling
        self._npower_iters = npower_iters

    def _check_matvec(self, matvec: MatVecOperator[Array]):
        """Validate the matrix-vector operator."""
        if not isinstance(matvec, MatVecOperator):
            raise ValueError("matvec must be an instance of MatVecOperator")

    @abstractmethod
    def compute(self, rank: int) -> Tuple[Array, Array, Array]:
        """
        Compute the randomized SVD.

        Parameters
        ----------
        rank : int
            Target rank of the approximation.

        Returns
        -------
        U : Array
            Left singular vectors. Shape: (nrows, rank)
        S : Array
            Singular values. Shape: (rank,)
        Vh : Array
            Right singular vectors (transposed). Shape: (rank, ncols)
        """
        raise NotImplementedError

    def adjust_sign(self, U: Array, Vh: Array) -> Tuple[Array, Array]:
        """Adjust signs for uniqueness."""
        return adjust_sign_svd(U, Vh, bkd=self._bkd)

    def _sample_column_space(self, rank: int) -> Array:
        """
        Sample the column space of the matrix.

        Parameters
        ----------
        rank : int
            Target rank.

        Returns
        -------
        Array
            Column space samples. Shape: (nrows, nsamples)
        """
        nsamples = rank + self._noversampling
        # Use transpose so omega samples are nested if nsamples are increased
        omega = self._bkd.asarray(
            np.random.normal(0, 1, (nsamples, self._matvec.ncols())).astype(np.float64)
        ).T

        # Sample column space
        Y = self._matvec.apply(omega)

        # Power iterations for better approximation
        for _ in range(self._npower_iters):
            G = self._matvec.apply_transpose(Y)
            Y = self._matvec.apply(G)

        return Y


class SinglePassRandomizedSVD(RandomizedSVD[Array]):
    """
    Single-pass randomized SVD algorithm.

    This algorithm requires only one pass over the matrix data,
    making it efficient for streaming or out-of-core applications.
    Requires that the operator supports right_apply.

    Parameters
    ----------
    matvec : MatVecOperator[Array]
        Matrix-vector product operator with right_apply implemented.
    noversampling : int, default=10
        Number of additional random samples.
    npower_iters : int, default=1
        Number of power iterations.
    """

    def compute(self, rank: int) -> Tuple[Array, Array, Array]:
        """
        Compute the single-pass randomized SVD.

        Returns
        -------
        U : Array
            Left singular vectors. Shape: (nrows, rank)
        S : Array
            Singular values. Shape: (rank,)
        Vh : Array
            Right singular vectors (transposed). Shape: (rank, ncols)
        """
        if not self._matvec.right_apply_implemented():
            raise ValueError("matvec must implement right_apply")

        cspace_samples = self._sample_column_space(rank)

        # Orthogonalize column space samples
        Q_np, _ = np.linalg.qr(self._bkd.to_numpy(cspace_samples), mode="reduced")
        Q = self._bkd.asarray(Q_np)

        # Compute B = Q.T @ A using right_apply
        B = self._matvec.right_apply(Q.T)

        # SVD of the small matrix B
        B_np = self._bkd.to_numpy(B)
        U_np, S_np, Vh_np = np.linalg.svd(B_np, full_matrices=False)

        U = self._bkd.asarray(U_np.astype(np.float64))
        S = self._bkd.asarray(S_np.astype(np.float64))
        Vh = self._bkd.asarray(Vh_np.astype(np.float64))

        # Project back: U = Q @ U
        U = Q @ U

        # Adjust signs and truncate to rank
        U, Vh = self.adjust_sign(U[:, :rank], Vh[:rank])
        return U, S[:rank], Vh


class DoublePassRandomizedSVD(RandomizedSVD[Array]):
    """
    Double-pass randomized SVD for symmetric matrices.

    This algorithm uses two passes over the matrix, which provides
    better accuracy for symmetric matrices.

    Parameters
    ----------
    matvec : SymmetricMatVecOperator[Array]
        Symmetric matrix-vector product operator.
    noversampling : int, default=10
        Number of additional random samples.
    npower_iters : int, default=1
        Number of power iterations.
    """

    def _check_matvec(self, matvec: MatVecOperator[Array]):
        """Validate that the operator is symmetric."""
        if not isinstance(matvec, SymmetricMatVecOperator):
            raise ValueError("matvec must be an instance of SymmetricMatVecOperator")

    def compute(self, rank: int) -> Tuple[Array, Array, Array]:
        """
        Compute the double-pass randomized SVD.

        Returns
        -------
        U : Array
            Left singular vectors. Shape: (nvars, rank)
        S : Array
            Singular values. Shape: (rank,)
        Vh : Array
            Right singular vectors (transposed). Shape: (rank, nvars)
        """
        # First pass: sample column space
        cspace_samples = self._sample_column_space(rank)
        Q1_np, _ = np.linalg.qr(self._bkd.to_numpy(cspace_samples), mode="reduced")
        Q1 = self._bkd.asarray(Q1_np)

        # Second pass: sample row space
        rspace_samples = self._matvec.apply_transpose(Q1)
        Q2_np, _ = np.linalg.qr(self._bkd.to_numpy(rspace_samples), mode="reduced")
        Q2 = self._bkd.asarray(Q2_np)

        # SVD of compressed matrix
        B = Q2.T @ rspace_samples
        B_np = self._bkd.to_numpy(B)
        U_np, S_np, Vh_np = np.linalg.svd(B_np, full_matrices=False)

        U = self._bkd.asarray(U_np.astype(np.float64))
        S = self._bkd.asarray(S_np.astype(np.float64))
        Vh = self._bkd.asarray(Vh_np.astype(np.float64))

        # Project row space: Vh = (Q1 @ Vh.T).T
        Vh = (Q1 @ Vh.T).T

        # Project column space: U = Q2 @ U
        U = Q2 @ U

        # Adjust signs and truncate to rank
        U, Vh = self.adjust_sign(U[:, :rank], Vh[:rank])
        return U, S[:rank], Vh


def randomized_symmetric_eigendecomposition(
    apply_operator: Callable[[Array], Array],
    nvars: int,
    rank: int,
    bkd: Backend[Array],
    noversampling: int = 10,
    npower_iters: int = 1,
) -> Tuple[Array, Array]:
    """
    Compute a low-rank eigenvalue decomposition using randomized methods.

    For a symmetric matrix A, compute an approximate eigendecomposition:

        A approx U @ diag(S) @ U.T

    using randomized sampling of the column space.

    Parameters
    ----------
    apply_operator : Callable[[Array], Array]
        Function that applies the symmetric matrix to vectors.
        Signature: apply_operator(vecs) -> A @ vecs
    nvars : int
        Dimension of the symmetric matrix.
    rank : int
        Rank of the approximation (number of eigenvalues to compute).
    bkd : Backend[Array]
        Computational backend.
    noversampling : int, default=10
        Number of additional random samples for improved accuracy.
    npower_iters : int, default=1
        Number of power iterations for improved accuracy when
        eigenvalues decay slowly.

    Returns
    -------
    eigenvalues : Array
        Top eigenvalues in descending order. Shape: (rank,)
    eigenvectors : Array
        Corresponding eigenvectors. Shape: (nvars, rank)

    Notes
    -----
    This implements the double-pass randomized SVD algorithm for
    symmetric matrices, which is equivalent to computing an
    eigenvalue decomposition.

    References
    ----------
    Halko, Martinsson, and Tropp (2011). "Finding Structure with
    Randomness: Probabilistic Algorithms for Constructing Approximate
    Matrix Decompositions."

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create a symmetric positive definite matrix
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> apply_A = lambda v: A @ v
    >>> eigenvalues, eigenvectors = randomized_symmetric_eigendecomposition(
    ...     apply_A, nvars=2, rank=2, bkd=bkd
    ... )
    """
    nsamples = rank + noversampling

    # Random matrix for sampling column space
    omega = bkd.asarray(np.random.normal(0, 1, (nvars, nsamples)).astype(np.float64))

    # Sample column space: Y = A @ omega
    Y = apply_operator(omega)

    # Power iterations for better approximation
    for _ in range(npower_iters):
        # Since A is symmetric, apply_transpose = apply
        G = apply_operator(Y)
        Y = apply_operator(G)

    # QR factorization to get orthonormal basis
    # Use numpy for QR since it's not differentiable anyway
    Y_np = bkd.to_numpy(Y)
    Q_np, _ = np.linalg.qr(Y_np, mode="reduced")
    Q = bkd.asarray(Q_np)

    # Form small matrix B = Q^T @ A @ Q
    AQ = apply_operator(Q)
    B = Q.T @ AQ

    # Eigenvalue decomposition of B
    # Use numpy for eigendecomposition
    B_np = bkd.to_numpy(B)
    eigenvalues_np, eigenvectors_np = np.linalg.eigh(B_np)

    # Sort in descending order
    idx = np.argsort(eigenvalues_np)[::-1]
    eigenvalues_np = eigenvalues_np[idx]
    eigenvectors_np = eigenvectors_np[:, idx]

    # Keep top rank eigenpairs
    eigenvalues = bkd.asarray(eigenvalues_np[:rank].astype(np.float64))
    Vr = bkd.asarray(eigenvectors_np[:, :rank].astype(np.float64))

    # Project back to full space: U_r = Q @ V_r
    eigenvectors = Q @ Vr

    return eigenvalues, eigenvectors


def get_low_rank_matrix(
    nrows: int,
    ncols: int,
    rank: int,
    bkd: Backend[Array],
) -> Array:
    """
    Construct a random matrix with a given rank.

    Useful for testing randomized SVD algorithms.

    Parameters
    ----------
    nrows : int
        Number of rows in the matrix.
    ncols : int
        Number of columns in the matrix.
    rank : int
        Rank of the matrix.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Low-rank matrix. Shape: (nrows, ncols)
    """
    if rank > min(nrows, ncols):
        raise ValueError(
            f"rank ({rank}) cannot exceed min(nrows, ncols) = {min(nrows, ncols)}"
        )

    # Generate a random matrix
    N = max(nrows, ncols)
    A = np.random.normal(0, 1, (N, N))

    # Make it symmetric positive definite
    A = A.T @ A

    # Compute eigendecomposition and zero out small eigenvalues
    eigvals, eigvecs = np.linalg.eigh(A)

    # Set smallest eigenvalues to zero (eigenvalues are in ascending order)
    eigvals[: (eigvals.shape[0] - rank)] = 0.0

    # Reconstruct the matrix
    A = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Resize to requested shape
    return bkd.asarray(A[:nrows, :ncols].astype(np.float64))

r"""Symmetric positive-semidefinite weights, independent of storage.

A weight :math:`W` defines the seminorm :math:`x^\top W x`. It is
symmetric and positive *semi*definite, unlike an inner product: a
mass matrix restricted to a subdomain is zero outside it and is still a
valid weight. Consumers see only :class:`SymmetricWeightProtocol`, the
action :math:`x \mapsto W x`; how :math:`W` is stored (a dense array, a
scipy sparse matrix, a torch sparse tensor, a diagonal, or no matrix at
all) is owned by the implementation that is injected. A new storage
format is a new class satisfying the protocol, with no change to any
consumer.

Every inner product in :mod:`pyapprox.util.linalg.inner_product`
satisfies this protocol, since a positive definite form is in particular
semidefinite.

Positive semidefiniteness is part of the contract but is not checked:
verifying it costs a factorization. Symmetry is checked at construction,
where raw data enters, because consumers rely on it (the gradient of
:math:`x^\top W x` is :math:`2 W x` only when :math:`W = W^\top`).
"""

from typing import Any, Generic, Protocol, runtime_checkable

from scipy.sparse import issparse, spmatrix

from pyapprox.util.backends.protocols import Array, Backend

_SYMMETRY_RTOL = 1e-12


def _check_square(shape: tuple[int, ...], name: str) -> None:
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"{name} must have shape (nstates, nstates), got {shape}")


def _check_symmetric(asymmetry: float, scale: float, name: str) -> None:
    if asymmetry > _SYMMETRY_RTOL * max(scale, 1.0):
        raise ValueError(
            f"{name} must be symmetric: max |W - W^T| = {asymmetry:.3e} "
            f"against max |W| = {scale:.3e}"
        )


@runtime_checkable
class SymmetricWeightProtocol(Protocol, Generic[Array]):
    r"""A symmetric positive-semidefinite weight :math:`W`.

    Arrays follow the samples-are-columns convention: a single vector is
    ``(nstates, 1)`` and a batch is ``(nstates, ncols)``.
    """

    def nstates(self) -> int:
        """Dimension of the space the weight acts on."""
        ...

    def apply(self, x: Array) -> Array:
        """Return ``W x``, shape ``(nstates, ncols)``."""
        ...


class DenseMatrixWeight(Generic[Array]):
    r"""A weight stored as a dense backend array.

    ``apply`` is a backend product, so a torch graph through ``x`` (and
    through ``W``, if it carries one) is preserved.

    Parameters
    ----------
    matrix : Array
        Symmetric positive-semidefinite matrix. Shape
        ``(nstates, nstates)``.
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If ``matrix`` is not square or not symmetric.
    """

    def __init__(self, matrix: Array, bkd: Backend[Array]) -> None:
        _check_square(tuple(matrix.shape), "matrix")
        _check_symmetric(
            bkd.to_float(bkd.max(bkd.abs(matrix - matrix.T))),
            bkd.to_float(bkd.max(bkd.abs(matrix))),
            "matrix",
        )
        self._matrix = matrix
        self._bkd = bkd

    def nstates(self) -> int:
        return int(self._matrix.shape[0])

    def apply(self, x: Array) -> Array:
        return self._bkd.dot(self._matrix, x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self.nstates()})"


class ScipySparseWeight(Generic[Array]):
    r"""A weight stored as a scipy sparse matrix.

    The native storage of an assembled finite element matrix, so it is
    never densified. ``apply`` is a scipy product: it accepts arrays of
    any backend but round-trips through numpy, so it does not preserve a
    torch graph through ``x``. Inject :class:`TorchSparseWeight` when one
    is needed.

    Parameters
    ----------
    matrix : scipy sparse matrix
        Symmetric positive-semidefinite matrix. Shape
        ``(nstates, nstates)``. Stored as CSR.
    bkd : Backend[Array]
        Computational backend of the arrays ``apply`` returns.

    Raises
    ------
    TypeError
        If ``matrix`` is not a scipy sparse matrix.
    ValueError
        If ``matrix`` is not square or not symmetric.
    """

    def __init__(self, matrix: spmatrix, bkd: Backend[Array]) -> None:
        if not issparse(matrix):
            raise TypeError(
                "ScipySparseWeight requires a scipy sparse matrix, got "
                f"{type(matrix).__name__}; use DenseMatrixWeight for a "
                "dense array"
            )
        _check_square(tuple(matrix.shape), "matrix")
        csr = matrix.tocsr()
        # Sparse W - W^T stays sparse: an O(nnz) check, never densified.
        _check_symmetric(float(abs(csr - csr.T).max()), float(abs(csr).max()), "matrix")
        self._matrix = csr
        self._bkd = bkd

    def nstates(self) -> int:
        return int(self._matrix.shape[0])

    def apply(self, x: Array) -> Array:
        return self._bkd.asarray(self._matrix @ self._bkd.to_numpy(x))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self.nstates()})"


class TorchSparseWeight(Generic[Array]):
    r"""A weight stored as a torch sparse CSR tensor.

    ``apply`` is ``torch.sparse.mm``, so a torch graph through ``x`` is
    preserved; :math:`W` is constant data. Use with ``TorchBkd`` arrays.

    Parameters
    ----------
    matrix : scipy sparse matrix
        Symmetric positive-semidefinite matrix, typically as assembled.
        Shape ``(nstates, nstates)``. Converted once, at construction.
    bkd : Backend[Array]
        A torch backend.

    Raises
    ------
    TypeError
        If ``matrix`` is not a scipy sparse matrix.
    ValueError
        If ``matrix`` is not square or not symmetric.
    """

    def __init__(self, matrix: spmatrix, bkd: Backend[Array]) -> None:
        import torch

        if not issparse(matrix):
            raise TypeError(
                "TorchSparseWeight requires a scipy sparse matrix, got "
                f"{type(matrix).__name__}"
            )
        _check_square(tuple(matrix.shape), "matrix")
        csr = matrix.tocsr()
        _check_symmetric(float(abs(csr - csr.T).max()), float(abs(csr).max()), "matrix")
        self._nstates = int(csr.shape[0])
        self._matrix: Any = torch.sparse_csr_tensor(
            torch.as_tensor(csr.indptr, dtype=torch.int64),
            torch.as_tensor(csr.indices, dtype=torch.int64),
            torch.as_tensor(csr.data, dtype=bkd.default_dtype()),
            size=csr.shape,
        )
        self._bkd = bkd

    def nstates(self) -> int:
        return self._nstates

    def apply(self, x: Array) -> Array:
        import torch

        # asarray rather than array: the latter detaches.
        return self._bkd.asarray(torch.sparse.mm(self._matrix, x))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self.nstates()})"

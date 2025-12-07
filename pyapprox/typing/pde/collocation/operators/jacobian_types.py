"""Sparse Jacobian representations for spectral collocation methods.

Provides efficient storage for Jacobians with special structure:
- DenseJacobian: Full dense matrix
- DiagJacobian: Block-diagonal structure (diagonal per input function)
- ZeroJacobian: Zero matrix (no dependency)
"""

from abc import ABC, abstractmethod
from typing import Generic, Union, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


class SparseJacobian(ABC, Generic[Array]):
    """Abstract base class for sparse Jacobian representations.

    Jacobians track dependencies between output and input fields.
    Shape convention: (noutputs, ninputs) where:
    - noutputs = nmesh_pts (for single output function)
    - ninputs = nmesh_pts * ninput_funs (for multiple input functions)

    Attributes
    ----------
    __array_priority__ : int
        Ensures SparseJacobian operations take precedence over numpy arrays.
    """

    __array_priority__ = 1

    def __init__(
        self, bkd: Backend[Array], shape: Tuple[int, int], sparse_jac: Array
    ):
        """Initialize sparse Jacobian.

        Parameters
        ----------
        bkd : Backend
            Computational backend.
        shape : Tuple[int, int]
            Full Jacobian shape (noutputs, ninputs).
        sparse_jac : Array
            Compact storage for Jacobian data.
        """
        self._bkd = bkd
        self._shape = shape
        self.set_sparse_jacobian(sparse_jac)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    @property
    def shape(self) -> Tuple[int, int]:
        """Return full Jacobian shape."""
        return self._shape

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self._shape})"

    @abstractmethod
    def copy(self) -> "SparseJacobian[Array]":
        """Return a copy of this Jacobian."""
        ...

    @abstractmethod
    def get_jacobian(self) -> Array:
        """Return full dense Jacobian matrix.

        Returns
        -------
        Array
            Dense matrix of shape (noutputs, ninputs).
        """
        ...

    @abstractmethod
    def set_sparse_jacobian(self, jac: Array) -> None:
        """Set the compact Jacobian storage.

        Parameters
        ----------
        jac : Array
            Compact representation (format depends on subclass).
        """
        ...

    @abstractmethod
    def __neg__(self) -> "SparseJacobian[Array]":
        """Return negated Jacobian."""
        ...

    @abstractmethod
    def __add__(
        self, other: "SparseJacobian[Array]"
    ) -> "SparseJacobian[Array]":
        """Add two Jacobians."""
        ...

    @abstractmethod
    def __sub__(
        self, other: "SparseJacobian[Array]"
    ) -> "SparseJacobian[Array]":
        """Subtract two Jacobians."""
        ...

    @abstractmethod
    def __mul__(
        self, other: Union[Array, float]
    ) -> "SparseJacobian[Array]":
        """Multiply Jacobian by scalar or array (row-wise scaling)."""
        ...

    def __rmul__(
        self, other: Union[Array, float]
    ) -> "SparseJacobian[Array]":
        """Right multiplication (same as left for scalars)."""
        return self.__mul__(other)

    @abstractmethod
    def __truediv__(
        self, other: Union[Array, float]
    ) -> "SparseJacobian[Array]":
        """Divide Jacobian by scalar or array."""
        ...

    @abstractmethod
    def rdot(self, other: Array) -> "SparseJacobian[Array]":
        """Compute A @ J in sparse format.

        Parameters
        ----------
        other : Array
            Matrix to multiply from left. Shape: (m, noutputs)

        Returns
        -------
        SparseJacobian
            Result with shape (m, ninputs).
        """
        ...


class DenseJacobian(SparseJacobian[Array]):
    """Dense Jacobian representation.

    Stores the full (noutputs, ninputs) matrix explicitly.
    """

    def copy(self) -> "DenseJacobian[Array]":
        return DenseJacobian(
            self._bkd, self._shape, self._bkd.copy(self._sparse_jac)
        )

    def set_sparse_jacobian(self, jac: Array) -> None:
        if jac.shape != self._shape:
            raise ValueError(
                f"Expected shape {self._shape}, got {jac.shape}"
            )
        self._sparse_jac = jac

    def get_jacobian(self) -> Array:
        return self._sparse_jac

    def __neg__(self) -> "DenseJacobian[Array]":
        return DenseJacobian(self._bkd, self._shape, -self._sparse_jac)

    def __mul__(
        self, other: Union[Array, float]
    ) -> "DenseJacobian[Array]":
        if isinstance(other, (int, float)) or (
            hasattr(other, "ndim") and other.ndim == 0
        ):
            return DenseJacobian(
                self._bkd, self._shape, self._sparse_jac * other
            )
        # Row-wise scaling: other shape (noutputs,)
        return DenseJacobian(
            self._bkd, self._shape, self._sparse_jac * other[:, None]
        )

    def __truediv__(
        self, other: Union[Array, float]
    ) -> "DenseJacobian[Array]":
        if isinstance(other, (int, float)) or (
            hasattr(other, "ndim") and other.ndim == 0
        ):
            return DenseJacobian(
                self._bkd, self._shape, self._sparse_jac / other
            )
        return DenseJacobian(
            self._bkd, self._shape, self._sparse_jac / other[:, None]
        )

    def rdot(self, other: Array) -> "DenseJacobian[Array]":
        if other.shape[1] != self._shape[0]:
            raise ValueError(
                f"Shape mismatch: {other.shape} @ {self._shape}"
            )
        new_shape = (other.shape[0], self._shape[1])
        return DenseJacobian(
            self._bkd, new_shape, self._bkd.dot(other, self._sparse_jac)
        )

    def __add__(
        self, other: "SparseJacobian[Array]"
    ) -> "SparseJacobian[Array]":
        if isinstance(other, ZeroJacobian):
            return self.copy()
        if isinstance(other, DiagJacobian):
            # Add diagonal blocks to dense using vectorized indexing
            dense_jac = self._bkd.copy(self._sparse_jac)
            nmesh = self._shape[0]
            ninput_funs = self._shape[1] // nmesh
            # Compute all column indices using floor division:
            # col_idx = [0, 1, ..., nmesh-1, nmesh, nmesh+1, ..., 2*nmesh-1, ...]
            col_idx = self._bkd.arange(self._shape[1])
            # row_idx = col_idx % nmesh gives [0,1,...,nmesh-1,0,1,...,nmesh-1,...]
            row_idx = col_idx % nmesh
            # Flatten diagonal values in correct order (block by block)
            diag_flat = self._bkd.flatten(
                self._bkd.transpose(other._sparse_jac)
            )
            # Vectorized update
            dense_jac[row_idx, col_idx] = dense_jac[row_idx, col_idx] + diag_flat
            return DenseJacobian(self._bkd, self._shape, dense_jac)
        # Dense + Dense
        return DenseJacobian(
            self._bkd, self._shape,
            self._sparse_jac + other._sparse_jac
        )

    def __sub__(
        self, other: "SparseJacobian[Array]"
    ) -> "SparseJacobian[Array]":
        if isinstance(other, ZeroJacobian):
            return self.copy()
        if isinstance(other, DiagJacobian):
            # Subtract diagonal blocks from dense using vectorized indexing
            dense_jac = self._bkd.copy(self._sparse_jac)
            nmesh = self._shape[0]
            # Compute all column indices
            col_idx = self._bkd.arange(self._shape[1])
            # row_idx = col_idx % nmesh gives [0,1,...,nmesh-1,0,1,...,nmesh-1,...]
            row_idx = col_idx % nmesh
            # Flatten diagonal values in correct order (block by block)
            diag_flat = self._bkd.flatten(
                self._bkd.transpose(other._sparse_jac)
            )
            # Vectorized update
            dense_jac[row_idx, col_idx] = dense_jac[row_idx, col_idx] - diag_flat
            return DenseJacobian(self._bkd, self._shape, dense_jac)
        return DenseJacobian(
            self._bkd, self._shape,
            self._sparse_jac - other._sparse_jac
        )


class DiagJacobian(SparseJacobian[Array]):
    """Block-diagonal Jacobian representation.

    For functions where output[i] depends only on input[i] for each
    input function. Stores compact representation of shape
    (nmesh_pts, ninput_funs) instead of full (nmesh_pts, nmesh_pts * ninput_funs).
    """

    def copy(self) -> "DiagJacobian[Array]":
        return DiagJacobian(
            self._bkd, self._shape, self._bkd.copy(self._sparse_jac)
        )

    def set_sparse_jacobian(self, jac: Array) -> None:
        nmesh_pts = self._shape[0]
        ninput_funs = self._shape[1] // nmesh_pts
        expected_shape = (nmesh_pts, ninput_funs)
        if jac.shape != expected_shape:
            raise ValueError(
                f"Expected compact shape {expected_shape}, got {jac.shape}"
            )
        self._sparse_jac = jac

    def get_jacobian(self) -> Array:
        """Expand to full dense matrix using vectorized indexing."""
        nmesh = self._shape[0]
        # Pre-allocate full matrix
        full = self._bkd.zeros(self._shape)
        # Compute all column indices
        col_idx = self._bkd.arange(self._shape[1])
        # row_idx = col_idx % nmesh gives [0,1,...,nmesh-1,0,1,...,nmesh-1,...]
        row_idx = col_idx % nmesh
        # Flatten diagonal values in correct order (transpose to get block order)
        diag_flat = self._bkd.flatten(self._bkd.transpose(self._sparse_jac))
        # Vectorized scatter
        full[row_idx, col_idx] = diag_flat
        return full

    def __neg__(self) -> "DiagJacobian[Array]":
        return DiagJacobian(self._bkd, self._shape, -self._sparse_jac)

    def __mul__(
        self, other: Union[Array, float]
    ) -> "DiagJacobian[Array]":
        if isinstance(other, (int, float)) or (
            hasattr(other, "ndim") and other.ndim == 0
        ):
            return DiagJacobian(
                self._bkd, self._shape, self._sparse_jac * other
            )
        return DiagJacobian(
            self._bkd, self._shape, self._sparse_jac * other[:, None]
        )

    def __truediv__(
        self, other: Union[Array, float]
    ) -> "DiagJacobian[Array]":
        if isinstance(other, (int, float)) or (
            hasattr(other, "ndim") and other.ndim == 0
        ):
            return DiagJacobian(
                self._bkd, self._shape, self._sparse_jac / other
            )
        return DiagJacobian(
            self._bkd, self._shape, self._sparse_jac / other[:, None]
        )

    def rdot(self, other: Array) -> "DenseJacobian[Array]":
        """Compute A @ diag(d) for each input function block.

        Uses broadcasting: A @ diag(d) = A * d (column scaling).
        For multiple input functions, computes all blocks at once.
        """
        if other.shape[1] != self._shape[0]:
            raise ValueError(
                f"Shape mismatch: {other.shape} @ {self._shape}"
            )
        nmesh = self._shape[0]
        ninput_funs = self._sparse_jac.shape[1]
        m = other.shape[0]
        new_shape = (m, self._shape[1])

        # Vectorized: broadcast other (m, nmesh) with diag (nmesh, ninput)
        # Result: (m, nmesh, ninput) -> reshape to (m, nmesh * ninput)
        # other[:, :, None] has shape (m, nmesh, 1)
        # self._sparse_jac[None, :, :] has shape (1, nmesh, ninput)
        # Product has shape (m, nmesh, ninput)
        scaled = other[:, :, None] * self._sparse_jac[None, :, :]
        # Transpose to (m, ninput, nmesh) then reshape to (m, ninput * nmesh)
        dense_jac = self._bkd.reshape(
            self._bkd.transpose(scaled, (0, 2, 1)),
            new_shape
        )
        return DenseJacobian(self._bkd, new_shape, dense_jac)

    def __add__(
        self, other: "SparseJacobian[Array]"
    ) -> "SparseJacobian[Array]":
        if isinstance(other, ZeroJacobian):
            return self.copy()
        if isinstance(other, DenseJacobian):
            return other.__add__(self)
        # Diag + Diag
        return DiagJacobian(
            self._bkd, self._shape,
            self._sparse_jac + other._sparse_jac
        )

    def __sub__(
        self, other: "SparseJacobian[Array]"
    ) -> "SparseJacobian[Array]":
        if isinstance(other, ZeroJacobian):
            return self.copy()
        if isinstance(other, DenseJacobian):
            # self - other = -(other - self)
            result = other.__sub__(self)
            return result.__neg__()
        return DiagJacobian(
            self._bkd, self._shape,
            self._sparse_jac - other._sparse_jac
        )


class ZeroJacobian(SparseJacobian[Array]):
    """Zero Jacobian representation.

    Represents a Jacobian that is all zeros (no dependency).
    No storage needed.
    """

    def __init__(self, bkd: Backend[Array], shape: Tuple[int, int]):
        self._bkd = bkd
        self._shape = shape
        self._sparse_jac = None

    def copy(self) -> "ZeroJacobian[Array]":
        return ZeroJacobian(self._bkd, self._shape)

    def set_sparse_jacobian(self, jac: Array) -> None:
        if jac is not None:
            raise ValueError("ZeroJacobian sparse_jac must be None")

    def get_jacobian(self) -> Array:
        return self._bkd.zeros(self._shape)

    def __neg__(self) -> "ZeroJacobian[Array]":
        return ZeroJacobian(self._bkd, self._shape)

    def __mul__(
        self, other: Union[Array, float]
    ) -> "ZeroJacobian[Array]":
        return ZeroJacobian(self._bkd, self._shape)

    def __truediv__(
        self, other: Union[Array, float]
    ) -> "ZeroJacobian[Array]":
        return ZeroJacobian(self._bkd, self._shape)

    def rdot(self, other: Array) -> "ZeroJacobian[Array]":
        if other.shape[1] != self._shape[0]:
            raise ValueError(
                f"Shape mismatch: {other.shape} @ {self._shape}"
            )
        new_shape = (other.shape[0], self._shape[1])
        return ZeroJacobian(self._bkd, new_shape)

    def __add__(
        self, other: "SparseJacobian[Array]"
    ) -> "SparseJacobian[Array]":
        if isinstance(other, ZeroJacobian):
            return ZeroJacobian(self._bkd, self._shape)
        return other.copy()

    def __sub__(
        self, other: "SparseJacobian[Array]"
    ) -> "SparseJacobian[Array]":
        if isinstance(other, ZeroJacobian):
            return ZeroJacobian(self._bkd, self._shape)
        return other.__neg__()

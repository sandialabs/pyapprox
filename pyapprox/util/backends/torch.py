from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Backend


# Implement the PyTorch backend
class TorchBkd(Backend[torch.Tensor]):  # Specify torch.Tensor type
    @staticmethod
    def dot(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        # for typing consistancy make sure dot always returns an array
        return torch.matmul(Amat, Bmat)

    @staticmethod
    def eye(
        nrows: int, ncols: Optional[int] = None, dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.eye(
            nrows, ncols if ncols is not None else nrows, dtype=dtype
        )

    @staticmethod
    def inv(matrix: torch.Tensor) -> torch.Tensor:
        result = torch.linalg.inv(matrix)
        assert isinstance(
            result, torch.Tensor
        ), "Expected torch.linalg.inv to return a torch.Tensor"
        return result

    @staticmethod
    def pinv(matrix: torch.Tensor) -> torch.Tensor:
        result = torch.linalg.pinv(matrix)
        assert isinstance(
            result, torch.Tensor
        ), "Expected torch.linalg.pinv to return a torch.Tensor"
        return result

    @staticmethod
    def array(
        array: Union[Sequence[Any], torch.Tensor, float, int],
        dtype: Optional[Any] = None,
    ) -> torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array.detach().clone().to(
                dtype=dtype or array.dtype
            )
        arr_np = np.asarray(array)
        result = torch.from_numpy(arr_np.copy())
        if dtype is not None:
            return result.to(dtype=dtype)
        if result.is_floating_point():
            return result.to(dtype=torch.float64)
        return result

    @staticmethod
    def asarray(
        array: Union[Sequence[Any], torch.Tensor, float, int],
        dtype: Optional[Any] = None,
    ) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype)

    @staticmethod
    def assert_allclose(
        actual: torch.Tensor,
        desired: torch.Tensor,
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: Optional[str] = None,
    ) -> None:
        torch.testing.assert_close(
            actual,
            desired,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            msg=err_msg,
        )

    @staticmethod
    def double_dtype() -> Any:
        return torch.double

    @staticmethod
    def complex_dtype() -> Any:
        return torch.cdouble

    @staticmethod
    def int64_dtype() -> Any:
        return torch.int64

    @staticmethod
    def stack(
        arrays: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        axis: int = 0,
    ) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def concatenate(
        arrays: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        axis: int = 0,
    ) -> torch.Tensor:
        return torch.cat(arrays, dim=axis)

    @staticmethod
    def hstack(
        arrays: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        return torch.hstack(arrays)

    @staticmethod
    def vstack(
        arrays: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        return torch.vstack(arrays)

    @staticmethod
    def linspace(start: float, stop: float, num: int) -> torch.Tensor:
        return torch.linspace(start, stop, num)

    @staticmethod
    def logspace(start: float, stop: float, num: int) -> torch.Tensor:
        return torch.logspace(start, stop, num)

    @staticmethod
    def to_numpy(array: torch.Tensor) -> NDArray[Any]:
        if isinstance(array, np.ndarray):
            return array
        return array.detach().numpy()

    @staticmethod
    def to_float(array: "float | torch.Tensor") -> float:
        if isinstance(array, (float, int)):
            return float(array)
        if array.numel() != 1:
            raise ValueError(
                "to_float requires a single-element tensor,"
                f" got shape {tuple(array.shape)}"
            )
        return array.detach().item()

    @staticmethod
    def to_int(array: "int | torch.Tensor") -> int:
        if isinstance(array, (float, int)):
            return int(array)
        if array.numel() != 1:
            raise ValueError(
                "to_int requires a single-element tensor,"
                f" got shape {tuple(array.shape)}"
            )
        return int(array.detach().item())

    @staticmethod
    def flatten(array: torch.Tensor) -> torch.Tensor:
        return array.flatten()

    @staticmethod
    def ravel(array: torch.Tensor) -> torch.Tensor:
        return torch.ravel(array)

    @staticmethod
    def meshgrid(
        *arrays: torch.Tensor, indexing: str = "xy"
    ) -> Tuple[torch.Tensor, ...]:
        return torch.meshgrid(*arrays, indexing=indexing)

    @staticmethod
    def reshape(array: torch.Tensor, newshape: Sequence[int]) -> torch.Tensor:
        return torch.reshape(array, newshape)

    @staticmethod
    def transpose(
        array: torch.Tensor,
        axes: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if axes is None:
            return array.T
        return array.permute(*axes)

    @staticmethod
    def sum(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return torch.sum(array, dim=axis, keepdim=keepdims)

    @staticmethod
    def cumsum(array: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        if axis is None:
            return torch.cumsum(array.flatten(), dim=0)
        return torch.cumsum(array, dim=axis)

    @staticmethod
    def sin(array: torch.Tensor) -> torch.Tensor:
        return torch.sin(array)

    @staticmethod
    def cos(array: torch.Tensor) -> torch.Tensor:
        return torch.cos(array)

    @staticmethod
    def arcsin(array: torch.Tensor) -> torch.Tensor:
        return torch.arcsin(array)

    @staticmethod
    def arccos(array: torch.Tensor) -> torch.Tensor:
        return torch.arccos(array)

    @staticmethod
    def arctan(array: torch.Tensor) -> torch.Tensor:
        return torch.arctan(array)

    @staticmethod
    def arctan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.arctan2(y, x)

    @staticmethod
    def sinh(array: torch.Tensor) -> torch.Tensor:
        return torch.sinh(array)

    @staticmethod
    def cosh(array: torch.Tensor) -> torch.Tensor:
        return torch.cosh(array)

    @staticmethod
    def tanh(array: torch.Tensor) -> torch.Tensor:
        return torch.tanh(array)

    @staticmethod
    def arcsinh(array: torch.Tensor) -> torch.Tensor:
        return torch.asinh(array)

    @staticmethod
    def arccosh(array: torch.Tensor) -> torch.Tensor:
        return torch.acosh(array)

    @staticmethod
    def arctanh(array: torch.Tensor) -> torch.Tensor:
        return torch.atanh(array)

    @staticmethod
    def full(
        shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.full(shape, fill_value, dtype=dtype)

    @staticmethod
    def zeros(
        shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(
        shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype)

    @staticmethod
    def empty(
        shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype)

    @staticmethod
    def arange(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.arange(*args, **kwargs)

    @staticmethod
    def prod(
        array: torch.Tensor,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        if axis is None:
            if keepdims is False:
                return torch.prod(array)
            return torch.prod(array, dim=None, keepdim=True)
        return torch.prod(array, dim=axis, keepdim=keepdims)

    @staticmethod
    def any_bool(
        array: torch.Tensor,
        keepdims: bool = False,
    ) -> bool:
        return bool(torch.any(array, keepdim=keepdims).item())

    @staticmethod
    def any_array(
        array: torch.Tensor,
        axis: int,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return torch.any(array, dim=axis, keepdim=keepdims)

    @staticmethod
    def all_bool(
        array: torch.Tensor,
        keepdims: bool = False,
    ) -> bool:
        return bool(torch.all(array, keepdim=keepdims).item())

    @staticmethod
    def all_array(
        array: torch.Tensor,
        axis: int,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return torch.all(array, dim=axis, keepdim=keepdims)

    @staticmethod
    def log(array: torch.Tensor) -> torch.Tensor:
        return torch.log(array)

    @staticmethod
    def exp(array: torch.Tensor) -> torch.Tensor:
        return torch.exp(array)

    @staticmethod
    def copy(array: torch.Tensor) -> torch.Tensor:
        return array.clone()

    @staticmethod
    def erf(array: torch.Tensor) -> torch.Tensor:
        return torch.erf(array)

    @staticmethod
    def erfinv(array: torch.Tensor) -> torch.Tensor:
        return torch.erfinv(array)

    @staticmethod
    def gammaln(array: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(array)

    @staticmethod
    def digamma(array: torch.Tensor) -> torch.Tensor:
        return torch.digamma(array)

    @staticmethod
    def isfinite(array: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(array)

    @staticmethod
    def nonzero(condition: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return torch.nonzero(condition, as_tuple=True)

    @staticmethod
    def norm(
        array: torch.Tensor,
        ord: Optional[Union[int, float, str]] = None,
        axis: Optional[Union[int, Tuple[int, int]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor, torch.linalg.norm(array, ord, axis, keepdims)
        )

    @staticmethod
    def sign(array: torch.Tensor) -> torch.Tensor:
        return torch.sign(array)

    @staticmethod
    def abs(array: torch.Tensor) -> torch.Tensor:
        return torch.abs(array)

    @staticmethod
    def round(array: torch.Tensor) -> torch.Tensor:
        return torch.round(array)

    @staticmethod
    def floor(array: torch.Tensor) -> torch.Tensor:
        return torch.floor(array)

    @staticmethod
    def ceil(array: torch.Tensor) -> torch.Tensor:
        return torch.ceil(array)

    @staticmethod
    def sqrt(array: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(array)

    @staticmethod
    def solve(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, torch.linalg.solve(Amat, Bmat))

    @staticmethod
    def solve_sparse(Amat, bvec: torch.Tensor) -> torch.Tensor:
        """Solve A @ x = b where A is a sparse matrix.

        Raises
        ------
        NotImplementedError
            Always — sparse solves are not supported with TorchBkd.
            torch.sparse.spsolve requires MPS/CUDA (not available on CPU),
            and falling back to scipy would silently break the autograd
            computation graph.
        """
        raise NotImplementedError(
            "TorchBkd.solve_sparse() is not supported. "
            "torch.sparse.spsolve requires MPS/CUDA and is not available "
            "on CPU. Falling back to scipy would silently break the "
            "autograd computation graph. Use NumpyBkd for sparse solves "
            "or convert to dense with .toarray() first."
        )

    @staticmethod
    def lstsq(
        a: torch.Tensor,
        b: torch.Tensor,
        rcond: Optional[float] = None,
    ) -> torch.Tensor:
        """Solve least squares problem min ||ax - b||_2."""
        result = torch.linalg.lstsq(a, b, rcond=rcond)
        return result.solution

    @staticmethod
    def flip(
        array: torch.Tensor, axis: Optional[Tuple[int]] = None
    ) -> torch.Tensor:
        if axis is None:
            dims = (0,)
        elif isinstance(axis, int):
            dims = (axis,)
        else:
            dims = axis
        return torch.flip(array, dims=dims)

    @staticmethod
    def sort(
        array: torch.Tensor, axis: int = -1
    ) -> torch.Tensor:
        return torch.sort(array, dim=axis).values

    @staticmethod
    def argsort(
        array: torch.Tensor, axis: int = -1
    ) -> torch.Tensor:
        return torch.argsort(array, dim=axis)

    @staticmethod
    def searchsorted(
        sorted_array: torch.Tensor, values: torch.Tensor, side: str = "left"
    ) -> torch.Tensor:
        if not sorted_array.is_contiguous():
            sorted_array = sorted_array.contiguous()
        return torch.searchsorted(sorted_array, values, side=side)  # type: ignore

    @staticmethod
    def clip(
        array: torch.Tensor, a_min: Any, a_max: Any
    ) -> torch.Tensor:
        return torch.clamp(array, a_min, a_max)

    @staticmethod
    def min(
        array: torch.Tensor, axis: Optional[int] = None, keepdims: bool = False
    ) -> torch.Tensor:
        if axis is None:
            return torch.min(array)
        values, _ = torch.min(array, dim=axis, keepdim=keepdims)
        return values

    @staticmethod
    def max(
        array: torch.Tensor, axis: Optional[int] = None, keepdims: bool = False
    ) -> torch.Tensor:
        if axis is None:
            return torch.max(array)
        values, _ = torch.max(array, dim=axis, keepdim=keepdims)
        return values

    @staticmethod
    def einsum(subscripts: str, *operands: torch.Tensor) -> torch.Tensor:
        return torch.einsum(subscripts, *operands)

    @staticmethod
    def moveaxis(
        array: torch.Tensor,
        source: Union[int, tuple[int, ...]],
        destination: Union[int, tuple[int, ...]],
    ) -> torch.Tensor:
        return torch.movedim(array, source, destination)

    @staticmethod
    def diag(array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.diag(array, diagonal=k)

    @staticmethod
    def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.outer(a, b)

    @staticmethod
    def diff(array: torch.Tensor, n: int = 1, axis: int = -1) -> torch.Tensor:
        return torch.diff(array, n=n, dim=axis)

    @staticmethod
    def allclose(
        array1: torch.Tensor,
        array2: torch.Tensor,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool:
        return torch.allclose(
            array1, array2, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    @staticmethod
    def jacobian(
        fun: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor
    ) -> torch.Tensor:
        return torch.autograd.functional.jacobian(fun, params)

    @staticmethod
    def hvp(
        fun: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        return torch.autograd.functional.hvp(fun, params, vec)[1]

    @staticmethod
    def jvp(
        fun: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        # set create_graph=True so that result is differentiable.
        return torch.autograd.functional.jvp(
            fun, params, vec, create_graph=True
        )[1]

    @staticmethod
    def hessian(
        fun: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor
    ) -> torch.Tensor:
        return torch.autograd.functional.hessian(fun, params)

    @staticmethod
    def cholesky(array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(array)

    @staticmethod
    def solve_triangular(
        matrix: NDArray[Any],
        rhs: NDArray[Any],
        lower: bool = True,
        unit_diagonal: bool = False,
    ) -> NDArray[Any]:
        return torch.linalg.solve_triangular(
            matrix, rhs, upper=(not lower), unitriangular=unit_diagonal
        )

    @staticmethod
    def cholesky_solve(
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        lower: bool = True,
    ) -> torch.Tensor:
        return torch.cholesky_solve(rhs, matrix, upper=(not lower))

    @staticmethod
    def eigvalsh(array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.eigvalsh(array)

    @staticmethod
    def trace(array: torch.Tensor) -> torch.Tensor:
        return torch.trace(array)

    @staticmethod
    def atleast_1d(array: torch.Tensor) -> torch.Tensor:
        return torch.atleast_1d(array)

    @staticmethod
    def atleast_2d(array: torch.Tensor) -> torch.Tensor:
        return torch.atleast_2d(array)

    @staticmethod
    def tile(array: torch.Tensor, reps: Tuple[int, ...]) -> torch.Tensor:
        return torch.tile(array, dims=reps)

    @staticmethod
    def isnan(array: torch.Tensor) -> torch.Tensor:
        return torch.isnan(array)

    @staticmethod
    def isinf(array: torch.Tensor) -> torch.Tensor:
        return torch.isinf(array)

    @staticmethod
    def unique(array: torch.Tensor) -> torch.Tensor:
        return torch.unique(array)

    @staticmethod
    def argmin(
        array: torch.Tensor, axis: Optional[int] = None
    ) -> torch.Tensor:
        if axis is None:
            return torch.argmin(array)
        return torch.argmin(array, dim=axis)

    @staticmethod
    def argmax(
        array: torch.Tensor, axis: Optional[int] = None
    ) -> torch.Tensor:
        if axis is None:
            return torch.argmax(array)
        return torch.argmax(array, dim=axis)

    @staticmethod
    def get_diagonal(
        array: torch.Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> torch.Tensor:
        return torch.diagonal(array, offset, dim1=axis1, dim2=axis2)

    @staticmethod
    def cdist(
        XA: torch.Tensor, XB: torch.Tensor, p: float = 2.0
    ) -> torch.Tensor:
        return torch.cdist(XA, XB, p)

    @staticmethod
    def tril(array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.tril(array, diagonal=k)

    @staticmethod
    def triu(array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.triu(array, diagonal=k)

    @staticmethod
    def kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.kron(a, b)

    @staticmethod
    def repeat(
        array: torch.Tensor, repeats: int, axis: Optional[int] = None
    ) -> torch.Tensor:
        if axis is None:
            return array.flatten().repeat_interleave(repeats)
        return array.repeat_interleave(repeats, dim=axis)

    @staticmethod
    def slogdet(
        array: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.slogdet(array)

    @staticmethod
    def det(array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.det(array)

    @staticmethod
    def mean(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        if axis is None:
            return torch.mean(array)
        return torch.mean(array, dim=axis, keepdim=keepdims)

    @staticmethod
    def var(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> torch.Tensor:
        # torch.var uses correction (Bessel's correction) instead of ddof
        # correction=0 gives population variance, correction=1 gives sample variance
        if axis is None:
            return torch.var(array, correction=ddof)
        return torch.var(array, dim=axis, keepdim=keepdims, correction=ddof)

    @staticmethod
    def std(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> torch.Tensor:
        # torch.std uses correction (Bessel's correction) instead of ddof
        # correction=0 gives population std, correction=1 gives sample std
        if axis is None:
            return torch.std(array, correction=ddof)
        return torch.std(array, dim=axis, keepdim=keepdims, correction=ddof)

    @staticmethod
    def cov(
        array: torch.Tensor,
        rowvar: bool = True,
        ddof: int = 1,
    ) -> torch.Tensor:
        # torch.cov expects rows as variables (like np.cov with rowvar=True)
        # and uses correction instead of ddof
        if not rowvar:
            array = array.T
        result = torch.cov(array, correction=ddof)
        # Ensure result is always 2D (torch.cov returns scalar for 1 variable)
        return torch.atleast_2d(result)

    @staticmethod
    def eigh(array: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigh(array)

    @staticmethod
    def svd(
        array: torch.Tensor, full_matrices: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.linalg.svd(array, full_matrices=full_matrices)

    @staticmethod
    def rank(array: torch.Tensor) -> int:
        return int(torch.linalg.matrix_rank(array))

    @staticmethod
    def split(
        array: torch.Tensor,
        indices_or_sections: torch.Tensor,
        axis: int = 0,
    ) -> List[torch.Tensor]:
        # torch.tensor_split takes indices directly
        return list(torch.tensor_split(array, indices_or_sections, dim=axis))

    @staticmethod
    def where(
        condition: torch.Tensor,
        x: Union[torch.Tensor, float, None] = None,
        y: Union[torch.Tensor, float, None] = None,
    ) -> torch.Tensor:
        if x is None and y is None:
            return torch.where(condition)
        return torch.where(condition, x, y)

    @staticmethod
    def ones_like(
        array: torch.Tensor, dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.ones_like(array, dtype=dtype)

    @staticmethod
    def zeros_like(
        array: torch.Tensor, dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.zeros_like(array, dtype=dtype)

    @staticmethod
    def default_dtype() -> Any:
        return torch.float64

    @staticmethod
    def maximum(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x1, x2)

    @staticmethod
    def minimum(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x1, x2)

    @staticmethod
    def qr(
        array: torch.Tensor, mode: str = "reduced"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.qr(array, mode=mode)

    @staticmethod
    def lu(
        array: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LU factorization with partial pivoting.

        Returns P, L, U such that A = P @ L @ U.

        Parameters
        ----------
        array : torch.Tensor
            Matrix to factorize.

        Returns
        -------
        P : torch.Tensor
            Permutation matrix.
        L : torch.Tensor
            Lower triangular matrix with unit diagonal.
        U : torch.Tensor
            Upper triangular matrix.
        """
        return torch.linalg.lu(array)

    @staticmethod
    def index_update(
        array: torch.Tensor,
        index: Union[int, Tuple[int, ...]],
        value: Any,
    ) -> torch.Tensor:
        """Return a copy of array with array[index] = value.

        This provides a functional interface for array updates that
        is compatible with autograd frameworks.

        Parameters
        ----------
        array : torch.Tensor
            Tensor to update.
        index : int or tuple of int
            Index to update.
        value : Any
            Value to set at index.

        Returns
        -------
        torch.Tensor
            Copy of tensor with updated value.
        """
        result = array.clone()
        result[index] = value
        return result

    @staticmethod
    def block(blocks: List[List[torch.Tensor]]) -> torch.Tensor:
        """Create a block matrix from nested list of tensors."""
        rows = [torch.hstack(row) for row in blocks]
        return torch.vstack(rows)

    @staticmethod
    def array_type() -> type:
        """Return the array type class for this backend."""
        return torch.Tensor

    @staticmethod
    def tril_indices(n: int, k: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return indices for lower-triangular part of an (n, n) array."""
        return torch.tril_indices(n, n, offset=k)

    @staticmethod
    def multidot(arrays: List[torch.Tensor]) -> torch.Tensor:
        """Compute the dot product of two or more arrays in a single call."""
        return torch.linalg.multi_dot(arrays)

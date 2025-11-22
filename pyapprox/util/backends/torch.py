import platform
import warnings
from typing import List, Optional, Tuple, Any, Sequence, Callable, Iterable
import torch

from pyapprox.util.backends.template import BackendMixin, AxisArg

torch.set_default_dtype(torch.double)


class TorchMixin(BackendMixin):
    # def __init__(self):
    #     # needed for autograd
    #     self._inputs = None

    @staticmethod
    def dot(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return Amat @ Bmat

    @staticmethod
    def eye(
        nrows: int,
        ncols: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if ncols is not None:
            return torch.eye(nrows, ncols, dtype=dtype)
        return torch.eye(nrows, dtype=dtype)

    @staticmethod
    def inv(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(matrix)

    @staticmethod
    def pinv(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.pinv(matrix)

    @staticmethod
    def solve(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(Amat, Bmat)

    @staticmethod
    def cholesky(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(matrix)

    @staticmethod
    def det(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.det(matrix)

    @staticmethod
    def cholesky_solve(
        chol: torch.Tensor, bvec: torch.Tensor, lower: bool = True
    ) -> torch.Tensor:
        return torch.cholesky_solve(bvec, chol, upper=(not lower))

    @staticmethod
    def qr(mat: torch.Tensor, mode: str = "complete") -> Any:
        return torch.linalg.qr(mat, mode=mode)

    @staticmethod
    def solve_triangular(
        Amat: torch.Tensor, bvec: torch.Tensor, lower: bool = True
    ) -> torch.Tensor:
        return torch.linalg.solve_triangular(Amat, bvec, upper=(not lower))

    @staticmethod
    def full(*args: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.full(*args, dtype=dtype)

    @staticmethod
    def zeros(*args: Any, dtype: torch.dtype = float) -> torch.Tensor:
        return torch.zeros(*args, dtype=dtype)

    @staticmethod
    def ones(*args: Any, dtype: torch.dtype = float) -> torch.Tensor:
        return torch.ones(*args, dtype=dtype)

    @staticmethod
    def empty(*args: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.empty(*args, dtype=dtype)

    @staticmethod
    def empty_like(*args: Any, dtype: torch.dtype = float) -> torch.Tensor:
        return torch.empty_like(*args, dtype=dtype)

    @staticmethod
    def exp(matrix: torch.Tensor) -> torch.Tensor:
        return torch.exp(matrix)

    @staticmethod
    def sqrt(matrix: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(matrix)

    @staticmethod
    def cos(matrix: torch.Tensor) -> torch.Tensor:
        return torch.cos(matrix)

    @staticmethod
    def arccos(matrix: torch.Tensor) -> torch.Tensor:
        return torch.arccos(matrix)

    @staticmethod
    def tan(matrix: torch.Tensor) -> torch.Tensor:
        return torch.tan(matrix)

    @staticmethod
    def arctan(matrix: torch.Tensor) -> torch.Tensor:
        return torch.arctan(matrix)

    @staticmethod
    def arctan2(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
        return torch.arctan2(matrix1, matrix2)

    @staticmethod
    def sin(matrix: torch.Tensor) -> torch.Tensor:
        return torch.sin(matrix)

    @staticmethod
    def arcsin(matrix: torch.Tensor) -> torch.Tensor:
        return torch.arcsin(matrix)

    @staticmethod
    def cosh(matrix: torch.Tensor) -> torch.Tensor:
        return torch.cosh(matrix)

    @staticmethod
    def sinh(matrix: torch.Tensor) -> torch.Tensor:
        return torch.sinh(matrix)

    @staticmethod
    def arccosh(matrix: torch.Tensor) -> torch.Tensor:
        return torch.arccosh(matrix)

    @staticmethod
    def arcsinh(matrix: torch.Tensor) -> torch.Tensor:
        return torch.arcsinh(matrix)

    @staticmethod
    def log(matrix: torch.Tensor) -> torch.Tensor:
        return torch.log(matrix)

    @staticmethod
    def log10(matrix: torch.Tensor) -> torch.Tensor:
        return torch.log10(matrix)

    @staticmethod
    def multidot(matrix_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.linalg.multi_dot(matrix_list)

    @staticmethod
    def prod(
        matrix_list: torch.Tensor, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        if axis is None:
            return torch.prod(matrix_list)
        return torch.prod(matrix_list, dim=axis)

    @staticmethod
    def hstack(arrays: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.hstack(arrays)

    @staticmethod
    def vstack(arrays: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.vstack(arrays)

    @staticmethod
    def stack(
        arrays: Sequence[torch.Tensor], axis: AxisArg = 0
    ) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def dstack(arrays: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.dstack(arrays)

    @staticmethod
    def arange(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.arange(*args, **kwargs)

    @staticmethod
    def linspace(
        *args: Any, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        return torch.linspace(*args, dtype=dtype)

    @staticmethod
    def logspace(
        *args: Any, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        return torch.logspace(*args, dtype=dtype)

    @staticmethod
    def ndim(mat: torch.Tensor) -> Any:
        return mat.ndim

    @staticmethod
    def repeat(
        mat: torch.Tensor, nreps: int, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        if mat.ndim == 1:
            return mat.repeat_interleave(nreps)
        if mat.ndim > 2:
            raise ValueError("ndim must be <= 2")
        if axis is None:
            return mat.repeat_interleave(nreps)
        if axis == 0:
            return mat.repeat(1, nreps).view(-1, mat.shape[1])
        if axis == 1:
            return mat.view(-1, 1).repeat(1, nreps).view(mat.shape[0], -1)
        raise ValueError("axis > 2 but mat.ndim == 2")

    @staticmethod
    def tile(mat: torch.Tensor, nreps: Tuple[Any, ...]) -> torch.Tensor:
        return torch.tile(mat, nreps)

    @staticmethod
    def cdist(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return torch.cdist(Amat, Bmat, p=2)

    @staticmethod
    def einsum(*args: Any) -> torch.Tensor:
        return torch.einsum(*args)

    @staticmethod
    def trace(mat: torch.Tensor) -> torch.Tensor:
        return torch.trace(mat)

    @staticmethod
    def copy(mat: torch.Tensor) -> torch.Tensor:
        # return mat.clone() will copy requires_grad field
        return mat.clone()

    @staticmethod
    def get_diagonal(mat: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(mat)

    @staticmethod
    def diag(array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.diag(array, diagonal=k)

    @staticmethod
    def isnan(mat: torch.Tensor) -> torch.Tensor:
        return torch.isnan(mat)

    @staticmethod
    def atleast1d(val: Any) -> torch.Tensor:
        return torch.atleast_1d(val)

    @staticmethod
    def atleast2d(val: Any) -> torch.Tensor:
        return torch.atleast_2d(val)

    @staticmethod
    def reshape(mat: torch.Tensor, newshape: Tuple[Any, ...]) -> torch.Tensor:
        return torch.reshape(mat, newshape)

    @staticmethod
    def where(
        cond: torch.Tensor,
        array1: torch.Tensor = None,
        array2: torch.Tensor = None,
    ) -> torch.Tensor:
        if array1 is not None:
            return torch.where(cond, array1, array2)
        return torch.where(cond)

    @staticmethod
    def detach(mat: torch.Tensor) -> torch.Tensor:
        return mat.detach()

    @staticmethod
    def tointeger(mat: torch.Tensor) -> torch.Tensor:
        return mat.int()

    @staticmethod
    def inf() -> torch.Tensor:
        return torch.inf

    @staticmethod
    def norm(
        mat: torch.Tensor, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        return torch.linalg.norm(mat, dim=axis)

    @staticmethod
    def any(mat: torch.Tensor, axis: Optional[AxisArg] = None) -> torch.Tensor:
        if axis is None:
            return torch.any(mat)
        return torch.any(mat, dim=axis)

    @staticmethod
    def all(mat: torch.Tensor, axis: Optional[AxisArg] = None) -> torch.Tensor:
        if axis is None:
            return torch.all(mat)
        return torch.all(mat, dim=axis)

    @staticmethod
    def kron(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return torch.kron(Amat, Bmat)

    @staticmethod
    def slogdet(Amat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.slogdet(Amat)

    @staticmethod
    def mean(
        mat: torch.Tensor, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        if axis is None:
            return torch.mean(mat)
        return torch.mean(mat, dim=axis)

    def var(
        mat: torch.Tensor, axis: Optional[AxisArg] = None, ddof: int = 0
    ) -> torch.Tensor:
        if axis is None:
            return torch.var(mat, correction=ddof)
        return torch.var(mat, dim=axis, correction=ddof)

    @staticmethod
    def std(
        mat: torch.Tensor, axis: Optional[AxisArg] = None, ddof: int = 0
    ) -> torch.Tensor:
        if axis is None:
            return torch.std(mat, correction=ddof)
        return torch.std(mat, dim=axis, correction=ddof)

    @staticmethod
    def cov(
        mat: torch.Tensor,
        ddof: int = 0,
        rowvar: bool = True,
        aweights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rowvar:
            return torch.cov(mat, correction=ddof, aweights=aweights)
        return torch.cov(mat.T, correction=ddof, aweights=aweights)

    @staticmethod
    def abs(mat: torch.Tensor) -> torch.Tensor:
        return torch.absolute(mat)

    @staticmethod
    def to_numpy(mat: torch.Tensor) -> Any:
        return mat.detach().numpy()

    @staticmethod
    def argsort(mat: torch.Tensor, axis: AxisArg = -1) -> torch.Tensor:
        return torch.argsort(mat, dim=axis)

    @staticmethod
    def sort(mat: torch.Tensor, axis: AxisArg = -1) -> torch.Tensor:
        return torch.sort(mat, dim=axis)[0]

    @staticmethod
    def flip(
        mat: torch.Tensor, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        if axis is None:
            _axis = (0,)
        elif type(axis) == int:
            _axis = (axis,)
        else:
            _axis = axis
        return torch.flip(mat, dims=_axis)

    @staticmethod
    def allclose(Amat: torch.Tensor, Bmat: torch.Tensor, **kwargs: Any) -> Any:
        return torch.allclose(Amat, Bmat, **kwargs)

    @staticmethod
    def isclose(
        Amat: torch.Tensor, Bmat: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return torch.isclose(Amat, Bmat, **kwargs)

    @staticmethod
    def lstsq(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lstsq(Amat, Bmat, rcond=None)[0]

    @staticmethod
    def argmax(array: torch.Tensor) -> torch.Tensor:
        return torch.argmax(array)

    @staticmethod
    def argmin(array: torch.Tensor) -> torch.Tensor:
        return torch.argmin(array)

    @staticmethod
    def max(
        array: torch.Tensor, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        if axis is None:
            return torch.max(array)
        # torch returns both max and indices
        return torch.max(array, dim=axis)[0]

    @staticmethod
    def maximum(array1: torch.Tensor, array2: torch.Tensor) -> torch.Tensor:
        return torch.maximum(array1, array2)

    @staticmethod
    def minimum(array1: torch.Tensor, array2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(array1, array2)

    @staticmethod
    def min(
        array: torch.Tensor, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        if axis is None:
            return torch.min(array)
        # torch returns both min and indices
        return torch.min(array, dim=axis)[0]

    @staticmethod
    def block(blocks: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
        return torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0)

    @staticmethod
    def sum(
        matrix: torch.Tensor, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        return torch.sum(matrix, dim=axis)

    @staticmethod
    def count_nonzero(
        matrix: torch.Tensor, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        return torch.count_nonzero(matrix, dim=axis)

    @staticmethod
    def array(
        array: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype)

    @staticmethod
    def eigh(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigh(matrix)

    @staticmethod
    def svd(
        matrix: torch.Tensor, full_matrices: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.linalg.svd(matrix, full_matrices=full_matrices)

    @staticmethod
    def isfinite(matrix: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(matrix)

    @staticmethod
    def cond(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cond(matrix)

    @staticmethod
    def rank(matrix: torch.Tensor) -> Any:
        return torch.linalg.matrix_rank(matrix)

    @staticmethod
    def jacobian(
        fun: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor
    ) -> torch.Tensor:
        return torch.autograd.functional.jacobian(fun, params)

    @staticmethod
    def grad(
        fun: Callable[[torch.Tensor], torch.Tensor], params: torch.Tensor
    ) -> torch.Tensor:
        params_copy = params.clone()
        params_copy.requires_grad = True
        val = fun(params_copy)
        val.backward()
        grad = TorchMixin.copy(params_copy.grad)
        params_copy.grad.zero_()
        # params.requires_grad = False
        return val, grad

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

    def hvp(
        fun: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        return torch.autograd.functional.hvp(fun, params, vec)[1]

    @staticmethod
    def up(
        matrix: torch.Tensor,
        indices: torch.Tensor,
        submatrix: torch.Tensor,
        axis: AxisArg = 0,
    ) -> torch.Tensor:
        if axis == 0:
            matrix[indices] = submatrix
            return matrix
        if axis == 1:
            matrix[:, indices] = submatrix
            return matrix
        if axis == -1:
            matrix[..., indices] = submatrix
            return matrix
        raise ValueError("axis must be in (0, 1, -1)")

    @staticmethod
    def moveaxis(
        array: torch.Tensor, source: int, destination: int
    ) -> torch.Tensor:
        return torch.moveaxis(array, source, destination)

    @staticmethod
    def floor(array: torch.Tensor) -> torch.Tensor:
        return torch.floor(array)

    @staticmethod
    def ceil(array: torch.Tensor) -> torch.Tensor:
        return torch.ceil(array)

    @staticmethod
    def asarray(
        array: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype)

    @staticmethod
    def unique(
        array: Iterable[Any], axis: Optional[AxisArg] = None, **kwargs: Any
    ) -> torch.Tensor:
        return torch.unique(array, dim=axis, **kwargs)

    @staticmethod
    def delete(
        array: torch.Tensor, inds: torch.Tensor, axis: Optional[int] = None
    ) -> torch.Tensor:
        if axis is None:
            _arr = array.flatten()
            axis = 0
        else:
            _arr = array
        inds = TorchMixin.atleast1d(TorchMixin.asarray(inds))
        skip = [i.item() for i in torch.arange(_arr.shape[axis])[inds]]
        retained = [
            i.item() for i in torch.arange(_arr.shape[axis]) if i not in skip
        ]
        indices = [
            slice(None) if i != axis else retained for i in range(_arr.ndim)
        ]
        return _arr[tuple(indices)]

    @staticmethod
    def jacobian_implemented() -> bool:
        return True

    @staticmethod
    def hessian_implemented() -> bool:
        return True

    @staticmethod
    def jvp_implemented() -> bool:
        return True

    @staticmethod
    def hvp_implemented() -> bool:
        return True

    @staticmethod
    def meshgrid(*arrays: torch.Tensor, indexing: str = "xy") -> torch.Tensor:
        return torch.meshgrid(*arrays, indexing=indexing)

    @staticmethod
    def tanh(array: torch.Tensor) -> torch.Tensor:
        return torch.tanh(array)

    @staticmethod
    def diff(array: torch.Tensor) -> torch.Tensor:
        return torch.diff(array)

    @staticmethod
    def int_dtype() -> torch.dtype:
        return torch.int64

    @staticmethod
    def cumsum(
        array: torch.Tensor, axis: AxisArg = 0, **kwargs: Any
    ) -> torch.Tensor:
        assert axis is not None
        return torch.cumsum(array, dim=axis, **kwargs)

    @staticmethod
    def complex_dtype() -> torch.dtype:
        return torch.cdouble

    @staticmethod
    def array_type() -> torch.Tensor:
        return torch.Tensor

    @staticmethod
    def real(array: torch.Tensor) -> torch.Tensor:
        return array.real

    @staticmethod
    def imag(array: torch.Tensor) -> torch.Tensor:
        return array.imag

    @staticmethod
    def round(array: torch.Tensor) -> torch.Tensor:
        return torch.round(array)

    @staticmethod
    def flatten(array: torch.Tensor) -> torch.Tensor:
        return array.flatten()

    @staticmethod
    def double_type() -> torch.dtype:
        return torch.double

    @staticmethod
    def bool_type() -> torch.dtype:
        return torch.bool

    @staticmethod
    def gammaln(mat: torch.Tensor) -> torch.Tensor:
        return torch.special.gammaln(mat)

    @staticmethod
    def split(
        mat: torch.Tensor, splits: torch.Tensor, axis: AxisArg = 0
    ) -> Any:
        return torch.tensor_split(mat, splits.tolist(), dim=axis)

    def chunks(mat: torch.Tensor, nchunks: int, axis: AxisArg = 0) -> Any:
        return torch.chunk(mat, nchunks, dim=axis)

    @staticmethod
    def sign(mat: torch.Tensor) -> torch.Tensor:
        return torch.sign(mat)

    @staticmethod
    def is_scalar_array(array: torch.Tensor) -> bool:
        return isinstance(array, torch.Tensor) and array.ndim == 0

    @staticmethod
    def quantile(
        array: torch.Tensor, q: float, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        return torch.quantile(array, q, dim=axis)

    @staticmethod
    def tril(array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.tril(array, diagonal=k)

    @staticmethod
    def tril_indices(
        n: int, k: int = 0, m: Optional[int] = None
    ) -> torch.Tensor:
        if m is None:
            m = n
        return torch.tril_indices(n, m, k)

    @staticmethod
    def triu(array: torch.Tensor, k: int = 0) -> torch.Tensor:
        return torch.triu(array, diagonal=k)

    @staticmethod
    def triu_indices(
        n: int, k: int = 0, m: Optional[int] = None
    ) -> torch.Tensor:
        if m is None:
            m = n
        return torch.triu_indices(n, m, k)

    @staticmethod
    def digamma(array: torch.Tensor) -> torch.Tensor:
        return torch.digamma(array)

    @staticmethod
    def erf(array: torch.Tensor) -> torch.Tensor:
        return torch.erf(array)

    @staticmethod
    def erfinv(array: torch.Tensor) -> torch.Tensor:
        return torch.erfinv(array)

    @staticmethod
    def reshape_fortran(
        array: torch.Tensor, shape: Tuple[Any, ...]
    ) -> torch.Tensor:
        array = array.permute(*reversed(range(len(array.shape))))
        array = array.reshape(*reversed(shape))
        return array.permute(*reversed(range(len(shape))))

    @staticmethod
    def gammainc(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.special.gammainc(a, x)

    @staticmethod
    def factorial(array: torch.Tensor) -> torch.Tensor:
        return torch.special.factorial(array)

    @staticmethod
    def clip(
        array: torch.Tensor, minval: float, maxval: float
    ) -> torch.Tensor:
        return torch.clip(array, minval, maxval)

    @staticmethod
    def cartesian_product(input_sets: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(input_sets) == 1:
            return input_sets[0][None, :]
        return TorchMixin.flip(
            torch.cartesian_prod(*input_sets[::-1]).T, axis=(0,)
        )

    @staticmethod
    def swapaxes(array: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        return torch.swapaxes(array, axis1, axis2)

    @staticmethod
    def block_diag(arrays: List[torch.Tensor]) -> torch.Tensor:
        return torch.block_diag(*arrays)

    @staticmethod
    def searchsorted(
        array: torch.Tensor, values: torch.Tensor, side: str = "left"
    ) -> torch.Tensor:
        return torch.searchsorted(array, values, side=side)

    @staticmethod
    def set_gpu_as_default() -> None:
        if platform.system() == "Darwin":
            torch.set_default_device("mps")
            torch.set_default_dtype(torch.float)
            warnings.warn(
                "GPUs can only be used with single precision on OSX",
                UserWarning,
            )
        else:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    @staticmethod
    def set_cpu_as_default() -> None:
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.double)
        warnings.warn(
            "Making CPUs default. This also set default dtype to double",
            UserWarning,
        )

    @staticmethod
    def fft(
        mat: torch.Tensor, axis: Optional[AxisArg] = None, **kwargs: Any
    ) -> torch.Tensor:
        if mat.ndim < 3:
            raise ValueError(
                "mat must explicitly express channel and sample " "dimensions"
            )
        _axis = list(range(mat.ndim - 2)) if axis is None else axis
        return torch.fft.fftn(mat, dim=_axis, **kwargs)

    @staticmethod
    def ifft(
        mat: torch.Tensor, axis: Optional[AxisArg] = None, **kwargs: Any
    ) -> torch.Tensor:
        if mat.ndim < 3:
            raise ValueError(
                "mat must explicitly express channel and sample " "dimensions"
            )
        _axis = list(range(mat.ndim - 2)) if axis is None else axis
        return torch.fft.ifftn(mat, dim=_axis, **kwargs)

    @staticmethod
    def fftshift(
        mat: torch.Tensor, axis: Optional[AxisArg] = None, **kwargs: Any
    ) -> torch.Tensor:
        if mat.ndim < 3:
            raise ValueError(
                "mat must explicitly express channel and sample " "dimensions"
            )
        _axis = list(range(mat.ndim - 2)) if axis is None else axis
        return torch.fft.fftshift(mat, dim=_axis, **kwargs)

    @staticmethod
    def ifftshift(
        mat: torch.Tensor, axis: Optional[AxisArg] = None, **kwargs: Any
    ) -> torch.Tensor:
        if mat.ndim < 3:
            raise ValueError(
                "mat must explicitly express channel and sample " "dimensions"
            )
        _axis = list(range(mat.ndim - 2)) if axis is None else axis
        return torch.fft.ifftshift(mat, dim=_axis, **kwargs)

    @staticmethod
    def cfloat() -> torch.dtype:
        return torch.complex128

    @staticmethod
    def transpose(
        mat: torch.Tensor, axis: Optional[AxisArg] = None
    ) -> torch.Tensor:
        if axis is None:
            axis = list(range(mat.ndim - 1, -1, -1))
        if not hasattr(axis, "__iter__"):
            axis = [axis]
        return torch.permute(mat, dims=axis)

    @staticmethod
    def size(mat: torch.Tensor) -> Any:
        return torch.numel(mat)

    @staticmethod
    def nan() -> torch.Tensor:
        return torch.nan

    @staticmethod
    def get_slices(
        mat: torch.Tensor, slices: Tuple[slice, ...]
    ) -> torch.Tensor:
        return mat[tuple(slices)]

    @staticmethod
    def concatenate(
        mats: List[torch.tensor], axis: AxisArg = 0
    ) -> torch.Tensor:
        return torch.cat(mats, dim=axis)

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

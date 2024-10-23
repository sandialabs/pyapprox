from typing import List

import torch

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin


class TorchLinAlgMixin(LinAlgMixin):
    def __init__(self):
        # needed for autograd
        self._inputs = None

    @staticmethod
    def dot(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return Amat @ Bmat

    @staticmethod
    def eye(nrows: int, dtype=torch.double) -> torch.Tensor:
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
    def det(matrix: torch.Tensor):
        return torch.linalg.det(matrix)

    @staticmethod
    def cholesky_solve(chol: torch.Tensor, bvec: torch.Tensor,
                       lower: bool = True) -> torch.Tensor:
        return torch.cholesky_solve(bvec, chol, upper=(not lower))

    @staticmethod
    def qr(mat: torch.Tensor, mode="complete"):
        return torch.linalg.qr(mat, mode=mode)

    @staticmethod
    def solve_triangular(Amat: torch.Tensor, bvec: torch.Tensor,
                         lower: bool = True) -> torch.Tensor:
        return torch.linalg.solve_triangular(Amat, bvec, upper=(not lower))

    @staticmethod
    def full(*args, dtype=torch.double):
        return torch.full(*args, dtype=dtype)

    @staticmethod
    def zeros(*args, dtype=float):
        return torch.zeros(*args, dtype=dtype)

    @staticmethod
    def ones(*args, dtype=float):
        return torch.ones(*args, dtype=dtype)

    @staticmethod
    def empty(*args, dtype=torch.double):
        return torch.empty(*args, dtype=dtype)

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
    def prod(matrix_list: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            return torch.prod(matrix_list)
        return torch.prod(matrix_list, dim=axis)

    @staticmethod
    def hstack(arrays) -> torch.Tensor:
        return torch.hstack(arrays)

    @staticmethod
    def vstack(arrays) -> torch.Tensor:
        return torch.vstack(arrays)

    @staticmethod
    def stack(arrays, axis=0) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def dstack(arrays) -> torch.Tensor:
        return torch.dstack(arrays)

    @staticmethod
    def arange(*args, **kwargs) -> torch.Tensor:
        return torch.arange(*args, **kwargs)

    @staticmethod
    def linspace(*args, dtype=torch.double):
        return torch.linspace(*args, dtype=dtype)

    @staticmethod
    def logspace(*args, dtype=torch.double):
        return torch.logspace(*args, dtype=dtype)

    @staticmethod
    def ndim(mat: torch.Tensor) -> int:
        return mat.ndim

    @staticmethod
    def repeat(mat: torch.Tensor, nreps: int) -> torch.Tensor:
        return mat.repeat(nreps)

    @staticmethod
    def tile(mat: torch.Tensor, nreps) -> torch.Tensor:
        return torch.tile(mat, nreps)

    @staticmethod
    def cdist(Amat: torch.tensor,
              Bmat: torch.tensor) -> torch.Tensor:
        return torch.cdist(Amat, Bmat, p=2)

    @staticmethod
    def einsum(*args) -> torch.Tensor:
        return torch.einsum(*args)

    @staticmethod
    def trace(mat: torch.Tensor) -> torch.Tensor:
        return torch.trace(mat)

    @staticmethod
    def copy(mat: torch.Tensor) -> torch.Tensor:
        return mat.clone()

    @staticmethod
    def get_diagonal(mat: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(mat)

    @staticmethod
    def diag(array, k=0):
        return torch.diag(array, diagonal=k)

    @staticmethod
    def isnan(mat) -> torch.Tensor:
        return torch.isnan(mat)

    @staticmethod
    def atleast1d(val, dtype=torch.double) -> torch.Tensor:
        return torch.atleast_1d(
            torch.as_tensor(val, dtype=dtype))

    @staticmethod
    def atleast2d(val, dtype=torch.double) -> torch.Tensor:
        return torch.atleast_2d(
            torch.as_tensor(val, dtype=dtype))

    @staticmethod
    def reshape(mat: torch.Tensor, newshape) -> torch.Tensor:
        return torch.reshape(mat, newshape)

    @staticmethod
    def where(cond: torch.Tensor) -> torch.Tensor:
        return torch.where(cond)

    @staticmethod
    def detach(mat: torch.Tensor) -> torch.Tensor:
        return mat.detach()

    @staticmethod
    def tointeger(mat: torch.Tensor) -> torch.Tensor:
        return mat.int()

    @staticmethod
    def inf():
        return torch.inf

    @staticmethod
    def norm(mat: torch.Tensor, axis=None) -> torch.Tensor:
        return torch.linalg.norm(mat, dim=axis)

    @staticmethod
    def any(mat: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            return torch.any(mat)
        return torch.any(mat, dim=axis)

    @staticmethod
    def all(mat: torch.Tensor, axis=None) -> torch.Tensor:
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
    def mean(mat: torch.Tensor, axis: int = None) -> torch.Tensor:
        if axis is None:
            return torch.mean(mat)
        return torch.mean(mat, dim=axis)

    @staticmethod
    def std(mat: torch.Tensor, axis: int = None,
                ddof: int = 0) -> torch.Tensor:
        if axis is None:
            return torch.std(mat, correction=ddof)
        return torch.std(mat, dim=axis, correction=ddof)

    @staticmethod
    def cov(mat: torch.Tensor, ddof=0, rowvar=True) -> torch.Tensor:
        if rowvar:
            return torch.cov(mat, correction=ddof)
        return torch.cov(mat.T, correction=ddof)

    @staticmethod
    def abs(mat: torch.Tensor) -> torch.Tensor:
        return torch.absolute(mat)

    @staticmethod
    def to_numpy(mat: torch.Tensor):
        return mat.numpy()

    @staticmethod
    def argsort(mat: torch.Tensor, axis=-1) -> torch.Tensor:
        return torch.argsort(mat, dim=axis)

    @staticmethod
    def sort(mat: torch.Tensor, axis=-1) -> torch.Tensor:
        return torch.sort(mat, dim=axis)[0]

    @staticmethod
    def flip(mat: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            axis = (0,)
        return torch.flip(mat, dims=axis)

    @staticmethod
    def allclose(Amat: torch.Tensor, Bmat: torch.Tensor,
                 **kwargs) -> bool:
        return torch.allclose(Amat, Bmat, **kwargs)

    @staticmethod
    def lstsq(Amat, Bmat):
        return torch.linalg.lstsq(Amat, Bmat, rcond=None)[0]

    @staticmethod
    def argmax(array):
        return torch.argmax(array)

    @staticmethod
    def argmin(array):
        return torch.argmin(array)

    @staticmethod
    def max(array, axis=None):
        if axis is None:
            return torch.max(array)
        # torch returns both max and indices
        return torch.max(array, dim=axis)[0]

    @staticmethod
    def maximum(array1, array2):
        return torch.maximum(array1, array2)

    @staticmethod
    def minimum(array1, array2):
        return torch.minimum(array1, array2)

    @staticmethod
    def min(array, axis=None):
        if axis is None:
            return torch.min(array)
        # torch returns both min and indices
        return torch.min(array, dim=axis)[0]

    @staticmethod
    def block(blocks):
        return torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0)

    @staticmethod
    def sum(matrix, axis=None):
        return torch.sum(matrix, dim=axis)

    @staticmethod
    def count_nonzero(matrix, axis=None):
        return torch.count_nonzero(matrix, dim=axis)

    @staticmethod
    def array(array, dtype=torch.double):
        return torch.as_tensor(array, dtype=dtype)

    @staticmethod
    def eigh(matrix):
        return torch.linalg.eigh(matrix)

    @staticmethod
    def svd(matrix, full_matrices=True):
        return torch.linalg.svd(matrix, full_matrices=full_matrices)

    @staticmethod
    def isfinite(matrix):
        return torch.isfinite(matrix)

    @staticmethod
    def cond(matrix):
        return torch.linalg.cond(matrix)

    @staticmethod
    def jacobian(fun, params):
        return torch.autograd.functional.jacobian(fun, params)

    @staticmethod
    def grad(fun, params):
        params.requires_grad = True
        val = fun(params)
        val.backward()
        grad = __class__.copy(params.grad)
        params.grad.zero_()
        return val, grad

    @staticmethod
    def hessian(fun, params):
        return torch.autograd.functional.hessian(fun, params)

    @staticmethod
    def up(matrix, indices, submatrix, axis=0):
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
    def moveaxis(array, source, destination):
        return torch.moveaxis(array, source, destination)

    @staticmethod
    def floor(array):
        return torch.floor(array)

    @staticmethod
    def asarray(array, dtype=torch.double):
        return torch.as_tensor(array, dtype=dtype)

    @staticmethod
    def unique(array, **kwargs):
        return torch.unique(array, **kwargs)

    @staticmethod
    def delete(array, obj, axis=None):
        if axis is None:
            axis = 0
        mask = __class__.ones(array.shape[axis], dtype=bool)
        mask[obj] = False
        if axis == 0:
            return array[mask]
        if axis == 1:
            return array[:, mask]
        if axis == -1:
            return array[..., mask]
        raise NotImplementedError("axis must be in (0, 1, -1)")

    @staticmethod
    def jacobian_implemented() -> bool:
        return True

    @staticmethod
    def hessian_implemented() -> bool:
        return True

    @staticmethod
    def meshgrid(*arrays, indexing="xy"):
        return torch.meshgrid(*arrays, indexing=indexing)

    @staticmethod
    def tanh(array):
        return torch.tanh(array)

    @staticmethod
    def diff(array):
        return torch.diff(array)

    @staticmethod
    def int():
        return torch.int64

    @staticmethod
    def cumsum(array, axis=0, **kwargs):
        assert axis is not None
        return array.cumsum(dim=axis, **kwargs)

    @staticmethod
    def complex_dtype():
        return torch.cdouble

    @staticmethod
    def array_type():
        return torch.Tensor

    @staticmethod
    def real(array):
        return array.real

    @staticmethod
    def imag(array):
        return array.imag

    @staticmethod
    def round(array):
        return torch.round(array)

    @staticmethod
    def flatten(array: torch.Tensor):
        return array.flatten()

from typing import List

import numpy as np
import scipy

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin


class NumpyLinAlgMixin(LinAlgMixin):
    @staticmethod
    def dot(Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return np.dot(Amat, Bmat)

    @staticmethod
    def eye(nrows: int, dtype=float) -> np.ndarray:
        return np.eye(nrows, dtype=dtype)

    @staticmethod
    def inv(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(matrix)

    @staticmethod
    def pinv(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(matrix)

    @staticmethod
    def solve(Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return np.linalg.solve(Amat, Bmat)

    @staticmethod
    def cholesky(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.cholesky(matrix)

    @staticmethod
    def det(matrix: np.ndarray):
        return np.linalg.det(matrix)

    @staticmethod
    def cholesky_solve(chol: np.ndarray, bvec: np.ndarray,
                       lower: bool = True) -> np.ndarray:
        return scipy.linalg.cho_solve((chol, lower), bvec)

    @staticmethod
    def qr(mat: np.ndarray, mode="complete"):
        return np.linalg.qr(mat, mode=mode)

    @staticmethod
    def solve_triangular(Amat: np.ndarray, bvec: np.ndarray,
                         lower: bool = True) -> np.ndarray:
        return scipy.linalg.solve_triangular(Amat, bvec, lower=lower)

    @staticmethod
    def full(*args, dtype=float):
        return np.full(*args, dtype=dtype)

    @staticmethod
    def zeros(*args, dtype=float):
        return np.zeros(*args, dtype=dtype)

    @staticmethod
    def ones(*args, dtype=float):
        return np.ones(*args, dtype=dtype)

    @staticmethod
    def empty(*args, dtype=float):
        return np.empty(*args, dtype=dtype)

    @staticmethod
    def exp(matrix: np.ndarray) -> np.ndarray:
        return np.exp(matrix)

    @staticmethod
    def sqrt(matrix: np.ndarray) -> np.ndarray:
        return np.sqrt(matrix)

    @staticmethod
    def cos(matrix: np.ndarray) -> np.ndarray:
        return np.cos(matrix)

    @staticmethod
    def arccos(matrix: np.ndarray) -> np.ndarray:
        return np.arccos(matrix)

    @staticmethod
    def tan(matrix: np.ndarray) -> np.ndarray:
        return np.tan(matrix)

    @staticmethod
    def arctan(matrix: np.ndarray) -> np.ndarray:
        return np.arctan(matrix)

    @staticmethod
    def arctan2(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        return np.arctan2(matrix1, matrix2)

    @staticmethod
    def sin(matrix: np.ndarray) -> np.ndarray:
        return np.sin(matrix)

    @staticmethod
    def arcsin(matrix: np.ndarray) -> np.ndarray:
        return np.arcsin(matrix)

    @staticmethod
    def cosh(matrix: np.ndarray) -> np.ndarray:
        return np.cosh(matrix)

    @staticmethod
    def sinh(matrix: np.ndarray) -> np.ndarray:
        return np.sinh(matrix)

    @staticmethod
    def arccosh(matrix: np.ndarray) -> np.ndarray:
        return np.arccosh(matrix)

    @staticmethod
    def arcsinh(matrix: np.ndarray) -> np.ndarray:
        return np.arcsinh(matrix)

    @staticmethod
    def log(matrix: np.ndarray) -> np.ndarray:
        return np.log(matrix)

    @staticmethod
    def log10(matrix: np.ndarray) -> np.ndarray:
        return np.log10(matrix)

    @staticmethod
    def multidot(matrix_list: List[np.ndarray]) -> np.ndarray:
        return np.linalg.multi_dot(matrix_list)

    @staticmethod
    def prod(matrix_list: np.ndarray, axis=None) -> np.ndarray:
        return np.prod(matrix_list, axis=axis)

    @staticmethod
    def hstack(arrays) -> np.ndarray:
        return np.hstack(arrays)

    @staticmethod
    def vstack(arrays) -> np.ndarray:
        return np.vstack(arrays)

    @staticmethod
    def stack(arrays, axis=0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    @staticmethod
    def dstack(arrays) -> np.ndarray:
        return np.dstack(arrays)

    @staticmethod
    def arange(*args, **kwargs) -> np.ndarray:
        return np.arange(*args, **kwargs)

    @staticmethod
    def linspace(*args, **kwargs):
        return np.linspace(*args, **kwargs)

    @staticmethod
    def logspace(*args):
        return np.logspace(*args)

    @staticmethod
    def ndim(mat: np.ndarray) -> int:
        return mat.ndim

    @staticmethod
    def repeat(mat: np.ndarray, nreps: int) -> np.ndarray:
        return np.tile(mat, nreps)

    @staticmethod
    def tile(mat: np.ndarray, nreps) -> np.ndarray:
        return np.tile(mat, nreps)

    @staticmethod
    def cdist(Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return scipy.spatial.distance.cdist(Amat, Bmat, metric="euclidean")

    @staticmethod
    def einsum(*args) -> np.ndarray:
        return np.einsum(*args)

    @staticmethod
    def trace(mat: np.ndarray) -> float:
        return np.trace(mat)

    @staticmethod
    def copy(mat: np.ndarray) -> np.ndarray:
        return mat.copy()

    @staticmethod
    def get_diagonal(mat: np.ndarray) -> np.ndarray:
        return np.diagonal(mat)

    @staticmethod
    def diag(array, k=0):
        return np.diag(array, k=k)

    @staticmethod
    def isnan(mat: np.ndarray) -> np.ndarray:
        return np.isnan(mat)

    @staticmethod
    def atleast1d(val, dtype=float) -> np.ndarray:
        return np.atleast_1d(val).astype(dtype)

    @staticmethod
    def atleast2d(val, dtype=float) -> np.ndarray:
        return np.atleast_2d(val).astype(dtype)

    @staticmethod
    def reshape(mat: np.ndarray, newshape) -> np.ndarray:
        return np.reshape(mat, newshape)

    @staticmethod
    def where(cond: np.ndarray) -> np.ndarray:
        return np.where(cond)

    @staticmethod
    def tointeger(mat: np.ndarray) -> np.ndarray:
        return np.asarray(mat, dtype=int)

    @staticmethod
    def inf():
        return np.inf

    @staticmethod
    def norm(mat: np.ndarray, axis=None) -> np.ndarray:
        return np.linalg.norm(mat, axis=axis)

    @staticmethod
    def any(mat: np.ndarray, axis=None) -> np.ndarray:
        return np.any(mat, axis=axis)

    @staticmethod
    def all(mat: np.ndarray, axis=None) -> np.ndarray:
        return np.all(mat, axis=axis)

    @staticmethod
    def kron(Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return np.kron(Amat, Bmat)

    @staticmethod
    def slogdet(Amat: np.ndarray) -> np.ndarray:
        return np.linalg.slogdet(Amat)

    @staticmethod
    def mean(mat: np.ndarray, axis: int = None) -> np.ndarray:
        return np.mean(mat, axis=axis)

    @staticmethod
    def std(mat: np.ndarray, axis: int = None,
                ddof: int = 0) -> np.ndarray:
        return np.std(mat, axis=axis, ddof=ddof)

    @staticmethod
    def cov(mat: np.ndarray, ddof=0, rowvar=True) -> np.ndarray:
        return np.cov(mat, ddof=ddof, rowvar=rowvar)

    @staticmethod
    def abs(mat: np.ndarray) -> np.ndarray:
        return np.absolute(mat)

    @staticmethod
    def to_numpy(mat: np.ndarray) -> np.ndarray:
        return mat

    @staticmethod
    def argsort(mat: np.ndarray, axis=-1) -> np.ndarray:
        return np.argsort(mat, axis=axis)

    @staticmethod
    def sort(mat: np.ndarray, axis=-1) -> np.ndarray:
        return np.sort(mat, axis=axis)

    @staticmethod
    def flip(mat, axis=None):
        return np.flip(mat, axis=axis)

    @staticmethod
    def allclose(Amat: np.ndarray, Bmat: np.ndarray,
                 **kwargs) -> bool:
        return np.allclose(Amat, Bmat, **kwargs)

    @staticmethod
    def lstsq(Amat, Bmat):
        return np.linalg.lstsq(Amat, Bmat, rcond=None)[0]

    @staticmethod
    def argmax(array):
        return np.argmax(array)

    @staticmethod
    def argmin(array):
        return np.argmin(array)

    @staticmethod
    def max(array, axis=None):
        return np.max(array, axis=axis)

    @staticmethod
    def maximum(array1, array2):
        return np.maximum(array1, array2)

    @staticmethod
    def minimum(array1, array2):
        return np.minimum(array1, array2)

    @staticmethod
    def min(array, axis=None):
        return np.min(array, axis=axis)

    @staticmethod
    def block(blocks):
        return np.block(blocks)

    @staticmethod
    def sum(matrix, axis=None):
        return np.sum(matrix, axis=axis)

    @staticmethod
    def count_nonzero(matrix, axis=None):
        return np.count_nonzero(matrix, axis=axis)

    @staticmethod
    def array(array, dtype=float):
        return np.array(array, dtype=dtype)

    @staticmethod
    def eigh(matrix):
        return np.linalg.eigh(matrix)

    @staticmethod
    def svd(matrix, full_matrices=True):
        return np.linalg.svd(
            matrix, compute_uv=True, full_matrices=full_matrices
        )

    @staticmethod
    def isfinite(matrix):
        return np.isfinite(matrix)

    @staticmethod
    def cond(matrix):
        return np.linalg.cond(matrix)

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
        raise ValueError("axis must be in (0, 1, -1)")

    @staticmethod
    def moveaxis(array, source, destination):
        return np.moveaxis(array, source, destination)

    @staticmethod
    def floor(array):
        return np.floor(array)

    @staticmethod
    def asarray(array, dtype=float):
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def unique(array, **kwargs):
        return np.unique(array, **kwargs)

    @staticmethod
    def delete(array, obj, axis=None):
        return np.delete(array, obj, axis=axis)

    @staticmethod
    def jacobian_implemented() -> bool:
        return False

    @staticmethod
    def hessian_implemented() -> bool:
        return False

    @staticmethod
    def meshgrid(*arrays, indexing="xy"):
        return np.meshgrid(*arrays, indexing=indexing)

    @staticmethod
    def tanh(array):
        return np.tanh(array)

    @staticmethod
    def diff(array):
        return np.diff(array)

    @staticmethod
    def int():
        return int

    @staticmethod
    def cumsum(array, axis=0, **kwargs):
        assert axis is not None
        return np.cumsum(array, axis=axis, **kwargs)

    @staticmethod
    def complex_dtype():
        return complex

    @staticmethod
    def array_type():
        return np.ndarray

    @staticmethod
    def real(array):
        return np.real(array)

    @staticmethod
    def imag(array):
        return np.imag(array)

    @staticmethod
    def round(array):
        return np.round(array)

    @staticmethod
    def flatten(array: np.ndarray):
        return array.flatten()

    @staticmethod
    def gammaln(mat: np.ndarray) -> np.ndarray:
        return scipy.special.gammaln(mat)

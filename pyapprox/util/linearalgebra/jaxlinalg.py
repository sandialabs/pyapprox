from typing import List

import jax
import jax.numpy as np

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin


# jac treats float as float32 unless the following is set to True
jax.config.update("jax_enable_x64", True)


class JaxLinAlgMixin(LinAlgMixin):
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
    def cholesky_solve(
        chol: np.ndarray, bvec: np.ndarray, lower: bool = True
    ) -> np.ndarray:
        return jax.scipy.linalg.cho_solve((chol, lower), bvec)

    @staticmethod
    def solve_triangular(
        Amat: np.ndarray, bvec: np.ndarray, lower: bool = True
    ) -> np.ndarray:
        return jax.scipy.linalg.solve_triangular(Amat, bvec, lower=lower)

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
    def sin(matrix: np.ndarray) -> np.ndarray:
        return np.sin(matrix)

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
    def linspace(*args):
        return np.linspace(*args)

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
    def cdist(Amat: np.ndarray, Bmat: np.ndarray, eps=1e-14) -> np.ndarray:
        # jax has no cdist function
        # return np.spatial.distance.cdist(Amat, Bmat, metric="euclidean")
        sq_dists = np.sum((Amat[:, None] - Bmat[None, :]) ** 2, -1)
        # jac returns nan of gradient when sq_dists is zero
        # so apply a clipping. This is not perfect but I dont know another
        # solution
        return np.where(sq_dists < eps, 1e-36, np.sqrt(sq_dists))

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
        return np.atleast_1d(np.asarray(val)).astype(dtype)

    @staticmethod
    def atleast2d(val, dtype=float) -> np.ndarray:
        return np.atleast_2d(np.asarray(val)).astype(dtype)

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
    def std(mat: np.ndarray, axis: int = None, ddof: int = 0) -> np.ndarray:
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
    def allclose(Amat: np.ndarray, Bmat: np.ndarray, **kwargs) -> bool:
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
    def isfinite(matrix):
        return np.isfinite(matrix)

    @staticmethod
    def cond(matrix):
        return np.linalg.cond(matrix)

    @staticmethod
    def up(matrix, indices, submatrix, axis=0):
        if axis == 0:
            return matrix.at[indices].set(submatrix)
        if axis == 1:
            return matrix.at[:, indices].set(submatrix)
        if axis == -1:
            return matrix.at[..., indices].set(submatrix)
        raise ValueError("axis must be in (0, 1, -1)")

    @staticmethod
    def jacobian(fun, params):
        return jax.jacfwd(fun)(params)

    @staticmethod
    def grad(self, fun, params):
        return jax.jacfwd(fun)(params)

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
        return True

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

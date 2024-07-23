from typing import List

import numpy as np
import scipy

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin


class NumpyLinAlgMixin(LinAlgMixin):
    def _la_dot(self, Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return np.dot(Amat, Bmat)

    def _la_eye(self, nrows: int, dtype=float) -> np.ndarray:
        return np.eye(nrows, dtype=dtype)

    def _la_inv(self, matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(matrix)

    def _la_cholesky(self, matrix: np.ndarray) -> np.ndarray:
        return np.linalg.cholesky(matrix)

    def _la_cholesky_solve(self, chol: np.ndarray, bvec: np.ndarray,
                           lower: bool = True) -> np.ndarray:
        return scipy.linalg.cho_solve((chol, lower), bvec)

    def _la_solve_triangular(self, Amat: np.ndarray, bvec: np.ndarray,
                             lower: bool = True) -> np.ndarray:
        return scipy.linalg.solve_triangular(Amat, bvec, lower=lower)

    def _la_full(self, *args, dtype=float):
        return np.full(*args, dtype=dtype)

    def _la_zeros(self, *args, dtype=float):
        return np.zeros(*args, dtype=dtype)

    def _la_ones(self, *args, dtype=float):
        return np.ones(*args, dtype=dtype)

    def _la_empty(self, *args, dtype=float):
        return np.empty(*args, dtype=dtype)

    def _la_exp(self, matrix: np.ndarray) -> np.ndarray:
        return np.exp(matrix)

    def _la_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        return np.sqrt(matrix)

    def _la_cos(self, matrix: np.ndarray) -> np.ndarray:
        return np.cos(matrix)

    def _la_arccos(self, matrix: np.ndarray) -> np.ndarray:
        return np.arccos(matrix)

    def _la_sin(self, matrix: np.ndarray) -> np.ndarray:
        return np.sin(matrix)

    def _la_log(self, matrix: np.ndarray) -> np.ndarray:
        return np.log(matrix)

    def _la_multidot(self, matrix_list: List[np.ndarray]) -> np.ndarray:
        return np.linalg.multi_dot(matrix_list)

    def _la_prod(self, matrix_list: np.ndarray, axis=None) -> np.ndarray:
        return np.prod(matrix_list, axis=axis)

    def _la_hstack(self, arrays) -> np.ndarray:
        return np.hstack(arrays)

    def _la_vstack(self, arrays) -> np.ndarray:
        return np.vstack(arrays)

    def _la_stack(self, arrays, axis=0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def _la_dstack(self, arrays) -> np.ndarray:
        return np.dstack(arrays)

    def _la_arange(self, *args, **kwargs) -> np.ndarray:
        return np.arange(*args)

    def _la_linspace(self, *args):
        return np.linspace(*args)

    def _la_ndim(self, mat: np.ndarray) -> int:
        return mat.ndim

    def _la_repeat(self, mat: np.ndarray, nreps: int) -> np.ndarray:
        return np.tile(mat, nreps)

    def _la_cdist(self, Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return scipy.spatial.distance.cdist(Amat, Bmat, metric="euclidean")

    def _la_einsum(self, *args) -> np.ndarray:
        return np.einsum(*args)

    def _la_trace(self, mat: np.ndarray) -> float:
        return np.trace(mat)

    def _la_copy(self, mat: np.ndarray) -> np.ndarray:
        return mat.copy()

    def _la_get_diagonal(self, mat: np.ndarray) -> np.ndarray:
        return np.diagonal(mat)

    def _la_diag(self, array, k=0):
        return np.diag(array, k=k)

    def _la_isnan(self, mat: np.ndarray) -> np.ndarray:
        return np.isnan(mat)

    def _la_atleast1d(self, val, dtype=float) -> np.ndarray:
        return np.atleast_1d(val).astype(dtype)

    def _la_atleast2d(self, val, dtype=float) -> np.ndarray:
        return np.atleast_2d(val).astype(dtype)

    def _la_reshape(self, mat: np.ndarray, newshape) -> np.ndarray:
        return np.reshape(mat, newshape)

    def _la_where(self, cond: np.ndarray) -> np.ndarray:
        return np.where(cond)

    def _la_tointeger(self, mat: np.ndarray) -> np.ndarray:
        return np.asarray(mat, dtype=int)

    def _la_inf(self):
        return np.inf

    def _la_norm(self, mat: np.ndarray, axis=None) -> np.ndarray:
        return np.linalg.norm(mat, axis=axis)

    def _la_any(self, mat: np.ndarray, axis=None) -> np.ndarray:
        return np.any(mat, axis=axis)

    def _la_all(self, mat: np.ndarray, axis=None) -> np.ndarray:
        return np.all(mat, axis=axis)

    def _la_kron(self, Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return np.kron(Amat, Bmat)

    def _la_slogdet(self, Amat: np.ndarray) -> np.ndarray:
        return np.linalg.slogdet(Amat)

    def _la_mean(self, mat: np.ndarray, axis: int = None) -> np.ndarray:
        return np.mean(mat, axis=axis)

    def _la_std(self, mat: np.ndarray, axis: int = None,
                ddof: int = 0) -> np.ndarray:
        return np.std(mat, axis=axis, ddof=ddof)

    def _la_cov(self, mat: np.ndarray, ddof=0, rowvar=True) -> np.ndarray:
        return np.cov(mat, ddof=ddof, rowvar=rowvar)

    def _la_abs(self, mat: np.ndarray) -> np.ndarray:
        return np.absolute(mat)

    def _la_to_numpy(self, mat: np.ndarray) -> np.ndarray:
        return mat

    def _la_argsort(self, mat: np.ndarray, axis=-1) -> np.ndarray:
        return np.argsort(mat, axis=axis)

    def _la_sort(self, mat: np.ndarray, axis=-1) -> np.ndarray:
        return np.sort(mat, axis=axis)

    def _la_flip(self, mat, axis=None):
        return np.flip(mat, axis=axis)

    def _la_allclose(self, Amat: np.ndarray, Bmat: np.ndarray,
                     **kwargs) -> bool:
        return np.allclose(Amat, Bmat, **kwargs)

    def _la_lstsq(self, Amat, Bmat):
        return np.linalg.lstsq(Amat, Bmat, rcond=None)[0]

    def _la_argmax(self, array):
        return np.argmax(array)

    def _la_max(self, array, axis=None):
        return np.max(array, axis=axis)

    def _la_min(self, array, axis=None):
        return np.min(array, axis=axis)

    def _la_block(self, blocks):
        return np.block(blocks)

    def _la_sum(self, matrix, axis=None):
        return np.sum(matrix, axis=axis)

    def _la_count_nonzero(self, matrix, axis=None):
        return np.count_nonzero(matrix, axis=axis)

    def _la_array(self, array, dtype=float):
        return np.array(array, dtype=dtype)

    def _la_eigh(self, matrix):
        return np.linalg.eigh(matrix)

    def _la_isfinite(self, matrix):
        return np.isfinite(matrix)

    def _la_cond(self, matrix):
        return np.linalg.cond(matrix)

from typing import List

import jax
import jax.numpy as np

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin


# jac treats float as float32 unless the following is set to True
jax.config.update("jax_enable_x64", True)


class JaxLinAlgMixin(LinAlgMixin):
    @staticmethod
    def _la_dot(Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return np.dot(Amat, Bmat)

    @staticmethod
    def _la_eye(nrows: int, dtype=float) -> np.ndarray:
        return np.eye(nrows, dtype=dtype)

    @staticmethod
    def _la_inv(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(matrix)

    @staticmethod
    def _la_pinv(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(matrix)

    @staticmethod
    def _la_solve(Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return np.linalg.solve(Amat, Bmat)

    @staticmethod
    def _la_cholesky(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.cholesky(matrix)

    @staticmethod
    def _la_cholesky_solve(
        chol: np.ndarray, bvec: np.ndarray, lower: bool = True
    ) -> np.ndarray:
        return jax.scipy.linalg.cho_solve((chol, lower), bvec)

    @staticmethod
    def _la_solve_triangular(
        Amat: np.ndarray, bvec: np.ndarray, lower: bool = True
    ) -> np.ndarray:
        return jax.scipy.linalg.solve_triangular(Amat, bvec, lower=lower)

    @staticmethod
    def _la_full(*args, dtype=float):
        return np.full(*args, dtype=dtype)

    @staticmethod
    def _la_zeros(*args, dtype=float):
        return np.zeros(*args, dtype=dtype)

    @staticmethod
    def _la_ones(*args, dtype=float):
        return np.ones(*args, dtype=dtype)

    @staticmethod
    def _la_empty(*args, dtype=float):
        return np.empty(*args, dtype=dtype)

    @staticmethod
    def _la_exp(matrix: np.ndarray) -> np.ndarray:
        return np.exp(matrix)

    @staticmethod
    def _la_sqrt(matrix: np.ndarray) -> np.ndarray:
        return np.sqrt(matrix)

    @staticmethod
    def _la_cos(matrix: np.ndarray) -> np.ndarray:
        return np.cos(matrix)

    @staticmethod
    def _la_arccos(matrix: np.ndarray) -> np.ndarray:
        return np.arccos(matrix)

    @staticmethod
    def _la_sin(matrix: np.ndarray) -> np.ndarray:
        return np.sin(matrix)

    @staticmethod
    def _la_log(matrix: np.ndarray) -> np.ndarray:
        return np.log(matrix)

    @staticmethod
    def _la_log10(matrix: np.ndarray) -> np.ndarray:
        return np.log10(matrix)

    @staticmethod
    def _la_multidot(matrix_list: List[np.ndarray]) -> np.ndarray:
        return np.linalg.multi_dot(matrix_list)

    @staticmethod
    def _la_prod(matrix_list: np.ndarray, axis=None) -> np.ndarray:
        return np.prod(matrix_list, axis=axis)

    @staticmethod
    def _la_hstack(arrays) -> np.ndarray:
        return np.hstack(arrays)

    @staticmethod
    def _la_vstack(arrays) -> np.ndarray:
        return np.vstack(arrays)

    @staticmethod
    def _la_stack(arrays, axis=0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    @staticmethod
    def _la_dstack(arrays) -> np.ndarray:
        return np.dstack(arrays)

    @staticmethod
    def _la_arange(*args, **kwargs) -> np.ndarray:
        return np.arange(*args)

    @staticmethod
    def _la_linspace(*args):
        return np.linspace(*args)

    @staticmethod
    def _la_logspace(*args):
        return np.logspace(*args)

    @staticmethod
    def _la_ndim(mat: np.ndarray) -> int:
        return mat.ndim

    @staticmethod
    def _la_repeat(mat: np.ndarray, nreps: int) -> np.ndarray:
        return np.tile(mat, nreps)

    @staticmethod
    def _la_cdist(Amat: np.ndarray, Bmat: np.ndarray, eps=1e-14) -> np.ndarray:
        # jax has no cdist function
        # return np.spatial.distance.cdist(Amat, Bmat, metric="euclidean")
        sq_dists = np.sum((Amat[:, None] - Bmat[None, :]) ** 2, -1)
        # jac returns nan of gradient when sq_dists is zero
        # so apply a clipping. This is not perfect but I dont know another
        # solution
        return np.where(sq_dists < eps, 1e-36, np.sqrt(sq_dists))

    @staticmethod
    def _la_einsum(*args) -> np.ndarray:
        return np.einsum(*args)

    @staticmethod
    def _la_trace(mat: np.ndarray) -> float:
        return np.trace(mat)

    @staticmethod
    def _la_copy(mat: np.ndarray) -> np.ndarray:
        return mat.copy()

    @staticmethod
    def _la_get_diagonal(mat: np.ndarray) -> np.ndarray:
        return np.diagonal(mat)

    @staticmethod
    def _la_diag(array, k=0):
        return np.diag(array, k=k)

    @staticmethod
    def _la_isnan(mat: np.ndarray) -> np.ndarray:
        return np.isnan(mat)

    @staticmethod
    def _la_atleast1d(val, dtype=float) -> np.ndarray:
        return np.atleast_1d(np.asarray(val)).astype(dtype)

    @staticmethod
    def _la_atleast2d(val, dtype=float) -> np.ndarray:
        return np.atleast_2d(np.asarray(val)).astype(dtype)

    @staticmethod
    def _la_reshape(mat: np.ndarray, newshape) -> np.ndarray:
        return np.reshape(mat, newshape)

    @staticmethod
    def _la_where(cond: np.ndarray) -> np.ndarray:
        return np.where(cond)

    @staticmethod
    def _la_tointeger(mat: np.ndarray) -> np.ndarray:
        return np.asarray(mat, dtype=int)

    @staticmethod
    def _la_inf():
        return np.inf

    @staticmethod
    def _la_norm(mat: np.ndarray, axis=None) -> np.ndarray:
        return np.linalg.norm(mat, axis=axis)

    @staticmethod
    def _la_any(mat: np.ndarray, axis=None) -> np.ndarray:
        return np.any(mat, axis=axis)

    @staticmethod
    def _la_all(mat: np.ndarray, axis=None) -> np.ndarray:
        return np.all(mat, axis=axis)

    @staticmethod
    def _la_kron(Amat: np.ndarray, Bmat: np.ndarray) -> np.ndarray:
        return np.kron(Amat, Bmat)

    @staticmethod
    def _la_slogdet(Amat: np.ndarray) -> np.ndarray:
        return np.linalg.slogdet(Amat)

    @staticmethod
    def _la_mean(mat: np.ndarray, axis: int = None) -> np.ndarray:
        return np.mean(mat, axis=axis)

    @staticmethod
    def _la_std(mat: np.ndarray, axis: int = None, ddof: int = 0) -> np.ndarray:
        return np.std(mat, axis=axis, ddof=ddof)

    @staticmethod
    def _la_cov(mat: np.ndarray, ddof=0, rowvar=True) -> np.ndarray:
        return np.cov(mat, ddof=ddof, rowvar=rowvar)

    @staticmethod
    def _la_abs(mat: np.ndarray) -> np.ndarray:
        return np.absolute(mat)

    @staticmethod
    def _la_to_numpy(mat: np.ndarray) -> np.ndarray:
        return mat

    @staticmethod
    def _la_argsort(mat: np.ndarray, axis=-1) -> np.ndarray:
        return np.argsort(mat, axis=axis)

    @staticmethod
    def _la_sort(mat: np.ndarray, axis=-1) -> np.ndarray:
        return np.sort(mat, axis=axis)

    @staticmethod
    def _la_flip(mat, axis=None):
        return np.flip(mat, axis=axis)

    @staticmethod
    def _la_allclose(Amat: np.ndarray, Bmat: np.ndarray, **kwargs) -> bool:
        return np.allclose(Amat, Bmat, **kwargs)

    @staticmethod
    def _la_lstsq(Amat, Bmat):
        return np.linalg.lstsq(Amat, Bmat, rcond=None)[0]

    @staticmethod
    def _la_argmax(array):
        return np.argmax(array)

    @staticmethod
    def _la_argmin(array):
        return np.argmin(array)

    @staticmethod
    def _la_max(array, axis=None):
        return np.max(array, axis=axis)

    @staticmethod
    def _la_min(array, axis=None):
        return np.min(array, axis=axis)

    @staticmethod
    def _la_block(blocks):
        return np.block(blocks)

    @staticmethod
    def _la_sum(matrix, axis=None):
        return np.sum(matrix, axis=axis)

    @staticmethod
    def _la_count_nonzero(matrix, axis=None):
        return np.count_nonzero(matrix, axis=axis)

    @staticmethod
    def _la_array(array, dtype=float):
        return np.array(array, dtype=dtype)

    @staticmethod
    def _la_eigh(matrix):
        return np.linalg.eigh(matrix)

    @staticmethod
    def _la_isfinite(matrix):
        return np.isfinite(matrix)

    @staticmethod
    def _la_cond(matrix):
        return np.linalg.cond(matrix)

    @staticmethod
    def _la_up(matrix, indices, submatrix, axis=0):
        if axis == 0:
            return matrix.at[indices].set(submatrix)
        if axis == 1:
            return matrix.at[:, indices].set(submatrix)
        if axis == -1:
            return matrix.at[..., indices].set(submatrix)
        raise ValueError("axis must be in (0, 1, -1)")

    @staticmethod
    def _la_jacobian(fun, params):
        return jax.jacfwd(fun)(params)

    @staticmethod
    def _la_grad(self, fun, params):
        return jax.jacfwd(fun)(params)

    @staticmethod
    def _la_moveaxis(array, source, destination):
        return np.moveaxis(array, source, destination)

    @staticmethod
    def _la_floor(array):
        return np.floor(array)

    @staticmethod
    def _la_asarray(array, dtype=float):
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def _la_unique(array, **kwargs):
        return np.unique(array, **kwargs)

    @staticmethod
    def _la_delete(array, obj, axis=None):
        return np.delete(array, obj, axis=axis)

    @staticmethod
    def _la_jacobian_implemented() -> bool:
        return True

    @staticmethod
    def _la_meshgrid(*arrays, indexing="xy"):
        return np.meshgrid(*arrays, indexing=indexing)

    @staticmethod
    def _la_tanh(array):
        return np.tanh(array)

    @staticmethod
    def _la_diff(array):
        return np.diff(array)

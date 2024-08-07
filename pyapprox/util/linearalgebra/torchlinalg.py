from typing import List

import torch

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin


class TorchLinAlgMixin(LinAlgMixin):
    def __init__(self):
        # needed for autograd
        self._inputs = None

    @staticmethod
    def _la_dot(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return Amat @ Bmat

    @staticmethod
    def _la_eye(nrows: int, dtype=torch.double) -> torch.Tensor:
        return torch.eye(nrows, dtype=dtype)

    @staticmethod
    def _la_inv(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(matrix)

    @staticmethod
    def _la_pinv(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.pinv(matrix)

    @staticmethod
    def _la_solve(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(Amat, Bmat)

    @staticmethod
    def _la_cholesky(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(matrix)

    @staticmethod
    def _la_cholesky_solve(chol: torch.Tensor, bvec: torch.Tensor,
                           lower: bool = True) -> torch.Tensor:
        return torch.cholesky_solve(bvec, chol, upper=(not lower))

    @staticmethod
    def _la_solve_triangular(Amat: torch.Tensor, bvec: torch.Tensor,
                             lower: bool = True) -> torch.Tensor:
        return torch.linalg.solve_triangular(Amat, bvec, upper=(not lower))

    @staticmethod
    def _la_full(*args, dtype=torch.double):
        return torch.full(*args, dtype=dtype)

    @staticmethod
    def _la_zeros(*args, dtype=float):
        return torch.zeros(*args, dtype=dtype)

    @staticmethod
    def _la_ones(*args, dtype=float):
        return torch.ones(*args, dtype=dtype)

    @staticmethod
    def _la_empty(*args, dtype=torch.double):
        return torch.empty(*args, dtype=dtype)

    @staticmethod
    def _la_exp(matrix: torch.Tensor) -> torch.Tensor:
        return torch.exp(matrix)

    @staticmethod
    def _la_sqrt(matrix: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(matrix)

    @staticmethod
    def _la_cos(matrix: torch.Tensor) -> torch.Tensor:
        return torch.cos(matrix)

    @staticmethod
    def _la_arccos(matrix: torch.Tensor) -> torch.Tensor:
        return torch.arccos(matrix)

    @staticmethod
    def _la_sin(matrix: torch.Tensor) -> torch.Tensor:
        return torch.sin(matrix)

    @staticmethod
    def _la_log(matrix: torch.Tensor) -> torch.Tensor:
        return torch.log(matrix)

    @staticmethod
    def _la_log10(matrix: torch.Tensor) -> torch.Tensor:
        return torch.log10(matrix)

    @staticmethod
    def _la_multidot(matrix_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.linalg.multi_dot(matrix_list)

    @staticmethod
    def _la_prod(matrix_list: torch.Tensor, axis=None) -> torch.Tensor:
        return torch.prod(matrix_list, dim=axis)

    @staticmethod
    def _la_hstack(arrays) -> torch.Tensor:
        return torch.hstack(arrays)

    @staticmethod
    def _la_vstack(arrays) -> torch.Tensor:
        return torch.vstack(arrays)

    @staticmethod
    def _la_stack(arrays, axis=0) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def _la_dstack(arrays) -> torch.Tensor:
        return torch.dstack(arrays)

    @staticmethod
    def _la_arange(*args, dtype=torch.double) -> torch.Tensor:
        return torch.arange(*args, dtype=dtype)

    @staticmethod
    def _la_linspace(*args, dtype=torch.double):
        return torch.linspace(*args, dtype=dtype)

    @staticmethod
    def _la_logspace(*args, dtype=torch.double):
        return torch.logspace(*args, dtype=dtype)

    @staticmethod
    def _la_ndim(mat: torch.Tensor) -> int:
        return mat.ndim

    @staticmethod
    def _la_repeat(mat: torch.Tensor, nreps: int) -> torch.Tensor:
        return mat.repeat(nreps)

    @staticmethod
    def _la_cdist(Amat: torch.tensor,
                  Bmat: torch.tensor) -> torch.Tensor:
        return torch.cdist(Amat, Bmat, p=2)

    @staticmethod
    def _la_einsum(*args) -> torch.Tensor:
        return torch.einsum(*args)

    @staticmethod
    def _la_trace(mat: torch.Tensor) -> torch.Tensor:
        return torch.trace(mat)

    @staticmethod
    def _la_copy(mat: torch.Tensor) -> torch.Tensor:
        return mat.clone()

    @staticmethod
    def _la_get_diagonal(mat: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(mat)

    @staticmethod
    def _la_diag(array, k=0):
        return torch.diag(array, diagonal=k)

    @staticmethod
    def _la_isnan(mat) -> torch.Tensor:
        return torch.isnan(mat)

    @staticmethod
    def _la_atleast1d(val, dtype=torch.double) -> torch.Tensor:
        return torch.atleast_1d(
            torch.as_tensor(val, dtype=dtype))

    @staticmethod
    def _la_atleast2d(val, dtype=torch.double) -> torch.Tensor:
        return torch.atleast_2d(
            torch.as_tensor(val, dtype=dtype))

    @staticmethod
    def _la_reshape(mat: torch.Tensor, newshape) -> torch.Tensor:
        return torch.reshape(mat, newshape)

    @staticmethod
    def _la_where(cond: torch.Tensor) -> torch.Tensor:
        return torch.where(cond)

    @staticmethod
    def _la_detach(mat: torch.Tensor) -> torch.Tensor:
        return mat.detach()

    @staticmethod
    def _la_tointeger(mat: torch.Tensor) -> torch.Tensor:
        return mat.int()

    @staticmethod
    def _la_inf():
        return torch.inf

    @staticmethod
    def _la_norm(mat: torch.Tensor, axis=None) -> torch.Tensor:
        return torch.linalg.norm(mat, dim=axis)

    @staticmethod
    def _la_any(mat: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            return torch.any(mat)
        return torch.any(mat, dim=axis)

    @staticmethod
    def _la_all(mat: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            return torch.all(mat)
        return torch.all(mat, dim=axis)

    @staticmethod
    def _la_kron(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return torch.kron(Amat, Bmat)

    @staticmethod
    def _la_slogdet(Amat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.slogdet(Amat)

    @staticmethod
    def _la_mean(mat: torch.Tensor, axis: int = None) -> torch.Tensor:
        if axis is None:
            return torch.mean(mat)
        return torch.mean(mat, dim=axis)

    @staticmethod
    def _la_std(mat: torch.Tensor, axis: int = None,
                ddof: int = 0) -> torch.Tensor:
        if axis is None:
            return torch.std(mat, correction=ddof)
        return torch.std(mat, dim=axis, correction=ddof)

    @staticmethod
    def _la_cov(mat: torch.Tensor, ddof=0, rowvar=True) -> torch.Tensor:
        if rowvar:
            return torch.cov(mat, correction=ddof)
        return torch.cov(mat.T, correction=ddof)

    @staticmethod
    def _la_abs(mat: torch.Tensor) -> torch.Tensor:
        return torch.absolute(mat)

    @staticmethod
    def _la_to_numpy(mat: torch.Tensor):
        return mat.numpy()

    @staticmethod
    def _la_argsort(mat: torch.Tensor, axis=-1) -> torch.Tensor:
        return torch.argsort(mat, dim=axis)

    @staticmethod
    def _la_sort(mat: torch.Tensor, axis=-1) -> torch.Tensor:
        return torch.sort(mat, dim=axis)

    @staticmethod
    def _la_flip(mat: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            axis = (0,)
        return torch.flip(mat, dims=axis)

    @staticmethod
    def _la_allclose(Amat: torch.Tensor, Bmat: torch.Tensor,
                     **kwargs) -> bool:
        return torch.allclose(Amat, Bmat, **kwargs)

    @staticmethod
    def _la_lstsq(Amat, Bmat):
        return torch.linalg.lstsq(Amat, Bmat, rcond=None)[0]

    @staticmethod
    def _la_argmax(array):
        return torch.argmax(array)

    @staticmethod
    def _la_argmin(array):
        return torch.argmin(array)

    @staticmethod
    def _la_max(array, axis=None):
        # torch returns both max and indices
        return torch.max(array, dim=axis)[0]

    @staticmethod
    def _la_min(array, axis=None):
        # torch returns both min and indices
        return torch.min(array, dim=axis)[0]

    @staticmethod
    def _la_block(blocks):
        return torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0)

    @staticmethod
    def _la_sum(matrix, axis=None):
        return torch.sum(matrix, dim=axis)

    @staticmethod
    def _la_count_nonzero(matrix, axis=None):
        return torch.count_nonzero(matrix, dim=axis)

    @staticmethod
    def _la_array(array, dtype=torch.double):
        return torch.as_tensor(array, dtype=dtype)

    @staticmethod
    def _la_eigh(matrix):
        return torch.linalg.eigh(matrix)

    @staticmethod
    def _la_isfinite(matrix):
        return torch.isfinite(matrix)

    @staticmethod
    def _la_cond(matrix):
        return torch.linalg.cond(matrix)

    @staticmethod
    def _la_jacobian(fun, params):
        return torch.autograd.functional.jacobian(fun, params)

    @staticmethod
    def _la_grad(fun, params):
        params.requires_grad = True
        val = fun(params)
        val.backward()
        grad = __class__._la_copy(params.grad)
        params.grad.zero_()
        return val, grad

    @staticmethod
    def _la_up(matrix, indices, submatrix, axis=0):
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
    def _la_moveaxis(array, source, destination):
        return torch.moveaxis(array, source, destination)

    @staticmethod
    def _la_floor(array):
        return torch.floor(array)

    @staticmethod
    def _la_asarray(array, dtype=torch.double):
        return torch.as_tensor(array, dtype=dtype)

    @staticmethod
    def _la_unique(array, **kwargs):
        return torch.unique(array, **kwargs)

    @staticmethod
    def _la_delete(array, obj, axis=None):
        mask = __class__._la_ones(array.shape[axis], dtype=bool)
        mask[obj] = False
        if axis is None:
            axis = 0
        if axis == 0:
            return array[mask]
        if axis == 1:
            return array[:, mask]
        if axis == -1:
            return array[..., mask]
        raise NotImplementedError("axis must be in (0, 1, -1)")

    @staticmethod
    def _la_jacobian_implemented() -> bool:
        return True

    @staticmethod
    def _la_meshgrid(*arrays, indexing="xy"):
        return torch.meshgrid(*arrays, indexing=indexing)

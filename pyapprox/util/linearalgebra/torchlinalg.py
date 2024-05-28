from typing import List

import torch

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin


class TorchLinAlgMixin(LinAlgMixin):
    def _la_dot(self, Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return Amat @ Bmat

    def _la_eye(self, nrows: int) -> torch.Tensor:
        return torch.eye(nrows)

    def _la_inv(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(matrix)

    def _la_cholesky(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(matrix)

    def _la_cholesky_solve(self, chol: torch.Tensor, bvec: torch.Tensor,
                           lower: bool = True) -> torch.Tensor:
        return torch.cholesky_solve(bvec, chol, upper=(not lower))

    def _la_solve_triangular(self, Amat: torch.Tensor, bvec: torch.Tensor,
                             lower: bool = True) -> torch.Tensor:
        return torch.linalg.solve_triangular(Amat, bvec, upper=(not lower))

    def _la_full(self, *args, dtype=torch.double):
        return torch.full(*args, dtype=dtype)

    def _la_empty(self, *args, dtype=torch.double):
        return torch.empty(*args, dtype=dtype)

    def _la_exp(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.exp(matrix)

    def _la_sqrt(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(matrix)

    def _la_cos(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.cos(matrix)

    def _la_arccos(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.arccos(matrix)

    def _la_sin(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.sin(matrix)

    def _la_log(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.log(matrix)

    def _la_multidot(self, matrix_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.linalg.multi_dot(matrix_list)

    def _la_prod(self, matrix_list: torch.Tensor, axis=None) -> torch.Tensor:
        return torch.prod(matrix_list, dim=axis)

    def _la_hstack(self, arrays) -> torch.Tensor:
        return torch.hstack(arrays)

    def _la_vstack(self, arrays) -> torch.Tensor:
        return torch.vstack(arrays)

    def _la_dstack(self, arrays) -> torch.Tensor:
        return torch.dstack(arrays)

    def _la_arange(self, *args, dtype=torch.double) -> torch.Tensor:
        return torch.arange(*args, dtype=dtype)

    def _la_linspace(self, *args, dtype=torch.double):
        return torch.linspace(*args, dtype=dtype)

    def _la_ndim(self, mat: torch.Tensor) -> int:
        return mat.ndim

    def _la_repeat(self, mat: torch.Tensor, nreps: int) -> torch.Tensor:
        return mat.repeat(nreps)

    def _la_cdist(self, Amat: torch.tensor,
                  Bmat: torch.tensor) -> torch.Tensor:
        return torch.cdist(Amat, Bmat, p=2)

    def _la_einsum(self, *args) -> torch.Tensor:
        return torch.einsum(*args)

    def _la_trace(self, mat: torch.Tensor) -> torch.Tensor:
        return torch.trace(mat)

    def _la_copy(self, mat: torch.Tensor) -> torch.Tensor:
        return mat.clone()

    def _la_get_diagonal(self, mat: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(mat)

    def _la_isnan(self, mat) -> torch.Tensor:
        return torch.isnan(mat)

    def _la_atleast1d(self, val, dtype=torch.double) -> torch.Tensor:
        return torch.atleast_1d(
            torch.as_tensor(val, dtype=dtype))

    def _la_atleast2d(self, val, dtype=torch.double) -> torch.Tensor:
        return torch.atleast_2d(
            torch.as_tensor(val, dtype=dtype))

    def _la_reshape(self, mat: torch.Tensor, newshape) -> torch.Tensor:
        return torch.reshape(mat, newshape)

    def _la_where(self, cond: torch.Tensor) -> torch.Tensor:
        return torch.where(cond)

    def _la_detach(self, mat: torch.Tensor) -> torch.Tensor:
        return mat.detach()

    def _la_tointeger(self, mat: torch.Tensor) -> torch.Tensor:
        return mat.int()

    def _la_inf(self):
        return torch.inf

    def _la_norm(self, mat: torch.Tensor, axis=None) -> torch.Tensor:
        return torch.linalg.norm(mat, dim=axis)

    def _la_any(self, mat: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            return torch.any(mat)
        return torch.any(mat, dim=axis)

    def _la_all(self, mat: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            return torch.all(mat)
        return torch.all(mat, dim=axis)

    def _la_kron(self, Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        return torch.kron(Amat, Bmat)

    def _la_slogdet(self, Amat: torch.Tensor) -> torch.Tensor:
        return torch.linalg.slogdet(Amat)

    def _la_mean(self, mat: torch.Tensor, axis: int = None) -> torch.Tensor:
        if axis is None:
            return torch.mean(mat)
        return torch.mean(mat, dim=axis)

    def _la_std(self, mat: torch.Tensor, axis: int = None,
                ddof: int = 0) -> torch.Tensor:
        if axis is None:
            return torch.std(mat, correction=ddof)
        return torch.std(mat, dim=axis, correction=ddof)

    def _la_cov(self, mat: torch.Tensor, ddof=0, rowvar=True) -> torch.Tensor:
        if rowvar:
            return torch.cov(mat, correction=ddof)
        return torch.cov(mat.T, correction=ddof)

    def _la_abs(self, mat: torch.Tensor) -> torch.Tensor:
        return torch.absolute(mat)

    def _la_to_numpy(self, mat: torch.Tensor):
        return mat.numpy()

    def _la_argsort(self, mat: torch.Tensor, axis=-1) -> torch.Tensor:
        return torch.argsort(mat, dim=axis)

    def _la_sort(self, mat: torch.Tensor, axis=-1) -> torch.Tensor:
        return torch.sort(mat, dim=axis)

    def _la_flip(self, mat: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            axis = (0,)
        return torch.flip(mat, dims=axis)

    def _la_allclose(self, Amat: torch.Tensor, Bmat: torch.Tensor,
                     **kwargs) -> bool:
        return torch.allclose(Amat, Bmat, **kwargs)

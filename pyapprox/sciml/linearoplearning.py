import numpy as np

from pyapprox.sciml.util.hyperparameter import (
    HyperParameter, HyperParameterList, IdentityHyperParameterTransform)
from pyapprox.sciml.kernels import HilbertSchmidtBasis
from pyapprox.sciml.util._torch_wrappers import (solve, einsum)


class HilbertSchmidtLinearOperator():
    def __init__(self, basis:  HilbertSchmidtBasis):
        self._basis = basis
        self._nbasis_terms = self._basis.nterms()**2
        coef = np.zeros((self._nbasis_terms))
        coef_bounds = [-np.inf, np.inf]
        self._coef = HyperParameter(
            "coef", coef.shape[0], coef, coef_bounds,
            IdentityHyperParameterTransform())
        self._hyp_list = HyperParameterList([self._coef])

    def _set_coefficients(self, active_coef):
        assert active_coef.ndim == 2 and active_coef.shape[1] == 1
        self._hyp_list.set_active_opt_params(active_coef[:, 0])

    def _deterministic_inner_product(self, values1, values2):
        # take inner product over ndof
        # values1 (ndof, nsamples1)
        # values2 (ndof, nsamples2)
        quad_w = self._basis.quadrule()[1]
        if values1.shape[0] != values2.shape[0]:
            raise ValueError(
                "values1.shape {0}".format(values1.shape) +
                " does not match values2.shape {0}".format(
                    values2.shape))
        integral = einsum("ij,ik->kj", quad_w*values1, values2)
        # Keep the following to show what einsum is doing
        # nsamples1, nsamples2 = values1.shape[1], values2.shape[1]
        # integral = np.empty((nsamples1, nsamples2))
        # for ii in range(nsamples1):
        #     for jj in range(nsamples2):
        #         integral[ii, jj] = np.sum(
        #             values1[:, ii]*values2[:, jj]*quad_w[:, 0])
        # integral = integral.T
        return integral

    def _basis_matrix(self, out_points, in_values):
        # out_points (nin_vars, nout_dof)
        # in_fun_values (nin_dof x nsamples)
        quad_x = self._basis.quadrule()[0]
        # out_basis_vals (nout_dof, nout_basis)
        out_basis_vals = self._basis(out_points)
        # in_prods (nsamples, nin_basis)
        in_prods = self._deterministic_inner_product(
            self._basis(quad_x), in_values)
        # outerproduct of inner and outer basis functions
        basis_matrix = einsum(
            "ij,kl->jlik", out_basis_vals, in_prods)
        nout_dof = out_points.shape[1]
        nsamples = in_values.shape[1]
        basis_matrix = basis_matrix.reshape(
            self._nbasis_terms, nout_dof, nsamples)
        # Keep the following to show what einsum and reshape is doing
        # basis_matrix (nbasis, nout_dof, nsamples)
        # basis_matrix = np.empty((self._nbasis_terms, nout_dof, nsamples))
        # cnt = 0
        # for ii in range(nin_basis):
        #     for jj in range(nout_basis):
        #         basis_matrix[cnt, :, :] = (
        #             out_basis_vals[:, jj:jj+1] @ in_prods[:, ii:ii+1].T)
        #         cnt += 1
        return basis_matrix

    def __call__(self, in_fun_values, out_points):
        # in_fun_values must be computed at self._in_quadrule[0]
        # basis_matrix (nbasis, nout_dof, nsamples)
        basis_mat = self._basis_matrix(out_points, in_fun_values)
        vals = einsum("ijk,i->jk", basis_mat, self._hyp_list.get_values())
        # Keep the following to show what einsum is doing
        # nout_dof = out_points.shape[1]
        # nsamples = in_fun_values.shape[1]
        # vals = np.empty((nout_dof, nsamples))
        # for ii in range(nout_dof):
        #     for jj in range(nsamples):
        #         vals[ii, jj] = basis_mat[:, ii, jj] @ self._coef[:, 0]
        return vals

    def _gram_matrix(self, basis_mat, out_weights):
        quad_w = self._basis.quadrule()[1]
        assert quad_w.ndim == 2 and quad_w.shape[1] == 1
        tmp = einsum(
            "ijk, ljk->ilk", basis_mat, quad_w[None, ...]*basis_mat)
        gram_mat = (tmp*out_weights[:, 0]).sum(axis=2)
        # Keep the following to show what einsum is doing
        # nbasis = basis_mat.shape[0]
        # gram_mat = np.empty((nbasis, nbasis))
        # for ii in range(nbasis):
        #     for jj in range(nbasis):
        #         gram_mat[ii, jj] = (np.sum(
        #             basis_mat[ii, ...]*quad_w*basis_mat[jj, ...],
        #             axis=0)*out_weights[:, 0]).sum(axis=0)
        return gram_mat

    def _rhs(self, train_out_values, basis_mat, out_weights):
        quad_w = self._basis.quadrule()[1]
        tmp = einsum(
            "ijk, jk->ik", basis_mat, quad_w*train_out_values)
        rhs = (tmp*out_weights[:, 0]).sum(axis=1)[:, None]
        # Keep the following to show what einsum is doing
        # nbasis = basis_mat.shape[0]
        # rhs = np.empty((nbasis, 1))
        # for ii in range(nbasis):
        #     tmp = (quad_w*basis_mat[ii, ...]*train_out_values).sum(axis=0)
        #     rhs[ii] = (tmp*out_weights[:, 0]).sum(axis=0)
        return rhs

    def fit(self, train_in_values, train_out_values, out_weights):
        quad_x = self._basis.quadrule()[0]
        basis_mat = self._basis_matrix(quad_x, train_in_values)
        gram_mat = self._gram_matrix(basis_mat, out_weights)
        rhs = self._rhs(train_out_values, basis_mat, out_weights)
        coef = solve(gram_mat, rhs)
        self._set_coefficients(coef)

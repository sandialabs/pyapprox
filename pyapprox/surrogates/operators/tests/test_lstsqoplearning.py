import unittest
import copy

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.operators.lstsqoplearning import (
    TensorOrthoPolyMultiLinearOperatorBasis, MultiLinearOperatorExpansion
)
from pyapprox.surrogates.bases.basis import (
    FixedTensorProductQuadratureRule
)
from pyapprox.surrogates.bases.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.bases.orthopoly import GaussQuadratureRule
from pyapprox.surrogates.bases.linearsystemsolvers import LstSqSolver


class BasisExpansionFunction:
    def __init__(self, bexp, fun_domain_points):
        self._bexp = bexp
        self._fun_domain_points = fun_domain_points

    def set_domain_points(self, fun_domain_points):
        self._fun_domain_points = fun_domain_points

    def __call__(self, coefs):
        # overwrite nqoi so we can pass in coefs with number of
        # columns given
        self._bexp._nqoi = coefs.shape[1]
        self._bexp.set_coefficients(coefs)
        return self._bexp(self._fun_domain_points)


def _evaluate_basis_expansion_functions(funs, coefs):
    fun_values = []
    cnt = 0
    for ii in range(len(funs)):
        fun = funs[ii]
        nterms = fun._bexp.basis.nterms()
        fun_values.append(
            fun(coefs[cnt:cnt+nterms])
        )
        cnt += nterms
    return fun_values


class TestLstSqOpLearning:
    def setUp(self):
        np.random.seed(1)

    def _setup_op_basis(
            self, nphys_vars, nin_funs, nout_funs, nin_terms_1d, nout_terms_1d,
            op_order, phys_marginal, coef_marginal
    ):
        bkd = self.get_backend()
        nterms_1d_per_infun = [[nin_terms_1d]*nphys_vars]*nin_funs

        op_basis = TensorOrthoPolyMultiLinearOperatorBasis(
            [[phys_marginal]*nphys_vars]*nin_funs,
            [[phys_marginal]*nphys_vars]*nout_funs,
            [[nin_terms_1d]*nphys_vars]*nin_funs,
            [[nout_terms_1d]*nphys_vars]*nout_funs,
        )
        ncoefs_per_infun = op_basis.ncoefficients_per_input_function()
        op_basis.set_basis(
            [[coef_marginal]*ncoefs for ncoefs in ncoefs_per_infun]
        )
        indices = bkd.cartesian_product(
            [bkd.arange(0, nt) for nt in [op_order+1]*sum(ncoefs_per_infun)]
        )
        # if op_order == 1:
        #    # linear op from paper
        #    indices = indices[:, indices.sum(axis=0) == 1]
        op_basis.set_coefficient_basis_indices(indices)
        return op_basis, nterms_1d_per_infun

    def test_multilinear_operator_basis(self):
        bkd = self.get_backend()
        nphys_vars, nin_funs, nout_funs = 1, 2, 1
        nin_terms_1d, nout_terms_1d = 2, 3
        op_order = 2
        phys_marginal = stats.uniform(-1, 2)
        coef_marginal = stats.norm(0, 1)
        op_basis, nterms_1d_per_infun = self._setup_op_basis(
            nphys_vars, nin_funs, nout_funs, nin_terms_1d, nout_terms_1d,
            op_order, phys_marginal, coef_marginal
        )

        # check out quad rules integrate output functions exactly.
        for ii in range(nout_funs):
            quad_samples, quad_weights = op_basis._out_quad_rules[ii]()
            basis_mat = op_basis._out_bases[ii](quad_samples)
            tmp = np.einsum(
                "ij, ik->jk", basis_mat, quad_weights*basis_mat)
            assert np.allclose(tmp, bkd.eye(basis_mat.shape[1]))

        # Test basis integrates input functions correctly. That is check
        # coefficients used to generate input functions are recovered.
        nfun_samples = 4
        # In general coef_marginal will be different for each coefficient as
        # there must be some decay on the coefficients of the input funs
        ncoef_dims = sum(
            [op_basis._in_bases[ii].nterms() for ii in range(nin_funs)]
        )
        coef_variable = IndependentMarginalsVariable(
            [coef_marginal]*ncoef_dims, backend=bkd
        )
        in_coefs = coef_variable.rvs(nfun_samples)

        in_funs = []
        for ii in range(nin_funs):
            bexp = PolynomialChaosExpansion(
                copy.deepcopy(op_basis._in_bases[ii])
            )
            bexp.basis.set_tensor_product_indices(nterms_1d_per_infun[ii])
            in_funs.append(
                BasisExpansionFunction(bexp, op_basis._in_quad_rules[ii]()[0])
            )

        infun_values = _evaluate_basis_expansion_functions(in_funs, in_coefs)

        recovered_in_coefs = op_basis._in_coef_from_infun_values(
            infun_values
        )
        assert bkd.allclose(
            recovered_in_coefs, in_coefs, rtol=1e-14, atol=1e-14
        )

        # Test grammian is computed correctly from basis
        ncoef_dims = sum(
            [op_basis._in_bases[ii].nterms() for ii in range(nin_funs)]
        )
        univariate_coef_quad_rules = [
            GaussQuadratureRule(coef_marginal) for jj in range(ncoef_dims)
        ]
        coef_quad_rule = FixedTensorProductQuadratureRule(
            ncoef_dims,
            univariate_coef_quad_rules,
            bkd.array([op_order*2]*ncoef_dims)
        )
        in_coef_quad_samples, in_coef_quad_weights = coef_quad_rule()

        infun_values = _evaluate_basis_expansion_functions(
            in_funs, in_coef_quad_samples
        )

        out_samples = [
            quad_rule()[0] for quad_rule in op_basis._out_quad_rules
        ]
        out_weights = [
            quad_rule()[1] for quad_rule in op_basis._out_quad_rules
        ]
        basis_mats = op_basis(infun_values, out_samples)
        for basis_mat, weights in zip(basis_mats, out_weights):
            np.set_printoptions(linewidth=1000)
            tmp = np.einsum(
                "ijk, ijl->jkl", basis_mat, weights[..., None]*basis_mat)
            gram_mat = (in_coef_quad_weights[..., None]*tmp).sum(axis=0)
            assert bkd.allclose(gram_mat, bkd.eye(gram_mat.shape[0]))

    def test_multilinear_operator_expansion(self):
        bkd = self.get_backend()
        nphys_vars, nin_funs, nout_funs = 1, 1, 1
        nin_terms_1d, nout_terms_1d = 3, 3
        op_order = 2
        phys_marginal = stats.uniform(-1, 2)
        coef_marginal = stats.norm(0, 1)
        op_basis, nterms_1d_per_infun = self._setup_op_basis(
            nphys_vars, nin_funs, nout_funs, nin_terms_1d, nout_terms_1d,
            op_order, phys_marginal, coef_marginal
        )

        in_funs = []
        for ii in range(nin_funs):
            bexp = PolynomialChaosExpansion(
                copy.deepcopy(op_basis._in_bases[ii])
            )
            bexp.basis.set_tensor_product_indices(nterms_1d_per_infun[ii])
            in_funs.append(
                BasisExpansionFunction(bexp, op_basis._in_quad_rules[ii]()[0])
            )

        nmc_samples = int(1e2)
        in_coef_variable = IndependentMarginalsVariable(
            [coef_marginal]*op_basis._coef_basis.nvars(), backend=bkd
        )
        in_coef_samples = in_coef_variable.rvs(nmc_samples)
        # in_coef_weights = bkd.full((nmc_samples, 1), 1/nmc_samples)

        # test only works when nin_terms_1d == nout_terms_1d
        infun_values = _evaluate_basis_expansion_functions(
            in_funs, in_coef_samples
        )
        out_coef_samples = bkd.copy(in_coef_samples)
        if op_order > 1:
            out_coef_samples[1, :] += (
                out_coef_samples[0, :] * out_coef_samples[1, :]
            )
            out_coef_samples[0, :] += 1
            out_funs = copy.deepcopy(in_funs)
            out_fun_values = _evaluate_basis_expansion_functions(
                out_funs, out_coef_samples)

        solver = LstSqSolver()
        op_exp = MultiLinearOperatorExpansion(op_basis, solver=solver)
        op_exp.fit(infun_values, out_fun_values)
        ntest_samples = 3
        plot_xx = bkd.linspace(-1, 1, 11)[None, :]
        ax = plt.figure().gca()
        test_infun_values = [
            vals[:, :ntest_samples] for vals in infun_values
        ]
        [out_fun.set_domain_points(plot_xx) for out_fun in out_funs]
        test_out_fun_values = _evaluate_basis_expansion_functions(
            out_funs, out_coef_samples
        )
        test_out_fun_values = [
            vals[:, :ntest_samples] for vals in test_out_fun_values
        ]
        ax.plot(plot_xx[0], op_exp(test_infun_values, [plot_xx])[0])
        ax.plot(plot_xx[0], test_out_fun_values[0], '--')
        # print(op_exp.basis.nterms())
        assert bkd.allclose(
            op_exp(test_infun_values, [plot_xx]), test_out_fun_values[0]
        )
        # plt.show()


class TestNumpyLstSqOpLearning(TestLstSqOpLearning, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest

import numpy as np
from scipy import stats

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.operators.lstsqoplearning import (
    MultiLinearOperatorBasis,
)
from pyapprox.surrogates.bases.orthopoly import (
    LegendrePolynomial1D,
    AffineMarginalTransform,
    GaussQuadratureRule,
    setup_univariate_orthogonal_polynomial_from_marginal
)
from pyapprox.surrogates.bases.basis import (
    OrthonormalPolynomialBasis,
    FixedTensorProductQuadratureRule,
)
from pyapprox.surrogates.bases.basisexp import PolynomialChaosExpansion


class TestLstsqOpLearning:
    def setUp(self):
        np.random.seed(1)

    def _setup_ortho_basis(self, marginal, nvars, nterms_1d=None):
        bkd = self.get_backend()
        polys_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=bkd
            )
            for ii in range(nvars)
        ]
        basis = OrthonormalPolynomialBasis(polys_1d)
        if nterms_1d is not None:
            basis.set_tensor_product_indices([nterms_1d] * nvars)
        return basis

    def _setup_ortho_expansion(
        self, marginal, nvars, nterms_1d, coefs=None
    ):
        bexp = PolynomialChaosExpansion(
            self._setup_ortho_basis(marginal, nvars, nterms_1d)
        )
        if coefs is not None:
            bexp.set_coefficients(coefs)
        return bexp

    def test_multi_linear_operator_basis(self):
        bkd = self.get_backend()
        nin_funs, nout_funs = 2, 1
        # nin_funs, nout_funs = 2, 2
        nin_terms_1d = 2
        nout_terms_1d = 3
        op_order = 2
        nvars = 1
        phys_marginal = stats.uniform(-1, 2)
        rand_marginal = stats.norm(0, 1)
        in_bases = [
            self._setup_ortho_basis(phys_marginal, nvars, nin_terms_1d)
            for ii in range(nin_funs)
        ]
        in_quad_rules = [
            FixedTensorProductQuadratureRule(
                nvars,
                [GaussQuadratureRule(phys_marginal) for jj in range(nvars)],
                bkd.array([2*nin_terms_1d]*nvars),
            )
            for ii in range(nin_funs)
        ]
        out_basis_exps = [
            self._setup_ortho_expansion(phys_marginal, nvars, nout_terms_1d)
            for ii in range(nout_funs)
        ]
        out_quad_rules = [
            FixedTensorProductQuadratureRule(
                nvars,
                [GaussQuadratureRule(phys_marginal) for jj in range(nvars)],
                bkd.array([2*nout_terms_1d]*nvars))
            for ii in range(nout_funs)
        ]

        for ii in range(nout_funs):
            quad_samples, quad_weights = out_quad_rules[ii]()
            basis_mat = out_basis_exps[ii].basis(quad_samples)
            tmp = np.einsum(
                "ij, ik->jk", basis_mat, quad_weights*basis_mat)
            assert np.allclose(tmp, bkd.eye(basis_mat.shape[1]))

        nin_coefs = bkd.sum(
            bkd.array([basis.nterms() for basis in in_bases], dtype=int)
        )
        coef_basis = self._setup_ortho_basis(rand_marginal, nin_coefs)
        indices = bkd.cartesian_product(
            [bkd.arange(0, nt) for nt in [op_order+1]*nin_coefs]
        )
        if op_order == 1:
            # linear op from paper
            indices = indices[:, indices.sum(axis=0) == 1]
        coef_basis.set_indices(indices)
        op_basis = MultiLinearOperatorBasis(
            nin_funs,
            in_bases,
            in_quad_rules,
            nout_funs,
            out_basis_exps,
            out_quad_rules,
            coef_basis,
        )

        nfun_samples = 4
        # In general rand_marginal will be different for each coefficient as
        # there must be some decay on the coefficients of the input funs
        ncoef_dims = sum([in_bases[ii].nterms() for ii in range(nin_funs)])
        coef_variable = IndependentMarginalsVariable(
            [rand_marginal]*ncoef_dims, backend=bkd
        )
        in_coefs = coef_variable.rvs(nfun_samples)

        class InFun:
            def __init__(self, bexp, quad_rule):
                self._bexp = bexp
                self._quad_rule = quad_rule

            def __call__(self, coefs):
                # overwrite nqoi so we can pass in coefs with number of
                # columns given
                self._bexp._nqoi = coefs.shape[1]
                self._bexp.set_coefficients(coefs)
                return self._bexp(self._quad_rule()[0])

        in_funs = [
            InFun(
                self._setup_ortho_expansion(
                    phys_marginal, nvars, nin_terms_1d
                ),
                quad_rule
            )
            for quad_rule in in_quad_rules
        ]
        in_fun_values = []
        cnt = 0
        for ii in range(nin_funs):
            in_fun_values.append(
                in_funs[ii](
                    in_coefs[cnt:cnt+in_bases[ii].nterms()]
                )
            )
            cnt += in_bases[ii].nterms()

        recovered_in_coefs = op_basis._in_coef_from_in_fun_values(
            in_fun_values
        )
        assert bkd.allclose(
            recovered_in_coefs, in_coefs, rtol=1e-14, atol=1e-14
        )

        ncoef_dims = sum([in_bases[ii].nterms() for ii in range(nin_funs)])
        univariate_coef_quad_rules = [
            GaussQuadratureRule(rand_marginal) for jj in range(ncoef_dims)
        ]
        coef_quad_rule = FixedTensorProductQuadratureRule(
            ncoef_dims,
            univariate_coef_quad_rules,
            bkd.array([op_order*2]*ncoef_dims)
        )
        in_coef_quad_samples, in_coef_quad_weights = coef_quad_rule()

        # nmc_samples = int(1e5)
        # in_coefs = variable.rvs(nmc_samples)
        # in_coef_weights = [
        #     bkd.full((nmc_samples, 1), 1/nmc_samples)
        #     for variable in coef_variables
        # ]

        in_fun_values = []
        cnt = 0
        for ii in range(nin_funs):
            in_fun_values.append(
                in_funs[ii](
                    in_coef_quad_samples[cnt:cnt+in_bases[ii].nterms()]
                )
            )
            cnt += in_bases[ii].nterms()

        out_samples = [quad_rule()[0] for quad_rule in out_quad_rules]
        out_weights = [quad_rule()[1] for quad_rule in out_quad_rules]
        basis_mats = op_basis(in_fun_values, out_samples)
        for basis_mat, weights in zip(basis_mats, out_weights):
            np.set_printoptions(linewidth=1000)
            tmp = np.einsum(
                "ijk, ijl->jkl", basis_mat, weights[..., None]*basis_mat)
            gram_mat = (in_coef_quad_weights[..., None]*tmp).sum(axis=0)
            assert bkd.allclose(gram_mat, bkd.eye(gram_mat.shape[0]))
      


class TestNumpyLstsqOpLearning(TestLstsqOpLearning, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

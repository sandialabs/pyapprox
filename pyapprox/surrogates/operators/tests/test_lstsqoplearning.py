import unittest

import numpy as np
from scipy import stats

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.operators.lstsqoplearning import (
    MultiLinearOperatorBasis,
)
from pyapprox.surrogates.bases.orthopoly import (
    LegendrePolynomial1D,
    AffineMarginalTransform,
    GaussQuadratureRule,
)
from pyapprox.surrogates.bases.basis import (
    OrthonormalPolynomialBasis,
    FixedTensorProductQuadratureRule,
)
from pyapprox.surrogates.bases.basisexp import PolynomialChaosExpansion


class TestLstsqOpLearning:
    def setUp(self):
        np.random.seed(1)

    def _setup_legendre_basis(self, marginal, nvars, nterms_1d=None):
        bkd = self.get_backend()
        trans = AffineMarginalTransform(
            marginal, enforce_bounds=True, backend=bkd
        )
        polys_1d = [
            LegendrePolynomial1D(trans=trans, backend=bkd)
            for ii in range(nvars)
        ]
        basis = OrthonormalPolynomialBasis(polys_1d)
        if nterms_1d is not None:
            basis.set_tensor_product_indices([nterms_1d] * nvars)
        return basis

    def _setup_legendre_expansion(
        self, marginal, nvars, nterms_1d, coefs=None
    ):
        bexp = PolynomialChaosExpansion(
            self._setup_legendre_basis(marginal, nvars, nterms_1d)
        )
        if coefs is not None:
            bexp.set_coefficients(coefs)
        return bexp

    def test_multi_linear_operator_basis(self):
        bkd = self.get_backend()
        nin_funs, nout_funs = 2, 2
        nterms_1d = 3
        nvars = 1
        marginal = stats.uniform(-1, 2)
        in_bases = [
            self._setup_legendre_basis(marginal, nvars, nterms_1d)
            for ii in range(nin_funs)
        ]
        in_quad_rules = [
            FixedTensorProductQuadratureRule(
                nvars,
                [GaussQuadratureRule(marginal) for jj in range(nvars)],
                bkd.array([2*nterms_1d]*nvars),
            )
            for ii in range(nin_funs)
        ]
        out_basis_exps = [
            self._setup_legendre_expansion(marginal, nvars, nterms_1d)
            for ii in range(nout_funs)
        ]
        out_quad_rules = [
            FixedTensorProductQuadratureRule(
                nvars,
                [GaussQuadratureRule(marginal) for jj in range(nvars)],
                bkd.array([2*nterms_1d]*nvars))
            for ii in range(nin_funs)
        ]
        nin_coefs = bkd.prod(
            bkd.array([basis.nterms() for basis in in_bases], dtype=int)
        )
        coef_basis = self._setup_legendre_basis(marginal, nin_coefs, None)
        op_basis = MultiLinearOperatorBasis(
            nin_funs,
            in_bases,
            in_quad_rules,
            nout_funs,
            out_basis_exps,
            out_quad_rules,
            coef_basis,
        )

        nfun_samples = 3
        in_coefs = [
            bkd.array(
                np.random.normal(0, 1, (nterms_1d**nvars, nfun_samples))
            )
            for ii in range(nin_funs)
        ]

        class InFun:
            def __init__(self, bexp, quad_rule):
                self._bexp = bexp
                self._quad_rule = quad_rule

            def __call__(self, coefs):
                print(self._bexp, coefs)
                self._bexp.set_coefficients(coefs)
                return self._bexp(self._quad_rule()[0])

        in_funs = [
            InFun(
                self._setup_legendre_expansion(
                    marginal, nvars, nterms_1d
                ),
                quad_rule
            )
            for quad_rule in in_quad_rules
        ]
        in_fun_values = [
            fun(coefs) for fun, coefs in zip(in_funs, in_coefs)
        ]

        recovered_in_coefs = op_basis._in_coef_from_in_fun_values(
            in_fun_values
        )
        print(recovered_in_coefs)
        print(recovered_in_coefs[0])
        # only recovered_in_coefs[0] will be correct
        assert bkd.allclose(
            recovered_in_coefs[0], in_coefs, rtol=1e-14, atol=1e-14
        )

        out_samples = bkd.linspace(*marginal.interval(1), 11)
        print(op_basis(in_fun_values, [out_samples]*nout_funs))


class TestNumpyLstsqOpLearning(TestLstsqOpLearning, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
from functools import partial

import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.univariate import (
    irregular_piecewise_linear_basis,
    irregular_piecewise_quadratic_basis,
    irregular_piecewise_cubic_basis,
    setup_univariate_piecewise_polynomial_basis,
    ClenshawCurtisQuadratureRule,
    UnivariateLagrangeBasis,
)


class TestUnivariateBasis:
    def setUp(self):
        np.random.seed(1)

    def test_irregular_piecewise_polynomial_basis(self):
        bkd = self.get_backend()

        def fun(xx):
            return bkd._la_sum(xx**2, axis=0)[:, None]

        nnodes = 11
        lb, ub = 0, 1
        # creat nodes with random spacing
        nodes = bkd._la_linspace(lb, ub, nnodes * 2)
        # fmt: off
        perm = bkd._la_asarray(
            np.random.permutation(2*nnodes-2), dtype=int
        )[:nnodes-2]
        nodes = bkd._la_sort(bkd._la_hstack((nodes[[0, -1]], nodes[1+perm])))
        # fmt: on

        # check basis interpolates values at nodes
        samples = nodes
        values = fun(nodes[None, :])
        basis = irregular_piecewise_linear_basis(nodes, samples, bkd=bkd)
        assert bkd._la_allclose(basis @ values, fun(samples[None, :]))

        # check basis accuracy is high with large nnodes
        nsamples = 31
        nnodes = 41
        nodes = bkd._la_linspace(lb, ub, nnodes)
        samples = bkd._la_asarray(np.random.uniform(lb, ub, (nsamples)))
        values = fun(nodes[None, :])
        basis = irregular_piecewise_linear_basis(nodes, samples, bkd=bkd)
        # check basis interpolates values at nodes
        assert bkd._la_allclose(
            basis @ values, fun(samples[None, :]), atol=2e-4
        )

        def fun(xx):
            return bkd._la_sum(xx**3, axis=0)[:, None]

        nnodes = 3
        lb, ub = 0, 1
        # create nodes with random spacing
        nodes = bkd._la_linspace(lb, ub, nnodes * 2)
        # fmt: off
        perm = bkd._la_asarray(
            np.random.permutation(2*nnodes-2), dtype=int
        )[:nnodes-2]
        nodes = bkd._la_sort(bkd._la_hstack((nodes[[0, -1]], nodes[1+perm])))
        # fmt: on

        # check basis interpolates values at nodes
        samples = nodes
        values = fun(nodes[None, :])
        basis = irregular_piecewise_quadratic_basis(nodes, samples, bkd=bkd)
        assert bkd._la_allclose(basis @ values, fun(samples[None, :]))

        # check basis accuracy is high with large nnodes
        nsamples = 31
        nnodes = 41
        nodes = bkd._la_linspace(lb, ub, nnodes)
        samples = bkd._la_asarray(np.random.uniform(lb, ub, (nsamples)))
        values = fun(nodes[None, :])
        basis = irregular_piecewise_quadratic_basis(nodes, samples, bkd=bkd)
        # check basis interpolates values at nodes
        assert bkd._la_allclose(
            basis @ values, fun(samples[None, :]), atol=1e-5
        )

        nnodes = 10
        nodes = bkd._la_linspace(lb, ub, nnodes * 2)
        # fmt: off
        perm = bkd._la_asarray(np.random.permutation(2*nnodes-2), dtype=int)[:nnodes-2]
        nodes = bkd._la_sort(bkd._la_hstack((nodes[[0, -1]], nodes[1+perm])))
        # fmt: on
        basis = irregular_piecewise_cubic_basis(nodes, nodes, bkd=bkd)
        values = fun(nodes[None, :])
        assert bkd._la_allclose(basis @ values, fun(nodes[None, :]))

        # check basis accuracy is high with large nnodes
        nsamples = 31
        nnodes = 34
        nodes = bkd._la_linspace(lb, ub, nnodes)
        samples = bkd._la_asarray(np.random.uniform(lb, ub, (nsamples)))
        values = fun(nodes[None, :])
        basis = irregular_piecewise_cubic_basis(nodes, samples, bkd=bkd)
        # check basis interpolates values at nodes
        assert bkd._la_allclose(
            basis @ values, fun(samples[None, :]), atol=1e-15
        )

    def _check_univariate_piecewise_polynomial_quadrature(
        self, name, degree, nnodes, tol
    ):
        bkd = self.get_backend()

        def fun(degree, xx):
            return bkd._la_sum(xx**degree, axis=0)[:, None]

        def integral(degree):
            if degree == 2:
                return 2 / 3
            if degree == 3:
                return 0
            if degree == 4:
                return 2 / 5

        bounds = [-1, 1]
        basis = setup_univariate_piecewise_polynomial_basis(
            name, bounds, backend=bkd)
        nodes = bkd._la_linspace(*bounds, nnodes)[None, :]
        basis.set_nodes(nodes)
        samples, weights = basis.quadrature_rule()
        assert np.allclose(
            fun(degree, samples).T @ weights, integral(degree), atol=tol
        )

        # randomize node spacing keeping both end points
        nodes = bkd._la_linspace(*bounds, 2 * nnodes)
        # fmt: off
        perm = bkd._la_asarray(
            np.random.permutation(2*nnodes-2), dtype=int
        )[:nnodes-2]
        nodes = bkd._la_sort(
            bkd._la_hstack((nodes[[0, -1]], nodes[1+perm]))
        )[None, :]
        # fmt: on
        basis.set_nodes(nodes)
        samples, weights = basis.quadrature_rule()
        assert np.allclose(
            fun(degree, samples).T @ weights, integral(degree), atol=tol
        )

    def test_univariate_piecewise_polynomial_quadrature(self):
        test_cases = [
            ["linear", 2, 101, 1e-3],
            ["quadratic", 2, 3, 1e-15],
            ["quadratic", 4, 91, 1e-5],
            ["cubic", 3, 4, 1e-15],
            ["cubic", 4, 46, 1e-5],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_univariate_piecewise_polynomial_quadrature(*test_case)

    def test_univariate_lagrange_basis_quadrature(self):
        nterms = 5
        bkd = self.get_backend()
        basis = UnivariateLagrangeBasis(
            ClenshawCurtisQuadratureRule(backend=bkd), nterms
        )

        def fun(degree, xx):
            return bkd._la_sum(xx**degree, axis=0)[:, None]

        def integral(degree):
            if degree == 2:
                return 1 / 3
            if degree == 3:
                return 0
            if degree == 4:
                return 1 / 5

        samples, weights = basis.quadrature_rule()
        degree = nterms-1
        assert np.allclose(
            fun(degree, samples).T @ weights, integral(degree)
        )


class TestNumpyUnivariateBasis(TestUnivariateBasis, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()


class TestTorchUnivariateBasis(TestUnivariateBasis, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)

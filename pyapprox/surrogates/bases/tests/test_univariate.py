import unittest

from scipy import stats
import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.univariate import (
    irregular_piecewise_linear_basis,
    irregular_piecewise_quadratic_basis,
    irregular_piecewise_cubic_basis,
    setup_univariate_piecewise_polynomial_basis,
    ClenshawCurtisQuadratureRule,
    UnivariateLagrangeBasis,
    ScipyUnivariateIntegrator,
    UnivariateUnboundedIntegrator,
)
from pyapprox.interface.model import ModelFromCallable


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

        # test semideep copy
        other = basis._semideep_copy()
        # check pointer to quad_rule is unchanged
        assert other != basis

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

        # test semideep copy
        other = basis._semideep_copy()
        # check pointer to quad_rule is unchanged
        assert other._quad_rule == basis._quad_rule
        assert other != basis

    def test_scipy_univariate_integrator(self):
        bkd = self.get_backend()
        bounds = [0, 1]
        integrator = ScipyUnivariateIntegrator(backend=bkd)
        integrator.set_bounds(bounds)
        integrand = ModelFromCallable(
            lambda x: bkd._la_cos(x[0])[:, None], backend=bkd
        )
        integrator.set_integrand(integrand)
        # use np to compare floats
        assert np.allclose(integrator(), np.sin(1))

        # check options are being passed to scipy.quad by using
        # set of options that will raise an error
        integrator.set_options(full_output=False, epsrel=-1, epsabs=-1)
        self.assertRaises(ValueError, integrator)

    def test_univariate_unbounded_integrator(self):
        # single qoi
        bkd = self.get_backend()
        marginal = stats.norm(0, 1)

        def integrand(sample):
            np_sample = bkd._la_to_numpy(sample)
            val = bkd._la_asarray(
                np_sample[0]**2*marginal.pdf(np_sample[0])
            )[:, None]
            return val
        quad_rule = ClenshawCurtisQuadratureRule(
            prob_measure=False, backend=bkd, store=True)
        integrator = UnivariateUnboundedIntegrator(quad_rule, backend=bkd)
        integrator.set_options(nquad_samples=2**3+1, maxiters=1000)
        integrator.set_bounds(marginal.interval(1))
        integrator.set_integrand(integrand)
        assert np.allclose(integrator(), 1)

        # multiple qoi
        marginal1 = stats.norm(0, 2)
        marginal2 = stats.expon(1)
        def integrand(sample):
            np_sample = bkd._la_to_numpy(sample)
            val1 = bkd._la_asarray(np_sample[0]**2*marginal1.pdf(np_sample[0]))
            val2 = bkd._la_asarray(np_sample[0]**2*marginal2.pdf(np_sample[0]))
            return bkd._la_stack([val1, val2], axis=1)
        quad_rule = ClenshawCurtisQuadratureRule(
            prob_measure=False, backend=bkd, store=True)
        integrator = UnivariateUnboundedIntegrator(quad_rule, backend=bkd)
        integrator.set_options(
            nquad_samples=2**3+1,
            maxiters=1000,
            maxinner_iters=6,
            interval_size=2
        )
        integrator.set_bounds(marginal.interval(1))
        integrator.set_integrand(integrand)
        integrals = integrator()
        assert np.allclose(
            integrals,
            [
                marginal1.var()+marginal1.mean()**2,
                marginal2.var()+marginal2.mean()**2
            ]
        )


class TestNumpyUnivariateBasis(TestUnivariateBasis, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()


# class TestTorchUnivariateBasis(TestUnivariateBasis, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)

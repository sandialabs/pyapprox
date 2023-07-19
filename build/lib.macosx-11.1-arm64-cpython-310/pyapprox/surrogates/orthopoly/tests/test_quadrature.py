import unittest
import numpy as np
from functools import partial
from scipy import stats
import sympy as sp

from pyapprox.surrogates.orthopoly.quadrature import (
    gauss_jacobi_pts_wts_1D, gauss_hermite_pts_wts_1D,
    clenshaw_curtis_pts_wts_1D, leja_growth_rule,
    constant_increment_growth_rule, gauss_quadrature,
    get_gauss_quadrature_rule_from_marginal
)
from pyapprox.variables.density import beta_pdf_on_ab, gaussian_pdf
from pyapprox.variables.marginals import (
    float_rv_discrete, get_probability_masses, transform_scale_parameters
)
from pyapprox.surrogates.orthopoly.leja_quadrature import (
    get_univariate_leja_quadrature_rule
    )
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    laguerre_recurrence
)


class TestQuadrature(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_gauss_jacobi_quadrature(self):
        """
        integrate x^2 x^a (1-x)^b/B(a+1,b+1) dx from x=0..1
        """
        alpha_poly = 0
        beta_poly = 0

        a = beta_poly+1
        b = alpha_poly+1

        true_mean = a/float(a+b)
        true_variance = a*b/float((a+b)**2*(a+b+1))

        x, w = gauss_jacobi_pts_wts_1D(2, alpha_poly, beta_poly)
        x = (x+1)/2.

        def function(x): return x**2
        assert np.allclose(np.dot(function(x), w)-true_mean**2, true_variance)

    def test_clenshaw_curtis_quadrature(self):
        a = 1
        b = 1

        true_mean = a/float(a+b)
        true_variance = a*b/float((a+b)**2*(a+b+1))

        x, w = clenshaw_curtis_pts_wts_1D(2)
        x = (x+1)/2.

        def function(x): return x**2
        assert np.allclose(np.dot(function(x), w)-true_mean**2, true_variance)

    def test_gauss_hermite_quadrature(self):
        """
        integrate x^2 1/sqrt(2*pi)exp(-x**2/2) dx from x=-inf..inf
        """

        true_mean = 0.
        true_variance = 1.

        x, w = gauss_hermite_pts_wts_1D(2)

        def function(x): return x**2
        assert np.allclose(np.dot(function(x), w)-true_mean**2, true_variance)

    def test_gaussian_leja_quadrature(self):
        level = 20
        quad_rule = get_univariate_leja_quadrature_rule(
            stats.norm(0, 1), leja_growth_rule,
            return_weights_for_all_levels=False)
        x_quad, w_quad = quad_rule(level)

        x = sp.Symbol('x')
        weight_function = gaussian_pdf(0, 1, x, sp)
        ranges = [-sp.oo, sp.oo]
        exact_integral = float(
            sp.integrate(weight_function*x**3, (x, ranges[0], ranges[1])))
        # print(exact_integral, x_quad, w_quad)
        assert np.allclose(exact_integral, np.dot(x_quad**3, w_quad))

    def test_beta_leja_quadrature(self):
        level = 12
        alpha_stat, beta_stat = 2, 10
        quad_rule = get_univariate_leja_quadrature_rule(
            stats.beta(alpha_stat, beta_stat), leja_growth_rule,
            return_weights_for_all_levels=False)
        x_quad, w_quad = quad_rule(level)

        x = sp.Symbol('x')
        weight_function = beta_pdf_on_ab(alpha_stat, beta_stat, -1, 1, x)
        ranges = [-1, 1]
        exact_integral = float(
            sp.integrate(weight_function*x**3, (x, ranges[0], ranges[1])))
        assert np.allclose(exact_integral, np.dot(x_quad**3, w_quad))

        level = 12
        alpha_stat, beta_stat = 2, 10
        x_quad, w_quad = quad_rule(level)
        x_quad = (x_quad+1)/2

        x = sp.Symbol('x')
        weight_function = beta_pdf_on_ab(alpha_stat, beta_stat, 0, 1, x)
        ranges = [0, 1]
        exact_integral = float(
            sp.integrate(weight_function*x**3, (x, ranges[0], ranges[1])))
        assert np.allclose(exact_integral, np.dot(x_quad**3, w_quad))

    def test_get_univariate_leja_rule_float_rv_discrete(self):
        nmasses = 20
        xk = np.array(range(1, nmasses+1), dtype='float')
        pk = np.ones(nmasses)/nmasses
        variable = float_rv_discrete(
            name='float_rv_discrete', values=(xk, pk))()

        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule,
            orthonormality_tol=1e-10,
            return_weights_for_all_levels=False)
        level = 3

        x, w = quad_rule(level)
        loc, scale = transform_scale_parameters(variable)
        x = x*scale+loc

        degree = x.shape[0]-1
        true_moment = (xk**degree).dot(pk)
        moment = (x**degree).dot(w)

        # print(moment, true_moment)
        assert np.allclose(moment, true_moment)

    def test_get_univariate_leja_rule_bounded_discrete(self):
        growth_rule = partial(constant_increment_growth_rule, 2)
        level = 3

        nmasses = 20
        xk = np.array(range(0, nmasses), dtype='float')
        pk = np.ones(nmasses)/nmasses
        var_cheb = float_rv_discrete(
            name='discrete_chebyshev', values=(xk, pk))()

        for variable in [var_cheb, stats.binom(17, 0.5),
                         stats.hypergeom(10+10, 10, 9)]:
            quad_rule = get_univariate_leja_quadrature_rule(
                variable, growth_rule)

            x, w = quad_rule(level)
            loc, scale = transform_scale_parameters(variable)
            x = x*scale+loc

            xk, pk = get_probability_masses(variable)
            print(x, xk, loc, scale)

            degree = (x.shape[0]-1)
            true_moment = (xk**degree).dot(pk)
            moment = (x**degree).dot(w[-1])

            print(moment, true_moment, variable.dist.name)
            assert np.allclose(moment, true_moment)

            # Note:
            # currently get_univariate_leja_quadrature_rule with christoffel
            # does not produce nested sequences without using initial_points

    def test_hermite_christoffel_leja_quadrature_rule(self):
        # warnings.filterwarnings('error')
        variable = stats.norm(2, 3)
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule, method='christoffel')
        level = 5
        samples, weights = quad_rule(level)
        # samples returned by quadrature rule will be on canonical domain
        # check first point was chosen correctly
        assert np.allclose(samples[0], (variable.ppf(0.75)-2)/3)
        # so integral is computed with resepect to standard normal
        assert np.allclose((samples**2).dot(weights[-1]), 1)

        # check samples are nested.
        samples1, weights1 = quad_rule(level+1)
        print(samples, samples1)
        assert np.allclose(samples1[:samples.shape[0]], samples)

    def test_uniform_christoffel_leja_quadrature_rule(self):
        # warnings.filterwarnings('error')
        variable = stats.uniform(-2, 3)
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule, method='christoffel')
        level = 5
        samples, weights = quad_rule(level)
        # samples returned by quadrature rule will be on canonical domain
        # check first point was chosen correctly
        assert np.allclose(samples[0], 2*(variable.ppf(0.5)+2)/3-1)
        # so integral is computed with resepect to uniform on [-1, 1]
        assert np.allclose((samples**2).dot(weights[-1]), 1/3)

    def test_hermite_pdf_weighted_leja_quadrature_rule(self):
        variable = stats.norm(2, 3)
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule, method='pdf')
        level = 5
        samples, weights = quad_rule(level)
        assert np.allclose(samples[0], (variable.ppf(0.75)-2)/3)
        assert np.allclose((samples**2).dot(weights[-1]), 1)

    def test_legendre_pdf_weighted_leja_quadrature_rule(self):
        variable = stats.uniform(-2, 3)
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule, method='pdf')
        level = 5
        samples, weights = quad_rule(level)
        assert np.allclose(samples[0], 2*(variable.ppf(0.5)+2)/3-1)
        assert np.allclose((samples**2).dot(weights[-1]), 1/3)

    def test_sampled_based_christoffel_leja_quadrature_rule(self):
        nsamples = int(1e4)
        samples = np.random.normal(0, 1, (1, nsamples))
        variable = float_rv_discrete(
            name='continuous_rv_sample',
            values=(samples[0, :], np.ones(nsamples)/nsamples))()
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule, method='christoffel',
            orthonormality_tol=1e-8)
        level = 5
        quad_samples, weights = quad_rule(level)
        # print(quad_samples)
        print((quad_samples**2).dot(weights[-1]))
        print((samples**2).mean())

        assert np.allclose(
            (quad_samples**2).dot(weights[-1]), (samples**2).mean())

    def test_gauss_laguerre_quadrature(self):
        a = 1
        rho = a-1
        N = 10

        true_mean = a
        true_variance = a

        ab = laguerre_recurrence(rho, N)
        print(ab)
        x, w = gauss_quadrature(ab, N)

        def function(x): return x**2
        # print(stats.gamma(a).mean())
        # print(x.dot(w), true_mean)
        # print((function(x).dot(w)-true_mean**2, true_variance))
        assert x.min() >= 0
        assert np.allclose(function(x).dot(w)-true_mean**2, true_variance)

    def test_get_univariate_quadrature_rule_from_marginal(self):
        max_nsamples = 10
        nsamples = 5
        marginal = stats.uniform(0, 1)
        canonical_quad_rule = get_gauss_quadrature_rule_from_marginal(
            marginal, max_nsamples, True)
        x, w = canonical_quad_rule(nsamples)
        x_exact, w_exact = gauss_jacobi_pts_wts_1D(nsamples, 0, 0)
        assert np.allclose(x, x_exact)
        assert np.allclose(w, w_exact)

        quad_rule = get_gauss_quadrature_rule_from_marginal(
            marginal, max_nsamples, False)
        x, w = quad_rule(nsamples)
        x_exact = (x_exact+1)/2
        assert np.allclose(x, x_exact)
        assert np.allclose(w, w_exact)

        max_nsamples = 10
        nsamples = 5
        marginal = stats.norm(1, 2)
        canonical_quad_rule = get_gauss_quadrature_rule_from_marginal(
            marginal, max_nsamples, True)
        x, w = canonical_quad_rule(nsamples)
        x_exact, w_exact = gauss_hermite_pts_wts_1D(nsamples)
        assert np.allclose(x, x_exact)
        assert np.allclose(w, w_exact)

        quad_rule = get_gauss_quadrature_rule_from_marginal(
            marginal, max_nsamples, False)
        x, w = quad_rule(nsamples)
        x_exact = 2*x_exact+1
        assert np.allclose(x, x_exact)
        assert np.allclose(w, w_exact)


if __name__ == "__main__":
    univariate_quadrature_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestQuadrature)
    unittest.TextTestRunner(verbosity=2).run(univariate_quadrature_test_suite)

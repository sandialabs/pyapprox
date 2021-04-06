import unittest
from pyapprox.univariate_quadrature import *
from scipy.special import gamma as gamma_fn
from scipy.special import beta as beta_fn
from pyapprox.utilities import beta_pdf_on_ab, gaussian_pdf
from pyapprox.variables import float_rv_discrete, get_distribution_info


class TestUnivariateQuadrature(unittest.TestCase):
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
        x_quad, w_quad = gaussian_leja_quadrature_rule(
            level, return_weights_for_all_levels=False)

        import sympy as sp
        x = sp.Symbol('x')
        weight_function = gaussian_pdf(0, 1, x, sp)
        ranges = [-sp.oo, sp.oo]
        exact_integral = float(
            sp.integrate(weight_function*x**3, (x, ranges[0], ranges[1])))
        assert np.allclose(exact_integral, np.dot(x_quad**3, w_quad))

    def test_beta_leja_quadrature(self):
        level = 12
        alpha_stat, beta_stat = 2, 10
        x_quad, w_quad = beta_leja_quadrature_rule(
            alpha_stat, beta_stat, level, return_weights_for_all_levels=False)

        import sympy as sp
        x = sp.Symbol('x')
        weight_function = beta_pdf_on_ab(alpha_stat, beta_stat, -1, 1, x)
        ranges = [-1, 1]
        exact_integral = float(
            sp.integrate(weight_function*x**3, (x, ranges[0], ranges[1])))
        assert np.allclose(exact_integral, np.dot(x_quad**3, w_quad))

        level = 12
        alpha_stat, beta_stat = 2, 10
        x_quad, w_quad = beta_leja_quadrature_rule(
            alpha_stat, beta_stat, level, return_weights_for_all_levels=False)
        x_quad = (x_quad+1)/2

        import sympy as sp
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
            numerically_generated_poly_accuracy_tolerance=1e-11)
        level = 3
        scales, shapes = get_distribution_info(variable)[1:]

        x, w = quad_rule(level)
        # x in [-1,1], scales for x in [0,1]
        loc, scale = scales['loc'], scales['scale']
        scale /= 2
        loc = loc+scale
        x = x*scale+loc

        true_moment = (xk**(x.shape[0]-1)).dot(pk)
        moment = (x**(x.shape[0]-1)).dot(w[-1])

        # print(moment)
        # print(true_moment)
        assert np.allclose(moment, true_moment)

    def test_get_univariate_leja_rule_bounded_discrete(self):
        from scipy import stats
        growth_rule = partial(constant_increment_growth_rule, 2)
        level = 3

        nmasses = 20
        xk = np.array(range(0, nmasses), dtype='float')
        pk = np.ones(nmasses)/nmasses
        var_cheb = float_rv_discrete(
            name='discrete_chebyshev', values=(xk, pk))()

        for variable in [var_cheb, stats.binom(20, 0.5),
                         stats.hypergeom(10+10, 10, 9)]:
            quad_rule = get_univariate_leja_quadrature_rule(
                variable, growth_rule)

            # polys of binom, hypergeometric have no canonical domain [-1,1]
            x, w = quad_rule(level)

            from pyapprox.variables import get_probability_masses
            xk, pk  =  get_probability_masses(variable)
            true_moment = (xk**(x.shape[0]-1)).dot(pk)
            moment = (x**(x.shape[0]-1)).dot(w[-1])
            
            assert np.allclose(moment, true_moment)

            # Note:
            # currently get_univariate_leja_quadrature_rule with christoffel
            # does not produce nested sequences without using initial_points

    def test_hermite_christoffel_leja_quadrature_rule(self):
        from scipy import stats
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
        samples1, weights1 = quad_rule(level+3)
        assert np.allclose(samples1[:samples.shape[0]], samples)

    def test_uniform_christoffel_leja_quadrature_rule(self):
        import warnings
        warnings.filterwarnings('error')
        from scipy import stats
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
        from scipy import stats
        variable = stats.norm(2, 3)
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule, method='pdf')
        level = 5
        samples, weights = quad_rule(level)
        assert np.allclose(samples[0], (variable.ppf(0.5)-2)/3)
        assert np.allclose((samples**2).dot(weights[-1]), 1)

    def test_hermite_pdf_weighted_leja_quadrature_rule(self):
        from scipy import stats
        variable = stats.uniform(-2, 3)
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule, method='pdf')
        level = 5
        samples, weights = quad_rule(level)
        assert np.allclose(samples[0], 2*(variable.ppf(0.5)+2)/3-1)
        assert np.allclose((samples**2).dot(weights[-1]), 1/3)

    def test_sampled_based_christoffel_leja_quadrature_rule(self):
        nsamples = int(1e6)
        samples = np.random.normal(0, 1, (1, nsamples))
        variable = float_rv_discrete(
            name='continuous_rv_sample',
            values=(samples[0, :], np.ones(nsamples)/nsamples))()
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule, method='christoffel',
            numerically_generated_poly_accuracy_tolerance=1e-8)
        level = 5
        quad_samples, weights = quad_rule(level)
        # print(quad_samples)
        # print((quad_samples**2).dot(weights[-1]))
        # print((samples**2).mean())

        assert np.allclose(
            (quad_samples**2).dot(weights[-1]), (samples**2).mean())


if __name__ == "__main__":
    #import warnings
    #warnings.filterwarnings('error')
    univariate_quadrature_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestUnivariateQuadrature)
    unittest.TextTestRunner(verbosity=2).run(univariate_quadrature_test_suite)

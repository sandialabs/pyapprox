import unittest
from pyapprox.univariate_quadrature import *
from scipy.special import gamma as gamma_fn
from scipy.special import beta as beta_fn
from pyapprox.utilities import beta_pdf_on_ab, gaussian_pdf
from pyapprox.variables import float_rv_discrete, get_distribution_info

class TestUnivariateQuadrature(unittest.TestCase):

    def test_gauss_jacobi_quadrature(self):
        """
        integrate x^2 x^a (1-x)^b/B(a+1,b+1) dx from x=0..1
        """
        alpha_poly=0
        beta_poly=0
        
        a = beta_poly+1
        b = alpha_poly+1
        
        true_mean = a/float(a+b)
        true_variance = a*b/float((a+b)**2*(a+b+1))
        
        x,w = gauss_jacobi_pts_wts_1D(2,alpha_poly,beta_poly)
        x=(x+1)/2.

        function = lambda x: x**2
        assert np.allclose(np.dot(function(x),w)-true_mean**2,true_variance)

    def test_clenshaw_curtis_quadrature(self):
        a = 1
        b = 1
        
        true_mean = a/float(a+b)
        true_variance = a*b/float((a+b)**2*(a+b+1))
        
        x,w = clenshaw_curtis_pts_wts_1D(2)
        x=(x+1)/2.

        function = lambda x: x**2
        assert np.allclose(np.dot(function(x),w)-true_mean**2,true_variance)

    def test_gauss_hermite_quadrature(self):
        """
        integrate x^2 1/sqrt(2*pi)exp(-x**2/2) dx from x=-inf..inf
        """
        
        true_mean = 0.
        true_variance = 1.
        
        x,w = gauss_hermite_pts_wts_1D(2)

        function = lambda x: x**2
        assert np.allclose(np.dot(function(x),w)-true_mean**2,true_variance)

    def test_gaussian_leja_quadrature(self):
        level = 20
        x_quad,w_quad = gaussian_leja_quadrature_rule(
            level,return_weights_for_all_levels=False)

        import sympy as sp
        x = sp.Symbol('x')
        weight_function = gaussian_pdf(0,1,x,sp)
        ranges = [-sp.oo,sp.oo]
        exact_integral = float(
            sp.integrate(weight_function*x**3,(x,ranges[0],ranges[1])))
        assert np.allclose(exact_integral, np.dot(x_quad**3,w_quad))


    def test_beta_leja_quadrature(self):
        level = 12
        alpha_stat,beta_stat = 2,10
        x_quad,w_quad = beta_leja_quadrature_rule(
            alpha_stat,beta_stat,level,return_weights_for_all_levels=False)

        import sympy as sp
        x = sp.Symbol('x')
        weight_function = beta_pdf_on_ab(alpha_stat,beta_stat,-1,1,x)
        ranges = [-1,1]
        exact_integral = float(
            sp.integrate(weight_function*x**3,(x,ranges[0],ranges[1])))
        assert np.allclose(exact_integral, np.dot(x_quad**3,w_quad))

        level = 12
        alpha_stat,beta_stat = 2,10
        x_quad,w_quad = beta_leja_quadrature_rule(
            alpha_stat,beta_stat,level,return_weights_for_all_levels=False)
        x_quad = (x_quad+1)/2

        import sympy as sp
        x = sp.Symbol('x')
        weight_function = beta_pdf_on_ab(alpha_stat,beta_stat,0,1,x)
        ranges = [0,1]
        exact_integral = float(
            sp.integrate(weight_function*x**3,(x,ranges[0],ranges[1])))
        assert np.allclose(exact_integral, np.dot(x_quad**3,w_quad))

    def test_get_univariate_leja_rule_float_rv_discrete(self):
        nmasses=20
        xk = np.array(range(1,nmasses+1),dtype='float')
        pk = np.ones(nmasses)/nmasses
        variable = float_rv_discrete(
               name='float_rv_discrete',values=(xk,pk))()
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(variable,growth_rule)
        level = 3
        scales,shapes=get_distribution_info(variable)[1:]
        print(scales)

        x,w=quad_rule(level)
        # x in [-1,1], scales for x in [0,1]
        loc,scale = scales['loc'],scales['scale']
        scale /= 2
        loc    = loc+scale
        x=x*scale+loc

        true_moment = (xk**(x.shape[0]-1)).dot(pk)
        moment = (x**(x.shape[0]-1)).dot(w[-1])
        
        #print(moment)
        #print(true_moment)
        assert np.allclose(moment,true_moment)

    def test_get_univariate_leja_rule_discrete_chebyshev(self):

        nmasses = 20
        xk = np.array(range(0, nmasses), dtype='float')
        pk = np.ones(nmasses)/nmasses
        variable = float_rv_discrete(
               name='discrete_chebyshev', values=(xk, pk))()
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(variable, growth_rule)
        level = 3
        scales, shapes = get_distribution_info(variable)[1:]

        x, w = quad_rule(level)

        true_moment = (xk**(x.shape[0]-1)).dot(pk)
        moment = (x**(x.shape[0]-1)).dot(w[-1])
        
        #print(moment)
        #print(true_moment)
        assert np.allclose(moment, true_moment)

    def test_hermite_christoffel_leja_quadrature_rule(self):
        import warnings
        warnings.filterwarnings('error')
        from scipy import stats
        variable = stats.norm(0, 1)
        growth_rule = partial(constant_increment_growth_rule, 2)
        quad_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule, method='christoffel')
        level = 5
        samples, weights = quad_rule(level)
        print(samples)


if __name__== "__main__":    
    univariate_quadrature_test_suite = \
      unittest.TestLoader().loadTestsFromTestCase(
         TestUnivariateQuadrature)
    unittest.TextTestRunner(verbosity=2).run(univariate_quadrature_test_suite)


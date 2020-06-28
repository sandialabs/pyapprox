import unittest
from scipy import stats
from pyapprox.approximate import *
from pyapprox.benchmarks.benchmarks import setup_benchmark
import pyapprox as pya
class TestApproximate(unittest.TestCase):

    def test_approximate_sparse_grid_default_options(self):
        nvars = 3
        benchmark = setup_benchmark('ishigami',a=7,b=0.1)
        univariate_variables = [stats.uniform(0,1)]*nvars
        approx = approximate(benchmark.fun,univariate_variables,'sparse-grid')
        nsamples = 100
        error = compute_l2_error(
            approx,benchmark.fun,approx.variable_transformation.variable,
            nsamples)
        assert error<1e-12

    def test_approximate_sparse_grid_user_options(self):
        nvars = 3
        benchmark = setup_benchmark('ishigami',a=7,b=0.1)
        univariate_variables = [stats.uniform(0,1)]*nvars
        errors = []
        def callback(approx):
            nsamples = 1000
            error = compute_l2_error(
                approx,benchmark.fun,approx.variable_transformation.variable,
                nsamples)
            errors.append(error)
        univariate_quad_rule_info = [
            pya.clenshaw_curtis_in_polynomial_order,
            pya.clenshaw_curtis_rule_growth]
        options = {'univariate_quad_rule_info':univariate_quad_rule_info,
                   'max_nsamples':110,'tol':0,'verbose':False}
        approx = approximate(
            benchmark.fun,univariate_variables,'sparse-grid',callback,options)
        assert np.min(errors)<1e-12

    def test_approximate_polynomial_chaos_default_options(self):
        nvars = 3
        benchmark = setup_benchmark('ishigami',a=7,b=0.1)
        univariate_variables = [stats.uniform(0,1)]*nvars
        approx = approximate(
            benchmark.fun,univariate_variables,method='polynomial-chaos')
        nsamples = 100
        error = compute_l2_error(
            approx,benchmark.fun,approx.variable_transformation.variable,
            nsamples)
        assert error<1e-12

if __name__== "__main__":    
    approximate_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestApproximate)
    unittest.TextTestRunner(verbosity=2).run(approximate_test_suite)

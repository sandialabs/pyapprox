import unittest
from scipy import stats
from pyapprox.approximate import *
from pyapprox.benchmarks.benchmarks import setup_benchmark
import pyapprox as pya
class TestApproximate(unittest.TestCase):

    def test_approximate_sparse_grid(self):
        nvars = 3
        benchmark = setup_benchmark('ishigami',a=7,b=0.1)
        univariate_variables = [stats.uniform(0,1)]*nvars
        def callback(approx):
            print(approx.num_equivalent_function_evaluations)
        univariate_quad_rule_info = [
            pya.clenshaw_curtis_in_polynomial_order,
            pya.clenshaw_curtis_rule_growth]
        univariate_quad_rule_info = None
        approx = approximate_sparse_grid(benchmark.fun,univariate_variables,callback=callback,refinement_indicator=variance_refinement_indicator,univariate_quad_rule_info=univariate_quad_rule_info,max_num_samples=100,tol=0,verbose=False)

        nvalidation_samples = 100
        validation_samples = pya.generate_independent_random_samples(approx.variable_transformation.variable,nvalidation_samples)
        validation_vals = benchmark.fun(validation_samples)
        approx_vals = approx(validation_samples)
        error=np.linalg.norm(approx_vals-validation_vals)/np.sqrt(
            validation_samples.shape[1])
        print(error)
        assert error<1e-12

if __name__== "__main__":    
    approximate_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestApproximate)
    unittest.TextTestRunner(verbosity=2).run(approximate_test_suite)

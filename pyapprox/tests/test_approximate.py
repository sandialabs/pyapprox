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
        approx = adaptive_approximate(
            benchmark.fun,univariate_variables,'sparse-grid')
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
        approx = adaptive_approximate(
            benchmark.fun,univariate_variables,'sparse-grid',callback,options)
        assert np.min(errors)<1e-12

    def test_approximate_polynomial_chaos_default_options(self):
        nvars = 3
        benchmark = setup_benchmark('ishigami',a=7,b=0.1)
        univariate_variables = [stats.uniform(0,1)]*nvars
        approx = adaptive_approximate(
            benchmark.fun,univariate_variables,method='polynomial-chaos')
        nsamples = 100
        error = compute_l2_error(
            approx,benchmark.fun,approx.variable_transformation.variable,
            nsamples)
        assert error<1e-12

    def test_cross_validate_pce_degree(self):
        num_vars = 2
        univariate_variables = [stats.uniform(-1,2)]*num_vars
        variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        var_trans = pya.AffineRandomVariableTransformation(variable)
        poly = pya.PolynomialChaosExpansion()
        poly_opts = pya.define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree=3
        poly.set_indices(pya.compute_hyperbolic_indices(num_vars,degree,1.0))
        num_samples = poly.num_terms()*2
        poly.set_coefficients(np.random.normal(0,1,(poly.indices.shape[1],1)))
        train_samples=pya.generate_independent_random_samples(
            variable,num_samples)
        train_vals = poly(train_samples)
        true_poly=poly

        poly = copy.deepcopy(true_poly)
        poly, best_degree = cross_validate_pce_degree(
            poly,train_samples,train_vals,1,degree+2)

        assert best_degree==degree

        num_validation_samples = 10
        validation_samples = pya.generate_independent_random_samples(
            variable,num_validation_samples)
        assert np.allclose(
            poly(validation_samples),true_poly(validation_samples))

    def test_pce_basis_expansion(self):
        num_vars = 2
        univariate_variables = [stats.uniform(-1,2)]*num_vars
        variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        var_trans = pya.AffineRandomVariableTransformation(variable)
        poly = pya.PolynomialChaosExpansion()
        poly_opts = pya.define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree,hcross_strength=7,0.4
        poly.set_indices(
            pya.compute_hyperbolic_indices(num_vars,degree,hcross_strength))
        num_samples = poly.num_terms()*2
        degrees = poly.indices.sum(axis=0)
        poly.set_coefficients(
            (np.random.normal(
                0,1,poly.indices.shape[1])/(degrees+1)**2)[:,np.newaxis])
        train_samples=pya.generate_independent_random_samples(
            variable,num_samples)
        train_vals = poly(train_samples)
        true_poly=poly

        poly = copy.deepcopy(true_poly)
        poly, best_degree = expanding_basis_omp_pce(
            poly,train_samples,train_vals,hcross_strength=hcross_strength,
            verbosity=1,max_num_terms=5)

        num_validation_samples = 10
        validation_samples = pya.generate_independent_random_samples(
            variable,num_validation_samples)
        validation_samples = train_samples
        #print(poly(validation_samples),'\n',true_poly(validation_samples))
        assert np.allclose(
            poly(validation_samples),true_poly(validation_samples))


if __name__== "__main__":    
    approximate_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestApproximate)
    unittest.TextTestRunner(verbosity=2).run(approximate_test_suite)

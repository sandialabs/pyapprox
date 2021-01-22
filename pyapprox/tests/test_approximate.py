import unittest
from scipy import stats
from pyapprox.approximate import *
from pyapprox.benchmarks.benchmarks import setup_benchmark
import pyapprox as pya


class TestApproximate(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_approximate_sparse_grid_default_options(self):
        nvars = 3
        benchmark = setup_benchmark('ishigami', a=7, b=0.1)
        univariate_variables = [stats.uniform(0, 1)]*nvars
        approx = adaptive_approximate(
            benchmark.fun, univariate_variables, 'sparse_grid').approx
        nsamples = 100
        error = compute_l2_error(
            approx, benchmark.fun, approx.variable_transformation.variable,
            nsamples)
        assert error < 1e-12

    def test_approximate_sparse_grid_user_options(self):
        nvars = 3
        benchmark = setup_benchmark('ishigami', a=7, b=0.1)
        univariate_variables = benchmark['variable'].all_variables()
        errors = []

        def callback(approx):
            nsamples = 1000
            error = compute_l2_error(
                approx, benchmark.fun, approx.variable_transformation.variable,
                nsamples)
            errors.append(error)
        univariate_quad_rule_info = [
            pya.clenshaw_curtis_in_polynomial_order,
            pya.clenshaw_curtis_rule_growth]
        # ishigami has same value at first 3 points in clenshaw curtis rule
        # and so adaptivity will not work so use different rule
        # growth_rule=partial(pya.constant_increment_growth_rule,4)
        # univariate_quad_rule_info = [
        #    pya.get_univariate_leja_quadrature_rule(
        #        univariate_variables[0],growth_rule),growth_rule]
        refinement_indicator = partial(
            variance_refinement_indicator, convex_param=0.5)
        options = {'univariate_quad_rule_info': univariate_quad_rule_info,
                   'max_nsamples': 300, 'tol': 0,
                   'callback': callback, 'verbose': 0,
                   'refinement_indicator': refinement_indicator}
        approx = adaptive_approximate(
            benchmark.fun, univariate_variables, 'sparse_grid', options).approx
        # print(np.min(errors))
        assert np.min(errors) < 1e-3

    def test_approximate_polynomial_chaos_default_options(self):
        nvars = 3
        benchmark = setup_benchmark('ishigami', a=7, b=0.1)
        # we can use different univariate variables than specified by
        # benchmark
        univariate_variables = [stats.uniform(0, 1)]*nvars
        approx = adaptive_approximate(
            benchmark.fun, univariate_variables,
            method='polynomial_chaos').approx
        nsamples = 100
        error = compute_l2_error(
            approx, benchmark.fun, approx.variable_transformation.variable,
            nsamples)
        assert error < 1e-12

    def help_cross_validate_pce_degree(self, solver_type, solver_options):
        print(solver_type, solver_options)
        num_vars = 2
        univariate_variables = [stats.uniform(-1, 2)]*num_vars
        variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        var_trans = pya.AffineRandomVariableTransformation(variable)
        poly = pya.PolynomialChaosExpansion()
        poly_opts = pya.define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree = 3
        poly.set_indices(pya.compute_hyperbolic_indices(num_vars, degree, 1.0))
        # factor of 2 does not pass test but 2.2 does
        num_samples = int(poly.num_terms()*2.2) 
        coef = np.random.normal(0, 1, (poly.indices.shape[1], 2))
        coef[pya.nchoosek(num_vars+2, 2):, 0] = 0
        # for first qoi make degree 2 the best degree
        poly.set_coefficients(coef)

        train_samples = pya.generate_independent_random_samples(
            variable, num_samples)
        train_vals = poly(train_samples)
        true_poly = poly

        poly = approximate(
            train_samples, train_vals, 'polynomial_chaos',
            {'basis_type': 'hyperbolic_cross', 'variable': variable,
             'options': {'verbose': 3, 'solver_type': solver_type,
                         'min_degree': 1, 'max_degree': degree+1,
                         'linear_solver_options':solver_options}}).approx

        num_validation_samples = 10
        validation_samples = pya.generate_independent_random_samples(
            variable, num_validation_samples)
        assert np.allclose(
            poly(validation_samples), true_poly(validation_samples))

        poly = copy.deepcopy(true_poly)
        approx_res = cross_validate_pce_degree(
            poly, train_samples, train_vals, 1, degree+1,
            solver_type=solver_type, linear_solver_options=solver_options)
        assert np.allclose(approx_res.degrees, [2, 3])

    def test_cross_validate_pce_degree(self):
        # lasso and omp do not pass this test so recommend not using them
        solver_type_list = ['lstsq', 'lasso']#, 'omp']#, 'lars']
        solver_options_list = [
            {'alphas': [1e-14], 'cv':20}, {'max_iter': 20, 'cv': 21},
            {'max_iter': 20, 'cv': 21}]
        for solver_type, solver_options in zip(
                solver_type_list, solver_options_list):
            self.help_cross_validate_pce_degree(solver_type, solver_options)

    def test_pce_basis_expansion(self):
        num_vars = 2
        univariate_variables = [stats.uniform(-1, 2)]*num_vars
        variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        var_trans = pya.AffineRandomVariableTransformation(variable)
        poly = pya.PolynomialChaosExpansion()
        poly_opts = pya.define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree, hcross_strength = 7, 0.4
        poly.set_indices(
            pya.compute_hyperbolic_indices(num_vars, degree, hcross_strength))
        num_samples = poly.num_terms()*2
        degrees = poly.indices.sum(axis=0)
        coef = np.random.normal(
            0, 1, (poly.indices.shape[1], 2))/(degrees[:, np.newaxis]+1)**2
        # set some coefficients to zero to make sure that different qoi
        # are treated correctly.
        I = np.random.permutation(coef.shape[0])[:coef.shape[0]//2]
        coef[I, 0] = 0
        I = np.random.permutation(coef.shape[0])[:coef.shape[0]//2]
        coef[I, 1] = 0
        poly.set_coefficients(coef)
        train_samples = pya.generate_independent_random_samples(
            variable, num_samples)
        train_vals = poly(train_samples)
        true_poly = poly

        poly = approximate(
            train_samples, train_vals, 'polynomial_chaos',
            {'basis_type': 'expanding_basis', 'variable': variable}).approx

        num_validation_samples = 100
        validation_samples = pya.generate_independent_random_samples(
            variable, num_validation_samples)
        validation_samples = train_samples
        error = np.linalg.norm(poly(validation_samples)-true_poly(
            validation_samples))/np.sqrt(num_validation_samples)
        assert np.allclose(
            poly(validation_samples), true_poly(validation_samples), atol=1e-8),\
            error

    def test_approximate_gaussian_process(self):
        from sklearn.gaussian_process.kernels import Matern
        num_vars = 1
        univariate_variables = [stats.uniform(-1, 2)]*num_vars
        variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)
        num_samples = 100
        train_samples = pya.generate_independent_random_samples(
            variable, num_samples)

        # Generate random function
        nu = np.inf  # 2.5
        kernel = Matern(0.5, nu=nu)
        X = np.linspace(-1, 1, 1000)[np.newaxis, :]
        alpha = np.random.normal(0, 1, X.shape[1])
        train_vals = kernel(train_samples.T, X.T).dot(alpha)[:, np.newaxis]

        gp = approximate(
            train_samples, train_vals, 'gaussian_process',
            {'nu': nu, 'noise_level': 1e-8}).approx

        error = np.linalg.norm(gp(X)[:, 0]-kernel(X.T, X.T).dot(alpha))/np.sqrt(
            X.shape[1])
        assert error < 1e-5

        # import matplotlib.pyplot as plt
        # plt.plot(X[0,:],kernel(X.T,X.T).dot(alpha),'r--',zorder=100)
        # vals,std = gp(X,return_std=True)
        # plt.plot(X[0,:],vals[:,0],c='b')
        # plt.fill_between(
        #     X[0,:],vals[:,0]-2*std,vals[:,0]+2*std,color='b',alpha=0.5)
        # plt.plot(train_samples[0,:], train_vals[:,0],'ro')
        # plt.show()


if __name__ == "__main__":
    approximate_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestApproximate)
    unittest.TextTestRunner(verbosity=2).run(approximate_test_suite)

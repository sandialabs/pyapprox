import unittest
from scipy import stats
import numpy as np
import copy
from functools import partial

from pyapprox.surrogates.gaussianprocess.kernels import Matern
from pyapprox.surrogates.approximate import (
    approximate, adaptive_approximate,
    cross_validate_pce_degree, compute_l2_error,
    cross_validate_approximation, LinearLeastSquaresCV,
    adaptive_approximate_polynomial_chaos_increment_degree
)
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_in_polynomial_order, clenshaw_curtis_rule_growth
)
from pyapprox.surrogates.interp.adaptive_sparse_grid import (
    variance_refinement_indicator
)
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, define_poly_options_from_variable_transformation,
    define_poly_options_from_variable
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.util.utilities import nchoosek
from pyapprox.variables.density import tensor_product_pdf


class TestApproximate(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_approximate_sparse_grid_default_options(self):
        nvars = 3
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        univariate_variables = [stats.uniform(0, 1)]*nvars
        approx = adaptive_approximate(
            benchmark.fun, univariate_variables, "sparse_grid").approx
        nsamples = 100
        error = compute_l2_error(
            approx, benchmark.fun, approx.var_trans.variable,
            nsamples)
        assert error < 1e-12

    def test_approximate_sparse_grid_discrete(self):
        def fun(samples):
            return np.cos(samples.sum(axis=0)/20)[:, None]
        nvars = 2
        univariate_variables = [stats.binom(20, 0.5)]*nvars
        approx = adaptive_approximate(
            fun, univariate_variables, "sparse_grid").approx
        nsamples = 100
        error = compute_l2_error(
            approx, fun, approx.var_trans.variable,
            nsamples)
        assert error < 1e-12
        # check leja samples are nested. Sparse grid uses christoffel
        # leja sequence that does not change preconditioner everytime
        # lu pivot is performed, but we can still enforce nestedness
        # by specifiying initial points. This tests make sure this is done
        # correctly
        for ll in range(1, len(approx.samples_1d[0])):
            n = approx.samples_1d[0][ll-1].shape[0]
            assert np.allclose(approx.samples_1d[0][ll][:n],
                               approx.samples_1d[0][ll-1])

    def test_approximate_sparse_grid_user_options(self):
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        univariate_variables = benchmark["variable"].marginals()
        errors = []

        def callback(approx):
            nsamples = 1000
            error = compute_l2_error(
                approx, benchmark.fun, approx.var_trans.variable,
                nsamples)
            errors.append(error)
        univariate_quad_rule_info = [
            clenshaw_curtis_in_polynomial_order,
            clenshaw_curtis_rule_growth, None, None]
        # ishigami has same value at first 3 points in clenshaw curtis rule
        # and so adaptivity will not work so use different rule
        # growth_rule=partial(constant_increment_growth_rule,4)
        # univariate_quad_rule_info = [
        #    get_univariate_leja_quadrature_rule(
        #        univariate_variables[0],growth_rule),growth_rule]
        refinement_indicator = partial(
            variance_refinement_indicator, convex_param=0.5)
        options = {"univariate_quad_rule_info": univariate_quad_rule_info,
                   "max_nsamples": 300, "tol": 0,
                   "callback": callback, "verbose": 0,
                   "refinement_indicator": refinement_indicator}
        adaptive_approximate(
            benchmark.fun, univariate_variables, "sparse_grid", options).approx
        # print(np.min(errors))
        assert np.min(errors) < 1e-3

    def test_approximate_polynomial_chaos_leja(self):
        nvars = 3
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        # we can use different univariate variables than specified by
        # benchmark. In this case we use the same but setup them uphear
        # to demonstrate this functionality
        univariate_variables = [stats.uniform(0, 1)]*nvars
        approx = adaptive_approximate(
            benchmark.fun, univariate_variables,
            method="polynomial_chaos",
            options={"method": "leja",
                     "options": {"max_nsamples": 100}}).approx
        nsamples = 100
        error = compute_l2_error(
            approx, benchmark.fun, approx.var_trans.variable,
            nsamples)
        assert error < 1e-12

    def test_approximate_polynomial_chaos_induced(self):
        nvars = 3
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        # we can use different univariate variables than specified by
        # benchmark. In this case we use the same but setup them uphear
        # to demonstrate this functionality
        univariate_variables = [stats.uniform(0, 1)]*nvars
        # approx = adaptive_approximate(
        #     benchmark.fun, univariate_variables,
        #     method="polynomial_chaos",
        #     options={"method": "induced",
        #              "options": {"max_nsamples": 200,
        #                          "induced_sampling": True,
        #                          "cond_tol": 1e8}}).approx
        # nsamples = 100
        # error = compute_l2_error(
        #     approx, benchmark.fun, approx.var_trans.variable,
        #     nsamples)
        # print(error)
        # assert error < 1e-5

        # probablility sampling
        approx = adaptive_approximate(
            benchmark.fun, univariate_variables,
            method="polynomial_chaos",
            options={"method": "induced",
                     "options": {"max_nsamples": 100,
                                 "induced_sampling": False,
                                 "cond_tol": 1e4,
                                 "max_level_1d": 4, "verbose": 3}}).approx
        nsamples = 100
        error = compute_l2_error(
            approx, benchmark.fun, approx.var_trans.variable,
            nsamples)
        print(error)
        assert error < 1e-5

    def test_approximate_polynomial_chaos_custom_poly_type(self):
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        nvars = benchmark.variable.num_vars()
        # this test purposefully select wrong variable to make sure
        # poly_type overide is activated
        univariate_variables = [stats.beta(5, 5, -np.pi, 2*np.pi)]*nvars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)
        # specify correct basis so it is not chosen from var_trans.variable
        poly_opts = {"var_trans": var_trans}
        # but rather from another variable which will invoke Legendre polys
        basis_opts = define_poly_options_from_variable(
            IndependentMarginalsVariable([stats.uniform()]*nvars))
        poly_opts["poly_types"] = basis_opts
        options = {"poly_opts": poly_opts, "variable": variable,
                   "options": {"max_num_step_increases": 1}}
        ntrain_samples = 400
        train_samples = np.random.uniform(
            -np.pi, np.pi, (nvars, ntrain_samples))
        train_vals = benchmark.fun(train_samples)
        approx = approximate(
            train_samples, train_vals,
            method="polynomial_chaos", options=options).approx
        nsamples = 100
        error = compute_l2_error(
            approx, benchmark.fun, approx.var_trans.variable,
            nsamples, rel=True)
        # print(error)
        assert error < 1e-4
        assert np.allclose(approx.mean(), benchmark.mean, atol=error)

    def help_cross_validate_pce_degree(self, solver_type, solver_options):
        print(solver_type, solver_options)
        num_vars = 2
        univariate_variables = [stats.uniform(-1, 2)]*num_vars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree = 3
        poly.set_indices(compute_hyperbolic_indices(num_vars, degree, 1.0))
        # factor of 2 does not pass test but 2.2 does
        num_samples = int(poly.num_terms()*2.2)
        coef = np.random.normal(0, 1, (poly.indices.shape[1], 2))
        coef[nchoosek(num_vars+2, 2):, 0] = 0
        # for first qoi make degree 2 the best degree
        poly.set_coefficients(coef)

        train_samples = variable.rvs(num_samples)
        train_vals = poly(train_samples)
        true_poly = poly

        poly = approximate(
            train_samples, train_vals, "polynomial_chaos",
            {"basis_type": "hyperbolic_cross", "variable": variable,
             "options": {"verbose": 3, "solver_type": solver_type,
                         "min_degree": 1, "max_degree": degree+1,
                         "linear_solver_options": solver_options}}).approx

        num_validation_samples = 10
        validation_samples = variable.rvs(num_validation_samples)
        assert np.allclose(
            poly(validation_samples), true_poly(validation_samples))

        poly = copy.deepcopy(true_poly)
        approx_res = cross_validate_pce_degree(
            poly, train_samples, train_vals, 1, degree+1,
            solver_type=solver_type, linear_solver_options=solver_options)
        assert np.allclose(approx_res.degrees, [2, 3])

    def test_cross_validate_pce_degree(self):
        # lasso and omp do not pass this test so recommend not using them
        solver_type_list = ["lstsq", "lstsq", "lasso"]  # , "omp"]#, "lars"]
        solver_options_list = [
            {"alphas": [1e-14], "cv":22}, {"cv": 10},
            {"max_iter": 20, "cv": 21}]
        for solver_type, solver_options in zip(
                solver_type_list, solver_options_list):
            self.help_cross_validate_pce_degree(solver_type, solver_options)

    def test_pce_basis_expansion(self):
        num_vars = 2
        univariate_variables = [stats.uniform(-1, 2)]*num_vars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree, hcross_strength = 7, 0.4
        poly.set_indices(
            compute_hyperbolic_indices(num_vars, degree, hcross_strength))
        num_samples = poly.num_terms()*2
        degrees = poly.indices.sum(axis=0)
        coef = np.random.normal(
            0, 1, (poly.indices.shape[1], 2))/(degrees[:, np.newaxis]+1)**2
        # set some coefficients to zero to make sure that different qoi
        # are treated correctly.
        II = np.random.permutation(coef.shape[0])[:coef.shape[0]//2]
        coef[II, 0] = 0
        II = np.random.permutation(coef.shape[0])[:coef.shape[0]//2]
        coef[II, 1] = 0
        poly.set_coefficients(coef)
        train_samples = variable.rvs(num_samples)
        train_vals = poly(train_samples)
        true_poly = poly

        poly = approximate(
            train_samples, train_vals, "polynomial_chaos",
            {"basis_type": "expanding_basis", "variable": variable,
             "options": {"max_num_expansion_steps_iter": 1, "verbose": 3,
                         "max_num_terms": 1000,
                         "max_num_step_increases": 2,
                         "max_num_init_terms": 33}}).approx

        num_validation_samples = 100
        validation_samples = variable.rvs(num_validation_samples)
        validation_samples = train_samples
        error = np.linalg.norm(poly(validation_samples)-true_poly(
            validation_samples))/np.sqrt(num_validation_samples)
        assert np.allclose(
            poly(validation_samples), true_poly(validation_samples),
            atol=1e-8), error

    def test_approximate_gaussian_process(self):
        num_vars = 1
        univariate_variables = [stats.uniform(-1, 2)]*num_vars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        num_samples = 100
        train_samples = variable.rvs(num_samples)

        # Generate random function
        nu = np.inf  # 2.5
        kernel = Matern(0.5, nu=nu)
        X = np.linspace(-1, 1, 1000)[np.newaxis, :]
        alpha = np.random.normal(0, 1, X.shape[1])
        train_vals = kernel(train_samples.T, X.T).dot(alpha)[:, np.newaxis]

        gp = approximate(
            train_samples, train_vals, "gaussian_process",
            {"nu": nu, "noise_level": 1e-8}).approx

        error = np.linalg.norm(gp(X)[:, 0]-kernel(X.T, X.T).dot(alpha)) /\
            np.sqrt(X.shape[1])
        assert error < 1e-5

        # import matplotlib.pyplot as plt
        # plt.plot(X[0,:],kernel(X.T,X.T).dot(alpha),"r--",zorder=100)
        # vals,std = gp(X,return_std=True)
        # plt.plot(X[0,:],vals[:,0],c="b")
        # plt.fill_between(
        #     X[0,:],vals[:,0]-2*std,vals[:,0]+2*std,color="b",alpha=0.5)
        # plt.plot(train_samples[0,:], train_vals[:,0],"ro")
        # plt.show()

    def test_adaptive_approximate_gaussian_process(self):
        num_vars = 1
        univariate_variables = [stats.uniform(-1, 2)]*num_vars

        # Generate random function
        nu = np.inf  # 2.5
        kernel = Matern(0.1, nu=nu)
        X = np.linspace(-1, 1, 1000)[np.newaxis, :]
        alpha = np.random.normal(0, 1, X.shape[1])

        def fun(x):
            return kernel(x.T, X.T).dot(alpha)[:, np.newaxis]
            # return np.cos(2*np.pi*x.sum(axis=0)/num_vars)[:, np.newaxis]

        errors = []
        validation_samples = np.random.uniform(-1, 1, (num_vars, 100))
        validation_values = fun(validation_samples)

        def callback(gp):
            gp_vals = gp(validation_samples)
            assert gp_vals.shape == validation_values.shape
            error = np.linalg.norm(gp_vals-validation_values)/np.linalg.norm(
                validation_values)
            print(error, gp.y_train_.shape[0])
            errors.append(error)

        # normalize_y must be False
        adaptive_approximate(
            fun, univariate_variables, "gaussian_process",
            {"nu": nu, "noise_level": None, "normalize_y": False,
             "alpha": 1e-10,
             "ncandidate_samples": 1e3, "callback": callback}).approx
        assert errors[-1] < 1e-8

    def test_adaptive_approximate_gaussian_process_normalize_inputs(self):
        num_vars = 1
        univariate_variables = [stats.beta(5, 10, 0, 2)]*num_vars

        # Generate random function
        nu = np.inf  # 2.5
        kernel = Matern(0.1, nu=nu)
        X = np.linspace(-1, 1, 1000)[np.newaxis, :]
        alpha = np.random.normal(0, 1, X.shape[1])

        def fun(x):
            return kernel(x.T, X.T).dot(alpha)[:, np.newaxis]
            # return np.cos(2*np.pi*x.sum(axis=0)/num_vars)[:, np.newaxis]

        errors = []
        variable = IndependentMarginalsVariable(univariate_variables)
        validation_samples = variable.rvs(100)
        validation_values = fun(validation_samples)

        print(np.linalg.norm(alpha), 'alpha')
        print(np.linalg.norm(validation_values), 'values')

        def callback(gp):
            gp_vals = gp(validation_samples)
            assert gp_vals.shape == validation_values.shape
            error = np.linalg.norm(gp_vals-validation_values)/np.linalg.norm(
                validation_values)
            print(error, gp.y_train_.shape[0])
            print(gp.kernel_)
            errors.append(error)

        weight_function = partial(
            tensor_product_pdf,
            univariate_pdfs=[v.pdf for v in univariate_variables])

        gp = adaptive_approximate(
            fun, univariate_variables, "gaussian_process",
            {"nu": nu, "noise_level": None, "normalize_y": False,
             "alpha": 1e-10,  "normalize_inputs": True, "length_scale": 0.5,
             "weight_function": weight_function, "n_restarts_optimizer": 10,
             "ncandidate_samples": 1e3, "callback": callback}).approx

        # import matplotlib.pyplot as plt
        # plt.plot(gp.X_train_.T[0, :], 0*gp.X_train_.T[0, :], "s")
        # plt.plot(gp.get_training_samples()[0, :], 0*gp.get_training_samples()[0, :], "x")
        # plt.plot(gp.sampler.candidate_samples[0, :], 0*gp.sampler.candidate_samples[0, :], "^")
        # plt.plot(validation_samples[0, :], validation_values[:, 0], "o")
        # var = univariate_variables[0]
        # lb, ub = var.interval(1)
        # xx = np.linspace(lb, ub, 101)
        # plt.plot(xx, var.pdf(xx), "r-")
        # plt.show()
        print(errors[-1])
        assert errors[-1] < 4e-7

    def test_approximate_fixed_pce(self):
        num_vars = 2
        univariate_variables = [stats.uniform(-1, 2)]*num_vars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree, hcross_strength = 7, 0.4
        poly.set_indices(
            compute_hyperbolic_indices(num_vars, degree, hcross_strength))
        num_samples = poly.num_terms()*2
        degrees = poly.indices.sum(axis=0)
        coef = np.random.normal(
            0, 1, (poly.indices.shape[1], 2))/(degrees[:, np.newaxis]+1)**2
        # set some coefficients to zero to make sure that different qoi
        # are treated correctly.
        II = np.random.permutation(coef.shape[0])[:coef.shape[0]//2]
        coef[II, 0] = 0
        II = np.random.permutation(coef.shape[0])[:coef.shape[0]//2]
        coef[II, 1] = 0
        poly.set_coefficients(coef)
        train_samples = variable.rvs(num_samples)
        train_vals = poly(train_samples)

        indices = compute_hyperbolic_indices(num_vars, 1, 1)
        nfolds = 10
        method = "polynomial_chaos"
        options = {"basis_type": "fixed", "variable": variable,
                   "options": {"linear_solver_options": {},
                               "indices": indices, "solver_type": "lstsq"}}
        approx_list, residues_list, cv_score = \
            cross_validate_approximation(
                train_samples, train_vals, options, nfolds, method,
                random_folds=False)

        solver = LinearLeastSquaresCV(cv=nfolds, random_folds=False)
        poly.set_indices(indices)
        basis_matrix = poly.basis_matrix(train_samples)
        solver.fit(basis_matrix, train_vals[:, 0:1])
        assert np.allclose(solver.cv_score_, cv_score[0])

        solver.fit(basis_matrix, train_vals[:, 1:2])
        assert np.allclose(solver.cv_score_, cv_score[1])

    def test_cross_validate_approximation_after_regularization_selection(self):
        """
        This test is useful as it shows how to use cross_validate_approximation
        to produce a list of approximations on each cross validation fold
        once regularization parameters have been chosen.
        These can be used to show variance in predictions of values,
        sensitivity indices, etc.

        Ideally this could be avoided if sklearn stored the coefficients
        and alphas for each fold and then we can just find the coefficients
        that correspond to the first time the path drops below the best_alpha
        """
        num_vars = 2
        univariate_variables = [stats.uniform(-1, 2)]*num_vars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree, hcross_strength = 7, 0.4
        poly.set_indices(
            compute_hyperbolic_indices(num_vars, degree, hcross_strength))
        num_samples = poly.num_terms()*2
        degrees = poly.indices.sum(axis=0)
        coef = np.random.normal(
            0, 1, (poly.indices.shape[1], 2))/(degrees[:, np.newaxis]+1)**2
        # set some coefficients to zero to make sure that different qoi
        # are treated correctly.
        II = np.random.permutation(coef.shape[0])[:coef.shape[0]//2]
        coef[II, 0] = 0
        II = np.random.permutation(coef.shape[0])[:coef.shape[0]//2]
        coef[II, 1] = 0
        poly.set_coefficients(coef)
        train_samples = variable.rvs(num_samples)
        train_vals = poly(train_samples)
        # true_poly = poly

        result = approximate(
            train_samples, train_vals, "polynomial_chaos",
            {"basis_type": "expanding_basis", "variable": variable})

        # Even with the same folds, iterative methods such as Lars, LarsLasso
        # and OMP will not have cv_score from approximate and cross validate
        # approximation exactly the same because iterative methods interpolate
        # residuals to compute cross validation scores
        nfolds = 10
        linear_solver_options = [
            {"alpha": result.reg_params[0]}, {"alpha": result.reg_params[1]}]
        indices = [result.approx.indices[:, np.where(np.absolute(c) > 0)[0]]
                   for c in result.approx.coefficients.T]
        options = {"basis_type": "fixed", "variable": variable,
                   "options": {"linear_solver_options": linear_solver_options,
                               "indices": indices}}
        approx_list, residues_list, cv_score = \
            cross_validate_approximation(
                train_samples, train_vals, options, nfolds, "polynomial_chaos",
                random_folds="sklearn")

        assert (np.all(cv_score < 7e-13) and np.all(result.scores < 4e-13))

    def test_approximate_neural_network(self):
        np.random.seed(2)
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        nvars = benchmark.variable.num_vars()
        nqoi = 1
        maxiter = 30000
        print(benchmark.variable)
        # var_trans = AffineTransform(
        #      [stats.uniform(-2, 4)]*nvars)
        var_trans = AffineTransform(benchmark.variable)
        network_opts = {"activation_func": "sigmoid",
                        "layers": [nvars, 75, nqoi],
                        "loss_func": "squared_loss",
                        "var_trans": var_trans, "lag_mult": 0}
        optimizer_opts = {"method": "L-BFGS-B",
                          "options": {"maxiter": maxiter, "iprint": -1,
                                      "gtol": 1e-6}}
        opts = {"network_opts": network_opts, "verbosity": 3,
                "optimizer_opts": optimizer_opts}
        ntrain_samples = 500
        train_samples = benchmark.variable.rvs(ntrain_samples)
        train_samples = var_trans.map_from_canonical(
            np.cos(np.random.uniform(0, np.pi, (nvars, ntrain_samples))))
        train_vals = benchmark.fun(train_samples)

        opts = {"network_opts": network_opts, "verbosity": 3,
                "optimizer_opts": optimizer_opts, "x0": 10}
        approx = approximate(
            train_samples, train_vals, "neural_network", opts).approx
        nsamples = 100
        error = compute_l2_error(
            approx, benchmark.fun, var_trans.variable,
            nsamples)
        print(error)
        assert error < 6.3e-2

    def test_adaptive_approximate_increment_degree(self):
        num_vars = 2
        univariate_variables = [stats.uniform(-1, 2)]*num_vars
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree = 3
        poly.set_indices(compute_hyperbolic_indices(num_vars, degree))
        poly.set_coefficients(
            np.random.normal(0, 1, (poly.indices.shape[1], 1)))
        fun = poly

        max_degree = degree+2
        result = adaptive_approximate_polynomial_chaos_increment_degree(
            fun, variable, max_degree, max_nsamples=31, cond_tol=1e4,
            sample_growth_factor=2, verbose=0,
            oversampling_ratio=None, solver_type='lstsq',
            callback=None)
        print('Ntrain samples', result.train_samples.shape[1])
        assert np.allclose(
            result.approx.coefficients[:poly.coefficients.shape[0]],
            poly.coefficients)


if __name__ == "__main__":
    approximate_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestApproximate)
    unittest.TextTestRunner(verbosity=2).run(approximate_test_suite)

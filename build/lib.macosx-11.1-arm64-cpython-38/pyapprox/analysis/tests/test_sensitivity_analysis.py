import unittest
from scipy import stats
import numpy as np
from functools import partial

from pyapprox.analysis.sensitivity_analysis import (
    get_sobol_indices, get_main_and_total_effect_indices_from_pce,
    get_morris_samples, downselect_morris_trajectories,
    get_morris_elementary_effects, get_morris_sensitivity_indices,
    print_morris_sensitivity_indices,
    gpc_sobol_sensitivities, sparse_grid_sobol_sensitivities,
    sampling_based_sobol_indices, repeat_sampling_based_sobol_indices,
    sampling_based_sobol_indices_from_gaussian_process,
    analytic_sobol_indices_from_gaussian_process,
    run_sensitivity_analysis
)
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.benchmarks.sensitivity_benchmarks import (
    ishigami_function, get_ishigami_funciton_statistics, sobol_g_function,
    get_sobol_g_function_statistics, morris_function
)
from pyapprox.surrogates.approximate import approximate, adaptive_approximate
from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_indices, tensor_product_indices
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.surrogates.polychaos.gpc import (
    define_poly_options_from_variable_transformation, PolynomialChaosExpansion,
    marginalize_polynomial_chaos_expansion
)
from pyapprox.expdesign.low_discrepancy_sequences import sobol_sequence
from pyapprox.util.utilities import cartesian_product


class TestSensitivityAnalysis(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_get_sobol_indices_from_pce(self):
        num_vars = 5
        degree = 5
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        coefficients = np.ones((indices.shape[1], 2), float)
        coefficients[:, 1] *= 2
        interaction_indices, interaction_values = \
            get_sobol_indices(
                coefficients, indices, max_order=num_vars)
        assert np.allclose(
            interaction_values.sum(axis=0), np.ones(2))

    def test_pce_sensitivities_of_ishigami_function(self):
        nsamples = 1500
        nvars, degree = 3, 18
        univariate_variables = [stats.uniform(-np.pi, 2*np.pi)]*nvars
        variable = IndependentMarginalsVariable(
            univariate_variables)

        var_trans = AffineTransform(variable)
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)
        indices = compute_hyperbolic_indices(nvars, degree, 1.0)
        poly.set_indices(indices)
        # print('No. PCE Terms',indices.shape[1])

        samples = variable.rvs(nsamples)
        values = ishigami_function(samples)

        basis_matrix = poly.basis_matrix(samples)
        coef = np.linalg.lstsq(basis_matrix, values, rcond=None)[0]
        poly.set_coefficients(coef)

        # nvalidation_samples = 1000
        # validation_samples = generate_independent_random_samples(
        #     var_trans.variable, nvalidation_samples)
        # validation_values = ishigami_function(validation_samples)
        # poly_validation_vals = poly(validation_samples)
        # abs_error = np.linalg.norm(
        #     poly_validation_vals-validation_values)/np.sqrt(nvalidation_samples)
        # print('Abs. Error',abs_error)

        pce_main_effects, pce_total_effects =\
            get_main_and_total_effect_indices_from_pce(
                poly.get_coefficients(), poly.get_indices())

        mean, variance, main_effects, total_effects, sobol_indices, \
            sobol_interaction_indices = get_ishigami_funciton_statistics()
        assert np.allclose(poly.mean(), mean)
        assert np.allclose(poly.variance(), variance)
        assert np.allclose(pce_main_effects, main_effects)
        assert np.allclose(pce_total_effects, total_effects)

        interaction_terms, pce_sobol_indices = get_sobol_indices(
            poly.get_coefficients(), poly.get_indices(), max_order=3)
        assert np.allclose(pce_sobol_indices, sobol_indices)

    def test_pce_sensitivities_of_sobol_g_function(self):
        nsamples = 2000
        nvars, degree = 3, 8
        a = np.array([1, 2, 5])[:nvars]
        univariate_variables = [stats.uniform(0, 1)]*nvars
        variable = IndependentMarginalsVariable(
            univariate_variables)

        var_trans = AffineTransform(variable)
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)
        indices = tensor_product_indices([degree]*nvars)
        poly.set_indices(indices)
        # print('No. PCE Terms',indices.shape[1])

        samples = variable.rvs(nsamples)
        samples = (np.cos(np.random.uniform(0, np.pi, (nvars, nsamples)))+1)/2
        values = sobol_g_function(a, samples)

        basis_matrix = poly.basis_matrix(samples)
        weights = 1/np.sum(basis_matrix**2, axis=1)[:, np.newaxis]
        coef = np.linalg.lstsq(basis_matrix*weights,
                               values*weights, rcond=None)[0]
        poly.set_coefficients(coef)

        nvalidation_samples = 1000
        validation_samples = variable.rvs(nvalidation_samples)
        validation_values = sobol_g_function(a, validation_samples)

        poly_validation_vals = poly(validation_samples)
        rel_error = np.linalg.norm(
            poly_validation_vals-validation_values)/np.linalg.norm(
                validation_values)
        print('Rel. Error', rel_error)

        pce_main_effects, pce_total_effects =\
            get_main_and_total_effect_indices_from_pce(
                poly.get_coefficients(), poly.get_indices())
        interaction_terms, pce_sobol_indices = get_sobol_indices(
            poly.get_coefficients(), poly.get_indices(), max_order=3)

        mean, variance, main_effects, total_effects, sobol_indices = \
            get_sobol_g_function_statistics(a, interaction_terms)
        assert np.allclose(poly.mean(), mean, atol=1e-2)
        # print((poly.variance(),variance))
        assert np.allclose(poly.variance(), variance, atol=1e-2)
        # print(pce_main_effects,main_effects)
        assert np.allclose(pce_main_effects, main_effects, atol=1e-2)
        # print(pce_total_effects,total_effects)
        assert np.allclose(pce_total_effects, total_effects, atol=1e-2)
        assert np.allclose(pce_sobol_indices, sobol_indices, atol=1e-2)

    def test_get_sobol_indices_from_pce_max_order(self):
        num_vars = 3
        degree = 4
        max_order = 2
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        coefficients = np.ones((indices.shape[1], 2), float)
        coefficients[:, 1] *= 2
        interaction_indices, interaction_values = \
            get_sobol_indices(coefficients, indices, max_order)

        assert len(interaction_indices) == 6
        true_interaction_indices = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
        for ii in range(len(interaction_indices)):
            assert np.allclose(
                true_interaction_indices[ii], interaction_indices[ii])

        true_variance = np.asarray(
            [indices.shape[1]-1, 2**2*(indices.shape[1]-1)])

        # get the number of interactions involving variables 0 and 1
        # test problem is symmetric so number is the same for all variables
        num_pairwise_interactions = np.where(
            np.all(indices[0:2, :] > 0, axis=0) &
            (indices[2, :] == 0))[0].shape[0]
        # II = np.where(np.all(indices[0:2, :] > 0, axis=0))[0]

        true_interaction_values = np.vstack((
            np.tile(np.arange(1, 3)[np.newaxis, :],
                    (num_vars, 1))**2*degree/true_variance,
            np.tile(np.arange(1, 3)[np.newaxis, :],
                    (num_vars, 1))**2*num_pairwise_interactions/true_variance))

        assert np.allclose(true_interaction_values, interaction_values)

        # plot_interaction_values( interaction_values, interaction_indices)

    def test_get_main_and_total_effect_indices_from_pce(self):
        num_vars = 3
        degree = num_vars
        # max_order = 2
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        coefficients = np.ones((indices.shape[1], 2), float)
        coefficients[:, 1] *= 2
        main_effects, total_effects = \
            get_main_and_total_effect_indices_from_pce(coefficients, indices)
        true_variance = np.asarray(
            [indices.shape[1]-1, 2**2*(indices.shape[1]-1)])
        true_main_effects = np.tile(
            np.arange(1, 3)[np.newaxis, :],
            (num_vars, 1))**2*degree/true_variance
        assert np.allclose(main_effects, true_main_effects)

        # get the number of interactions variable 0 is involved in
        # test problem is symmetric so number is the same for all variables
        num_interactions_per_variable = np.where(indices[0, :] > 0)[0].shape[0]
        true_total_effects = np.tile(
            np.arange(1, 3)[np.newaxis, :],
            (num_vars, 1))**2*num_interactions_per_variable/true_variance
        assert np.allclose(true_total_effects, total_effects)

        # plot_total_effects(total_effects)
        # plot_main_effects(main_effects)

    def test_morris_elementary_effects(self):
        nvars = 20
        function = morris_function

        nvars = 6
        coefficients = np.array([78, 12, 0.5, 2, 97, 33])
        function = partial(sobol_g_function, coefficients)

        nlevels, ncandidate_trajectories, ntrajectories = 4, 40, 4
        candidate_samples = get_morris_samples(
            nvars, nlevels, ncandidate_trajectories)

        samples = downselect_morris_trajectories(
            candidate_samples, ntrajectories)

        values = function(samples)
        elem_effects = get_morris_elementary_effects(samples, values)
        mu, sigma = get_morris_sensitivity_indices(elem_effects)
        print_morris_sensitivity_indices(mu, sigma)

        variable = IndependentMarginalsVariable([stats.beta(2, 3)]*nvars)
        result = run_sensitivity_analysis("morris", function, variable, 10)

    def test_gpc_sobol_sensitivities(self):
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)

        num_samples = 1000
        train_samples = benchmark.variable.rvs(num_samples)
        train_vals = benchmark.fun(train_samples)

        pce = approximate(
            train_samples, train_vals, 'polynomial_chaos',
            {'basis_type': 'hyperbolic_cross', 'variable': benchmark.variable,
             'options': {'max_degree': 8}}).approx

        res = gpc_sobol_sensitivities(pce, benchmark.variable)
        assert np.allclose(res.main_effects, benchmark.main_effects, atol=2e-3)

        res = run_sensitivity_analysis("pce_sobol", pce, benchmark.variable)
        assert np.allclose(res.main_effects, benchmark.main_effects, atol=2e-3)

    def test_sparse_grid_sobol_sensitivities(self):
        benchmark = setup_benchmark("oakley")

        options = {'max_nsamples': 2000, 'verbose': 0}
        approx = adaptive_approximate(
            benchmark.fun, benchmark.variable.marginals(),
            'sparse_grid', options).approx

        from pyapprox.surrogates.approximate import compute_l2_error
        nsamples = 100
        error = compute_l2_error(
            approx, benchmark.fun, approx.var_trans.variable,
            nsamples, rel=True)
        # print(error)
        assert error < 7e-3

        res = sparse_grid_sobol_sensitivities(approx)
        assert np.allclose(res.main_effects, benchmark.main_effects, atol=4e-4)

    def test_qmc_sobol_sensitivity_analysis_ishigami(self):
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)

        nsamples = 10000
        nvars = benchmark.variable.num_vars()
        order = 3
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, np.where(
            interaction_terms.max(axis=0) == 1)[0]]

        sampling_method = 'sobol'
        sobol_indices, total_effect_indices, var, mean = \
            sampling_based_sobol_indices(
                benchmark.fun, benchmark.variable, interaction_terms, nsamples,
                sampling_method, qmc_start_index=100)

        assert np.allclose(mean, benchmark.mean, atol=2e-3)

        main_effects = sobol_indices[:nvars]
        assert np.allclose(main_effects, benchmark.main_effects, atol=2e-3)

        assert np.allclose(
            total_effect_indices, benchmark.total_effects, atol=2e-3)

        for ii in range(interaction_terms.shape[1]):
            index = interaction_terms[:, ii]
            assert np.allclose(
                np.where(index > 0)[0],
                benchmark.sobol_interaction_indices[ii])
        assert np.allclose(sobol_indices, benchmark.sobol_indices,
                           rtol=5e-3, atol=1e-3)

        result = run_sensitivity_analysis(
            "sobol", benchmark.fun, benchmark.variable,
            interaction_terms, nsamples, sampling_method, qmc_start_index=100)
        main_effects = result["sobol_indices"]["median"][:benchmark.variable.num_vars()]
        assert np.allclose(main_effects, benchmark.main_effects, atol=2e-3)

    def test_repeat_qmc_sobol_sensitivity_analysis_ishigami(self):
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)

        nsamples = 10000
        nvars = benchmark.variable.num_vars()
        order = 3
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, np.where(
            interaction_terms.max(axis=0) == 1)[0]]

        sampling_method = 'sobol'
        nsobol_realizations = 5
        result = repeat_sampling_based_sobol_indices(
            benchmark.fun, benchmark.variable, interaction_terms, nsamples,
            sampling_method, nsobol_realizations=nsobol_realizations)
        rep_sobol_indices, rep_total_effect_indices, rep_var, rep_mean = (
            result["sobol_indices"]["values"],
            result["total_effects"]["values"],
            result["variance"]["values"], result["mean"]["values"])

        assert np.allclose(rep_mean.mean(axis=0), benchmark.mean, atol=2e-3)
        # check that there is variation in output. If not then we are not
        # generating different samlpe sets for each sobol realization
        assert rep_mean.std(axis=0) > 0

        sobol_indices = rep_sobol_indices.mean(axis=0)
        main_effects = sobol_indices[:nvars]
        assert np.allclose(main_effects, benchmark.main_effects, atol=2e-3)

        total_effect_indices = rep_total_effect_indices.mean(axis=0)
        assert np.allclose(
            total_effect_indices, benchmark.total_effects, atol=2e-3)

        for ii in range(interaction_terms.shape[1]):
            index = interaction_terms[:, ii]
            assert np.allclose(
                np.where(index > 0)[0],
                benchmark.sobol_interaction_indices[ii])

        sobol_indices = rep_sobol_indices.mean(axis=0)
        assert np.allclose(sobol_indices, benchmark.sobol_indices,
                           rtol=5e-3, atol=1e-3)

    def test_qmc_sobol_sensitivity_analysis_oakley(self):
        benchmark = setup_benchmark("oakley")

        nsamples = 100000
        nvars = benchmark.variable.num_vars()
        order = 1
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, np.where(
            interaction_terms.max(axis=0) == 1)[0]]

        fun = benchmark.fun
        for sampling_method in ['sobol', 'random', 'halton']:
            sobol_indices, total_effect_indices, var, mean = \
                sampling_based_sobol_indices(
                    fun, benchmark.variable, interaction_terms, nsamples,
                    sampling_method)

            # print(mean-benchmark.mean)
            assert np.allclose(mean, benchmark.mean, atol=2e-2)
            # print(var-benchmark.variance)
            assert np.allclose(benchmark.variance, var, rtol=2e-2)
            main_effects = sobol_indices[:nvars]
            # print(main_effects-benchmark.main_effects)
            assert np.allclose(main_effects, benchmark.main_effects, atol=2e-2)

    def test_sampling_based_sobol_indices_from_gaussian_process(self):
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        nvars = benchmark.variable.num_vars()

        # nsobol_samples and ntrain_samples effect assert tolerances
        ntrain_samples = 500
        nsobol_samples = int(1e4)
        train_samples = benchmark.variable.rvs(ntrain_samples)
        # from pyapprox import CholeskySampler
        # sampler = CholeskySampler(nvars, 10000, benchmark.variable)
        # kernel = Matern(
        #     np.array([1]*nvars), length_scale_bounds='fixed', nu=np.inf)
        # sampler.set_kernel(kernel)
        # train_samples = sampler(ntrain_samples)[0]

        train_vals = benchmark.fun(train_samples)
        approx = approximate(
            train_samples, train_vals, 'gaussian_process', {
                'nu': np.inf, 'normalize_y': True}).approx

        from pyapprox.surrogates.approximate import compute_l2_error
        error = compute_l2_error(
            approx, benchmark.fun, benchmark.variable,
            nsobol_samples, rel=True)
        print('error', error)
        # assert error < 4e-2

        order = 2
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, np.where(
            interaction_terms.max(axis=0) == 1)[0]]

        result = sampling_based_sobol_indices_from_gaussian_process(
            approx, benchmark.variable, interaction_terms,
            nsobol_samples, sampling_method='sobol',
            ngp_realizations=1000, normalize=True, nsobol_realizations=3,
            stat_functions=(np.mean, np.std), ninterpolation_samples=1000,
            ncandidate_samples=2000)

        mean_mean = result['mean']['mean']
        mean_sobol_indices = result['sobol_indices']['mean']
        mean_total_effects = result['total_effects']['mean']
        mean_main_effects = mean_sobol_indices[:nvars]

        print(benchmark.mean-mean_mean)
        print(benchmark.main_effects[:, 0]-mean_main_effects)
        print(benchmark.total_effects[:, 0]-mean_total_effects)
        print(benchmark.sobol_indices[:-1, 0]-mean_sobol_indices)
        assert np.allclose(mean_mean, benchmark.mean, atol=3e-2)
        assert np.allclose(mean_main_effects,
                           benchmark.main_effects[:, 0], atol=1e-2)
        assert np.allclose(mean_total_effects,
                           benchmark.total_effects[:, 0], atol=1e-2)
        assert np.allclose(mean_sobol_indices,
                           benchmark.sobol_indices[:-1, 0], atol=1e-2)

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 3, figsize=(3*8, 6))
        # mean_main_effects = mean_sobol_indices[:nvars]
        # std_main_effects = std_sobol_indices[:nvars]
        # axs[0].set_title(r'$\mathrm{Main\;Effects}$')
        # axs[2].set_title(r'$\mathrm{Total\;Effects}$')
        # axs[1].set_title(r'$\mathrm{Sobol\;Indices}$')
        # bp0 = plot_sensitivity_indices_with_confidence_intervals(
        #     mean_main_effects, std_main_effects,
        #     [r'$z_{%d}$'%(ii+1) for ii in range(nvars)], axs[0],
        #     benchmark.main_effects)
        # # axs[0].legend([bp0['means'][0]], ['$\mathrm{Truth}$'])
        # bp2 = plot_sensitivity_indices_with_confidence_intervals(
        #     mean_total_effects, std_total_effects,
        #     [r'$z_{%d}$'%(ii+1) for ii in range(nvars)], axs[2],
        #     benchmark.total_effects)
        # axs[2].legend([bp2['means'][0]], ['$\mathrm{Truth}$'])
        # I = np.argsort(mean_sobol_indices+2*std_sobol_indices)
        # mean_sobol_indices = mean_sobol_indices[I]
        # std_sobol_indices = std_sobol_indices[I]
        # rv = 'z'
        # labels = []
        # interaction_terms = [
        #     np.where(index>0)[0] for index in interaction_terms.T]
        # for ii in range(I.shape[0]):
        #     l = '($'
        #     for jj in range(len(interaction_terms[ii])-1):
        #         l += '%s_{%d},' % (rv, interaction_terms[ii][jj]+1)
        #     l += '%s_{%d}$)' % (rv, interaction_terms[ii][-1]+1)
        #     labels.append(l)
        # labels = [labels[ii] for ii in I]
        # bp1 = plot_sensitivity_indices_with_confidence_intervals(
        #     mean_sobol_indices, std_sobol_indices, labels, axs[1],
        #     benchmark.sobol_indices[I])
        # # axs[1].legend([bp1['means'][0]], ['$\mathrm{Truth}$'])
        # plt.tight_layout()
        # plt.show()

    def test_analytic_sobol_indices_from_gaussian_process(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.surrogates.approximate import approximate
        benchmark = setup_benchmark("ishigami", a=.1, b=0.1)

        # def fun(xx):
        #     vals = np.sum(xx, axis=0)[:, None]
        #     return vals
        # benchmark.variable = IndependentMarginalsVariable(
        #     [stats.norm(0, 1)]*1)
        #     #[stats.uniform(0, 1)]*1)
        # benchmark.fun = fun

        nvars = benchmark.variable.num_vars()
        ntrain_samples = 500
        train_samples = sobol_sequence(
            nvars, ntrain_samples, variable=benchmark.variable, start_index=1)

        train_vals = benchmark.fun(train_samples)
        # print(train_vals)
        approx = approximate(
            train_samples, train_vals, 'gaussian_process', {
                'nu': np.inf, 'normalize_y': False, 'alpha': 1e-6}).approx

        nsobol_samples = int(1e4)
        from pyapprox.surrogates.approximate import compute_l2_error
        error = compute_l2_error(
            approx, benchmark.fun, benchmark.variable,
            nsobol_samples, rel=True)
        print('error', error)

        order = 2
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, np.where(
            interaction_terms.max(axis=0) == 1)[0]]

        result = analytic_sobol_indices_from_gaussian_process(
            approx, benchmark.variable, interaction_terms,
            # ngp_realizations=0,
            ngp_realizations=1000,
            summary_stats=["mean", "std"],
            ninterpolation_samples=2000, ncandidate_samples=3000,
            use_cholesky=False, alpha=1e-7,
            nquad_samples=50)

        mean_mean = result['mean']['mean']
        mean_sobol_indices = result['sobol_indices']['mean']
        mean_total_effects = result['total_effects']['mean']
        mean_main_effects = mean_sobol_indices[:nvars]
        # print(mean_sobol_indices[:nvars], 's')
        # print(benchmark.main_effects[:, 0])

        # print(result['mean']['values'][-1])
        # print(result['variance']['values'][-1])
        # print(benchmark.main_effects[:, 0]-mean_main_effects)
        # print(benchmark.total_effects[:, 0]-mean_total_effects)
        # print(benchmark.sobol_indices[:-1, 0]-mean_sobol_indices)
        assert np.allclose(mean_mean, benchmark.mean, rtol=1e-3, atol=3e-3)
        assert np.allclose(mean_main_effects,
                           benchmark.main_effects[:, 0], rtol=1e-3, atol=3e-3)
        assert np.allclose(mean_total_effects,
                           benchmark.total_effects[:, 0], rtol=1e-3, atol=3e-3)
        assert np.allclose(
            mean_sobol_indices,
            benchmark.sobol_indices[:-1, 0], rtol=1e-3, atol=3e-3)

        result = run_sensitivity_analysis(
            "gp_sobol", approx, benchmark.variable, interaction_terms)
        assert np.allclose(mean_mean, benchmark.mean, rtol=1e-3, atol=3e-3)

    def test_marginalize_polynomial_chaos_expansions(self):
        univariate_variables = [
            stats.uniform(-1, 2), stats.norm(0, 1), stats.uniform(-1, 2)]
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)
        num_vars = len(univariate_variables)

        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)

        degree = 2
        indices = compute_hyperbolic_indices(num_vars, degree, 1)
        poly.set_indices(indices)
        poly.set_coefficients(np.ones((indices.shape[1], 1)))

        pce_main_effects, pce_total_effects =\
            get_main_and_total_effect_indices_from_pce(
                poly.get_coefficients(), poly.get_indices())
        print(poly.num_terms())

        for ii in range(num_vars):
            # Marginalize out 2 variables
            xx = np.linspace(-1, 1, 101)
            inactive_idx = np.hstack(
                (np.arange(ii), np.arange(ii+1, num_vars)))
            marginalized_pce = marginalize_polynomial_chaos_expansion(
                poly, inactive_idx, center=True)
            mvals = marginalized_pce(xx[None, :])
            variable_ii = variable.marginals()[ii:ii+1]
            var_trans_ii = AffineTransform(variable_ii)
            poly_ii = PolynomialChaosExpansion()
            poly_opts_ii = \
                define_poly_options_from_variable_transformation(
                    var_trans_ii)
            poly_ii.configure(poly_opts_ii)
            indices_ii = compute_hyperbolic_indices(1, degree, 1.)
            poly_ii.set_indices(indices_ii)
            poly_ii.set_coefficients(np.ones((indices_ii.shape[1], 1)))
            pvals = poly_ii(xx[None, :])
            # import matplotlib.pyplot as plt
            # plt.plot(xx, pvals)
            # plt.plot(xx, mvals, '--')
            # plt.show()
            assert np.allclose(mvals, pvals-poly.mean())
            assert np.allclose(
                poly_ii.variance()/poly.variance(), pce_main_effects[ii])
            poly_ii.coefficients /= np.sqrt(poly.variance())
            assert np.allclose(poly_ii.variance(), pce_main_effects[ii])

            # Marginalize out 1 variable
            xx = cartesian_product([xx]*2)
            inactive_idx = np.array([ii])
            marginalized_pce = marginalize_polynomial_chaos_expansion(
                poly, inactive_idx, center=True)
            mvals = marginalized_pce(xx)
            variable_ii = variable.marginals()[:ii] +\
                variable.marginals()[ii+1:]
            var_trans_ii = AffineTransform(variable_ii)
            poly_ii = PolynomialChaosExpansion()
            poly_opts_ii = \
                define_poly_options_from_variable_transformation(
                    var_trans_ii)
            poly_ii.configure(poly_opts_ii)
            indices_ii = compute_hyperbolic_indices(2, degree, 1.)
            poly_ii.set_indices(indices_ii)
            poly_ii.set_coefficients(np.ones((indices_ii.shape[1], 1)))
            pvals = poly_ii(xx)
            assert np.allclose(mvals, pvals-poly.mean())


if __name__ == "__main__":
    sensitivity_analysis_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSensitivityAnalysis)
    unittest.TextTestRunner(verbosity=2).run(sensitivity_analysis_test_suite)

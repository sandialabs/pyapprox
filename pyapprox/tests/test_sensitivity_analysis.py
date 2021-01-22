import unittest
from pyapprox.sensitivity_analysis import *
from pyapprox.benchmarks.sensitivity_benchmarks import *
from scipy.stats import uniform
import pyapprox as pya


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
        univariate_variables = [uniform(-np.pi, 2*np.pi)]*nvars
        variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)

        var_trans = pya.AffineRandomVariableTransformation(variable)
        poly = pya.PolynomialChaosExpansion()
        poly_opts = pya.define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)
        indices = pya.compute_hyperbolic_indices(nvars, degree, 1.0)
        poly.set_indices(indices)
        #print('No. PCE Terms',indices.shape[1])

        samples = pya.generate_independent_random_samples(
            var_trans.variable, nsamples)
        values = ishigami_function(samples)

        basis_matrix = poly.basis_matrix(samples)
        coef = np.linalg.lstsq(basis_matrix, values, rcond=None)[0]
        poly.set_coefficients(coef)

        nvalidation_samples = 1000
        validation_samples = pya.generate_independent_random_samples(
            var_trans.variable, nvalidation_samples)
        validation_values = ishigami_function(validation_samples)
        poly_validation_vals = poly(validation_samples)
        abs_error = np.linalg.norm(
            poly_validation_vals-validation_values)/np.sqrt(nvalidation_samples)
        #print('Abs. Error',abs_error)

        pce_main_effects, pce_total_effects =\
            pya.get_main_and_total_effect_indices_from_pce(
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
        univariate_variables = [uniform(0, 1)]*nvars
        variable = pya.IndependentMultivariateRandomVariable(
            univariate_variables)

        var_trans = pya.AffineRandomVariableTransformation(variable)
        poly = pya.PolynomialChaosExpansion()
        poly_opts = pya.define_poly_options_from_variable_transformation(
            var_trans)
        poly.configure(poly_opts)
        indices = pya.tensor_product_indices([degree]*nvars)
        poly.set_indices(indices)
        #print('No. PCE Terms',indices.shape[1])

        samples = pya.generate_independent_random_samples(
            var_trans.variable, nsamples)
        samples = (np.cos(np.random.uniform(0, np.pi, (nvars, nsamples)))+1)/2
        values = sobol_g_function(a, samples)

        basis_matrix = poly.basis_matrix(samples)
        weights = 1/np.sum(basis_matrix**2, axis=1)[:, np.newaxis]
        coef = np.linalg.lstsq(basis_matrix*weights,
                               values*weights, rcond=None)[0]
        poly.set_coefficients(coef)

        nvalidation_samples = 1000
        validation_samples = pya.generate_independent_random_samples(
            var_trans.variable, nvalidation_samples)
        validation_values = sobol_g_function(a, validation_samples)

        poly_validation_vals = poly(validation_samples)
        rel_error = np.linalg.norm(
            poly_validation_vals-validation_values)/np.linalg.norm(
                validation_values)
        print('Rel. Error', rel_error)

        pce_main_effects, pce_total_effects =\
            pya.get_main_and_total_effect_indices_from_pce(
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
        I = np.where(np.all(indices[0:2, :] > 0, axis=0))[0]

        true_interaction_values = np.vstack((
            np.tile(np.arange(1, 3)[np.newaxis, :],
                    (num_vars, 1))**2*degree/true_variance,
            np.tile(np.arange(1, 3)[np.newaxis, :],
                    (num_vars, 1))**2*num_pairwise_interactions/true_variance))

        assert np.allclose(true_interaction_values, interaction_values)

        #plot_interaction_values( interaction_values, interaction_indices)

    def test_get_main_and_total_effect_indices_from_pce(self):
        num_vars = 3
        degree = num_vars
        max_order = 2
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
        from functools import partial
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
        # ix1 = 0
        # for ii in range(ntrajectories):
        #     ix2 = ix1+nvars+1
        #     plt.plot(samples[0,ix1:ix2],samples[1,ix1:ix2],'-o')
        #     ix1=ix2
        # plt.xlim([0,1]); plt.ylim([0,1]); plt.show()

    def test_analyze_sensitivity_sparse_grid(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.adaptive_sparse_grid import isotropic_refinement_indicator
        benchmark = setup_benchmark("oakley")
        options = {'approx_options': {'max_nsamples': 2000}, 'max_order': 2}
        # 'refinement_indicator':isotropic_refinement_indicator}
        res = adaptive_analyze_sensitivity(
            benchmark.fun, benchmark.variable.all_variables(), "sparse_grid",
            options=options)

        # print(res.main_effects-benchmark.main_effects)
        assert np.allclose(res.main_effects, benchmark.main_effects, atol=2e-4)

    def test_analyze_sensitivity_polynomial_chaos(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.approximate import approximate
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)

        num_samples = 1000
        train_samples = pya.generate_independent_random_samples(
            benchmark.variable, num_samples)
        train_vals = benchmark.fun(train_samples)

        pce = approximate(
            train_samples, train_vals, 'polynomial_chaos',
            {'basis_type': 'hyperbolic_cross', 'variable': benchmark.variable,
             'options': {'max_degree': 8}}).approx

        res = analyze_sensitivity_polynomial_chaos(pce)
        assert np.allclose(res.main_effects, benchmark.main_effects, atol=2e-3)

    def test_analyze_sensitivity_sparse_grid(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.approximate import adaptive_approximate
        benchmark = setup_benchmark("oakley")

        options = {'max_nsamples': 2000, 'verbose': 0}
        approx = adaptive_approximate(
            benchmark.fun, benchmark.variable.all_variables(),
            'sparse_grid', options).approx

        from pyapprox.approximate import compute_l2_error
        nsamples = 100
        error = compute_l2_error(
            approx, benchmark.fun, approx.variable_transformation.variable,
            nsamples, rel=True)
        # print(error)
        assert error < 3e-3

        res = analyze_sensitivity_sparse_grid(approx)
        assert np.allclose(res.main_effects, benchmark.main_effects, atol=2e-4)

    def test_qmc_sobol_sensitivity_analysis_ishigami(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.approximate import approximate
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)

        nsamples = 10000
        nvars = benchmark.variable.num_vars()
        order = 3
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, 
            np.where(interaction_terms.max(axis=0)==1)[0]]

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
                np.where(index>0)[0], benchmark.sobol_interaction_indices[ii])
        assert np.allclose(sobol_indices, benchmark.sobol_indices,
                           rtol=5e-3, atol=1e-3)

    def test_repeat_qmc_sobol_sensitivity_analysis_ishigami(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.approximate import approximate
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)

        nsamples = 10000
        nvars = benchmark.variable.num_vars()
        order = 3
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, 
            np.where(interaction_terms.max(axis=0)==1)[0]]

        sampling_method = 'sobol'
        nsobol_realizations = 5
        rep_sobol_indices, rep_total_effect_indices, rep_var, rep_mean = \
            repeat_sampling_based_sobol_indices(
                benchmark.fun, benchmark.variable, interaction_terms, nsamples,
                sampling_method, nsobol_realizations=nsobol_realizations)

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
                np.where(index>0)[0], benchmark.sobol_interaction_indices[ii])

        sobol_indies = rep_sobol_indices.mean(axis=0)
        assert np.allclose(sobol_indices, benchmark.sobol_indices,
                           rtol=5e-3, atol=1e-3)

    def test_qmc_sobol_sensitivity_analysis_oakley(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.approximate import approximate
        benchmark = setup_benchmark("oakley")

        nsamples = 10000
        nvars = benchmark.variable.num_vars()
        order = 1
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, 
            np.where(interaction_terms.max(axis=0)==1)[0]]

        sampling_method = 'sobol'
        sobol_indices, total_effect_indices, var, mean = \
            sampling_based_sobol_indices(
                benchmark.fun, benchmark.variable, interaction_terms, nsamples,
                sampling_method)

        assert np.allclose(mean, benchmark.mean, atol=2e-2)

        assert np.allclose(benchmark.variance, var, rtol=2e-2)

        main_effects = sobol_indices[:nvars]
        assert np.allclose(main_effects, benchmark.main_effects, atol=2e-2)


    def test_sampling_based_sobol_indices_from_gaussian_process(self):
        from pyapprox.benchmarks.benchmarks import setup_benchmark
        from pyapprox.approximate import approximate
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        nvars = benchmark.variable.num_vars()

        # nsobol_samples and ntrain_samples effect assert tolerances
        ntrain_samples = 300
        nsobol_samples = 1000
        train_samples = pya.generate_independent_random_samples(
           benchmark.variable, ntrain_samples)
        # from pyapprox import CholeskySampler
        # sampler = CholeskySampler(nvars, 10000, benchmark.variable)
        # kernel = pya.Matern(
        #     np.array([1]*nvars), length_scale_bounds='fixed', nu=np.inf)
        # sampler.set_kernel(kernel)
        # train_samples = sampler(ntrain_samples)[0]

        train_vals = benchmark.fun(train_samples)
        approx = approximate(
            train_samples, train_vals, 'gaussian_process', {
                'nu':np.inf, 'normalize_y': True}).approx

        from pyapprox.approximate import compute_l2_error
        error = compute_l2_error(
            approx, benchmark.fun, benchmark.variable,
            nsobol_samples, rel=True)
        print(error)
        # assert error < 4e-2

        order = 2
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[:, 
            np.where(interaction_terms.max(axis=0)==1)[0]]

        mean_sobol_indices, mean_total_effects, mean_variance, mean_mean, \
            std_sobol_indices, std_total_effects, std_variance, std_mean = \
                sampling_based_sobol_indices_from_gaussian_process(
                    approx, benchmark.variable, interaction_terms,
                    nsobol_samples, sampling_method='sobol',
                    ngp_realizations=1000, normalize=True,
                    nsobol_realizations=3)

        assert np.allclose(mean_mean, benchmark.mean, atol=3e-2)
        
        mean_main_effects = mean_sobol_indices[:nvars]
        assert np.allclose(mean_main_effects,
                           benchmark.main_effects[:, 0], atol=4e-2)
        assert np.allclose(mean_total_effects,
                           benchmark.total_effects[:, 0], atol=2e-2)
        assert np.allclose(mean_sobol_indices,
                           benchmark.sobol_indices[:-1, 0], atol=5e-2)
        #print(benchmark.main_effects[:, 0]-mean_main_effects)
        #print(benchmark.total_effects[:, 0]-mean_total_effects)
        #print(benchmark.sobol_indices[:-1, 0]-mean_sobol_indices)

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


if __name__ == "__main__":
    sensitivity_analysis_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSensitivityAnalysis)
    unittest.TextTestRunner(verbosity=2).run(sensitivity_analysis_test_suite)

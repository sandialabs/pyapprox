import unittest
from scipy import stats
import numpy as np
from functools import partial

from pyapprox.benchmarks import (
    IshigamiBenchmark,
    OakleyBenchmark,
    SobolGBenchmark,
)
from pyapprox.expdesign.sequences import SobolSequence
from pyapprox.analysis.sensitivity_analysis import (
    SobolSequenceBasedSensitivityAnalysis,
    HaltonSequenceBasedSensitivityAnalysis,
    MonteCarloBasedSensitivityAnalysis,
    MorrisSensitivityAnalysis,
    BinBasedVarianceSensitivityAnalysis,
    PolynomialChaosSensivitityAnalysis,
    LagrangeSparseGridSensitivityAnalysis,
    sampling_based_sobol_indices_from_gaussian_process,
    analytic_sobol_indices_from_gaussian_process,
)
from pyapprox.surrogates.bases.basisexp import (
    setup_polynomial_chaos_expansion_from_variable,
)
from pyapprox.surrogates.sparsegrids.combination import (
    MaxLevelSparseGridSubSpaceAdmissibilityCriteria,
    MaxErrorSparseGridSubspaceAdmissibilityCriteria,
    MultipleSparseGridSubSpaceAdmissibilityCriteria,
    setup_leja_lagrange_sparse_grid_from_variable,
)


class TestSensitivityAnalysis(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_sample_based_sobol_sensitivity_analysis_ishigami(self):
        benchmark = IshigamiBenchmark(a=7, b=0.1)
        nsamples = 10000
        analyzer = SobolSequenceBasedSensitivityAnalysis(
            benchmark.variable(), 100
        )
        analyzer.set_interaction_terms_of_interest(
            benchmark.sobol_interaction_indices()
        )
        samples = analyzer.generate_samples(nsamples)
        values = benchmark.model()(samples)
        analyzer.compute(values)

        assert np.allclose(analyzer.mean(), benchmark.mean(), atol=2e-3)
        assert np.allclose(
            analyzer.main_effects(), benchmark.main_effects(), atol=2e-3
        )
        assert np.allclose(
            analyzer.total_effects(), benchmark.total_effects(), atol=2e-3
        )
        assert np.allclose(
            analyzer.sobol_indices(),
            benchmark.sobol_indices(),
            rtol=5e-3,
            atol=1e-3,
        )

    def _check_sample_based_sobol_sensitivity_analysis_oakley(
        self, analyzer_cls, *args
    ):
        nsamples = 100000
        benchmark = OakleyBenchmark()
        analyzer = analyzer_cls(benchmark.variable(), *args)
        order = 1
        analyzer.set_interaction_terms_of_interest(
            analyzer.isotropic_interaction_terms(order)
        )
        samples = analyzer.generate_samples(nsamples)
        values = benchmark.model()(samples)
        analyzer.compute(values)

        # print(result.mean-benchmark.mean())
        assert np.allclose(analyzer.mean(), benchmark.mean(), atol=2e-2)
        # print(result.variance-benchmark.variance())
        assert np.allclose(
            analyzer.variance(), benchmark.variance(), rtol=2e-2
        )
        # print(result.main_effects-benchmark.main_effects)
        assert np.allclose(
            analyzer.main_effects(), benchmark.main_effects(), atol=2e-2
        )

    def test_sample_based_sobol_sensitivity_analysis_oakley(self):
        test_cases = [
            [SobolSequenceBasedSensitivityAnalysis, 100],
            [HaltonSequenceBasedSensitivityAnalysis, 100],
            [MonteCarloBasedSensitivityAnalysis],
        ]
        for test_case in test_cases:
            self._check_sample_based_sobol_sensitivity_analysis_oakley(
                *test_case
            )

    def test_morris_method(self):
        benchmark = IshigamiBenchmark()

        nlevels, ncandidate_trajectories, ntrajectories = 4, None, 20
        analyzer = MorrisSensitivityAnalysis(benchmark.variable(), nlevels)
        samples = analyzer.generate_samples(
            ntrajectories, ncandidate_trajectories
        )
        values = benchmark.model()(samples)
        analyzer.compute(values)
        analyzer.print_sensitivity_indices()
        # regression test
        assert np.allclose(
            analyzer.mu()[:, 0],
            np.array([10.82845216, 0.7875, 1.87463883]),
        )
        assert np.allclose(
            analyzer.mu_star()[:, 0],
            np.array([10.82845216, 7.875, 6.87367571]),
        )
        assert np.allclose(
            analyzer.sigma()[:, 0],
            np.array([5.55220104, 8.03908012, 9.31270356]),
        )

    def test_bin_based_variance_sensivitity_analysis(self):
        benchmark = IshigamiBenchmark()
        analyzer = BinBasedVarianceSensitivityAnalysis(benchmark.variable())
        nsamples = int(1e6)
        seq = SobolSequence(
            benchmark.variable().nvars(), variable=benchmark.variable()
        )
        samples = seq.rvs(nsamples)
        values = benchmark.model()(samples)
        analyzer.compute(samples, values)
        assert np.allclose(
            analyzer.main_effects(),
            benchmark.main_effects(),
            rtol=1e-2,
            atol=1e-3,
        )

    def _check_pce_sensitivities(
        self, benchmark, degree, nsamples, l2_tol, sa_tol
    ):
        pce = setup_polynomial_chaos_expansion_from_variable(
            benchmark.variable(), benchmark.model().nqoi()
        )
        pce.basis().set_hyperbolic_indices(degree, 1.0)
        assert nsamples > pce.basis().nterms()
        samples = benchmark.variable().rvs(nsamples)
        values = benchmark.model()(samples)
        pce.fit(samples, values)
        nvalidation_samples = 1000
        validation_samples = benchmark.variable().rvs(nvalidation_samples)
        validation_values = benchmark.model()(validation_samples)
        pce_vals = pce(validation_samples)
        abs_error = np.linalg.norm(pce_vals - validation_values) / np.sqrt(
            nvalidation_samples
        )
        # print("Abs. Error", abs_error)
        assert abs_error < l2_tol

        analyzer = PolynomialChaosSensivitityAnalysis(pce.nvars())
        analyzer.set_interaction_terms_of_interest(
            benchmark.sobol_interaction_indices()
        )
        analyzer.compute(pce)
        assert np.allclose(pce.mean(), benchmark.mean(), rtol=sa_tol)
        assert np.allclose(pce.variance(), benchmark.variance(), rtol=sa_tol)
        assert np.allclose(
            analyzer.main_effects(), benchmark.main_effects(), rtol=sa_tol
        )
        assert np.allclose(
            analyzer.total_effects(), benchmark.total_effects(), rtol=sa_tol
        )
        assert np.allclose(
            analyzer.sobol_indices(), benchmark.sobol_indices(), rtol=sa_tol
        )

    def test_pce_sensitivities(self):
        test_cases = [
            [IshigamiBenchmark(), 18, 2000, 1e-5, 1e-7],
            [SobolGBenchmark(nvars=2), 20, 4000, 3e-2, 5e-3],
        ]
        for test_case in test_cases:
            self._check_pce_sensitivities(*test_case)

    def test_sparse_grid_sobol_sensitivities(self):
        benchmark = IshigamiBenchmark()
        level = 8
        admissibility_criteria = (
            MultipleSparseGridSubSpaceAdmissibilityCriteria(
                (
                    MaxLevelSparseGridSubSpaceAdmissibilityCriteria(
                        level, 1.0
                    ),
                    MaxErrorSparseGridSubspaceAdmissibilityCriteria(1e-8),
                )
            )
        )
        sg = setup_leja_lagrange_sparse_grid_from_variable(
            benchmark.model().nqoi(),
            benchmark.variable(),
            admissibility_criteria,
        )
        sg.build(benchmark.model())
        print(sg.get_train_samples().shape)
        analyzer = LagrangeSparseGridSensitivityAnalysis(benchmark.variable())
        analyzer.set_interaction_terms_of_interest(
            benchmark.sobol_interaction_indices()
        )
        analyzer.compute(sg)
        sa_tol = 1e-5
        print(analyzer.main_effects(), benchmark.main_effects())
        assert np.allclose(
            analyzer.main_effects(), benchmark.main_effects(), rtol=sa_tol
        )
        assert np.allclose(
            analyzer.total_effects(), benchmark.total_effects(), rtol=sa_tol
        )
        assert np.allclose(
            analyzer.sobol_indices(), benchmark.sobol_indices(), rtol=sa_tol
        )

    def test_sampling_based_sobol_indices_from_gaussian_process(self):
        benchmark = setup_benchmark("ishigami", a=7, b=0.1)
        nvars = benchmark.variable.nvars()

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
            train_samples,
            train_vals,
            "gaussian_process",
            {"nu": np.inf, "normalize_y": True},
        ).approx

        from pyapprox.surrogates.approximate import compute_l2_error

        error = compute_l2_error(
            approx, benchmark.fun, benchmark.variable, nsobol_samples, rel=True
        )
        print("error", error)
        # assert error < 4e-2

        order = 2
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[
            :, np.where(interaction_terms.max(axis=0) == 1)[0]
        ]

        result = sampling_based_sobol_indices_from_gaussian_process(
            approx,
            benchmark.variable,
            interaction_terms,
            nsobol_samples,
            sampling_method="sobol",
            ngp_realizations=1000,
            normalize=True,
            nsobol_realizations=3,
            stat_functions=(np.mean, np.std),
            ninterpolation_samples=1000,
            ncandidate_samples=2000,
        )

        mean_mean = result["mean"]["mean"]
        mean_sobol_indices = result["sobol_indices"]["mean"]
        mean_total_effects = result["total_effects"]["mean"]
        mean_main_effects = mean_sobol_indices[:nvars]

        print(benchmark.mean - mean_mean)
        print(benchmark.main_effects[:, 0] - mean_main_effects)
        print(benchmark.total_effects[:, 0] - mean_total_effects)
        print(benchmark.sobol_indices[:-1, 0] - mean_sobol_indices)
        assert np.allclose(mean_mean, benchmark.mean, atol=3e-2)
        assert np.allclose(
            mean_main_effects, benchmark.main_effects[:, 0], atol=1e-2
        )
        assert np.allclose(
            mean_total_effects, benchmark.total_effects[:, 0], atol=1e-2
        )
        assert np.allclose(
            mean_sobol_indices, benchmark.sobol_indices[:-1, 0], atol=1e-2
        )

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

        benchmark = setup_benchmark("ishigami", a=0.1, b=0.1)

        # def fun(xx):
        #     vals = np.sum(xx, axis=0)[:, None]
        #     return vals
        # benchmark.variable = IndependentMarginalsVariable(
        #     [stats.norm(0, 1)]*1)
        #     #[stats.uniform(0, 1)]*1)
        # benchmark.fun = fun

        nvars = benchmark.variable.nvars()
        ntrain_samples = 500
        train_samples = sobol_sequence(
            nvars, ntrain_samples, variable=benchmark.variable, start_index=1
        )

        train_vals = benchmark.fun(train_samples)
        # print(train_vals)
        approx = approximate(
            train_samples,
            train_vals,
            "gaussian_process",
            {"nu": np.inf, "normalize_y": False, "alpha": 1e-6},
        ).approx

        nsobol_samples = int(1e4)
        from pyapprox.surrogates.approximate import compute_l2_error

        error = compute_l2_error(
            approx, benchmark.fun, benchmark.variable, nsobol_samples, rel=True
        )
        print("error", error)

        order = 2
        interaction_terms = compute_hyperbolic_indices(nvars, order)
        interaction_terms = interaction_terms[
            :, np.where(interaction_terms.max(axis=0) == 1)[0]
        ]

        result = analytic_sobol_indices_from_gaussian_process(
            approx,
            benchmark.variable,
            interaction_terms,
            # ngp_realizations=0,
            ngp_realizations=1000,
            summary_stats=["mean", "std"],
            ninterpolation_samples=2000,
            ncandidate_samples=3000,
            use_cholesky=False,
            alpha=1e-7,
            nquad_samples=50,
        )

        mean_mean = result["mean"]["mean"]
        mean_sobol_indices = result["sobol_indices"]["mean"]
        mean_total_effects = result["total_effects"]["mean"]
        mean_main_effects = mean_sobol_indices[:nvars]
        # print(mean_sobol_indices[:nvars], 's')
        # print(benchmark.main_effects[:, 0])

        # print(result['mean']['values'][-1])
        # print(result['variance']['values'][-1])
        # print(benchmark.main_effects[:, 0]-mean_main_effects)
        # print(benchmark.total_effects[:, 0]-mean_total_effects)
        # print(benchmark.sobol_indices[:-1, 0]-mean_sobol_indices)
        assert np.allclose(mean_mean, benchmark.mean, rtol=1e-3, atol=3e-3)
        assert np.allclose(
            mean_main_effects,
            benchmark.main_effects[:, 0],
            rtol=1e-3,
            atol=3e-3,
        )
        assert np.allclose(
            mean_total_effects,
            benchmark.total_effects[:, 0],
            rtol=1e-3,
            atol=3e-3,
        )
        assert np.allclose(
            mean_sobol_indices,
            benchmark.sobol_indices[:-1, 0],
            rtol=1e-3,
            atol=3e-3,
        )

        result = run_sensitivity_analysis(
            "gp_sobol", approx, benchmark.variable, interaction_terms
        )
        assert np.allclose(mean_mean, benchmark.mean, rtol=1e-3, atol=3e-3)


if __name__ == "__main__":
    sensitivity_analysis_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestSensitivityAnalysis)
    )
    unittest.TextTestRunner(verbosity=2).run(sensitivity_analysis_test_suite)

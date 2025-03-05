import unittest

import numpy as np
import matplotlib.pyplot as plt

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
    EnsembleGaussianProcessSensivitityAnalysis,
)
from pyapprox.surrogates.bases.basisexp import (
    setup_polynomial_chaos_expansion_from_variable,
)
from pyapprox.surrogates.sparsegrids.combination import (
    MaxLevelSparseGridSubSpaceAdmissibilityCriteria,
    MaxErrorSparseGridSubspaceAdmissibilityCriteria,
    MultipleSparseGridSubSpaceAdmissibilityCriteria,
    LejaLagrangeAdaptiveCombinationSparseGrid,
)
from pyapprox.surrogates.kernels import MaternKernel
from pyapprox.surrogates.autogp.exactgp import (
    ExactGaussianProcess,
    GaussianProcessIdentityTransform,
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
        level = 7

        admissibility_criteria = (
            MultipleSparseGridSubSpaceAdmissibilityCriteria(
                [
                    MaxLevelSparseGridSubSpaceAdmissibilityCriteria(
                        level, 1.0
                    ),
                    MaxErrorSparseGridSubspaceAdmissibilityCriteria(1e-8),
                ]
            )
        )
        sg = LejaLagrangeAdaptiveCombinationSparseGrid(
            benchmark.variable(), benchmark.model().nqoi()
        )
        # Must not use default of mean or the refinement will terminate early
        # in the 3rd dimension
        init_sequences = [
            np.array([[marginal.ppf(0.6)]])
            for marginal in benchmark.variable().marginals()
        ]
        univariate_quad_rules = sg.unique_univariate_leja_quadrature_rules(
            init_sequences
        )
        sg.setup(
            admissibility_criteria, univariate_quad_rules=univariate_quad_rules
        )
        sg.build(benchmark.model())
        analyzer = LagrangeSparseGridSensitivityAnalysis(benchmark.variable())
        analyzer.set_interaction_terms_of_interest(
            benchmark.sobol_interaction_indices()
        )
        analyzer.compute(sg)
        sa_tol = 1e-5
        assert np.allclose(
            analyzer.main_effects(), benchmark.main_effects(), rtol=sa_tol
        )
        assert np.allclose(
            analyzer.total_effects(), benchmark.total_effects(), rtol=sa_tol
        )
        assert np.allclose(
            analyzer.sobol_indices(), benchmark.sobol_indices(), rtol=sa_tol
        )

    def _check_gp_sensitivities(self, benchmark, nsamples, l2_tol, sa_tol):

        # samples = benchmark.variable().rvs(nsamples)
        seq = SobolSequence(benchmark.nvars(), 0, benchmark.variable())
        samples = seq.rvs(nsamples)
        values = benchmark.model()(samples)
        nvalidation_samples = 1000
        validation_samples = benchmark.variable().rvs(nvalidation_samples)
        validation_values = benchmark.model()(validation_samples)

        kernel_reg = 1e-7
        from pyapprox.surrogates.gaussianprocess.gaussian_process import (
            GaussianProcess,
            Matern,
            ConstantKernel as CKernel,
            WhiteKernel,
            marginalize_gaussian_process,
            compute_expected_sobol_indices,
        )

        pyakernel = Matern(np.array([0.5, 0.5, 0.5]), (1e-2, 2), nu=np.inf)
        pyagp = GaussianProcess(pyakernel, alpha=kernel_reg)
        pyagp.fit(samples, values)
        print(
            "pya error",
            np.linalg.norm(
                (pyagp(validation_samples))
                - benchmark.model()(validation_samples)
            )
            / np.sqrt(nvalidation_samples),
        )
        print(pyagp.kernel_)
        print(
            pyagp.log_marginal_likelihood(
                pyagp.kernel_.theta, eval_gradient=True
            ),
            pyagp.kernel_.theta,
            "obj",
        )
        pyasobolindices, pyatotaleffects, pyamean, pyavar = (
            compute_expected_sobol_indices(
                pyagp,
                benchmark.variable(),
                benchmark.sobol_interaction_indices(),
            )
        )
        print(pyasobolindices)
        print(pyatotaleffects)

        # setup gp
        kernel = MaternKernel(
            np.inf,
            pyagp.kernel_.length_scale,
            [1e-1, 2],
            benchmark.nvars(),
            fixed=True,
        )
        # constant_kernel = ConstantKernel(
        #     1, (1e-1, 1e1), transform=LogHyperParameterTransform(), fixed=False
        # )
        # kernel = constant_kernel * kernel
        out_trans = GaussianProcessIdentityTransform()
        print(benchmark.nvars(), benchmark.variable().nvars())
        gp = ExactGaussianProcess(
            benchmark.variable().nvars(),
            kernel,
            trend=None,
            kernel_reg=kernel_reg,
        )
        # in_trans = AffineTransform(benchmark.variable())
        # gp.set_input_transform(in_trans)
        gp.set_output_transform(out_trans)
        gp.set_optimizer(ncandidates=1, verbosity=0)
        # train gp

        gp.fit(samples, values)

        gp_vals = gp(validation_samples)
        abs_error = np.linalg.norm(gp_vals - validation_values) / np.sqrt(
            nvalidation_samples
        )
        print("Abs. Error", abs_error)

        assert abs_error < l2_tol

        # compute sensivitity indices from gp
        # analyzer = GaussianProcessSensivitityAnalysis(benchmark.variable())
        analyzer = EnsembleGaussianProcessSensivitityAnalysis(
            benchmark.variable(), nrealizations=100
        )
        analyzer.set_interaction_terms_of_interest(
            benchmark.sobol_interaction_indices()
        )
        analyzer.compute(gp)

        assert np.allclose(analyzer.mean(), benchmark.mean(), rtol=sa_tol)
        assert np.allclose(
            analyzer.variance(), benchmark.variance(), rtol=sa_tol
        )
        # make sure there is variation in the computed statistics
        print(analyzer.variance("min"), analyzer.variance("max"))
        assert analyzer.variance("min") != analyzer.variance("max")
        assert np.allclose(
            analyzer.main_effects(),
            benchmark.main_effects(),
            rtol=sa_tol,
            atol=1e-5,
        )
        assert np.allclose(
            analyzer.total_effects(), benchmark.total_effects(), rtol=sa_tol
        )
        assert np.allclose(
            analyzer.sobol_indices(),
            benchmark.sobol_indices(),
            rtol=sa_tol,
            atol=1e-5,
        )

        # make sure plots run
        axs = plt.subplots(1, 3, figsize=(3 * 8, 6))[1]
        analyzer.plot_main_effects(axs[0])
        analyzer.plot_total_effects(axs[1])
        analyzer.plot_sobol_indices(axs[2])

    def test_gp_sensitivities(self):
        test_cases = [
            [IshigamiBenchmark(a=0.1, b=0.02), 1000, 2e-3, 5e-3],
        ]
        for test_case in test_cases:
            self._check_gp_sensitivities(*test_case)


if __name__ == "__main__":
    sensitivity_analysis_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestSensitivityAnalysis)
    )
    unittest.TextTestRunner(verbosity=2).run(sensitivity_analysis_test_suite)

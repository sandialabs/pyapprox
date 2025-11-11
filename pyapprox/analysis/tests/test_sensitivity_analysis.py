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
    PolynomialChaosSensitivityAnalysis,
    LagrangeSparseGridSensitivityAnalysis,
    EnsembleGaussianProcessSensitivityAnalysis,
)
from pyapprox.surrogates.affine.basisexp import (
    setup_polynomial_chaos_expansion_from_variable,
)
from pyapprox.surrogates.sparsegrids.combination import (
    MaxLevelSparseGridSubSpaceAdmissibilityCriteria,
    MaxErrorSparseGridSubspaceAdmissibilityCriteria,
    MultipleSparseGridSubSpaceAdmissibilityCriteria,
    LejaLagrangeAdaptiveCombinationSparseGrid,
)
from pyapprox.surrogates.kernels import MaternKernel
from pyapprox.surrogates.gaussianprocess.exactgp import (
    ExactGaussianProcess,
    GaussianProcessIdentityTransform,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


class TestSensitivityAnalysis:
    def setUp(self):
        np.random.seed(1)

    def test_sample_based_sobol_sensitivity_analysis_ishigami(self):
        bkd = self.get_backend()
        benchmark = IshigamiBenchmark(bkd, a=7, b=0.1)
        nsamples = 10000
        analyzer = SobolSequenceBasedSensitivityAnalysis(
            benchmark.prior(), 100
        )
        analyzer.set_interaction_terms_of_interest(
            benchmark.sobol_interaction_indices()
        )
        samples = analyzer.generate_samples(nsamples)
        values = benchmark.model()(samples)
        analyzer.compute(values)

        assert bkd.allclose(analyzer.mean(), benchmark.mean(), atol=2e-3)
        assert bkd.allclose(
            analyzer.main_effects(), benchmark.main_effects(), atol=2e-3
        )
        assert bkd.allclose(
            analyzer.total_effects(), benchmark.total_effects(), atol=2e-3
        )
        assert bkd.allclose(
            analyzer.sobol_indices(),
            benchmark.sobol_indices(),
            rtol=5e-3,
            atol=1e-3,
        )

    def _check_sample_based_sobol_sensitivity_analysis_oakley(
        self, analyzer_cls, *args
    ):
        bkd = self.get_backend()
        nsamples = 100000
        benchmark = OakleyBenchmark(bkd)
        analyzer = analyzer_cls(benchmark.prior(), *args)
        order = 1
        analyzer.set_interaction_terms_of_interest(
            analyzer.isotropic_interaction_terms(order)
        )
        samples = analyzer.generate_samples(nsamples)
        values = benchmark.model()(samples)
        analyzer.compute(values)

        assert bkd.allclose(analyzer.mean(), benchmark.mean(), atol=2e-2)
        assert bkd.allclose(
            analyzer.variance(), benchmark.variance(), rtol=2e-2
        )
        assert bkd.allclose(
            analyzer.main_effects(), benchmark.main_effects(), atol=2e-2
        )

    def test_sample_based_sobol_sensitivity_analysis_oakley(self):
        test_cases = [
            [SobolSequenceBasedSensitivityAnalysis, 100],
            [HaltonSequenceBasedSensitivityAnalysis, 100],
            [MonteCarloBasedSensitivityAnalysis],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_sample_based_sobol_sensitivity_analysis_oakley(
                *test_case
            )

    def test_morris_method(self):
        bkd = self.get_backend()
        benchmark = IshigamiBenchmark(bkd)

        nlevels, ncandidate_trajectories, ntrajectories = 4, None, 20
        analyzer = MorrisSensitivityAnalysis(benchmark.prior(), nlevels)
        samples = analyzer.generate_samples(
            ntrajectories, ncandidate_trajectories
        )
        values = benchmark.model()(samples)
        analyzer.compute(values)
        analyzer.print_sensitivity_indices()
        # regression test
        assert bkd.allclose(
            analyzer.mu()[:, 0],
            bkd.array([10.82845216, 0.7875, 1.87463883]),
        )
        assert bkd.allclose(
            analyzer.mu_star()[:, 0],
            bkd.array([10.82845216, 7.875, 6.87367571]),
        )
        assert bkd.allclose(
            analyzer.sigma()[:, 0],
            bkd.array([5.55220104, 8.03908012, 9.31270356]),
        )

    def test_bin_based_variance_sensitivity_analysis(self):
        bkd = self.get_backend()
        benchmark = IshigamiBenchmark(bkd)
        analyzer = BinBasedVarianceSensitivityAnalysis(benchmark.prior())
        nsamples = int(1e6)
        seq = SobolSequence(
            benchmark.prior().nvars(), variable=benchmark.prior(), bkd=bkd
        )
        samples = seq.rvs(nsamples)
        values = benchmark.model()(samples)
        analyzer.compute(samples, values)
        assert bkd.allclose(
            analyzer.main_effects(),
            benchmark.main_effects(),
            rtol=1e-2,
            atol=1e-3,
        )

    def _check_pce_sensitivities(
        self, benchmark, degree, nsamples, l2_tol, sa_tol
    ):
        bkd = self.get_backend()
        pce = setup_polynomial_chaos_expansion_from_variable(
            benchmark.prior(), benchmark.model().nqoi()
        )
        pce.basis().set_hyperbolic_indices(degree, 1.0)
        assert nsamples > pce.basis().nterms()
        samples = benchmark.prior().rvs(nsamples)
        values = benchmark.model()(samples)
        pce.fit(samples, values)
        nvalidation_samples = 1000
        validation_samples = benchmark.prior().rvs(nvalidation_samples)
        validation_values = benchmark.model()(validation_samples)
        pce_vals = pce(validation_samples)
        abs_error = bkd.norm(pce_vals - validation_values) / np.sqrt(
            nvalidation_samples
        )
        # print("Abs. Error", abs_error)
        assert abs_error < l2_tol

        analyzer = PolynomialChaosSensitivityAnalysis(pce.nvars(), pce._bkd)
        analyzer.set_interaction_terms_of_interest(
            benchmark.sobol_interaction_indices()
        )
        analyzer.compute(pce)
        assert bkd.allclose(pce.mean(), benchmark.mean(), rtol=sa_tol)
        assert bkd.allclose(pce.variance(), benchmark.variance(), rtol=sa_tol)
        assert bkd.allclose(
            analyzer.main_effects(), benchmark.main_effects(), rtol=sa_tol
        )
        assert bkd.allclose(
            analyzer.total_effects(), benchmark.total_effects(), rtol=sa_tol
        )
        assert bkd.allclose(
            analyzer.sobol_indices(), benchmark.sobol_indices(), rtol=sa_tol
        )

    def test_pce_sensitivities(self):
        bkd = self.get_backend()
        test_cases = [
            [IshigamiBenchmark(bkd), 18, 2000, 1e-5, 1e-7],
            [SobolGBenchmark(bkd, nvars=2), 20, 4000, 3e-2, 5e-3],
        ]
        for test_case in test_cases:
            self._check_pce_sensitivities(*test_case)

    def test_sparse_grid_sobol_sensitivities(self):
        bkd = self.get_backend()
        benchmark = IshigamiBenchmark(bkd)
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
            benchmark.prior(), benchmark.model().nqoi()
        )
        # Must not use default of mean or the refinement will terminate early
        # in the 3rd dimension
        init_sequences = [
            marginal.ppf(bkd.array([0.6]))[None, :]
            for marginal in benchmark.prior().marginals()
        ]
        univariate_quad_rules = sg.unique_univariate_leja_quadrature_rules(
            init_sequences
        )
        sg.setup(
            admissibility_criteria, univariate_quad_rules=univariate_quad_rules
        )
        sg.build(benchmark.model())
        analyzer = LagrangeSparseGridSensitivityAnalysis(benchmark.prior())
        analyzer.set_interaction_terms_of_interest(
            benchmark.sobol_interaction_indices()
        )
        analyzer.compute(sg)
        sa_tol = 1e-5
        assert bkd.allclose(
            analyzer.main_effects(), benchmark.main_effects(), rtol=sa_tol
        )
        assert bkd.allclose(
            analyzer.total_effects(), benchmark.total_effects(), rtol=sa_tol
        )
        assert bkd.allclose(
            analyzer.sobol_indices(), benchmark.sobol_indices(), rtol=sa_tol
        )

    def _check_gp_sensitivities(self, benchmark, nsamples, l2_tol, sa_tol):
        bkd = self.get_backend()
        # samples = benchmark.prior().rvs(nsamples)
        seq = SobolSequence(benchmark.nvars(), 0, benchmark.prior(), bkd=bkd)
        samples = seq.rvs(nsamples)
        values = benchmark.model()(samples)
        nvalidation_samples = 1000
        validation_samples = benchmark.prior().rvs(nvalidation_samples)
        validation_values = benchmark.model()(validation_samples)

        kernel_reg = 1e-7
        # setup gp
        kernel = MaternKernel(
            np.inf,
            bkd.array([0.5, 0.5, 0.5]),
            [1e-1, 2],
            benchmark.nvars(),
            fixed=False,
            backend=bkd,
        )
        # constant_kernel = ConstantKernel(
        #     1, (1e-1, 1e1), transform=LogHyperParameterTransform(), fixed=False, backend=bkd
        # )
        # kernel = constant_kernel * kernel
        out_trans = GaussianProcessIdentityTransform(backend=bkd)
        gp = ExactGaussianProcess(
            benchmark.prior().nvars(),
            kernel,
            trend=None,
            kernel_reg=kernel_reg,
        )
        # in_trans = AffineTransform(benchmark.prior())
        # gp.set_input_transform(in_trans)
        gp.set_output_transform(out_trans)
        gp.set_optimizer(ncandidates=1, verbosity=0)
        # train gp

        gp.fit(samples, values)

        gp_vals = gp(validation_samples)
        abs_error = bkd.norm(gp_vals - validation_values) / np.sqrt(
            nvalidation_samples
        )
        # print("Abs. Error", abs_error)
        assert abs_error < l2_tol

        # compute sensitivity indices from gp
        analyzer = EnsembleGaussianProcessSensitivityAnalysis(
            benchmark.prior(), nrealizations=100
        )
        analyzer.set_interaction_terms_of_interest(
            benchmark.sobol_interaction_indices()
        )
        analyzer.compute(gp)

        assert bkd.allclose(analyzer.mean(), benchmark.mean(), rtol=sa_tol)
        assert bkd.allclose(
            analyzer.variance(), benchmark.variance(), rtol=sa_tol
        )
        # make sure there is variation in the computed statistics
        print(analyzer.variance("min"), analyzer.variance("max"))
        assert analyzer.variance("min") != analyzer.variance("max")
        assert bkd.allclose(
            analyzer.main_effects(),
            benchmark.main_effects(),
            rtol=sa_tol,
            atol=1e-5,
        )
        assert bkd.allclose(
            analyzer.total_effects(), benchmark.total_effects(), rtol=sa_tol
        )
        assert bkd.allclose(
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
        bkd = self.get_backend()
        test_cases = [
            [IshigamiBenchmark(bkd, a=0.1, b=0.02), 1000, 2e-3, 5e-3],
        ]
        for test_case in test_cases:
            self._check_gp_sensitivities(*test_case)


class TestNumpySensitivityAnalysis(TestSensitivityAnalysis, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchSensitivityAnalysis(TestSensitivityAnalysis, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

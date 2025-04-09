import unittest

import numpy as np
from scipy import stats

from pyapprox.interface.model import ModelListCostFunction
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.kernels import MaternKernel
from pyapprox.surrogates.gaussianprocess.mokernels import (
    MultiLevelKernel,
    construct_tensor_product_monomial_scaling,
)
from pyapprox.surrogates.gaussianprocess.exactgp import (
    ExactGaussianProcess,
    MOExactGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.activelearning import (
    CholeskySampler,
    TensorProductQuadratureGreedyIntegratedVarianceSampler,
    MonteCarloGreedyIntegratedVarianceSampler,
    BruteForceTensorProductQuadratureGreedyIntegratedVarianceSampler,
    BruteForceMonteCarloGreedyIntegratedVarianceSampler,
    AdaptiveGaussianProcess,
    SamplingScheduleFromList,
    MultiOutputMonteCarloGreedyIntegratedVarianceSampler,
    BruteForceMultiOutputMonteCarloGreedyIntegratedVarianceSampler,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


class TestActiveLearning:
    def setUp(self):
        np.random.seed(1)

    def test_cholesky_sampling_update(self):
        bkd = self.get_backend()
        nvars = 1
        variable = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)] * nvars, backend=bkd
        )
        kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)
        gp = ExactGaussianProcess(nvars, kernel)
        sampler = CholeskySampler(variable)
        sampler.set_gaussian_process(gp)

        nsamples = 10
        samples = sampler(nsamples)

        # set seed so that random candidates are the same
        np.random.seed(1)
        sampler2 = CholeskySampler(variable)
        sampler2.set_gaussian_process(gp)
        # generate half the required samples
        samples2 = sampler2(nsamples // 2)
        # make sure sample can generate the correct remaining samples
        # via an update
        samples2 = bkd.hstack([samples2, sampler2(nsamples)])
        assert np.allclose(samples2, samples)

    def test_cholesky_sampler_update_with_changed_kernel(self):
        bkd = self.get_backend()
        nvars = 1
        variable = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)] * nvars, backend=bkd
        )
        kernel1 = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)
        kernel2 = MaternKernel(np.inf, 0.1, [1e-1, 1], nvars, backend=bkd)
        gp = ExactGaussianProcess(nvars, kernel1)

        nsamples = 10
        sampler = CholeskySampler(variable)
        sampler.set_gaussian_process(gp)
        samples = sampler(nsamples)

        # set seed so that random candidates are the same
        np.random.seed(1)
        sampler2 = CholeskySampler(variable)
        sampler2.set_gaussian_process(gp)
        samples2 = sampler2(nsamples // 2)
        gp._kernel = kernel2
        sampler2._kernel_changed = True
        samples2 = bkd.hstack([samples2, sampler2(nsamples)])
        assert bkd.allclose(
            samples2[:, : nsamples // 2], samples[:, : nsamples // 2]
        )
        assert not bkd.allclose(samples2, samples)

    def test_cholesky_sampler_update_with_changed_weight_function(self):
        bkd = self.get_backend()
        nvars = 1
        variable = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)] * nvars, backend=bkd
        )
        kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)

        def wfunction1(x):
            return bkd.ones(x.shape[1])

        def wfunction2(x):
            return x[0, :] ** 2

        nsamples = 10
        sampler = CholeskySampler(variable)
        gp = ExactGaussianProcess(nvars, kernel)
        sampler.set_gaussian_process(gp)
        sampler.set_weight_function(wfunction1)
        samples = sampler(nsamples)

        # set seed so that random candidates are the same
        np.random.seed(1)
        sampler2 = CholeskySampler(variable)
        sampler2.set_gaussian_process(gp)
        sampler2.set_weight_function(wfunction1)
        samples2 = sampler2(nsamples // 2)
        sampler2.set_weight_function(wfunction2)
        samples2 = bkd.hstack([samples2, sampler2(nsamples)])
        assert not bkd.allclose(samples2, samples)

    def _check_greedy_ivar_sampling_update(
        self, bruteforce_sampler_cls, sampler_cls, regression_test_samples
    ):
        bkd = self.get_backend()
        nvars = 1
        variable = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)] * nvars, backend=bkd
        )
        kernel = MaternKernel(np.inf, 0.1, [1e-1, 1], nvars, backend=bkd)
        gp = ExactGaussianProcess(nvars, kernel)
        sampler = bruteforce_sampler_cls(variable)
        sampler.set_gaussian_process(gp)
        ncandidates = 101
        candidate_samples = bkd.linspace(-1, 1, ncandidates)[None, :]
        sampler.set_candidate_samples(candidate_samples)
        init_pivots = bkd.array([ncandidates // 2], dtype=int)
        sampler.set_initial_pivots(init_pivots)

        nsamples = 11
        np.random.seed(1)
        samples = sampler(nsamples)
        print(samples, "greedy_ivar")
        print(regression_test_samples)
        assert bkd.allclose(samples, regression_test_samples)

        sorted_samples = bkd.sort(samples)
        # samples should be symmetric
        assert bkd.allclose(
            sorted_samples[: nsamples // 2], sorted_samples[nsamples // 2 :]
        )

        # set seed so that random candidates are the same
        np.random.seed(1)
        sampler2 = bruteforce_sampler_cls(variable)
        sampler2.set_gaussian_process(gp)
        sampler2.set_candidate_samples(candidate_samples)
        sampler2.set_initial_pivots(init_pivots)
        # generate half the required samples
        samples2 = sampler2(nsamples // 2)
        # make sure sample can generate the correct remaining samples
        # via an update
        samples2 = bkd.hstack([samples2, sampler2(nsamples)])
        assert np.allclose(samples2, samples)

        np.set_printoptions(precision=16, linewidth=1000)
        nsamples = 11
        ncandidates = 31
        candidate_samples = bkd.linspace(-1, 1, ncandidates)[None, :]
        init_pivots = bkd.array([ncandidates // 2], dtype=int)
        brute_sampler = bruteforce_sampler_cls(variable, nugget=0)
        brute_sampler.set_gaussian_process(gp)
        brute_sampler.set_candidate_samples(candidate_samples)
        brute_sampler.set_initial_pivots(init_pivots)
        sampler = sampler_cls(variable, nugget=0)
        sampler.set_gaussian_process(gp)
        sampler.set_candidate_samples(candidate_samples)
        sampler.set_initial_pivots(init_pivots)
        np.random.seed(1)
        brute_samples = brute_sampler(nsamples)
        np.random.seed(1)
        samples = sampler(nsamples)
        # sort samples because there are small variations in priotities
        # of approximately machine precision between two methods that causes
        # syymetric points to be chosen in different order
        # print(samples)
        # print(brute_samples)
        # print(bkd.sort(samples) - bkd.sort(brute_samples))
        assert bkd.allclose(bkd.sort(samples), bkd.sort(brute_samples))

    def test_greedy_ivar_sampling_update(self):
        bkd = self.get_backend()
        regression_test_samples = bkd.array(
            [
                [
                    0.0,
                    0.48,
                    -0.48,
                    -0.78,
                    0.78,
                    -0.24,
                    0.24,
                    -0.92,
                    0.92,
                    0.62,
                    -0.62,
                ]
            ]
        )
        np.random.seed(1)
        self._check_greedy_ivar_sampling_update(
            BruteForceTensorProductQuadratureGreedyIntegratedVarianceSampler,
            TensorProductQuadratureGreedyIntegratedVarianceSampler,
            regression_test_samples,
        )

        regression_test_samples = bkd.array(
            [
                [
                    0.0,
                    -0.54,
                    0.40,
                    0.74,
                    -0.82,
                    -0.26,
                    0.18,
                    0.9,
                    0.56,
                    -0.4,
                    -0.7,
                ]
            ]
        )
        np.random.seed(1)
        self._check_greedy_ivar_sampling_update(
            BruteForceMonteCarloGreedyIntegratedVarianceSampler,
            MonteCarloGreedyIntegratedVarianceSampler,
            regression_test_samples,
        )

    def _init_monomial(self, nvars, degree, val, bounds, fixed, bkd):
        return construct_tensor_product_monomial_scaling(
            nvars, [degree + 1] * nvars, val, bounds, fixed, bkd
        )

    def test_multioutput_ivar_update(self):

        bkd = self.get_backend()
        nvars = 1
        nmodels = 3
        variable = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)] * nvars, backend=bkd
        )
        degree = 0
        # note designs created will not always be symmetric because
        # many points will have the same objective value +/- numerical
        # noise when the correlation length is small. Because reduction
        # in variance is localized around the point and may not interaction
        # with previously chosen points. Also a new point right of a
        # existing point can reduce the variance to the left of the existing
        # point thus producing greater integrated variance reduction
        # than a point that is placed at the point of maximum variance
        kernels = [
            MaternKernel(
                np.inf, 0.3, [1e-1, 1], nvars, fixed=True, backend=bkd
            )
            for nn in range(nmodels)
        ]
        scalings = [
            self._init_monomial(nvars, degree, 2, [-1, 2], True, bkd),
            self._init_monomial(nvars, degree, -3, [-3, 3], True, bkd),
        ]
        # todo test all multi-output kernels
        kernel = MultiLevelKernel(kernels, scalings)
        gp = MOExactGaussianProcess(nvars, kernel)
        cost_function = ModelListCostFunction(bkd.arange(1, nmodels + 1))
        sampler = (
            BruteForceMultiOutputMonteCarloGreedyIntegratedVarianceSampler(
                variable, cost_function, nugget=0, nquad_samples=100000
            )
        )
        sampler.set_gaussian_process(gp)
        ncandidates_per_model = 21
        candidate_samples = [
            bkd.linspace(-1, 1, ncandidates_per_model)[None, :]
            for nn in range(nmodels)
        ]
        sampler.set_candidate_samples(candidate_samples)
        # init_pivots refer to index in array that concatenates all candidates
        # following adds the midpoint of the highest indexed output
        init_pivots = bkd.array(
            [
                ncandidates_per_model * (nmodels - 1)
                + ncandidates_per_model // 2
            ],
            dtype=int,
        )
        sampler.set_initial_pivots(init_pivots)

        nsamples = 7
        samples = sampler(nsamples)
        gp.fit(
            samples,
            [s[0][:, None] * 0 for s in samples],
        )
        # compute objective val directly by integrating pointwise
        # posterior variance of gp
        obj_val = -(
            (
                gp._evaluate_canonical_prior(sampler._stat._cquadx, True)[1][
                    nmodels - 1
                ]
                ** 2
            ).mean()
            - (
                gp.evaluate(sampler._stat._cquadx, True)[1][nmodels - 1] ** 2
            ).mean()
        )
        # print(samples)
        # note use of cquadx above in canonical space this only works
        # because in_trans is the identity
        assert bkd.allclose(sampler._best_obj_vals[-1], obj_val)

    def test_adaptive_gaussian_process(self):
        bkd = self.get_backend()
        nvars = 1

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        variable = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)] * nvars, backend=bkd
        )
        kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)
        sampling_schedule = SamplingScheduleFromList([8, 3])
        sampler = CholeskySampler(variable)
        gp = AdaptiveGaussianProcess(
            nvars, kernel, sampling_schedule=sampling_schedule
        )
        gp.set_optimizer(ncandidates=10)
        gp.set_sampler(sampler)
        gp.build(fun)
        # import matplotlib.pyplot as plt
        # ax = plt.figure().gca()
        # gp.plot(ax, bounds=[-1, 1])
        # ax.plot(gp.get_train_samples()[0], gp.get_train_values(), "o")
        train_samples = bkd.array(
            [
                -1.00000000e00,
                9.96093750e-01,
                -8.31658642e-04,
                -5.39062500e-01,
                6.11328125e-01,
                -8.16406250e-01,
                8.43750000e-01,
                2.83132418e-01,
                -2.95140740e-01,
                -9.33593750e-01,
                9.40039978e-01,
            ]
        )[None, :]
        # regression test. Note when the last point added produces
        # a matrix that is close to singular then there can be differences
        # between numpy and torch train samples
        print(gp.get_train_samples(), "ADAPTIVE GP TRAIN SAMPLES")
        print(train_samples)
        # plt.show()

        assert bkd.allclose(gp.get_train_samples(), train_samples)


class TestNumpyActiveLearning(TestActiveLearning, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchActiveLearning(TestActiveLearning, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

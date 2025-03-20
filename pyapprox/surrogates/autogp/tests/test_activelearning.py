import unittest

import numpy as np
from scipy import stats

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.kernels import MaternKernel
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.autogp.activelearning import (
    CholeskySampler,
    GreedyIntegratedVarianceSampler,
    AdaptiveGaussianProcess,
    SamplingScheduleFromList,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin


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
        # sampler = CholeskySampler(nvars, 100, variables)
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

    def test_greedy_ivar_sampling_update(self):
        bkd = self.get_backend()
        nvars = 1
        variable = IndependentMarginalsVariable(
            [stats.uniform(-1, 2)] * nvars, backend=bkd
        )
        kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)
        gp = ExactGaussianProcess(nvars, kernel)
        # sampler = CholeskySampler(nvars, 100, variables)
        sampler = GreedyIntegratedVarianceSampler(variable)
        sampler.set_gaussian_process(gp)

        nsamples = 10
        samples = sampler(nsamples)

        # set seed so that random candidates are the same
        np.random.seed(1)
        sampler2 = GreedyIntegratedVarianceSampler(variable)
        sampler2.set_gaussian_process(gp)
        # generate half the required samples
        samples2 = sampler2(nsamples // 2)
        # make sure sample can generate the correct remaining samples
        # via an update
        samples2 = bkd.hstack([samples2, sampler2(nsamples)])
        assert np.allclose(samples2, samples)

        import matplotlib.pyplot as plt

        # plt.plot(samples[0], 0 * samples[0], "o")
        # plt.show()
        raise NotImplementedError(
            "Test runs but points are not distributed correctly"
        )

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
        gp.set_sampler(sampler)
        gp.build(fun)
        import matplotlib.pyplot as plt

        ax = plt.figure().gca()
        gp.plot(ax, bounds=[-1, 1])
        ax.plot(gp.get_train_samples()[0], gp.get_train_values(), "o")
        train_samples = bkd.array(
            [
                -1.00000000e00,
                9.94645701e-01,
                -8.31658642e-04,
                -5.39062500e-01,
                6.09509127e-01,
                -8.16406250e-01,
                8.43750000e-01,
                2.81250000e-01,
                -3.00781250e-01,
                -9.35353867e-01,
                9.40039978e-01,
            ]
        )[None, :]
        # regression test. Note when the last point added produces
        # a matrix that is close to singular then there can be differences
        # between numpy and torch train samples
        print(gp.get_train_samples())
        print(train_samples)
        assert bkd.allclose(gp.get_train_samples(), train_samples)


class TestNumpyActiveLearning(TestActiveLearning, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchActiveLearning(TestActiveLearning, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

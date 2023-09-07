import unittest
import numpy as np

from pyapprox.util.utilities import check_gradients
from pyapprox.surrogates.autogp.kernels import (
    MaternKernel, Monomial, ConstantKernel)
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.autogp.variationalgp import InducingGaussianProcess


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_exact_gp_training(self):
        nvars = 1
        kernel = MaternKernel(np.inf, 1, [1e-1, 1], nvars)
        gp = ExactGaussianProcess(nvars, kernel)

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        ntrain_samples = 10
        train_samples = np.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        gp.set_training_data(train_samples, train_values)
        bounds = gp.hyp_list.get_active_opt_bounds().numpy()
        x0 = np.random.uniform(
            bounds[:, 0], bounds[:, 1])
        errors = check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None])
        assert errors.min()/errors.max() < 1e-7

        print(gp.kernel)
        gp.fit(train_samples, train_values)

        ntest_samples = 5
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))
        test_vals = fun(test_samples)

        gp_vals, gp_std = gp(test_samples, return_std=True)
        assert np.allclose(gp_vals, test_vals, atol=1e-2)

        mean = Monomial(nvars, 2, 1.0, (-1e3, 1e3), name='mean')
        kernel = ConstantKernel(0.1, (1e-3, 1e1))*MaternKernel(
            np.inf, 1, [1e-1, 1], nvars)
        from pyapprox.surrogates.autogp.transforms import IdentityTransform
        gp = ExactGaussianProcess(
            nvars, kernel, mean=mean, values_trans=IdentityTransform(1))

        gp.set_training_data(train_samples, train_values)
        bounds = gp.hyp_list.get_active_opt_bounds().numpy()
        # make infinite boudns finite when generating initial condition
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        errors = check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None])
        assert errors.min()/errors.max() < 1e-7

        gp.fit(train_samples, train_values)

        ntest_samples = 10
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))
        test_vals = fun(test_samples)

        gp_vals, gp_std = gp(test_samples, return_std=True)
        assert np.allclose(gp_vals, test_vals, atol=1e-14)
        assert np.allclose(gp.mean.hyp_list.get_values(), [0, 0, 1])
        print(gp.kernel)
        print(gp.mean.hyp_list)
        import matplotlib.pyplot as plt
        xx = np.linspace(-1, 1, 101)[None, :]
        plt.plot(xx[0], gp.mean(xx))
        plt.plot(xx[0], fun(xx))
        plt.show()

        assert False #TODO add tests checking training values scaling

    def test_variational_gp_training(self):
        nvars = 1
        kernel = MaternKernel(np.inf, 1, [1e-1, 1], nvars)
        gp = InducingGaussianProcess(nvars, kernel)

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        ntrain_samples = 10
        train_samples = np.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        gp.set_training_data(train_samples, train_values)
        bounds = gp.kernel.hyp_list.get_active_opt_bounds().numpy()
        x0 = np.random.uniform(
            bounds[:, 0], bounds[:, 1])
        check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None])

        print(gp.kernel)
        gp.fit(train_samples, train_values)

        ntest_samples = 10
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))
        test_vals = fun(test_samples)

        gp_vals, gp_std = gp(test_samples, return_std=True)
        assert np.allclose(gp_vals, test_vals, atol=1e-2)


if __name__ == "__main__":
    gaussian_process_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianProcess)
    unittest.TextTestRunner(verbosity=2).run(gaussian_process_test_suite)

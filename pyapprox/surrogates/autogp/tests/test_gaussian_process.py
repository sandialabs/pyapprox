import unittest
import numpy as np

from pyapprox.util.utilities import check_gradients
from pyapprox.surrogates.autogp.kernels import (
    MaternKernel, Monomial, ConstantKernel)
from pyapprox.surrogates.autogp.hyperparameter import (
    LogHyperParameterTransform, HyperParameter)
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.autogp.variationalgp import (
    InducingGaussianProcess, InducingSamples)
from pyapprox.surrogates.autogp.transforms import (
    IdentityValuesTransform, StandardDeviationValuesTransform)


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _check_exact_gp_training(self, mean, values_trans, constant):
        nvars = 1
        if mean is not None:
            assert mean.nvars == nvars

        kernel = MaternKernel(np.inf, 1, [1e-1, 1], nvars)
        if constant is not None:
            constant_kernel = ConstantKernel(
                0.1, (1e-3, 1e1), transform=LogHyperParameterTransform())
            kernel = constant_kernel*kernel

        gp = ExactGaussianProcess(
            nvars, kernel, mean=mean, values_trans=values_trans)

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
        assert errors.min()/errors.max() < 1e-6

        gp.fit(train_samples, train_values)

        ntest_samples = 5
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))
        test_vals = fun(test_samples)

        gp_vals, gp_std = gp(test_samples, return_std=True)

        if mean is not None and mean.degree == 2:
            assert np.allclose(gp_vals, test_vals, atol=1e-14)
            xx = np.linspace(-1, 1, 101)[None, :]
            assert np.allclose(gp.values_trans.map_from_canonical(
                gp._canonical_mean(xx)), fun(xx))
        else:
            assert np.allclose(gp_vals, test_vals, atol=1e-2)

    def test_exact_gp_training(self):
        test_cases = [
            [None, IdentityValuesTransform(), None],
            [Monomial(1, 2, 1.0, (-1e3, 1e3), name='mean'),
             IdentityValuesTransform(), None],
            [None, StandardDeviationValuesTransform(), None],
            [Monomial(1, 2, 1.0, (-1e3, 1e3), name='mean'),
             StandardDeviationValuesTransform(), None],
        ]
        for test_case in test_cases[-1:]:
            self._check_exact_gp_training(*test_case)

    def test_variational_gp_training(self):
        ntrain_samples = 10
        nvars, ninducing_samples = 1, 5
        kernel = MaternKernel(np.inf, 0.5, [1e-1, 1], nvars)
        inducing_samples = np.linspace(-1, 1, ninducing_samples)[None, :]
        noise = HyperParameter(
            'noise', 1, 1, (1e-6, 1), LogHyperParameterTransform())
        inducing_samples = InducingSamples(
            nvars, ninducing_samples, inducing_samples=inducing_samples,
            noise=noise)
        gp = InducingGaussianProcess(
            nvars, kernel, inducing_samples,
            kernel_reg=1e-10, values_trans=IdentityValuesTransform())

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        train_samples = np.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        gp.set_training_data(train_samples, train_values)
        # bounds = gp.hyp_list.get_active_opt_bounds().numpy()
        # x0 = gp._get_random_optimizer_initial_guess(bounds)
        x0 = gp.hyp_list.get_active_opt_params().numpy()
        errors = check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None])
        assert errors.min()/errors.max() < 1e-6

        gp.fit(train_samples, train_values, max_nglobal_opt_iters=3)
        print(gp)

        import matplotlib.pyplot as plt
        xx = np.linspace(-1, 1, 101)[None, :]
        plt.plot(xx[0], gp(xx, False), '--')
        plt.plot(gp.inducing_samples.get_samples(),
                 0*gp.inducing_samples.get_samples(), 's')
        plt.plot(xx[0], fun(xx)[:, 0], 'k-')
        plt.plot(gp.train_samples[0], gp.train_values, 'o')
        gp_mu, gp_std = gp(xx, return_std=True)
        gp_mu = gp_mu[:, 0]
        gp_std = gp_std[:, 0]
        print(gp_mu.shape, gp_std.shape)
        plt.fill_between(xx[0], gp_mu-3*gp_std, gp_mu+3*gp_std, alpha=0.1,
                         color='blue')
        #plt.show()

        ntest_samples = 10
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))
        test_vals = fun(test_samples)
        gp_mu, gp_std = gp(test_samples, return_std=True)
        print(gp_mu-test_vals)
        assert np.allclose(gp_mu, test_vals, atol=6e-3)


if __name__ == "__main__":
    gaussian_process_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianProcess)
    unittest.TextTestRunner(verbosity=2).run(gaussian_process_test_suite)

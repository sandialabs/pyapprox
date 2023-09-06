import unittest
import numpy as np

from pyapprox.util.utilities import check_gradients
from pyapprox.surrogates.autogp.kernels import MaternKernel
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess


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

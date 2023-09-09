import unittest
import numpy as np

from pyapprox.util.utilities import check_gradients
from pyapprox.surrogates.autogp.kernels import (
    MaternKernel, Monomial, ConstantKernel)
from pyapprox.surrogates.autogp.mokernels import (
    SphericalCovariance, ICMKernel)
from pyapprox.surrogates.autogp.hyperparameter import (
    LogHyperParameterTransform, HyperParameter)
from pyapprox.surrogates.autogp.exactgp import (
    ExactGaussianProcess, MOExactGaussianProcess)
from pyapprox.surrogates.autogp.variationalgp import (
    InducingGaussianProcess, InducingSamples,
    _invert_noisy_low_rank_nystrom_approximation,
    _log_prob_gaussian_with_noisy_nystrom_covariance)
from pyapprox.surrogates.autogp.transforms import (
    IdentityValuesTransform, StandardDeviationValuesTransform)
from pyapprox.surrogates.autogp._torch_wrappers import asarray


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        pass

    def _check_invert_noisy_low_rank_nystrom_approximation(self, N, M):
        const = 2.0
        vmat = np.random.normal(0, 1, (N, M))
        full = const*np.eye(N) + vmat @ vmat.T
        full_inverse = np.linalg.inv(full)

        lowrank_inverse, log_cov_det, L = (
            _invert_noisy_low_rank_nystrom_approximation(asarray(const), vmat))
        assert np.allclose(log_cov_det, np.linalg.slogdet(full)[1])
        assert np.allclose(lowrank_inverse, full_inverse)

        noise = 1
        tmp = np.random.normal(0, 1, (N, N))
        C_NN = tmp.T@tmp
        C_MN = C_NN[:M]
        C_MM = C_NN[:M, :M]
        Q = asarray(C_MN.T @ np.linalg.inv(C_MM) @ C_MN + noise*np.eye(N))

        values = asarray(np.ones((N, 1)))
        from torch.distributions import MultivariateNormal
        p_y = MultivariateNormal(values[:, 0]*0, covariance_matrix=Q)
        logpdf1 = p_y.log_prob(values[:, 0])

        L_UU = asarray(np.linalg.cholesky(C_MM))
        logpdf2 = _log_prob_gaussian_with_noisy_nystrom_covariance(
            asarray(noise), L_UU, asarray(C_MN.T), values)
        assert np.allclose(logpdf1, logpdf2)

    def test_invert_noisy_low_rank_nystrom_approximation(self):
        test_cases = [
            [3, 2], [4, 2], [15, 6]]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_invert_noisy_low_rank_nystrom_approximation(*test_case)

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
        values_trans = IdentityValuesTransform()
        gp = InducingGaussianProcess(
            nvars, kernel, inducing_samples,
            kernel_reg=1e-10, values_trans=values_trans)

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

        gp.fit(train_samples, train_values, max_nglobal_opt_iters=1)
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

    def test_variational_gp_collapse_to_exact_gp(self):
        nvars = 1
        ntrain_samples = 6
        kernel = MaternKernel(np.inf, 1, [1e-1, 1], nvars)
        values_trans = IdentityValuesTransform()

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        train_samples = np.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        ntest_samples = 6
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))

        exact_gp = ExactGaussianProcess(
            nvars, kernel, mean=None, values_trans=values_trans, kernel_reg=0)
        exact_gp.set_training_data(train_samples, train_values)
        exact_gp.fit(train_samples, train_values, max_nglobal_opt_iters=1)
        exact_gp_vals, exact_gp_std = exact_gp(test_samples, return_std=True)

        inducing_samples = train_samples
        ninducing_samples = ntrain_samples
        # fix hyperparameters so they are not changed from exact_gp
        # or setting provided if not found in exact_gp
        noise = HyperParameter(
            'noise', 1, 1e-15, [np.nan, np.nan], LogHyperParameterTransform())
        inducing_samples = InducingSamples(
            nvars, ninducing_samples, inducing_samples=inducing_samples,
            inducing_sample_bounds=[np.nan, np.nan], noise=noise)
        values_trans = IdentityValuesTransform()
        # use correlation length learnt by exact gp
        vi_kernel = kernel
        vi_gp = InducingGaussianProcess(
            nvars, vi_kernel, inducing_samples,
            kernel_reg=0, values_trans=values_trans)
        vi_gp.fit(train_samples, train_values, max_nglobal_opt_iters=1)
        vi_gp_vals, vi_gp_std = vi_gp(test_samples, return_std=True)

        # print(vi_gp_vals-exact_gp_vals)
        assert np.allclose(vi_gp_vals, exact_gp_vals, atol=1e-12)
        # print(vi_gp_std-exact_gp_std)
        # I think larger tolerance needed because sqrt of covariance
        # is being taken inside funcitns
        assert np.allclose(vi_gp_std, exact_gp_std, atol=2e-9)

    def test_icm_gp(self):
        nvars, noutputs = 1, 2

        def fun0(xx):
            delta0 = 0.0
            return np.cos(2*np.pi*xx.T+delta0)

        def fun1(xx):
            delta1 = 0.5
            return np.cos(2*np.pi*xx.T+delta1)

        funs = [fun0, fun1]

        radii, radii_bounds = np.arange(1, noutputs+1), [0.1, 10]
        angles = np.pi/4
        latent_kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
        output_kernel = SphericalCovariance(
            noutputs, radii, radii_bounds, angles=angles)

        kernel = ICMKernel(latent_kernel, output_kernel, noutputs)

        nsamples_per_output = [5, 3]
        samples_per_output = [
            np.random.uniform(-1, 1, (nvars, nsamples))
            for nsamples in nsamples_per_output]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)]

        gp = MOExactGaussianProcess(
            nvars, kernel, mean=None, values_trans=IdentityValuesTransform(),
            kernel_reg=0)
        gp.fit(samples_per_output, values_per_output)

        import matplotlib.pyplot as plt
        ax = plt.subplots(1, 1)[1]
        gp.plot(ax, [-1, 1])
        plt.show()


if __name__ == "__main__":
    gaussian_process_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianProcess)
    unittest.TextTestRunner(verbosity=2).run(gaussian_process_test_suite)

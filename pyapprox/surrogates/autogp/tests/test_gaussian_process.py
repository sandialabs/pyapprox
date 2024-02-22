import unittest
import numpy as np
from functools import partial

from pyapprox.util.utilities import check_gradients
from pyapprox.surrogates.autogp.kernels import (
    MaternKernel, Monomial, ConstantKernel, GaussianNoiseKernel)
from pyapprox.surrogates.autogp.mokernels import (
    SphericalCovariance, ICMKernel, MultiPeerKernel, CollaborativeKernel)
from pyapprox.surrogates.autogp.hyperparameter import (
    LogHyperParameterTransform, HyperParameter)
from pyapprox.surrogates.autogp.exactgp import (
    ExactGaussianProcess, MOExactGaussianProcess, MOPeerExactGaussianProcess,
    MOICMPeerExactGaussianProcess)
from pyapprox.surrogates.autogp.variationalgp import (
    InducingGaussianProcess, InducingSamples,
    _log_prob_gaussian_with_noisy_nystrom_covariance)
from pyapprox.surrogates.autogp.transforms import (
    IdentityValuesTransform, StandardDeviationValuesTransform)
from pyapprox.surrogates.autogp._torch_wrappers import asarray


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        pass

    def _check_invert_noisy_low_rank_nystrom_approximation(self, N, M):
        noise_std = 2
        tmp = np.random.normal(0, 1, (N, N))
        C_NN = tmp.T@tmp
        C_MN = C_NN[:M]
        C_MM = C_NN[:M, :M]
        Q = asarray(
            C_MN.T @ np.linalg.inv(C_MM) @ C_MN + noise_std**2*np.eye(N))

        values = asarray(np.ones((N, 1)))
        from torch.distributions import MultivariateNormal
        p_y = MultivariateNormal(values[:, 0]*0, covariance_matrix=Q)
        logpdf1 = p_y.log_prob(values[:, 0])

        L_UU = asarray(np.linalg.cholesky(C_MM))
        logpdf2 = _log_prob_gaussian_with_noisy_nystrom_covariance(
            asarray(noise_std), L_UU, asarray(C_MN.T), values)
        assert np.allclose(logpdf1, logpdf2)

        if N != M:
            return

        assert np.allclose(Q, C_NN + noise_std**2*np.eye(N))

        values = values.numpy()
        Q_inv = np.linalg.inv(Q)

        import scipy
        Delta = scipy.linalg.solve_triangular(
            L_UU, C_MN.T, lower=True)/noise_std
        Omega = np.eye(M) + Delta@Delta.T
        L_Omega = np.linalg.cholesky(Omega)
        log_det = 2*np.log(np.diag(L_Omega)).sum()+2*N*np.log(noise_std)
        gamma = scipy.linalg.solve_triangular(
            L_Omega, Delta @ values, lower=True)
        assert np.allclose(log_det, np.linalg.slogdet(Q)[1])

        coef = Q_inv @ values
        assert np.allclose(
            values.T@coef,
            values.T@values/noise_std**2-gamma.T@gamma/noise_std**2)

        mll = -0.5 * (
            values.T@coef +
            np.linalg.slogdet(Q)[1] +
            N*np.log(2*np.pi)
        )
        assert np.allclose(mll, logpdf2)

    def test_invert_noisy_low_rank_nystrom_approximation(self):
        test_cases = [
            [3, 2], [4, 2], [15, 6], [3, 3]]
        for test_case in test_cases[-1:]:
            np.random.seed(1)
            self._check_invert_noisy_low_rank_nystrom_approximation(*test_case)

    def _check_exact_gp_training(self, mean, values_trans, constant):
        nvars = 1
        if mean is not None:
            assert mean.nvars() == nvars

        kernel = MaternKernel(np.inf, 1., [1e-1, 1], nvars)

        kernel = kernel
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
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None],
            disp=False)
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
                gp._canonical_mean(xx)), fun(xx), atol=5e-6)
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
        for test_case in test_cases:
            self._check_exact_gp_training(*test_case)

    def test_compare_with_deprecated_gp(self):
        nvars = 1
        noise = 0.0 #1
        sigma = 1
        lenscale = 0.5
        kernel = (ConstantKernel(sigma, [np.nan, np.nan]) *
                  MaternKernel(np.inf, lenscale, [np.nan, np.nan], nvars) +
                  GaussianNoiseKernel(noise, [np.nan, np.nan]))

        gp = ExactGaussianProcess(
            nvars, kernel, mean=None, values_trans=IdentityValuesTransform())

        # def fun(xx):
        #     return (xx**2).sum(axis=0)[:, None]

        def fun(xx, noisy=True):
            vals = np.cos(2*np.pi*xx.T)
            if not noisy:
                return vals
            return vals + np.random.normal(0, np.sqrt(noise), xx.T.shape)

        ntrain_samples = 6
        train_samples = np.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        from pyapprox.surrogates.gaussianprocess.gaussian_process import (
            GaussianProcess, Matern,  ConstantKernel as CKernel, WhiteKernel)
        pyakernel = (CKernel(sigma, 'fixed') *
                     Matern(lenscale, length_scale_bounds='fixed', nu=np.inf) +
                     WhiteKernel(noise, 'fixed'))

        assert np.allclose(kernel(train_samples), pyakernel(train_samples.T))

        gp.fit(train_samples, train_values)

        pyagp = GaussianProcess(pyakernel, alpha=0.)
        pyagp.fit(train_samples, train_values)
        # print(gp)
        # print(pyagp)

        ntest_samples = 5
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))
        test_samples = np.linspace(-1, 1, 5)[None, :]

        pyagp_vals, pyagp_std = pyagp(test_samples, return_std=True)
        gp_vals, gp_std = gp(test_samples, return_std=True)
        # print(gp_std[:, 0]-pyagp_std)
        assert np.allclose(gp_std[:, 0], pyagp_std, atol=1e-6)

        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 1)[1]
        # gp.plot(ax, [-1, 1], plt_kwargs={"c": "r", "ls": "-"}, npts_1d=101)
        # pyagp.plot_1d(101, [-1, 1], ax=ax)
        # xx = np.linspace(-1, 1, 101)[None, :]
        # plt.plot(xx[0], fun(xx, False))
        # plt.plot(gp.train_samples[0], gp.train_values, 'o')
        # plt.show()

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
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None],
            disp=False)
        assert errors.min()/errors.max() < 1e-6

        gp.fit(train_samples, train_values, max_nglobal_opt_iters=1)
        # print(gp)

        # import matplotlib.pyplot as plt
        # xx = np.linspace(-1, 1, 101)[None, :]
        # plt.plot(xx[0], gp(xx, False), '--')
        # plt.plot(gp.inducing_samples.get_samples(),
        #          0*gp.inducing_samples.get_samples(), 's')
        # plt.plot(xx[0], fun(xx)[:, 0], 'k-')
        # plt.plot(gp.train_samples[0], gp.train_values, 'o')
        # gp_mu, gp_std = gp(xx, return_std=True)
        # gp_mu = gp_mu[:, 0]
        # gp_std = gp_std[:, 0]
        # plt.fill_between(xx[0], gp_mu-3*gp_std, gp_mu+3*gp_std, alpha=0.1,
        #                  color='blue')
        # plt.show()

        ntest_samples = 10
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))
        test_vals = fun(test_samples)
        gp_mu, gp_std = gp(test_samples, return_std=True)
        # print(gp_mu-test_vals)
        assert np.allclose(gp_mu, test_vals, atol=6e-3)

    def test_variational_gp_collapse_to_exact_gp(self):
        nvars = 1
        ntrain_samples = 6
        noise_var = 1e-8
        kernel = (MaternKernel(np.inf, 1, [1e-1, 1], nvars))
        values_trans = IdentityValuesTransform()

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        train_samples = np.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        ntest_samples = 6
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))

        exact_gp = ExactGaussianProcess(
            nvars, kernel+GaussianNoiseKernel(noise_var, [np.nan, np.nan]),
            mean=None, values_trans=values_trans, kernel_reg=0)
        exact_gp.set_training_data(train_samples, train_values)
        exact_gp.fit(train_samples, train_values, max_nglobal_opt_iters=1)
        exact_gp_vals, exact_gp_std = exact_gp(test_samples, return_std=True)

        inducing_samples = train_samples
        ninducing_samples = ntrain_samples
        # fix hyperparameters so they are not changed from exact_gp
        # or setting provided if not found in exact_gp
        noise = HyperParameter(
            'noise_std', 1, np.sqrt(noise_var), [np.nan, np.nan],
            LogHyperParameterTransform())
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
        assert np.allclose(vi_gp_std, exact_gp_std, atol=5e-5)

    def test_icm_gp(self):
        nvars, noutputs = 1, 2

        def fun0(xx):
            delta0 = 0.0
            return np.cos(2*np.pi*xx.T+delta0)

        def fun1(xx):
            delta1 = 0.5
            return np.cos(2*np.pi*xx.T+delta1)

        funs = [fun0, fun1]

        radii, radii_bounds = np.arange(1, noutputs+1), [1, 10]
        angles = np.pi/4
        latent_kernel = MaternKernel(np.inf, 0.5, [1e-1, 2], nvars)
        output_kernel = SphericalCovariance(
            noutputs, radii, radii_bounds, angles=angles,
            angle_bounds=[0, np.pi])

        kernel = ICMKernel(latent_kernel, output_kernel, noutputs)

        nsamples_per_output = [12, 12]
        samples_per_output = [
            np.random.uniform(-1, 1, (nvars, nsamples))
            for nsamples in nsamples_per_output]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)]

        gp = MOExactGaussianProcess(
            nvars, kernel, mean=None, values_trans=IdentityValuesTransform(),
            kernel_reg=1e-8)
        gp.fit(samples_per_output, values_per_output, max_nglobal_opt_iters=3)

        # check correlation between models is estimated correctly.
        # SphericalCovariance is not guaranteed to recover the statistical
        # correlation, but for this case it can
        from pyapprox.util.utilities import get_correlation_from_covariance
        cov_matrix = output_kernel.get_covariance_matrix()
        corr_matrix = get_correlation_from_covariance(cov_matrix.numpy())
        samples = np.random.uniform(-1, 1, (1, 101))
        values = np.hstack([fun(samples) for fun in funs])
        assert np.allclose(
            corr_matrix,
            get_correlation_from_covariance(np.cov(values.T, ddof=1)),
            atol=1e-2)

        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 1)[1]
        # # gp.plot(ax, [-1, 1])
        # gp.plot(ax, [-1, 1], output_id=0, plt_kwargs={"c": "r", "ls": "-"})
        # gp.plot(ax, [-1, 1], output_id=1)
        # xx = np.linspace(-1, 1, 101)[None, :]
        # ax.plot(xx[0], funs[0](xx), '--')
        # ax.plot(xx[0], funs[1](xx), ':')
        # ax.plot(gp.train_samples[0][0], gp.train_values[0], 'o')
        # ax.plot(gp.train_samples[1][0], gp.train_values[1], 's')
        # plt.show()

    def test_peer_gaussian_process(self):
        nvars, noutputs = 1, 4
        degree = 0
        kernels = [MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
                   for ii in range(noutputs)]
        scalings = [
            Monomial(nvars, degree, 1, [-1, 2], name=f'scaling{ii}')
            for ii in range(noutputs-1)]
        kernel = MultiPeerKernel(kernels, scalings)

        def peer_fun(delta, xx):
            return np.cos(2*np.pi*xx.T+delta)

        def target_fun(peer_funs, xx):
            # return (
            #     np.hstack([f(xx) for f in peer_funs]).sum(axis=1)[:, None] +
            #     np.exp(-xx.T**2*2))
            return np.cos(2*np.pi*xx.T)

        peer_deltas = np.linspace(0, 1, noutputs-1)
        peer_funs = [partial(peer_fun, delta) for delta in peer_deltas]
        funs = peer_funs + [partial(target_fun, peer_funs)]

        # nsamples_per_output = np.array([5 for ii in range(noutputs-1)]+[4])*2
        nsamples_per_output = np.array([7 for ii in range(noutputs-1)]+[5])
        samples_per_output = [
            np.random.uniform(-1, 1, (nvars, nsamples))
            for nsamples in nsamples_per_output]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)]

        gp = MOExactGaussianProcess(
            nvars, kernel, mean=None, values_trans=IdentityValuesTransform(),
            kernel_reg=0)
        gp.fit(samples_per_output, values_per_output, max_nglobal_opt_iters=3)

        # import matplotlib.pyplot as plt
        # axs = plt.subplots(
        #     1, noutputs, figsize=(noutputs*8, 6), sharey=True)[1]
        # xx = np.linspace(-1, 1, 101)[None, :]
        # for ii in range(noutputs):
        #     gp.plot(axs[ii], [-1, 1], output_id=ii)
        #     axs[ii].plot(xx[0], funs[ii](xx), '--')
        #     axs[ii].plot(gp.train_samples[ii][0], gp.train_values[ii], 'o')
        # plt.show()

        # check that when using hyperparameters found by dense GP the PeerGP
        # return the same likelihood value and prediction mean and std. dev.
        peer_gp = MOPeerExactGaussianProcess(
            nvars, kernel, mean=None, values_trans=IdentityValuesTransform(),
            kernel_reg=0)
        peer_gp.set_training_data(samples_per_output, values_per_output)
        assert np.allclose(
            gp._neg_log_likelihood_with_hyperparameter_mean(),
            peer_gp._neg_log_likelihood_with_hyperparameter_mean())
        xx = np.linspace(-1, 1, 31)[None, :]
        gp_mean, gp_std = gp([xx]*noutputs, return_std=True)
        peer_gp_mean, peer_gp_std = peer_gp([xx]*noutputs, return_std=True)
        assert np.allclose(peer_gp_mean, gp_mean)
        assert np.allclose(peer_gp_std, gp_std)

    def test_icm_peer_gp(self):
        nvars, noutputs = 1, 4

        def peer_fun(delta, xx):
            return np.cos(2*np.pi*xx.T+delta)

        def target_fun(peer_funs, xx):
            #return (
            #    np.hstack([f(xx) for f in peer_funs]).sum(axis=1)[:, None] +
            #    np.exp(-xx.T**2*2))
            return np.cos(2*np.pi*xx.T)

        # radii, radii_bounds = np.ones(noutputs), [1, 10]
        radii, radii_bounds = np.arange(1, 1+noutputs), [1, 10]
        angles = np.pi/2
        latent_kernel = MaternKernel(np.inf, 0.5, [1e-1, 2], nvars)
        output_kernel = SphericalCovariance(
            noutputs, radii, radii_bounds, angles=angles,
            angle_bounds=[0, np.pi])

        kernel = ICMKernel(latent_kernel, output_kernel, noutputs)

        peer_deltas = np.linspace(0.2, 1, noutputs-1)
        peer_funs = [partial(peer_fun, delta) for delta in peer_deltas]
        funs = peer_funs + [partial(target_fun, peer_funs)]

        nsamples_per_output = np.array([7 for ii in range(noutputs-1)]+[5])
        # nsamples_per_output = np.array([5 for ii in range(noutputs-1)]+[4])*2
        # nsamples_per_output = np.array([3 for ii in range(noutputs-1)]+[2])
        samples_per_output = [
            np.random.uniform(-1, 1, (nvars, nsamples))
            for nsamples in nsamples_per_output]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)]

        gp = MOICMPeerExactGaussianProcess(
            nvars, kernel, output_kernel,
            values_trans=IdentityValuesTransform(), kernel_reg=0)
        gp_params = gp.hyp_list.get_active_opt_params()

        from pyapprox.util.utilities import check_gradients
        bounds = gp.hyp_list.get_active_opt_bounds().numpy()
        x0 = np.random.uniform(
            bounds[:, 0], bounds[:, 1])
        icm_cons = gp._get_constraints(noutputs)
        errors = check_gradients(
            lambda x: icm_cons[0]['fun'](x[:, 0], *icm_cons[0]['args']),
            lambda x: icm_cons[0]['jac'](x[:, 0], *icm_cons[0]['args']),
            x0[:, None], disp=True)
        assert errors.min()/errors.max() < 1e-6
        # reset values to good guess
        gp.hyp_list.set_active_opt_params(gp_params)
        print(output_kernel)

        gp.set_training_data(samples_per_output, values_per_output)
        x0 = gp.hyp_list.get_active_opt_params().numpy()
        errors = check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None],
            disp=False)
        gp.hyp_list.set_active_opt_params(gp_params)
        assert errors.min()/errors.max() < 3.2e-6

        gp.fit(samples_per_output, values_per_output, max_nglobal_opt_iters=3)
        cov_matrix = output_kernel.get_covariance_matrix()
        print(cov_matrix)
        for ii in range(2, noutputs):
            for jj in range(1, ii):
                np.abs(cov_matrix[ii, jj]) < 1e-10

        # import matplotlib.pyplot as plt
        # axs = plt.subplots(
        #     1, noutputs, figsize=(noutputs*8, 6), sharey=True)[1]
        # xx = np.linspace(-1, 1, 101)[None, :]
        # for ii in range(noutputs):
        #     gp.plot(axs[ii], [-1, 1], output_id=ii)
        #     axs[ii].plot(xx[0], funs[ii](xx), '--')
        #     axs[ii].plot(gp.train_samples[ii][0], gp.train_values[ii], 'o')
        # plt.show()

    def test_collaborative_gp(self):
        nvars, noutputs = 1, 4
        def peer_fun(delta, xx):
            return np.cos(2*np.pi*xx.T+delta)

        def target_fun(peer_funs, xx):
            return (
                np.hstack([f(xx) for f in peer_funs]).sum(axis=1)[:, None] +
                np.exp(-xx.T**2*2))
            # return np.cos(2*np.pi*xx.T)

        radii, radii_bounds = np.ones(noutputs), [1, 2]
        angles = np.pi/4
        latent_kernel = MaternKernel(np.inf, 0.5, [1e-1, 2], nvars)
        output_kernel = SphericalCovariance(
            noutputs, radii, radii_bounds, angles=angles,
            angle_bounds=[0, np.pi])

        output_kernels = [output_kernel]
        latent_kernels = [latent_kernel]
        discrepancy_kernels = [
            ConstantKernel(
                0.1, (1e-1, 1), transform=LogHyperParameterTransform()) *
            MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
            for ii in range(noutputs)]
        co_kernel = CollaborativeKernel(
            latent_kernels, output_kernels, discrepancy_kernels, noutputs)

        peer_deltas = np.linspace(0.2, 1, noutputs-1)
        peer_funs = [partial(peer_fun, delta) for delta in peer_deltas]
        funs = peer_funs + [partial(target_fun, peer_funs)]

        nsamples_per_output = np.array([7 for ii in range(noutputs-1)]+[5])
        # nsamples_per_output = np.array([5 for ii in range(noutputs-1)]+[4])*2
        # nsamples_per_output = np.array([3 for ii in range(noutputs-1)]+[2])
        samples_per_output = [
            np.random.uniform(-1, 1, (nvars, nsamples))
            for nsamples in nsamples_per_output]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)]

        gp = MOExactGaussianProcess(
            nvars, co_kernel, mean=None,
            values_trans=IdentityValuesTransform(), kernel_reg=0)
        gp_params = gp.hyp_list.get_active_opt_params()

        gp.set_training_data(samples_per_output, values_per_output)
        x0 = gp.hyp_list.get_active_opt_params().numpy()
        errors = check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None],
            disp=True)
        gp.hyp_list.set_active_opt_params(gp_params)
        assert errors.min()/errors.max() < 1e-6

        # slsqp requires feasiable initial guess. It does not throw an
        # error if guess is infeasiable it just runs and produces garbage.
        # For now just start test from feasiable initial parameter values
        # and run optimization once.
        # todo change initial guess to always be feasiable
        gp.fit(samples_per_output, values_per_output, max_nglobal_opt_iters=1)
        cov_matrix = output_kernel.get_covariance_matrix()
        print(cov_matrix)
        for ii in range(2, noutputs):
            for jj in range(1, ii):
                assert True#np.abs(cov_matrix[ii, jj]) < 1e-10

        # import matplotlib.pyplot as plt
        # axs = plt.subplots(
        #     1, noutputs, figsize=(noutputs*8, 6), sharey=True)[1]
        # xx = np.linspace(-1, 1, 101)[None, :]
        # for ii in range(noutputs):
        #     gp.plot(axs[ii], [-1, 1], output_id=ii)
        #     axs[ii].plot(xx[0], funs[ii](xx), '--')
        #     axs[ii].plot(gp.train_samples[ii][0], gp.train_values[ii], 'o')
        # plt.show()


if __name__ == "__main__":
    gaussian_process_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianProcess)
    unittest.TextTestRunner(verbosity=2).run(gaussian_process_test_suite)

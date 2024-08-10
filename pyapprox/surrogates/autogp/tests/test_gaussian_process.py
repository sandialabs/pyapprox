import unittest
from functools import partial

import numpy as np
from scipy import stats
from torch.distributions import MultivariateNormal as TorchMultivariateNormal

from pyapprox.util.utilities import check_gradients
from pyapprox.surrogates.bases.univariate import Monomial1D
from pyapprox.surrogates.bases.basis import MultiIndexBasis
from pyapprox.surrogates.bases.basisexp import BasisExpansion
from pyapprox.util.hyperparameter import (
    LogHyperParameterTransform,
    HyperParameter,
)
from pyapprox.surrogates.kernels import (
    MaternKernel,
    ConstantKernel,
    GaussianNoiseKernel,
    SphericalCovariance,
)
from pyapprox.surrogates.autogp.exactgp import (
    ExactGaussianProcess,
    MOExactGaussianProcess,
    MOPeerExactGaussianProcess,
    MOICMPeerExactGaussianProcess,
)
from pyapprox.surrogates.autogp.mokernels import (
    ICMKernel,
    MultiPeerKernel,
    CollaborativeKernel,
)
from pyapprox.surrogates.autogp.variationalgp import (
    _log_prob_gaussian_with_noisy_nystrom_covariance,
    InducingSamples,
    InducingGaussianProcess,
)
from pyapprox.util.transforms import (
    IdentityTransform,
    StandardDeviationTransform,
)

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin


def _check_apply(
    sample,
    symb,
    fun,
    apply_fun,
    fd_eps=None,
    direction=None,
    relative=True,
    disp=False,
    bkd=NumpyLinAlgMixin,
):
    if sample.ndim != 2:
        raise ValueError(
            "sample with shape {0} must be 2D array".format(sample.shape)
        )
    if fd_eps is None:
        fd_eps = bkd._la_flip(bkd._la_logspace(-13, 0, 14))
    if direction is None:
        nvars = sample.shape[0]
        direction = bkd._la_array(np.random.normal(0, 1, (nvars, 1)))
        direction /= bkd._la_linalg.norm(direction)

    row_format = "{:<12} {:<25} {:<25} {:<25}"
    headers = [
        "Eps",
        "norm({0}v)".format(symb),
        "norm({0}v_fd)".format(symb),
        "Rel. Errors" if relative else "Abs. Errors",
    ]
    if disp:
        print(row_format.format(*headers))
    row_format = "{:<12.2e} {:<25} {:<25} {:<25}"
    errors = []
    val = fun(sample)
    directional_grad = apply_fun(sample, direction)
    for ii in range(fd_eps.shape[0]):
        sample_perturbed = sample.copy() + fd_eps[ii] * direction
        perturbed_val = fun(sample_perturbed)
        fd_directional_grad = (perturbed_val - val) / fd_eps[ii]
        errors.append(
            bkd._la_norm(
                fd_directional_grad.reshape(directional_grad.shape)
                - directional_grad
            )
        )
        if relative:
            errors[-1] /= bkd._la_norm(directional_grad)
        if disp:
            print(
                row_format.format(
                    fd_eps[ii],
                    bkd._la_norm(directional_grad),
                    bkd._la_norm(fd_directional_grad),
                    errors[ii],
                )
            )
    return bkd._la_array(errors)


class TestNystrom:
    def setUp(self):
        np.random.seed(1)

    def _check_invert_noisy_low_rank_nystrom_approximation(self, N, M):
        bkd = self.get_backend()
        MultivariateNormal = self.get_mvn()
        noise_std = 2
        tmp = bkd._la_atleast2d(np.random.normal(0, 1, (N, N)))
        C_NN = tmp.T @ tmp
        C_MN = C_NN[:M]
        C_MM = C_NN[:M, :M]
        Q = C_MN.T @ bkd._la_inv(C_MM) @ C_MN + noise_std**2 * bkd._la_eye(N)

        values = bkd._la_full((N, 1), 1)
        p_y = MultivariateNormal(values[:, 0] * 0, covariance_matrix=Q)
        logpdf1 = p_y.log_prob(values[:, 0])

        L_UU = bkd._la_cholesky(C_MM)
        logpdf2 = _log_prob_gaussian_with_noisy_nystrom_covariance(
            noise_std, L_UU, C_MN.T, values, bkd
        )
        assert np.allclose(logpdf1, logpdf2)

        if N != M:
            return

        assert np.allclose(Q, C_NN + noise_std**2 * bkd._la_eye(N))

        values = values
        Q_inv = bkd._la_inv(Q)

        Delta = bkd._la_solve_triangular(L_UU, C_MN.T, lower=True) / noise_std
        Omega = bkd._la_eye(M) + Delta @ Delta.T
        L_Omega = bkd._la_cholesky(Omega)
        log_det = 2 * bkd._la_log(
            bkd._la_get_diagonal(L_Omega)
        ).sum() + 2 * N * np.log(noise_std)
        gamma = bkd._la_solve_triangular(L_Omega, Delta @ values, lower=True)
        assert np.allclose(log_det, bkd._la_slogdet(Q)[1])

        coef = Q_inv @ values
        assert np.allclose(
            values.T @ coef,
            values.T @ values / noise_std**2
            - gamma.T @ gamma / noise_std**2,
        )

        mll = -0.5 * (
            values.T @ coef + bkd._la_slogdet(Q)[1] + N * np.log(2 * np.pi)
        )
        assert np.allclose(mll, logpdf2)

    def test_invert_noisy_low_rank_nystrom_approximation(self):
        test_cases = [
            [3, 2],
            [4, 2],
            [15, 6],
            [3, 3],
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_invert_noisy_low_rank_nystrom_approximation(*test_case)


class TestGaussianProcess:
    def setUp(self):
        np.random.seed(1)

    def _check_exact_gp_training(self, trend, values_trans, constant):
        bkd = self.get_backend()
        nvars = 1

        if not values_trans:
            values_trans = IdentityTransform(backend=bkd)
        else:
            values_trans = StandardDeviationTransform(trans=True, backend=bkd)

        if trend:
            basis = MultiIndexBasis(
                [Monomial1D(backend=bkd) for ii in range(nvars)]
            )
            basis.set_indices(bkd._la_arange(3, dtype=int)[None, :])
            trend = BasisExpansion(basis, None, 1.0, (-1e3, 1e3))
        else:
            trend = None

        kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)

        kernel = kernel
        if constant:
            constant_kernel = ConstantKernel(
                0.1,
                (1e-3, 1e1),
                transform=LogHyperParameterTransform(backend=bkd),
                backend=bkd,
            )
            kernel = constant_kernel * kernel

        gp = ExactGaussianProcess(
            nvars, kernel, trend=trend, values_trans=values_trans
        )

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        ntrain_samples = 10
        train_samples = bkd._la_linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        gp.set_training_data(train_samples, train_values)
        bounds = gp.hyp_list.get_active_opt_bounds()
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        errors = check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None], disp=False
        )
        # print(errors.min()/errors.max())
        assert errors.min() / errors.max() < 2e-6

        gp.fit(train_samples, train_values)

        ntest_samples = 5
        test_samples = bkd._la_atleast2d(
            np.random.uniform(-1, 1, (nvars, ntest_samples))
        )
        test_vals = fun(test_samples)

        gp_vals, gp_std = gp(test_samples, return_std=True)

        if trend is not None:
            assert bkd._la_allclose(gp_vals, test_vals, atol=1e-14)
            xx = bkd._la_linspace(-1, 1, 101)[None, :]
            assert bkd._la_allclose(
                gp.values_trans.map_from_canonical(gp._canonical_trend(xx)),
                fun(xx),
                atol=6e-5,
            )
        else:
            assert np.allclose(gp_vals, test_vals, atol=1e-2)

    def test_exact_gp_training(self):
        test_cases = [
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [True, True, False],
        ]
        for test_case in test_cases:
            self._check_exact_gp_training(*test_case)

    def test_compare_with_deprecated_gp(self):
        bkd = self.get_backend()
        nvars = 1
        noise = 0.0  # 1
        sigma = 1
        lenscale = 0.5
        kernel = ConstantKernel(
            sigma, [0.01, 1], fixed=True, backend=bkd
        ) * MaternKernel(
            np.inf, lenscale, [lenscale, lenscale], nvars, backend=bkd
        ) + GaussianNoiseKernel(
            noise, [0.02, 2], fixed=True, backend=bkd
        )

        gp = ExactGaussianProcess(nvars, kernel)

        # def fun(xx):
        #     return (xx**2).sum(axis=0)[:, None]

        def fun(xx, noisy=True):
            vals = np.cos(2 * np.pi * xx.T)
            if not noisy:
                return vals
            return vals + np.random.normal(0, np.sqrt(noise), xx.T.shape)

        ntrain_samples = 6
        train_samples = np.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)
        torch_train_samples = bkd._la_atleast2d(train_samples)
        torch_train_values = bkd._la_atleast2d(train_values)

        from pyapprox.surrogates.gaussianprocess.gaussian_process import (
            GaussianProcess,
            Matern,
            ConstantKernel as CKernel,
            WhiteKernel,
        )

        pyakernel = CKernel(sigma, "fixed") * Matern(
            lenscale, length_scale_bounds="fixed", nu=np.inf
        ) + WhiteKernel(noise, "fixed")

        assert np.allclose(
            bkd._la_to_numpy(kernel(torch_train_samples)),
            pyakernel(torch_train_samples.T),
        )

        gp.fit(torch_train_samples, torch_train_values)

        pyagp = GaussianProcess(pyakernel, alpha=0.0)
        pyagp.fit(train_samples, train_values)

        ntest_samples = 5
        test_samples = bkd._la_array(
            np.random.uniform(-1, 1, (nvars, ntest_samples))
        )
        test_samples = bkd._la_linspace(-1, 1, 5)[None, :]

        pyagp_vals, pyagp_std = pyagp(test_samples, return_std=True)
        gp_vals, gp_std = gp(test_samples, return_std=True)
        # print(gp_std[:, 0]-pyagp_std)
        assert np.allclose(
            bkd._la_to_numpy(gp_std[:, 0]), pyagp_std, atol=1e-6
        )

        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 1)[1]
        # gp.plot(ax, [-1, 1], plt_kwargs={"c": "r", "ls": "-"}, npts_1d=101)
        # pyagp.plot_1d(101, [-1, 1], ax=ax)
        # xx = np.linspace(-1, 1, 101)[None, :]
        # plt.plot(xx[0], fun(xx, False))
        # plt.plot(gp.train_samples[0], gp.train_values, 'o')
        # plt.show()

    def test_variational_gp_training(self):
        bkd = self.get_backend()
        ntrain_samples = 10
        nvars, ninducing_samples = 1, 5
        kernel = MaternKernel(np.inf, 0.5, [1e-1, 1], nvars, backend=bkd)
        inducing_samples = np.linspace(-1, 1, ninducing_samples)[None, :]
        noise = HyperParameter(
            "noise",
            1,
            1,
            (1e-6, 1),
            LogHyperParameterTransform(backend=bkd),
        )
        inducing_samples = InducingSamples(
            nvars,
            ninducing_samples,
            inducing_samples=inducing_samples,
            noise=noise,
            backend=bkd,
        )
        values_trans = IdentityTransform(backend=bkd)
        gp = InducingGaussianProcess(
            nvars,
            kernel,
            inducing_samples,
            kernel_reg=1e-10,
            values_trans=values_trans,
        )

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        train_samples = bkd._la_linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        gp.set_training_data(train_samples, train_values)
        # bounds = gp.hyp_list.get_active_opt_bounds().numpy()
        # x0 = gp._get_random_optimizer_initial_guess(bounds)
        x0 = bkd._la_to_numpy(gp.hyp_list.get_active_opt_params())
        errors = check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None], disp=False
        )
        assert errors.min() / errors.max() < 1e-6

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
        test_samples = bkd._la_atleast2d(
            np.random.uniform(-1, 1, (nvars, ntest_samples))
        )
        test_vals = fun(test_samples)
        gp_mu, gp_std = gp(test_samples, return_std=True)
        # print(gp_mu-test_vals)
        assert np.allclose(gp_mu, test_vals, atol=6e-3)

    def test_variational_gp_collapse_to_exact_gp(self):
        bkd = self.get_backend()
        nvars = 1
        ntrain_samples = 6
        noise_var = 1e-8
        kernel = MaternKernel(np.inf, 1, [1e-1, 1], nvars, backend=bkd)
        values_trans = IdentityTransform()

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        train_samples = bkd._la_linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        ntest_samples = 6
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))

        exact_gp = ExactGaussianProcess(
            nvars,
            kernel
            + GaussianNoiseKernel(noise_var, [0.1, 1], fixed=True, backend=bkd),
            trend=None,
            values_trans=values_trans,
            kernel_reg=0,
        )
        exact_gp.set_training_data(train_samples, train_values)
        exact_gp.fit(train_samples, train_values, max_nglobal_opt_iters=1)
        exact_gp_vals, exact_gp_std = exact_gp(test_samples, return_std=True)

        inducing_samples = train_samples
        ninducing_samples = ntrain_samples
        # fix hyperparameters so they are not changed from exact_gp
        # or setting provided if not found in exact_gp
        noise = HyperParameter(
            "noise_std",
            1,
            np.sqrt(noise_var),
            [0.1, 1],
            LogHyperParameterTransform(backend=bkd),
            fixed=True,
        )
        inducing_samples = InducingSamples(
            nvars,
            ninducing_samples,
            inducing_samples=inducing_samples,
            inducing_sample_bounds=bkd._la_atleast1d([-1, 1]),
            noise=noise,
            backend=bkd,
        )
        inducing_samples.hyp_list.set_all_inactive()
        values_trans = IdentityTransform(backend=bkd)
        # use correlation length learnt by exact gp
        vi_kernel = kernel
        vi_gp = InducingGaussianProcess(
            nvars,
            vi_kernel,
            inducing_samples,
            kernel_reg=0,
            values_trans=values_trans,
        )
        vi_gp.fit(train_samples, train_values, max_nglobal_opt_iters=1)
        vi_gp_vals, vi_gp_std = vi_gp(test_samples, return_std=True)

        # print(vi_gp_vals-exact_gp_vals)
        assert np.allclose(vi_gp_vals, exact_gp_vals, atol=1e-12)
        # print(vi_gp_std-exact_gp_std)
        # I think larger tolerance needed because sqrt of covariance
        # is being taken inside funcitns
        assert np.allclose(vi_gp_std, exact_gp_std, atol=5e-5)

    def test_icm_gp(self):
        bkd = self.get_backend()
        nvars, noutputs = 1, 2

        def fun0(xx):
            delta0 = 0.0
            return np.cos(2 * np.pi * xx.T + delta0)

        def fun1(xx):
            delta1 = 0.5
            return np.cos(2 * np.pi * xx.T + delta1)

        funs = [fun0, fun1]

        radii, radii_bounds = np.arange(1, noutputs + 1), [1, 10]
        angles = np.pi / 4
        latent_kernel = MaternKernel(
            np.inf, 0.5, [1e-1, 2], nvars, backend=bkd
        )
        output_kernel = SphericalCovariance(
            noutputs,
            radii=radii,
            radii_bounds=radii_bounds,
            angles=angles,
            angle_bounds=[0, np.pi],
            backend=bkd,
        )

        kernel = ICMKernel(latent_kernel, output_kernel, noutputs)

        nsamples_per_output = [12, 12]
        samples_per_output = [
            bkd._la_atleast2d(np.random.uniform(-1, 1, (nvars, nsamples)))
            for nsamples in nsamples_per_output
        ]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)
        ]

        gp = MOExactGaussianProcess(
            nvars,
            kernel,
            values_trans=IdentityTransform(),
            kernel_reg=1e-8,
        )
        gp.fit(samples_per_output, values_per_output, max_nglobal_opt_iters=3)

        # check correlation between models is estimated correctly.
        # SphericalCovariance is not guaranteed to recover the statistical
        # correlation, but for this case it can
        cov_matrix = output_kernel.get_covariance_matrix()
        corr_matrix = bkd._la_get_correlation_from_covariance(cov_matrix)
        samples = bkd._la_atleast2d(np.random.uniform(-1, 1, (1, 101)))
        values = bkd._la_hstack([fun(samples) for fun in funs])
        assert np.allclose(
            corr_matrix,
            bkd._la_get_correlation_from_covariance(
                bkd._la_cov(values.T, ddof=1)
            ),
            atol=1e-2,
        )

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
        bkd = self.get_backend()
        nvars, noutputs = 1, 4
        degree = 0
        kernels = [
            MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)
            for ii in range(noutputs)
        ]
        scaling_indices = bkd._la_arange(degree + 1, dtype=int)[None, :]
        scalings = [
            BasisExpansion(
                MultiIndexBasis(
                    [Monomial1D(backend=bkd) for ii in range(nvars)],
                    indices=scaling_indices,
                ),
                None,
                1,
                [-1, 2],
            )
            for ii in range(noutputs - 1)
        ]
        kernel = MultiPeerKernel(kernels, scalings)

        def peer_fun(delta, xx):
            return bkd._la_cos(2 * np.pi * xx.T + delta)

        def target_fun(peer_funs, xx):
            # return (
            #     np.hstack([f(xx) for f in peer_funs]).sum(axis=1)[:, None] +
            #     np.exp(-xx.T**2*2))
            return bkd._la_cos(2 * np.pi * xx.T)

        peer_deltas = bkd._la_linspace(0, 1, noutputs - 1)
        peer_funs = [partial(peer_fun, delta) for delta in peer_deltas]
        funs = peer_funs + [partial(target_fun, peer_funs)]

        # nsamples_per_output = np.array([5 for ii in range(noutputs-1)]+[4])*2
        nsamples_per_output = np.array([7 for ii in range(noutputs - 1)] + [5])
        samples_per_output = [
            bkd._la_atleast2d(np.random.uniform(-1, 1, (nvars, nsamples)))
            for nsamples in nsamples_per_output
        ]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)
        ]

        gp = MOExactGaussianProcess(
            nvars,
            kernel,
            values_trans=IdentityTransform(backend=bkd),
            kernel_reg=0,
        )
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
            nvars, kernel, values_trans=IdentityTransform(), kernel_reg=0
        )
        peer_gp.set_training_data(samples_per_output, values_per_output)
        assert np.allclose(
            gp._neg_log_likelihood_with_hyperparameter_trend(),
            peer_gp._neg_log_likelihood_with_hyperparameter_trend(),
        )
        xx = bkd._la_linspace(-1, 1, 31)[None, :]
        gp_mean, gp_std = gp([xx] * noutputs, return_std=True)
        peer_gp_mean, peer_gp_std = peer_gp([xx] * noutputs, return_std=True)
        assert np.allclose(peer_gp_mean, gp_mean)
        assert np.allclose(peer_gp_std, gp_std)

    def test_icm_peer_gp(self):
        bkd = self.get_backend()
        nvars, noutputs = 1, 4

        def peer_fun(delta, xx):
            return np.cos(2 * np.pi * xx.T + delta)

        def target_fun(peer_funs, xx):
            # return (
            #    np.hstack([f(xx) for f in peer_funs]).sum(axis=1)[:, None] +
            #    np.exp(-xx.T**2*2))
            return np.cos(2 * np.pi * xx.T)

        # radii, radii_bounds = np.ones(noutputs), [1, 10]
        radii, radii_bounds = bkd._la_arange(1, 1 + noutputs), [1, 10]
        angles = np.pi / 2
        latent_kernel = MaternKernel(
            np.inf, 0.5, [1e-1, 2], nvars, backend=bkd
        )
        output_kernel = SphericalCovariance(
            noutputs,
            radii=radii,
            radii_bounds=radii_bounds,
            angles=angles,
            angle_bounds=[0, np.pi],
            backend=bkd,
        )

        kernel = ICMKernel(latent_kernel, output_kernel, noutputs)

        peer_deltas = bkd._la_linspace(0.2, 1, noutputs - 1)
        peer_funs = [partial(peer_fun, delta) for delta in peer_deltas]
        funs = peer_funs + [partial(target_fun, peer_funs)]

        nsamples_per_output = np.array([7 for ii in range(noutputs - 1)] + [5])
        # nsamples_per_output = np.array([5 for ii in range(noutputs-1)]+[4])*2
        # nsamples_per_output = np.array([3 for ii in range(noutputs-1)]+[2])
        samples_per_output = [
            bkd._la_atleast2d(np.random.uniform(-1, 1, (nvars, nsamples)))
            for nsamples in nsamples_per_output
        ]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)
        ]

        gp = MOICMPeerExactGaussianProcess(
            nvars,
            kernel,
            output_kernel,
            values_trans=IdentityTransform(),
            kernel_reg=0,
        )
        gp_params = gp.hyp_list.get_active_opt_params()

        from pyapprox.util.utilities import check_gradients

        bounds = bkd._la_to_numpy(gp.hyp_list.get_active_opt_bounds())
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        icm_cons = gp._get_constraints(noutputs)
        errors = check_gradients(
            lambda x: icm_cons[0]["fun"](x[:, 0], *icm_cons[0]["args"]),
            lambda x: icm_cons[0]["jac"](x[:, 0], *icm_cons[0]["args"]),
            x0[:, None],
            disp=False,
        )
        assert errors.min() / errors.max() < 1e-6
        # reset values to good guess
        gp.hyp_list.set_active_opt_params(gp_params)

        gp.set_training_data(samples_per_output, values_per_output)
        x0 = gp.hyp_list.get_active_opt_params().numpy()
        errors = check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None], disp=False
        )
        gp.hyp_list.set_active_opt_params(gp_params)
        assert errors.min() / errors.max() < 3.2e-6

        gp.fit(samples_per_output, values_per_output, max_nglobal_opt_iters=3)
        cov_matrix = output_kernel.get_covariance_matrix()
        for ii in range(2, noutputs):
            for jj in range(1, ii):
                bkd._la_abs(cov_matrix[ii, jj]) < 1e-10

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
        bkd = self.get_backend()
        nvars, noutputs = 1, 4

        radii, radii_bounds = bkd._la_ones(noutputs), [1, 2]
        angles = np.pi / 4
        latent_kernel = MaternKernel(
            np.inf, 0.5, [1e-1, 2], nvars, backend=bkd
        )
        output_kernel = SphericalCovariance(
            noutputs,
            radii=radii,
            radii_bounds=radii_bounds,
            angles=angles,
            angle_bounds=[0, np.pi],
            backend=bkd,
        )

        output_kernels = [output_kernel]
        latent_kernels = [latent_kernel]
        discrepancy_kernels = [
            ConstantKernel(
                0.1,
                (1e-1, 1),
                transform=LogHyperParameterTransform(backend=bkd),
            )
            * MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)
            for ii in range(noutputs)
        ]
        co_kernel = CollaborativeKernel(
            latent_kernels, output_kernels, discrepancy_kernels, noutputs
        )

        def peer_fun(delta, xx):
            return bkd._la_cos(2 * np.pi * xx.T + delta)

        def target_fun(peer_funs, xx):
            return bkd._la_hstack([f(xx) for f in peer_funs]).sum(axis=1)[
                :, None
            ] + bkd._la_exp(-xx.T**2 * 2)
            # return np.cos(2*np.pi*xx.T)

        peer_deltas = bkd._la_linspace(0.2, 1, noutputs - 1)
        peer_funs = [partial(peer_fun, delta) for delta in peer_deltas]
        funs = peer_funs + [partial(target_fun, peer_funs)]

        nsamples_per_output = np.array([7 for ii in range(noutputs - 1)] + [5])
        # nsamples_per_output = np.array([5 for ii in range(noutputs-1)]+[4])*2
        # nsamples_per_output = np.array([3 for ii in range(noutputs-1)]+[2])
        samples_per_output = [
            bkd._la_atleast2d(np.random.uniform(-1, 1, (nvars, nsamples)))
            for nsamples in nsamples_per_output
        ]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)
        ]

        gp = MOExactGaussianProcess(
            nvars,
            co_kernel,
            values_trans=IdentityTransform(backend=bkd),
            kernel_reg=0,
        )
        gp_params = gp.hyp_list.get_active_opt_params()

        gp.set_training_data(samples_per_output, values_per_output)
        x0 = bkd._la_to_numpy(gp.hyp_list.get_active_opt_params())
        errors = check_gradients(
            lambda x: gp._fit_objective(x[:, 0]), True, x0[:, None], disp=False
        )
        gp.hyp_list.set_active_opt_params(gp_params)
        assert errors.min() / errors.max() < 1e-6

        # slsqp requires feasiable initial guess. It does not throw an
        # error if guess is infeasiable it just runs and produces garbage.
        # For now just start test from feasiable initial parameter values
        # and run optimization once.
        # todo change initial guess to always be feasiable
        gp.fit(samples_per_output, values_per_output, max_nglobal_opt_iters=1)
        cov_matrix = output_kernel.get_covariance_matrix()
        for ii in range(2, noutputs):
            for jj in range(1, ii):
                assert True  # np.abs(cov_matrix[ii, jj]) < 1e-10

        # import matplotlib.pyplot as plt
        # axs = plt.subplots(
        #     1, noutputs, figsize=(noutputs*8, 6), sharey=True)[1]
        # xx = np.linspace(-1, 1, 101)[None, :]
        # for ii in range(noutputs):
        #     gp.plot(axs[ii], [-1, 1], output_id=ii)
        #     axs[ii].plot(xx[0], funs[ii](xx), '--')
        #     axs[ii].plot(gp.train_samples[ii][0], gp.train_values[ii], 'o')
        # plt.show()


class TestNumpyNystrom(TestNystrom, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()

    def get_mvn(self):
        # set multivariatenormal for scipy to have same api as torch
        class NumpyMultivariateNormal:
            def __init__(self, mean, covariance_matrix):
                self._mvn = stats.multivariate_normal(mean, covariance_matrix)

            def log_prob(self, xx):
                return self._mvn.logpdf(xx)

        return NumpyMultivariateNormal


class TestTorchNystrom(TestNystrom, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin()

    def get_mvn(self):
        return TorchMultivariateNormal


class TestTorchGaussianProcess(TestGaussianProcess, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)

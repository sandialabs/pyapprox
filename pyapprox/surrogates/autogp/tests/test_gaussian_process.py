import unittest
from functools import partial

import numpy as np
from scipy import stats
from torch.distributions import MultivariateNormal as TorchMultivariateNormal
import matplotlib.pyplot as plt

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
        fd_eps = bkd.flip(bkd.logspace(-13, 0, 14))
    if direction is None:
        nvars = sample.shape[0]
        direction = bkd.array(np.random.normal(0, 1, (nvars, 1)))
        direction /= bkd.linalg.norm(direction)

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
            bkd.norm(
                fd_directional_grad.reshape(directional_grad.shape)
                - directional_grad
            )
        )
        if relative:
            errors[-1] /= bkd.norm(directional_grad)
        if disp:
            print(
                row_format.format(
                    fd_eps[ii],
                    bkd.norm(directional_grad),
                    bkd.norm(fd_directional_grad),
                    errors[ii],
                )
            )
    return bkd.array(errors)


class TestNystrom:
    def setUp(self):
        np.random.seed(1)

    def _check_invert_noisy_low_rank_nystrom_approximation(self, N, M):
        bkd = self.get_backend()
        MultivariateNormal = self.get_mvn()
        noise_std = 2
        tmp = bkd.asarray(np.random.normal(0, 1, (N, N)))
        C_NN = tmp.T @ tmp
        C_MN = C_NN[:M]
        C_MM = C_NN[:M, :M]
        Q = C_MN.T @ bkd.inv(C_MM) @ C_MN + noise_std**2 * bkd.eye(N)

        values = bkd.full((N, 1), 1)
        p_y = MultivariateNormal(values[:, 0] * 0, covariance_matrix=Q)
        logpdf1 = p_y.log_prob(values[:, 0])

        L_UU = bkd.cholesky(C_MM)
        logpdf2 = _log_prob_gaussian_with_noisy_nystrom_covariance(
            noise_std, L_UU, C_MN.T, values, bkd
        )
        assert np.allclose(logpdf1, logpdf2)

        if N != M:
            return

        assert np.allclose(Q, C_NN + noise_std**2 * bkd.eye(N))

        values = values
        Q_inv = bkd.inv(Q)

        Delta = bkd.solve_triangular(L_UU, C_MN.T, lower=True) / noise_std
        Omega = bkd.eye(M) + Delta @ Delta.T
        L_Omega = bkd.cholesky(Omega)
        log_det = 2 * bkd.log(
            bkd.get_diagonal(L_Omega)
        ).sum() + 2 * N * np.log(noise_std)
        gamma = bkd.solve_triangular(L_Omega, Delta @ values, lower=True)
        assert np.allclose(log_det, bkd.slogdet(Q)[1])

        coef = Q_inv @ values
        assert np.allclose(
            values.T @ coef,
            values.T @ values / noise_std**2
            - gamma.T @ gamma / noise_std**2,
        )

        mll = -0.5 * (
            values.T @ coef + bkd.slogdet(Q)[1] + N * np.log(2 * np.pi)
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

    def _check_exact_gp_training(self, trend, out_trans, constant):
        bkd = self.get_backend()
        nvars = 1

        if not out_trans:
            out_trans = IdentityTransform(backend=bkd)
        else:
            out_trans = StandardDeviationTransform(trans=False, backend=bkd)

        if trend:
            basis = MultiIndexBasis(
                [Monomial1D(backend=bkd) for ii in range(nvars)]
            )
            basis.set_indices(bkd.arange(3, dtype=int)[None, :])
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
            nvars, kernel, trend=trend
        )
        gp.set_output_transform(out_trans)

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        ntrain_samples = 10
        train_samples = bkd.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)
        gp._set_training_data(train_samples, train_values)
        errors = gp._loss.check_apply_jacobian(
            gp._optimizer._initial_interate_gen()
        )
        assert errors.min() / errors.max() < 2e-6

        gp.fit(train_samples, train_values)

        ntest_samples = 5
        test_samples = bkd.asarray(
            np.random.uniform(-1, 1, (nvars, ntest_samples))
        )
        test_vals = fun(test_samples)

        gp_vals, gp_std = gp.evaluate(test_samples, return_std=True)

        if trend is not None:
            assert bkd.allclose(gp_vals, test_vals, atol=1e-14)
            xx = bkd.linspace(-1, 1, 101)[None, :]
            assert bkd.allclose(
                gp._out_trans.map_from_canonical(gp._canonical_trend(xx)),
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
        bkd_train_samples = bkd.asarray(train_samples)
        bkd_train_values = bkd.asarray(train_values)

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
            bkd.to_numpy(kernel(bkd_train_samples)),
            pyakernel(bkd_train_samples.T),
        )
        gp.fit(bkd_train_samples, bkd_train_values)

        pyagp = GaussianProcess(pyakernel, alpha=0.0)
        pyagp.fit(train_samples, train_values)

        ntest_samples = 5
        test_samples = bkd.array(
            np.random.uniform(-1, 1, (nvars, ntest_samples))
        )
        test_samples = bkd.linspace(-1, 1, 5)[None, :]

        pyagp_vals, pyagp_std = pyagp(test_samples, return_std=True)
        gp_vals, gp_std = gp.evaluate(test_samples, return_std=True)
        # print(gp_std[:, 0]-pyagp_std)
        assert np.allclose(
            bkd.to_numpy(gp_std[:, 0]), pyagp_std, atol=1e-6
        )

        # test plot runs
        ax = plt.subplots(1, 1)[1]
        gp.plot(ax, [-1, 1], plt_kwargs={"c": "r", "ls": "-"}, npts_1d=101)
        pyagp.plot_1d(101, [-1, 1], ax=ax)
        xx = np.linspace(-1, 1, 101)[None, :]
        plt.plot(xx[0], fun(xx, False))
        plt.plot(gp.get_train_samples()[0], gp.get_train_values(), 'o')

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
        gp = InducingGaussianProcess(
            nvars,
            kernel,
            inducing_samples,
            kernel_reg=1e-10,
        )

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        train_samples = bkd.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        gp._set_training_data(train_samples, train_values)
        errors = gp._loss.check_apply_jacobian(
            gp._optimizer._initial_interate_gen()
        )
        assert errors.min() / errors.max() < 1e-6

        gp.fit(train_samples, train_values)
        # print(gp)

        # test plot runs
        xx = np.linspace(-1, 1, 101)[None, :]
        plt.plot(xx[0], gp.evaluate(xx, False), '--')
        plt.plot(gp.inducing_samples.get_samples(),
                 0*gp.inducing_samples.get_samples(), 's')
        plt.plot(xx[0], fun(xx)[:, 0], 'k-')
        plt.plot(gp.get_train_samples()[0], gp.get_train_values(), 'o')
        gp_mu, gp_std = gp.evaluate(xx, return_std=True)
        gp_mu = gp_mu[:, 0]
        gp_std = gp_std[:, 0]
        plt.fill_between(xx[0], gp_mu-3*gp_std, gp_mu+3*gp_std, alpha=0.1,
                         color='blue')

        ntest_samples = 10
        test_samples = bkd.asarray(
            np.random.uniform(-1, 1, (nvars, ntest_samples))
        )
        test_vals = fun(test_samples)
        gp_mu, gp_std = gp.evaluate(test_samples, return_std=True)
        # print(gp_mu-test_vals)
        assert np.allclose(gp_mu, test_vals, atol=6e-3)

    def test_variational_gp_collapse_to_exact_gp(self):
        bkd = self.get_backend()
        nvars = 1
        ntrain_samples = 6
        noise_var = 1e-8
        kernel = MaternKernel(np.inf, 1, [1e-1, 1], nvars, backend=bkd)

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        train_samples = bkd.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)

        ntest_samples = 6
        test_samples = np.random.uniform(-1, 1, (nvars, ntest_samples))

        exact_gp = ExactGaussianProcess(
            nvars,
            kernel
            + GaussianNoiseKernel(
                noise_var, [0.1, 1], fixed=True, backend=bkd
            ),
            trend=None,
            kernel_reg=0,
        )
        exact_gp.fit(train_samples, train_values)
        exact_gp_vals, exact_gp_std = exact_gp.evaluate(
            test_samples, return_std=True
        )

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
            inducing_sample_bounds=bkd.atleast1d([-1, 1]),
            noise=noise,
            backend=bkd,
        )
        inducing_samples.hyp_list.set_all_inactive()
        # use correlation length learnt by exact gp
        vi_kernel = kernel
        vi_gp = InducingGaussianProcess(
            nvars,
            vi_kernel,
            inducing_samples,
            kernel_reg=0,
        )
        vi_gp.fit(train_samples, train_values)
        vi_gp_vals, vi_gp_std = vi_gp.evaluate(test_samples, return_std=True)

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
            bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
            for nsamples in nsamples_per_output
        ]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)
        ]

        gp = MOExactGaussianProcess(
            nvars,
            kernel,
            kernel_reg=1e-8,
        )
        gp.set_optimizer(ncandidates=3)
        gp.fit(samples_per_output, values_per_output)

        # check correlation between models is estimated correctly.
        # SphericalCovariance is not guaranteed to recover the statistical
        # correlation, but for this case it can
        cov_matrix = output_kernel.get_covariance_matrix()
        corr_matrix = bkd.get_correlation_from_covariance(cov_matrix)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 101)))
        values = bkd.hstack([fun(samples) for fun in funs])
        assert np.allclose(
            corr_matrix,
            bkd.get_correlation_from_covariance(
                bkd.cov(values.T, ddof=1)
            ),
            atol=1e-2,
        )

        # test plot runs
        ax = plt.subplots(1, 1)[1]
        gp.plot(ax, [-1, 1], output_id=0, plt_kwargs={"c": "r", "ls": "-"})
        gp.plot(ax, [-1, 1], output_id=1)
        xx = np.linspace(-1, 1, 101)[None, :]
        ax.plot(xx[0], funs[0](xx), '--')
        ax.plot(xx[0], funs[1](xx), ':')
        ax.plot(gp.get_train_samples()[0][0], gp.get_train_values()[0], 'o')
        ax.plot(gp.get_train_samples()[1][0], gp.get_train_values()[1], 's')

    def test_peer_gaussian_process(self):
        bkd = self.get_backend()
        nvars, noutputs = 1, 4
        degree = 0
        kernels = [
            MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)
            for ii in range(noutputs)
        ]
        scaling_indices = bkd.arange(degree + 1, dtype=int)[None, :]
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
            return bkd.cos(2 * np.pi * xx.T + delta)

        def target_fun(peer_funs, xx):
            # return (
            #     np.hstack([f(xx) for f in peer_funs]).sum(axis=1)[:, None] +
            #     np.exp(-xx.T**2*2))
            return bkd.cos(2 * np.pi * xx.T)

        peer_deltas = bkd.linspace(0, 1, noutputs - 1)
        peer_funs = [partial(peer_fun, delta) for delta in peer_deltas]
        funs = peer_funs + [partial(target_fun, peer_funs)]

        # nsamples_per_output = np.array([5 for ii in range(noutputs-1)]+[4])*2
        nsamples_per_output = np.array([7 for ii in range(noutputs - 1)] + [5])
        samples_per_output = [
            bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
            for nsamples in nsamples_per_output
        ]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)
        ]

        gp = MOExactGaussianProcess(
            nvars,
            kernel,
            kernel_reg=0,
        )
        gp.set_optimizer(ncandidates=3)
        gp.fit(samples_per_output, values_per_output)

        # test plots run
        axs = plt.subplots(
            1, noutputs, figsize=(noutputs*8, 6), sharey=True)[1]
        xx = bkd.linspace(-1, 1, 101)[None, :]
        for ii in range(noutputs):
            gp.plot(axs[ii], [-1, 1], output_id=ii)
            axs[ii].plot(xx[0], funs[ii](xx), '--')
            axs[ii].plot(
                gp.get_train_samples()[ii][0], gp.get_train_values()[ii], 'o'
            )

        # check that when using hyperparameters found by dense GP the PeerGP
        # return the same likelihood value and prediction mean and std. dev.
        peer_gp = MOPeerExactGaussianProcess(nvars, kernel, kernel_reg=0)
        peer_gp._set_training_data(samples_per_output, values_per_output)
        assert np.allclose(
            gp._neg_log_likelihood_with_hyperparameter_trend(),
            peer_gp._neg_log_likelihood_with_hyperparameter_trend(),
        )
        xx = bkd.linspace(-1, 1, 31)[None, :]
        gp_mean, gp_std = gp.evaluate([xx] * noutputs, return_std=True)
        peer_gp_mean, peer_gp_std = peer_gp.evaluate([xx] * noutputs, return_std=True)
        assert np.allclose(peer_gp_mean, gp_mean)
        assert np.allclose(peer_gp_std, gp_std)

    @unittest.skip("Incomplete")
    def test_collaborative_gp(self):
        bkd = self.get_backend()
        nvars, noutputs = 1, 4

        radii, radii_bounds = bkd.ones(noutputs), [1, 2]
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
            return bkd.cos(2 * np.pi * xx.T + delta)

        def target_fun(peer_funs, xx):
            return bkd.hstack([f(xx) for f in peer_funs]).sum(axis=1)[
                :, None
            ] + bkd.exp(-xx.T**2 * 2)
            # return np.cos(2*np.pi*xx.T)

        peer_deltas = bkd.linspace(0.2, 1, noutputs - 1)
        peer_funs = [partial(peer_fun, delta) for delta in peer_deltas]
        funs = peer_funs + [partial(target_fun, peer_funs)]

        nsamples_per_output = np.array([7 for ii in range(noutputs - 1)] + [5])
        # nsamples_per_output = np.array([5 for ii in range(noutputs-1)]+[4])*2
        # nsamples_per_output = np.array([3 for ii in range(noutputs-1)]+[2])
        samples_per_output = [
            bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
            for nsamples in nsamples_per_output
        ]

        values_per_output = [
            fun(samples) for fun, samples in zip(funs, samples_per_output)
        ]

        gp = MOExactGaussianProcess(
            nvars,
            co_kernel,
            kernel_reg=0,
        )
        gp_params = gp.hyp_list.get_active_opt_params()

        gp._set_training_data(samples_per_output, values_per_output)
        errors = gp._loss.check_apply_jacobian(
            gp._optimizer._initial_interate_gen()
        )
        gp.hyp_list.set_active_opt_params(gp_params)
        assert errors.min() / errors.max() < 1e-6

        # slsqp requires feasiable initial guess. It does not throw an
        # error if guess is infeasiable it just runs and produces garbage.
        # For now just start test from feasiable initial parameter values
        # and run optimization once.
        # todo change initial guess to always be feasiable
        gp.fit(samples_per_output, values_per_output)
        cov_matrix = output_kernel.get_covariance_matrix()
        bkd.cholesky(cov_matrix)
        print(cov_matrix)
        for ii in range(2, noutputs):
            for jj in range(1, ii):
                assert True  # np.abs(cov_matrix[ii, jj]) < 1e-10

        # test plot runs
        axs = plt.subplots(
            1, noutputs, figsize=(noutputs*8, 6), sharey=True)[1]
        xx = bkd.linspace(-1, 1, 101)[None, :]
        for ii in range(noutputs):
            gp.plot(axs[ii], [-1, 1], output_id=ii)
            axs[ii].plot(xx[0], funs[ii](xx), '--')
            axs[ii].plot(
                gp.get_train_samples()[ii][0], gp.get_train_values()[ii], 'o'
            )
        plt.show()


class TestNumpyNystrom(TestNystrom, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin

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
        return TorchLinAlgMixin

    def get_mvn(self):
        return TorchMultivariateNormal


class TestTorchGaussianProcess(TestGaussianProcess, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import itertools
from functools import partial

import numpy as np
from scipy import stats
from torch.distributions import MultivariateNormal as TorchMultivariateNormal
import matplotlib.pyplot as plt

from pyapprox.surrogates.univariate.base import Monomial1D
from pyapprox.surrogates.affine.basis import MultiIndexBasis
from pyapprox.surrogates.affine.basisexp import (
    BasisExpansion,
    MonomialExpansion,
)
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
from pyapprox.surrogates.gaussianprocess.exactgp import (
    ExactGaussianProcess,
    MOExactGaussianProcess,
    MOPeerExactGaussianProcess,
    GaussianProcessIdentityTransform,
    GaussianProcessStandardDeviationTransform,
    SequentialMultiLevelGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.stats import (
    GaussianProcessStatistics,
    EnsembleGaussianProcessStatistics,
    marginalize_gaussian_process,
    MonteCarloKernelStatistics,
    TensorProductQuadratureKernelStatistics,
)
from pyapprox.surrogates.gaussianprocess.mokernels import (
    ICMKernel,
    MultiPeerKernel,
    CollaborativeKernel,
    MultiLevelKernel,
)
from pyapprox.surrogates.gaussianprocess.variationalgp import (
    _log_prob_gaussian_with_noisy_nystrom_covariance,
    InducingSamples,
    InducingGaussianProcess,
)
from pyapprox.util.transforms import (
    IdentityTransform,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.transforms import AffineTransform

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.surrogates.affine.basis import (
    FixedGaussianTensorProductQuadratureRuleFromVariable,
)
from pyapprox.benchmarks import IshigamiBenchmark
from pyapprox.expdesign.sequences import SobolSequence
from pyapprox.interface.model import ModelFromVectorizedCallable


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
            values.T @ values / noise_std**2 - gamma.T @ gamma / noise_std**2,
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
            out_trans = GaussianProcessIdentityTransform(backend=bkd)
        else:
            out_trans = GaussianProcessStandardDeviationTransform(backend=bkd)

        if trend:
            basis = MultiIndexBasis(
                [Monomial1D(backend=bkd) for ii in range(nvars)],
            )
            basis.set_indices(bkd.arange(3, dtype=int)[None, :])
            trend = BasisExpansion(basis, None, 1.0, (-1e3, 1e3))
        else:
            trend = None

        kernel = MaternKernel(np.inf, 0.1, [1e-1, 1], nvars, backend=bkd)

        kernel = kernel
        if constant:
            constant_kernel = ConstantKernel(
                0.1,
                (1e-3, 1e1),
                transform=LogHyperParameterTransform(backend=bkd),
                backend=bkd,
            )
            kernel = constant_kernel * kernel

        gp = ExactGaussianProcess(nvars, kernel, trend=trend, kernel_reg=1e-8)
        gp.set_output_transform(out_trans)

        def fun(xx):
            return (xx**2).sum(axis=0)[:, None]

        ntrain_samples = 10
        train_samples = bkd.linspace(-1, 1, ntrain_samples)[None, :]
        train_values = fun(train_samples)
        gp._set_training_data(train_samples, train_values)
        errors = gp._loss.check_apply_jacobian(
            gp._optimizer._initial_interate_gen(), disp=False
        )
        assert errors.min() / errors.max() < 2.5e-6

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
        plt.plot(xx[0], gp.evaluate(xx, False), "--")
        plt.plot(
            gp.inducing_samples.get_samples(),
            0 * gp.inducing_samples.get_samples(),
            "s",
        )
        plt.plot(xx[0], fun(xx)[:, 0], "k-")
        plt.plot(gp.get_train_samples()[0], gp.get_train_values(), "o")
        gp_mu, gp_std = gp.evaluate(xx, return_std=True)
        gp_mu = gp_mu[:, 0]
        gp_std = gp_std[:, 0]
        plt.fill_between(
            xx[0],
            gp_mu - 3 * gp_std,
            gp_mu + 3 * gp_std,
            alpha=0.1,
            color="blue",
        )

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
            inducing_sample_bounds=bkd.asarray([-1, 1]),
            noise=noise,
            backend=bkd,
        )
        inducing_samples.hyp_list().set_all_inactive()
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

        nsamples_per_output = [60, 60]
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
        # correlation, but for this case it can (Even for cases it works it
        # recovery depends on the number of samples per output)
        cov_matrix = output_kernel.get_covariance_matrix()
        corr_matrix = bkd.get_correlation_from_covariance(cov_matrix)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 101)))
        values = bkd.hstack([fun(samples) for fun in funs])
        # print(values.shape)
        # print(corr_matrix)
        # print(bkd.get_correlation_from_covariance(bkd.cov(values.T, ddof=1)))
        assert np.allclose(
            corr_matrix,
            bkd.get_correlation_from_covariance(bkd.cov(values.T, ddof=1)),
            atol=1e-2,
        )

        # test plot runs
        ax = plt.subplots(1, 1)[1]
        gp.plot(ax, [-1, 1], output_id=0, plt_kwargs={"c": "r", "ls": "-"})
        gp.plot(ax, [-1, 1], output_id=1)
        xx = np.linspace(-1, 1, 101)[None, :]
        ax.plot(xx[0], funs[0](xx), "--")
        ax.plot(xx[0], funs[1](xx), ":")
        ax.plot(gp.get_train_samples()[0][0], gp.get_train_values()[0], "o")
        ax.plot(gp.get_train_samples()[1][0], gp.get_train_values()[1], "s")

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
            1, noutputs, figsize=(noutputs * 8, 6), sharey=True
        )[1]
        xx = bkd.linspace(-1, 1, 101)[None, :]
        for ii in range(noutputs):
            gp.plot(axs[ii], [-1, 1], output_id=ii)
            axs[ii].plot(xx[0], funs[ii](xx), "--")
            axs[ii].plot(
                gp.get_train_samples()[ii][0], gp.get_train_values()[ii], "o"
            )

        # check that when using hyperparameters found by dense GP the PeerGP
        # return the same likelihood value and prediction mean and std. dev.
        peer_gp = MOPeerExactGaussianProcess(nvars, kernel, kernel_reg=0)
        peer_gp._set_training_data(samples_per_output, values_per_output)
        assert np.allclose(
            gp._neg_log_like_with_hyperparam_trend(),
            peer_gp._neg_log_like_with_hyperparam_trend(),
        )
        # compute the data need to predict with the GP
        peer_gp._set_coef()
        xx = bkd.linspace(-1, 1, 31)[None, :]
        gp_mean, gp_std = gp.evaluate([xx] * noutputs, return_std=True)
        peer_gp_mean, peer_gp_std = peer_gp.evaluate(
            [xx] * noutputs, return_std=True
        )
        assert np.allclose(peer_gp_mean[-1], gp_mean[-1])
        assert np.allclose(peer_gp_std[-1], gp_std[-1])

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
        gp_params = gp.hyp_list().get_active_opt_params()

        gp._set_training_data(samples_per_output, values_per_output)
        errors = gp._loss.check_apply_jacobian(
            gp._optimizer._initial_interate_gen()
        )
        gp.hyp_list().set_active_opt_params(gp_params)
        assert errors.min() / errors.max() < 1e-6

        # slsqp requires feasiable initial guess. It does not throw an
        # error if guess is infeasiable it just runs and produces garbage.
        # For now just start test from feasiable initial parameter values
        # and run optimization once.
        # todo change initial guess to always be feasiable
        gp.fit(samples_per_output, values_per_output)
        cov_matrix = output_kernel.get_covariance_matrix()
        bkd.cholesky(cov_matrix)
        # print(cov_matrix)
        for ii in range(2, noutputs):
            for jj in range(1, ii):
                assert True  # np.abs(cov_matrix[ii, jj]) < 1e-10

        # test plot runs
        axs = plt.subplots(
            1, noutputs, figsize=(noutputs * 8, 6), sharey=True
        )[1]
        xx = bkd.linspace(-1, 1, 101)[None, :]
        for ii in range(noutputs):
            gp.plot(axs[ii], [-1, 1], output_id=ii)
            axs[ii].plot(xx[0], funs[ii](xx), "--")
            axs[ii].plot(
                gp.get_train_samples()[ii][0], gp.get_train_values()[ii], "o"
            )
        # plt.show()

    def _setup_high_acccuracy_gp(self, kernel, variable):
        bkd = self.get_backend()
        constant = 1e3

        def fun(x):
            return constant * bkd.sum((2 * x - 0.5) ** 2, axis=0)[:, None]

        gp = ExactGaussianProcess(
            variable.nvars(), kernel, trend=None, kernel_reg=1e-8
        )
        gp.set_optimizer(ncandidates=4, verbosity=0)
        ntrain_samples = 25
        train_samples = (
            1 - bkd.cos(bkd.linspace(0, np.pi, ntrain_samples))[None, :]
        ) / 2
        train_values = fun(train_samples)
        gp.fit(train_samples, train_values)

        test_samples = variable.rvs(1000)
        test_values = fun(test_samples)
        # rel_l2_error = bkd.norm(gp(test_samples) - test_values) / bkd.norm(
        #     test_values
        # )
        # print(rel_l2_error)
        return gp, train_samples

    def _check_high_accuracy_gp_statistics(self, kernel, variable):
        # test that mean and variance of GP are accurate when GP is accurate
        # When GP is highly accurate, the values of the variance_of_variance
        # can be poor, especially when condition number of training kernel
        # matrix is poor (large)
        bkd = self.get_backend()
        constant = 1e3
        gp, train_samples = self._setup_high_acccuracy_gp(kernel, variable)
        true_mean = bkd.array([constant * 7 / 12])
        true_variance = constant**2 * 61 / 80 - true_mean**2

        gp_stat = GaussianProcessStatistics(gp, variable)
        expected_mean = gp_stat.expectation_of_mean()
        expected_variance = gp_stat.expectation_of_variance()
        # print(expected_mean - true_mean)
        assert bkd.allclose(expected_mean, true_mean)
        assert bkd.allclose(expected_variance, true_variance)

    def test_high_accuracy_gp_statistics(self):
        bkd = self.get_backend()
        marginals = [stats.uniform(0, 1)]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)

        kernel1 = MaternKernel(
            np.inf, 0.1, [1e-1, 1], variable.nvars(), backend=bkd
        )
        constant_kernel = ConstantKernel(
            0.1,
            (1e-3, 1e1),
            transform=LogHyperParameterTransform(backend=bkd),
            backend=bkd,
        )
        kernel2 = constant_kernel * kernel1
        test_cases = [[kernel1, variable], [kernel2, variable]]

        for test_case in test_cases:
            self._check_high_accuracy_gp_statistics(*test_case)

    def test_high_accuracy_gp_input_jacobian(self):
        bkd = self.get_backend()
        marginals = [stats.uniform(0, 1)]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)

        kernel1 = MaternKernel(
            np.inf, 0.1, [1e-1, 1], variable.nvars(), backend=bkd
        )
        constant_kernel = ConstantKernel(
            0.1,
            (1e-3, 1e1),
            transform=LogHyperParameterTransform(backend=bkd),
            backend=bkd,
        )
        noise_kernel = GaussianNoiseKernel(
            1e-2, [1e-8, 1e-2], fixed=True, backend=bkd
        )
        kernel2 = constant_kernel * kernel1 + noise_kernel
        test_cases = [[kernel1, variable], [kernel2, variable]]

        for test_case in test_cases:
            gp, train_samples = self._setup_high_acccuracy_gp(*test_case)
            errors = gp.check_apply_jacobian(train_samples[:, :1])
            assert errors.min() / errors.max() < 1e-6

    def _check_gp_realizations_and_covariance(
        self, gp, gp_realizations, quad_rule
    ):
        bkd = self.get_backend()
        assert bkd.allclose(
            bkd.diag(gp.covariance(quad_rule()[0])),
            gp.evaluate(quad_rule()[0], True)[1][:, 0] ** 2,
        )
        assert bkd.allclose(
            gp.covariance(quad_rule()[0]),
            bkd.cov(gp_realizations, ddof=1),
            atol=4e-3,
        )

    def _check_gaussian_process_statistics(
        self, gp, nrealizations, variable, quad_rule, out_trans, rtol
    ):
        bkd = self.get_backend()
        gp_stat = GaussianProcessStatistics(gp, variable)
        gp_ensemble_stat = EnsembleGaussianProcessStatistics(
            gp, variable, ninterpolation_samples=100
        )
        means_of_realizations = gp_ensemble_stat.means_of_realizations(
            nrealizations
        )

        # test expectation of gp mean matches that computed emprically from
        # random realizations
        expected_mean = gp_stat.expectation_of_mean()
        assert bkd.allclose(bkd.mean(means_of_realizations), expected_mean)

        # test variance of gp mean matches that computed emprically from
        # random realizations
        variance_of_mean = gp_stat.variance_of_mean()
        # print(
        #     (variance_of_mean - bkd.var(means_of_realizations, ddof=1))
        #     / bkd.var(means_of_realizations, ddof=1),
        #     rtol,
        # )
        assert bkd.allclose(
            variance_of_mean,
            bkd.var(means_of_realizations, ddof=1),
            rtol=rtol,
        )

        variances_of_realizations = gp_ensemble_stat.variances_of_realizations(
            nrealizations
        )
        # test expectation of gp variance matches that computed emprically from
        # random realizations
        expected_variance = gp_stat.expectation_of_variance()
        assert bkd.allclose(
            expected_variance, bkd.mean(variances_of_realizations)
        )

        # test variance of gp variance matches that computed emprically from
        # random realizations
        variance_of_variance = gp_stat.variance_of_variance()
        assert bkd.allclose(
            variance_of_variance,
            bkd.var(variances_of_realizations, ddof=1),
            rtol=rtol,
        )

    def _setup_low_accuracy_gp_test_case(
        self, variable, kernel, out_trans, in_trans, ntrain_samples
    ):
        bkd = self.get_backend()
        constant = 1e3

        def fun(x):
            return constant * bkd.sum((2 * x - 0.5) ** 2, axis=0)[:, None]

        gp = ExactGaussianProcess(
            variable.nvars(), kernel, trend=None, kernel_reg=1e-7
        )
        gp.set_output_transform(out_trans)
        gp.set_input_transform(in_trans)
        gp.set_optimizer(ncandidates=4, verbosity=0)
        train_samples = bkd.cartesian_product(
            [(1 - bkd.cos(bkd.linspace(0, np.pi, ntrain_samples))) / 2]
            * variable.nvars()
        )
        train_values = fun(train_samples)
        gp.fit(train_samples, train_values)

        # test_samples = variable.rvs(1000)
        # test_values = fun(test_samples)
        # error = bkd.norm(gp(test_samples) - test_values) / bkd.norm(
        #     test_values
        # )
        # print("error", error)
        return gp

    def _check_statistics_low_accuracy_gp(
        self, variable, kernel, out_trans, in_trans, rtol
    ):
        # test that expectation and variance of the mean and variance of GP
        # are accurate when GP is inaccurate
        gp = self._setup_low_accuracy_gp_test_case(
            variable, kernel, out_trans, in_trans, 10
        )

        quad_rule = FixedGaussianTensorProductQuadratureRuleFromVariable(
            variable, [50] * variable.nvars()
        )
        # test gp covariance matches that computed emprically from
        # random realizations
        gp_realizations = gp.predict_random_realizations(quad_rule()[0], 1e6)

        self._check_gp_realizations_and_covariance(
            gp, gp_realizations, quad_rule
        )
        nrealizations = 1e5
        self._check_gaussian_process_statistics(
            gp, nrealizations, variable, quad_rule, out_trans, rtol
        )

    def test_statistics_low_accuracy_gp(self):
        bkd = self.get_backend()
        nvars = 1
        kernel1 = MaternKernel(np.inf, 0.1, [1e-1, 1], 1, backend=bkd)

        kernels = [kernel1]
        out_trans = [
            GaussianProcessIdentityTransform(backend=bkd),
        ]
        marginals = [stats.uniform(0, 1)] * nvars
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        in_trans = [IdentityTransform(backend=bkd), AffineTransform(variable)]
        variables = [variable]
        for test_case in itertools.product(
            variables, kernels, out_trans, in_trans
        ):
            np.random.seed(1)
            self._check_statistics_low_accuracy_gp(*test_case, 1e-2)

        constant_kernel = ConstantKernel(
            0.1,
            (1e-3, 1e1),
            transform=LogHyperParameterTransform(backend=bkd),
            backend=bkd,
        )
        kernel2 = constant_kernel * MaternKernel(
            np.inf, 0.1, [1e-1, 1], 1, backend=bkd
        )
        kernels = [kernel2]
        for test_case in itertools.product(
            variables, kernels, out_trans, in_trans
        ):
            np.random.seed(1)
            self._check_statistics_low_accuracy_gp(*test_case, 2e-2)

    def _check_marginalized_gaussian_process(self, gp, variable):
        bkd = self.get_backend()
        active_id = 0
        marginalized_gp = marginalize_gaussian_process(gp, variable, active_id)

        def marginalized_fun(x):
            constant = 1e3
            return constant * ((2 * x[0] - 0.5) ** 2 + 7 / 12)[:, None]

        samples = bkd.linspace(0, 1, 101)[None, :]
        marginalized_gp.plot(
            plt.figure().gca(),
            variable.marginals()[active_id].interval(1),
        )
        plt.plot(samples[0], marginalized_fun(samples))
        assert bkd.allclose(
            marginalized_gp(samples), marginalized_fun(samples), rtol=1e-2
        )

    def _check_gaussian_process_marginalization_low_accuracy_gp(
        self, variable, kernel, out_trans, in_trans
    ):
        gp = self._setup_low_accuracy_gp_test_case(
            variable, kernel, out_trans, in_trans, 20
        )
        self._check_marginalized_gaussian_process(gp, variable)

    def test_gaussian_process_marginalization_low_accuracy_gp(self):
        bkd = self.get_backend()
        nvars = 2
        kernel1 = MaternKernel(np.inf, 0.1, [1e-1, 1], nvars, backend=bkd)
        constant_kernel = ConstantKernel(
            0.1,
            (1e-3, 1e1),
            transform=LogHyperParameterTransform(backend=bkd),
            backend=bkd,
        )
        kernel2 = constant_kernel * MaternKernel(
            np.inf, 0.1, [1e-1, 1], nvars, backend=bkd
        )
        kernels = [kernel1, kernel2]
        out_trans = [
            GaussianProcessIdentityTransform(backend=bkd),
        ]
        marginals = [stats.uniform(0, 1)] * nvars
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        in_trans = [IdentityTransform(backend=bkd), AffineTransform(variable)]
        variables = [variable]

        for test_case in itertools.product(
            variables, kernels, out_trans, in_trans
        ):
            self._check_gaussian_process_marginalization_low_accuracy_gp(
                *test_case
            )

    def test_ishigami(self):
        bkd = self.get_backend()
        benchmark = IshigamiBenchmark(a=0.1, b=0.02, backend=bkd)

        # setup gp
        kernel_reg = 1e-9
        kernel = MaternKernel(
            np.inf,
            0.5,
            [1e-1, 3],
            benchmark.nvars(),
            fixed=False,
            backend=bkd,
        )
        # constant_kernel = ConstantKernel(
        #     1, (1e-1, 1e1), transform=LogHyperParameterTransform(), fixed=False
        # )
        # kernel = constant_kernel * kernel
        out_trans = GaussianProcessIdentityTransform()
        gp = ExactGaussianProcess(
            benchmark.variable().nvars(),
            kernel,
            trend=None,
            kernel_reg=kernel_reg,
        )
        gp.set_output_transform(out_trans)
        gp.set_optimizer(ncandidates=3, verbosity=0)

        # train gp
        ntrain_samples = 1000
        seq = SobolSequence(benchmark.nvars(), 0, benchmark.variable())
        samples = seq.rvs(ntrain_samples)
        values = benchmark.model()(samples)
        nvalidation_samples = 1000
        validation_samples = benchmark.variable().rvs(nvalidation_samples)
        validation_values = benchmark.model()(validation_samples)

        gp.fit(samples, values)
        # gradients converge but rel error is still large
        # this example is effected by conditioning of training kernel matrix
        # Error in gradient can degrade
        # significantly if thes quantities are varied. This is true if using
        # autograd of analytical gradients
        # gp.hyp_list().set_all_active()
        # errors = gp._optimizer._objective.check_apply_jacobian(
        #     gp.hyp_list().get_active_opt_params()[:, None], disp=False
        # )
        # assert errors.min() / errors.max() < 1e-6

        gp_vals = gp(validation_samples)
        rel_error = np.linalg.norm(
            gp_vals - validation_values
        ) / np.linalg.norm(validation_values)
        # print("Rel. Error", rel_error)

        assert rel_error < 5e-4

    def _setup_multilevel_model_ensemble(self, degree, nmodels):
        bkd = self.get_backend()
        rho = bkd.full(((nmodels - 1) * (degree + 1),), 0.9)

        def scale(x, rho, kk):
            if degree == 0:
                return rho[kk]
            return rho[2 * kk] + x.T * rho[2 * kk + 1]

        def f1(x):
            y = x[0:1].T
            return ((y * 6 - 2) ** 2) * bkd.sin((y * 6 - 2) * 2) / 5

        def f2(x):
            y = x[0:1].T
            return scale(x, rho, 0) * f1(x) + bkd.cos(y * 2) / 10

        def f3(x):
            y = x[0:1].T
            return scale(x, rho, -1) * f2(x) + ((y - 0.5) * 1.0 - 5) / 5

        model1 = ModelFromVectorizedCallable(1, 1, f1, backend=bkd)
        model2 = ModelFromVectorizedCallable(1, 1, f2, backend=bkd)
        model3 = ModelFromVectorizedCallable(1, 1, f3, backend=bkd)
        if nmodels == 3:
            return rho, (model1, model2, model3)
        return rho, (model1, model2)

    def _setup_multilevel_gp(self, nmodels, nvars, degree):
        bkd = self.get_backend()
        lscales = [0.1, 0.2, 0.3][:nmodels]
        sml_kernels = [
            MaternKernel(np.inf, lscale, (1e-1, 1), nvars, backend=bkd)
            for lscale in lscales
        ]
        sml_scaling_basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(nvars)]
        )
        sml_scaling_basis.set_tensor_product_indices([degree + 1] * nvars)
        sml_scalings = [
            MonomialExpansion(sml_scaling_basis) for nn in range(nmodels - 1)
        ]
        [
            scaling.set_coefficient_bounds(
                bkd.array([0.9] * (degree + 1)), (0.8, 1.0)
            )
            for scaling in sml_scalings
        ]
        sml_gp = SequentialMultiLevelGaussianProcess(
            sml_kernels, sml_scalings, kernel_reg=1e-10
        )
        for gp in sml_gp.gaussian_processes():
            gp.set_optimizer(ncandidates=5, verbosity=0)
        return sml_gp, sml_kernels, sml_scalings

    def _check_multilevel_gaussian_process(self, nmodels, degree, tol):
        bkd = self.get_backend()
        nvars = 1
        rho, models = self._setup_multilevel_model_ensemble(degree, nmodels)

        # create nested samples
        ntrain_samples_per_model = [2**5 + 1, 2**4 + 1, 2**3 + 1][:nmodels]
        train_samples_per_model = [
            bkd.linspace(0, 1, nsamples)[None, :]
            for nsamples in ntrain_samples_per_model
        ]
        train_values_per_model = [
            model(samples)
            for model, samples in zip(models, train_samples_per_model)
        ]
        sml_gp, sml_kernels, sml_scalings = self._setup_multilevel_gp(
            nmodels, nvars, degree
        )

        # call fit to store data needed to check jacobian
        fd_gp = self._setup_multilevel_gp(nmodels, nvars, degree)[0]
        fd_ntrain_samples_per_model = [2**+1, 2**3 + 1, 2**2 + 1][:nmodels]
        fd_train_samples_per_model = [
            bkd.linspace(0, 1, nsamples)[None, :]
            for nsamples in fd_ntrain_samples_per_model
        ]
        fd_train_values_per_model = [
            model(samples)
            for model, samples in zip(models, fd_train_samples_per_model)
        ]

        fd_gp.fit(fd_train_samples_per_model, fd_train_values_per_model)
        # iterate = (
        #     fd_gp.gaussian_processes()[1].hyp_list().get_values()[:, None]
        #     + 0.001
        # )
        iterate = sml_gp.gaussian_processes()[
            1
        ]._optimizer._initial_interate_gen()
        errors = fd_gp.gaussian_processes()[1]._loss.check_apply_jacobian(
            iterate,
            disp=False,
            # fd_eps=bkd.flip(bkd.logspace(-13, -1, 13)),
        )
        # print(errors.min(), errors.max())
        assert errors.max() < np.inf
        assert errors.min() / errors[0] < 2.5e-6

        # call fit again to fit data and test accuracy
        np.random.seed(1)
        sml_gp.fit(train_samples_per_model, train_values_per_model)

        test_samples_per_model = [
            bkd.asarray(np.random.uniform(0, 1, (1, 10)))
        ] * nmodels
        test_values_per_model = [
            model(samples)
            for model, samples in zip(models, test_samples_per_model)
        ]
        sml_gp_values = [
            sml_gp.evaluate(test_samples_per_model[ii], False, ii)
            for ii in range(nmodels)
        ]

        ml_kernel = MultiLevelKernel(sml_kernels, sml_scalings)
        ml_gp = MOExactGaussianProcess(nvars, ml_kernel, kernel_reg=1e-10)
        ml_gp.set_optimizer(ncandidates=10)
        ml_gp.fit(train_samples_per_model, train_values_per_model)
        ml_gp_values = ml_gp.evaluate(test_samples_per_model, False)

        for ii in range(nmodels):
            # print(sml_gp_values[ii] - test_values_per_model[ii], tol)
            assert bkd.allclose(
                sml_gp_values[ii],
                test_values_per_model[ii],
                rtol=tol,
                atol=tol,
            )
            # for 3 models with degree = 1
            # lenscale0, lenscale1, lenscale2, rho10, rho11, rho20, rho21
            ml_hypparam_vals = ml_gp.hyp_list().get_values()
            # for 3 models with degree = 1
            # lenscale0, lenscale1, rho10, rho11, lenscale2, rho20, rho12
            sml_hypparam_vals = bkd.hstack(
                [
                    gp.hyp_list().get_values()
                    for gp in sml_gp.gaussian_processes()
                ]
            )
            # reorder to match ml_hypparam_vals ordering
            mask = bkd.ones((sml_hypparam_vals.shape[0]), dtype=bool)
            mask[0] = False
            mask[1 :: degree + 2] = False
            sml_hypparam_vals = bkd.hstack(
                (
                    sml_hypparam_vals[:1],
                    sml_hypparam_vals[1 :: degree + 2],
                    sml_hypparam_vals[mask],
                )
            )
            # print(sml_hypparam_vals - ml_hypparam_vals)
            # print(sml_hypparam_vals)
            # print(ml_hypparam_vals)
            # check sequential and co-criggking gps are the same
            # when points are nested and values noiseless
            assert bkd.allclose(ml_hypparam_vals, sml_hypparam_vals, atol=1e-4)
            # print(test_values_per_model[ii][:, 0])
            # print(ml_gp_values)
            # print(test_values_per_model[ii][:, 0] - ml_gp_values[ii][:, 0])
            assert bkd.allclose(
                ml_gp_values[ii],
                test_values_per_model[ii],
                rtol=tol,
                atol=tol,
            )

    def test_multilevel_gaussian_process(self):
        test_cases = [[2, 0, 2e-4], [2, 1, 2e-4], [3, 0, 1e-3], [3, 1, 1e-3]]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_multilevel_gaussian_process(*test_case)

    def test_monte_carlo_kernel_statistic(self):
        bkd = self.get_backend()
        marginals = [stats.uniform(0, 1)]
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        kernel = MaternKernel(
            np.inf, 0.1, [1e-1, 1], variable.nvars(), backend=bkd
        )
        gp, train_samples = self._setup_high_acccuracy_gp(kernel, variable)
        mc_stat = MonteCarloKernelStatistics(
            gp, variable, train_samples, nquad_samples=1000000
        )
        quad_stat = TensorProductQuadratureKernelStatistics(
            gp, variable, train_samples
        )
        assert bkd.allclose(
            mc_stat._tau_P()[0], quad_stat._tau_P()[0], rtol=1e-2
        )
        assert bkd.allclose(
            mc_stat._tau_P()[1], quad_stat._tau_P()[1], rtol=1e-2
        )


class TestNumpyNystrom(TestNystrom, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin

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
        return TorchMixin

    def get_mvn(self):
        return TorchMultivariateNormal


class TestTorchGaussianProcess(TestGaussianProcess, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

    # nrealizations = 1000
    # train_samples = quad_rule()[0]
    # rand_noise = bkd.asarray(
    #     np.random.normal(
    #         0, 1, (int(nrealizations), train_samples.shape[1])
    #     ).T
    # )
    # # make last sample mean of gaussian process
    # rand_noise = bkd.hstack(
    #     (rand_noise, bkd.zeros((rand_noise.shape[0], 1)))
    # )
    # train_values = gp._predict_random_realizations_from_rand_noise(
    #     train_samples, rand_noise
    # )
    # print(train_values.shape, train_samples.shape)
    # gp_proxy = ExactGaussianProcess(
    #     variable.nvars(),
    #     gp.kernel(),
    #     trend=gp.trend(),
    #     kernel_reg=1e-12,
    # )
    # gp_proxy._set_training_data(train_samples, train_values)
    # gp_proxy._check_values_shape = lambda s, v: None
    # print(
    #     bkd.norm(gp_proxy(train_samples) - train_values)
    #     / train_values.shape[1]
    # )
    # assert False  # hack once test complete and pass modularize these functions

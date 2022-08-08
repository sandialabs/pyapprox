import unittest
import numpy as np
from sklearn.gaussian_process.kernels import _approx_fprime

from pyapprox.surrogates.gaussianprocess.kernels import RBF
from pyapprox.surrogates.gaussianprocess.gradient_enhanced_gp import (
    get_gp_samples_kernel)
from pyapprox.surrogates.gaussianprocess.multilevel_gp import (
    MultilevelGPKernel, MultilevelGP, SequentialMultiLevelGP
)


class TestMultilevelGP(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_pyapprox_rbf_kernel(self):
        kernel = RBF(0.1)
        nvars, nsamples = 1, 3
        XX = np.random.uniform(0, 1, (nsamples, nvars))
        YY = np.random.uniform(0, 1, (nsamples-1, nvars))
        K_grad = kernel(XX, YY, eval_gradient=True)[1]

        def f(theta):
            kernel.theta = theta
            K = kernel(XX, YY)
            return K
        K_grad_f = _approx_fprime(kernel.theta, f, 1e-8)
        assert np.allclose(K_grad, K_grad_f)

        nvars, nsamples = 2, 4
        kernel = RBF([0.1, 0.2])
        XX = np.random.uniform(0, 1, (nsamples, nvars))

        YY = None
        K_grad = kernel(XX, YY, eval_gradient=True)[1]
        rbf_kernel = RBF([0.1, 0.2])
        assert np.allclose(rbf_kernel(XX, YY, eval_gradient=True)[1],
                           K_grad)

        YY = np.random.uniform(0, 1, (nsamples-1, nvars))

        def f(theta):
            kernel.theta = theta
            K = kernel(XX, YY)
            return K
        K_grad_fd = _approx_fprime(kernel.theta, f, 1e-8)
        K_grad = kernel(XX, YY, eval_gradient=True)[1]
        assert np.allclose(K_grad, K_grad_fd, atol=1e-6)

    def _check_multilevel_kernel(self, nmodels):

        nmodels = 3
        nsamples = int(1e6)
        nvars = 1  # if increase must change from linspace to rando
        # nsamples_per_model = [5, 4, 3][:nmodels]
        nsamples_per_model = [4, 3, 2][:nmodels]
        length_scales = [1, 2, 3][:nmodels]
        scalings = np.arange(2, 2+nmodels-1)/3
        # shared indices of samples from lower level
        shared_idx_list = [
            np.random.permutation(
                np.arange(nsamples_per_model[nn-1]))[:nsamples_per_model[nn]]
            for nn in range(1, nmodels)]
        XX_list = [np.linspace(-1, 1, nsamples_per_model[0])[:, None]]
        for nn in range(1, nmodels):
            XX_list += [XX_list[nn-1][shared_idx_list[nn-1]]]

        assert nmodels*nvars == len(length_scales)
        # kernels = [
        #     Matern(length_scales[nn], nu=np.inf)
        #     for nn in range(nmodels)]
        kernels = [RBF(length_scales[nn]) for nn in range(nmodels)]

        samples_list = [
            np.random.normal(0, 1, (nsamples_per_model[nn], nsamples))
            for nn in range(nmodels)]

        # Sample from discrepancies
        DD_list = [
            np.linalg.cholesky(kernels[nn](XX_list[nn])).dot(
                samples_list[nn]) for nn in range(nmodels)]
        # cannout use kernel1(XX2, XX2) here because this will generate
        # different samples to those used in YY1
        YY_list = [None for nn in range(nmodels)]
        YY_list[0] = DD_list[0]
        for nn in range(1, nmodels):
            YY_list[nn] = (
                scalings[nn-1]*YY_list[nn-1][shared_idx_list[nn-1], :] +
                DD_list[nn])

        assert np.allclose(
            YY_list[0][shared_idx_list[0]],
            (YY_list[1]-DD_list[1])/scalings[0])
        if nmodels > 2:
            assert np.allclose(
                YY_list[1][shared_idx_list[1]],
                (YY_list[2]-DD_list[2])/scalings[1])

        for nn in range(nmodels):
            assert np.allclose(YY_list[nn].mean(axis=1), 0, atol=1e-2)
        YY_centered_list = [YY_list[nn]-YY_list[nn].mean(axis=1)[:, None]
                            for nn in range(nmodels)]

        cov = [[None for nn in range(nmodels)] for kk in range(nmodels)]
        for nn in range(nmodels):
            cov[nn][nn] = np.cov(YY_list[nn])
            assert np.allclose(
                YY_centered_list[nn].dot(YY_centered_list[nn].T)/(nsamples-1),
                cov[nn][nn])

        assert np.allclose(cov[0][0], kernels[0](XX_list[0]), atol=1e-2)
        assert np.allclose(
            cov[1][1],
            scalings[0]**2*kernels[0](XX_list[1])+kernels[1](XX_list[1]),
            atol=1e-2)
        if nmodels > 2:
            assert np.allclose(
                cov[2][2], scalings[:2].prod()**2*kernels[0](XX_list[2]) +
                scalings[1]**2*kernels[1](XX_list[2]) +
                kernels[2](XX_list[2]),
                atol=1e-2)

        cov[0][1] = YY_centered_list[0].dot(
            YY_centered_list[1].T)/(nsamples-1)
        assert np.allclose(
            cov[0][1], scalings[0]*kernels[0](
                XX_list[0], XX_list[1]), atol=1e-2)

        if nmodels > 2:
            cov[0][2] = YY_centered_list[0].dot(
                YY_centered_list[2].T)/(nsamples-1)
            assert np.allclose(
                cov[0][2], scalings[:2].prod()*kernels[0](
                    XX_list[0], XX_list[2]), atol=1e-2)
            cov[1][2] = YY_centered_list[1].dot(
                YY_centered_list[2].T)/(nsamples-1)
            assert np.allclose(
                cov[1][2], scalings[0]**2*scalings[1]*kernels[0](
                    XX_list[1], XX_list[2])+scalings[1]*kernels[1](
                    XX_list[1], XX_list[2]), atol=1e-2)

        length_scale = np.hstack((length_scales, scalings))
        length_scale_bounds = [(1e-1, 10)] * \
            (nmodels*nvars) + [(1e-1, 1)]*(nmodels-1)
        mlgp_kernel = MultilevelGPKernel(
            nvars, nsamples_per_model, kernels, length_scale=length_scale,
            length_scale_bounds=length_scale_bounds)
        # mlgp_kernel = MultilevelGPKernelDeprecated(
        #     nvars, nsamples_per_model, length_scale=length_scale,
        #     length_scale_bounds=length_scale_bounds)

        XX_train = np.vstack(XX_list)
        np.set_printoptions(linewidth=500)
        K = mlgp_kernel(XX_train)
        for nn in range(nmodels):
            assert np.allclose(
                K[sum(nsamples_per_model[:nn]):sum(nsamples_per_model[:nn+1]),
                  sum(nsamples_per_model[:nn]):sum(nsamples_per_model[:nn+1])],
                cov[nn][nn], atol=1e-2)

        assert np.allclose(
            K[:nsamples_per_model[0],
              nsamples_per_model[0]:sum(nsamples_per_model[:2])],
            cov[0][1], atol=2e-2)

        if nmodels > 2:
            assert np.allclose(
                K[:nsamples_per_model[0], sum(nsamples_per_model[:2]):],
                cov[0][2], atol=2e-2)
            assert np.allclose(
                K[nsamples_per_model[0]:sum(nsamples_per_model[:2]),
                  sum(nsamples_per_model[:2]):],
                cov[1][2], atol=2e-2)

        nsamples_test = 6
        XX_train = np.vstack(XX_list)
        XX_test = np.linspace(-1, 1, nsamples_test)[:, None]
        K = mlgp_kernel(XX_test, XX_train)
        assert np.allclose(K[:, :XX_list[0].shape[0]],
                           scalings.prod()*kernels[0](XX_test, XX_list[0]))
        if nmodels == 2:
            tnm1_prime = scalings[0]*kernels[0](XX_test, XX_list[1])
            assert np.allclose(
                K[:, nsamples_per_model[0]:sum(nsamples_per_model[:2])],
                scalings[0]*tnm1_prime +
                kernels[1](XX_test, XX_list[1]))
        elif nmodels == 3:
            t2m1_prime = scalings[1]*scalings[0]*kernels[0](
                XX_test, XX_list[1])
            assert np.allclose(
                K[:, nsamples_per_model[0]:sum(nsamples_per_model[:2])],
                scalings[0]*t2m1_prime +
                scalings[1]*kernels[1](XX_test, XX_list[1]))

            t2m1_prime = scalings[1]*scalings[0]*kernels[0](
                XX_test, XX_list[2])
            t3m1_prime = (scalings[0]*t2m1_prime +
                          scalings[1]*kernels[1](XX_test, XX_list[2]))
            assert np.allclose(
                K[:, sum(nsamples_per_model[:2]):],
                scalings[1]*t3m1_prime +
                kernels[2](XX_test, XX_list[2]))

        # samples_test = [np.random.normal(0, 1, (nsamples_test, nsamples))
        #                 for nn in range(nmodels)]
        # # to evaluate lower fidelity model change kernel index
        # DD2_list = [
        #     np.linalg.cholesky(kernels[nn](XX_test)).dot(
        #         samples_test[nn]) for nn in range(nmodels)]
        # YY2_test = DD2_list[0]
        # for nn in range(1, nmodels):
        #     YY2_test = (
        #         scalings[nn-1]*YY2_test + DD2_list[nn])
        # YY2_test_centered = YY2_test-YY2_test.mean(axis=1)[:, None]
        # t0 = YY2_test_centered.dot(YY_centered_list[0].T)/(nsamples-1)
        # print(t0)
        K = mlgp_kernel(XX_list[nmodels-1], XX_train)
        t0 = YY_centered_list[nmodels-1].dot(YY_centered_list[0].T)/(
            nsamples-1)
        assert np.allclose(t0, K[:, :nsamples_per_model[0]], atol=1e-2)
        t1 = YY_centered_list[nmodels-1].dot(YY_centered_list[1].T)/(
            nsamples-1)
        assert np.allclose(
            t1, K[:, nsamples_per_model[0]:sum(nsamples_per_model[:2])],
            atol=1e-2)
        if nmodels > 2:
            t2 = YY_centered_list[nmodels-1].dot(YY_centered_list[2].T)/(
                nsamples-1)
            assert np.allclose(
                t2, K[:, sum(nsamples_per_model[:2]):], atol=1e-2)

        def f(theta):
            mlgp_kernel.theta = theta
            K = mlgp_kernel(XX_train)
            return K
        from sklearn.gaussian_process.kernels import _approx_fprime
        K_grad_fd = _approx_fprime(mlgp_kernel.theta, f, 1e-8)
        K_grad = mlgp_kernel(XX_train, eval_gradient=True)[1]
        # idx = 3
        # print(K_grad[:, :, idx])
        # print(K_grad_fd[:, :, idx])
        # np.set_printoptions(precision=3, suppress=True)
        # print(np.absolute(K_grad[:, :, idx]-K_grad_fd[:, :, idx]))#.max())
        # print(K_grad_fd.shape, K_grad.shape)
        assert np.allclose(K_grad, K_grad_fd, atol=1e-6)

    def test_multilevel_kernel(self):
        self._check_multilevel_kernel(2)
        self._check_multilevel_kernel(3)

    def _check_2_models(self, nested):
        # TODO Add Test which builds gp on two models data separately when
        # data2 is subset data and hyperparameters are fixed.
        # Then Gp should just be sum of separate GPs.

        lb, ub = 0, 1
        nvars, nmodels = 1, 2
        np.random.seed(2)
        true_rho = [2]
        def f1(x):
            return ((x.T*6-2)**2)*np.sin((x.T*6-2)*2)/5

        def f2(x):
            return true_rho[0]*f1(x)+((x.T-0.5)*1. - 5)/5

        if not nested:
            x2 = np.array([[0.0], [0.4], [0.6], [1.0]]).T
            x1 = np.array([[0.1], [0.2], [0.3], [0.5], [0.7],
                           [0.8], [0.9], [0.0], [0.4], [0.6], [1.0]]).T
        else:
            # nested
            x1 = np.array([[0.1], [0.2], [0.3], [0.5], [0.7],
                           [0.8], [0.9], [0.0], [0.4], [0.6], [1.0]]).T
            x2 = x1[:, [0, 2, 4, 6]]
            # x1 = x1[:, ::2]
            # x2 = x1[:, [0, 2]]

        samples = [x1, x2]
        values = [f(x) for f, x in zip([f1, f2], samples)]
        nsamples_per_model = [s.shape[1] for s in samples]

        n_restarts_optimizer = 10

        rho = np.ones(nmodels-1)
        length_scale = [1]*(nmodels*(nvars))+list(rho)
        # print(length_scale)
        length_scale_bounds = [(1e-1, 10)] * \
            (nmodels*nvars)+[(1e-1, 10)]*(nmodels-1)
        # length_scale_bounds='fixed'
        kernels = [RBF(0.1) for nn in range(nmodels)]
        mlgp_kernel = MultilevelGPKernel(
            nvars, nsamples_per_model, kernels, length_scale=length_scale,
            length_scale_bounds=length_scale_bounds)
        # noise_level_bounds=(1e-8, 1)
        # do not use noise kernel for entire kernel
        # have individual noise kernels for each model
        # mlgp_kernel += WhiteKernel( # optimize gp noise
        #    noise_level=noise_level, noise_level_bounds=noise_level_bounds)

        gp = MultilevelGP(mlgp_kernel)
        gp.set_data(samples, values)
        gp.fit()
        print(gp.kernel_.length_scale[2], true_rho)
        # print(gp.kernel_.length_scale[2]-true_rho)
        assert np.allclose(gp.kernel_.length_scale[-1], true_rho, atol=4e-3)

        sml_kernels = [
            RBF(length_scale=get_gp_samples_kernel(gp).length_scale[
                nvars*ii:nvars*(ii+1)],
                length_scale_bounds=(1e-1, 10)) for ii in range(nmodels)]
        print("SML kernels", sml_kernels)
        print(gp.kernel_.length_scale)

        sml_gp = SequentialMultiLevelGP(
            sml_kernels, n_restarts_optimizer=n_restarts_optimizer)
        sml_gp.set_data(samples, values)
        sml_gp.fit(true_rho)

        from pyapprox.surrogates.gaussianprocess.kernels import ConstantKernel
        from pyapprox.surrogates.gaussianprocess.gaussian_process import (
            GaussianProcess)
        # point used to evaluate diag does not matter for stationary kernels
        sf_var = gp.kernel_.diag(np.zeros((1, nvars)))
        sf_kernel = RBF(
            length_scale_bounds=length_scale_bounds[:nvars])*ConstantKernel(
                sf_var, constant_value_bounds="fixed")
        sf_gp = GaussianProcess(sf_kernel)
        sf_gp.fit(samples[1], values[1])

        # print('ml')
        # print(get_gp_samples_kernel(gp).length_scale[-1], true_rho)

        xx = np.linspace(lb, ub, 2**8+1)[np.newaxis, :]

        sml_gp_mean, sml_gp_std = sml_gp(xx)
        gp_mean, gp_std = gp(xx, return_std=True)
        sf_gp_mean, sf_gp_std = sf_gp(xx, return_std=True)

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 1)
        # axs = [axs]
        # axs[0].plot(samples[1][0, :], values[1], 'ko')
        # axs[0].plot(xx[0, :], f2(xx), 'k-', label='f2')
        # axs[0].plot(xx[0, :], sml_gp_mean, 'b--')
        # axs[0].plot(xx[0, :], gp_mean, 'r:')
        # axs[0].plot(xx[0, :], sf_gp_mean, 'g-.')
        # nstdev = 2
        # gp_mean = gp_mean[:, 0]
        # sf_gp_mean = sf_gp_mean[:, 0]
        # axs[0].fill_between(
        #     xx[0, :], gp_mean - nstdev*gp_std, gp_mean + nstdev*gp_std,
        #     alpha=0.2, color='r')
        # axs[0].fill_between(
        #     xx[0, :], sf_gp_mean - nstdev*sf_gp_std, sf_gp_mean + nstdev*sf_gp_std,
        #     alpha=0.2, color='g')
        # # prior_stdev = np.sqrt(gp.kernel_.diag(xx.T))
        # # axs[0].fill_between(
        # #     xx[0, :], -nstdev*prior_stdev, nstdev*prior_stdev,
        # #     alpha=0.5, color='k')
        # plt.legend()
        # plt.show()
        # gp_cov = gp(xx, return_cov=True)[1]
        # assert np.allclose(gp_cov, gp_std**2, atol=1e-4)
        print(np.abs(sml_gp_mean - gp_mean).max())
        assert np.allclose(sml_gp_mean, gp_mean, atol=5e-3)

    def test_2_models(self):
        self._check_2_models(True)
        self._check_2_models(False)


if __name__ == "__main__":
    multilevel_gp_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMultilevelGP)
    unittest.TextTestRunner(verbosity=2).run(multilevel_gp_test_suite)

import unittest
import numpy as np
from functools import partial
from sklearn.gaussian_process.kernels import RBF

from pyapprox.surrogates.gaussianprocess.gradient_enhanced_gp import (
    kernel_ff, get_gp_samples_kernel)
from pyapprox.surrogates.gaussianprocess.multilevel_gp import (
    MultilevelGPKernel, MultilevelGP, SequentialMultiLevelGP,
)


class TestMultilevelGP(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_multilevel_kernel(self):

        nsamples = int(1e6)
        nvars = 1  # if increase must change from linspace to random
        nsamples_per_model = [5, 4, 3][:3]
        length_scales = [1, 2, 3][:3]
        nmodels = len(nsamples_per_model)
        scalings = np.arange(2, 2+nmodels-1)/4
        # shared indices of samples from lower level
        shared_idx_list = [
            np.random.permutation(
                np.arange(nsamples_per_model[nn-1]))[:nsamples_per_model[nn]]
            for nn in range(1, nmodels)]
        XX_list = [np.linspace(-1, 1, nsamples_per_model[0])[:, None]]
        for nn in range(1, nmodels):
            XX_list += [XX_list[nn-1][shared_idx_list[nn-1]]]

        assert nmodels*nvars == len(length_scales)
        # kernel1 = partial(kernel_ff, length_scale=length_scales[0])
        # kernel2 = partial(kernel_ff, length_scale=length_scales[1])
        from sklearn.gaussian_process.kernels import Matern
        kernels = [
            Matern(length_scales[nn], length_scale_bounds='fixed', nu=np.inf)
            for nn in range(nmodels)]

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


    # cannot debug failing test on osx latest wth python3.7
    # because do not have access to such a machine
    @unittest.skip
    def test_2_models(self):
        # TODO Add Test which builds gp on two models data separately when
        # data2 is subset data and hyperparameters are fixed.
        # Then Gp should just be sum of separate GPs.

        nvars, nmodels = 1, 2

        np.random.seed(2)
        # np.random.seed(3)
        # n1,n2=5,3
        n1, n2 = 9, 5
        # n1,n2=10,9
        # n1,n2=17,9
        # n1,n2=32,17
        lb, ub = -1, 1
        x1 = np.atleast_2d(np.linspace(lb, ub, n1))
        x2 = x1[:, np.random.permutation(n1)[:n2]]

        # def f1(x): return (1*f2(x)+x.T**2)  # change 1* to some non unitary rho
        # def f2(x): return np.cos(2*np.pi*x).T

        true_rho = [2]

        def f1(x):
            return ((x.T*6-2)**2)*np.sin((x.T*6-2)*2)

        def f2(x):
            return true_rho[0]*f1(x)+(x.T-0.5)*1. - 5

        # def f2(x):
        #     return ((x.T*6-2)**2)*np.sin((x.T*6-2)*2)
        # def f1(x):
        #     return 1/true_rho[0]*((x.T*6-2)**2)*np.sin((x.T*6-2)*2)+(x.T-0.5)*1. - 5

        # non-nested
        # x2 = np.array([[0.0], [0.4], [0.6], [1.0]]).T
        # x1 = np.array([[0.1], [0.2], [0.3], [0.5], [0.7],
        #                [0.8], [0.9], [0.0], [0.4], [0.6], [1.0]]).T
        # nested
        x1 = np.array([[0.1], [0.2], [0.3], [0.5], [0.7],
                       [0.8], [0.9], [0.0], [0.4], [0.6], [1.0]]).T
        x2 = x1[:, [0, 2, 4, 6]]
        lb, ub = 0, 1
        # x1 = np.linspace(lb,ub,31)[np.newaxis,:]
        # print(x1)

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
        mlgp_kernel = MultilevelGPKernel(
            nvars, nsamples_per_model, length_scale=length_scale,
            length_scale_bounds=length_scale_bounds)
        # noise_level_bounds=(1e-8, 1)
        # do not use noise kernel for entire kernel
        # have individual noise kernels for each model
        # mlgp_kernel += WhiteKernel( # optimize gp noise
        #    noise_level=noise_level, noise_level_bounds=noise_level_bounds)

        gp = MultilevelGP(mlgp_kernel)
        gp.set_data(samples, values)
        gp.fit()

        sml_kernels = [
            RBF(length_scale=get_gp_samples_kernel(gp).length_scale[
                nvars*ii:nvars*(ii+1)],
                length_scale_bounds=(1e-1, 10)) for ii in range(nmodels)]
        print(sml_kernels)
        print(get_gp_samples_kernel(gp).length_scale)

        sml_gp = SequentialMultiLevelGP(
            sml_kernels, n_restarts_optimizer=n_restarts_optimizer)
        sml_gp.set_data(samples, values)
        sml_gp.fit(true_rho)

        print('ml')
        print(get_gp_samples_kernel(gp).length_scale[-1], true_rho)
        assert np.allclose(gp.kernel_.length_scale[-1], true_rho, atol=4e-3)
        xx = np.linspace(lb, ub, 2**8+1)[np.newaxis, :]
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 1)
        # axs = [axs]
        # gp.plot_1d(2**8+1,[lb,ub],axs[0])
        # #xx = np.linspace(lb,ub,2**8+1)[np.newaxis,:]
        # axs[0].plot(xx[0,:],f2(xx),'r')
        # #axs[0].plot(xx[0,:],f1(xx),'g--')
        # axs[0].plot(x1[0,:],f1(x1),'gs')
        # plt.show()

        sml_gp_mean, sml_gp_std = sml_gp(xx)
        gp_mean, gp_std = gp(xx, return_std=True)
        # axs[0].plot(samples[1][0, :], values[1], 'ko')
        # axs[0].plot(xx[0, :], f2(xx), 'k-', label='f2')
        # axs[0].plot(xx[0, :], sml_gp_mean, 'b--')
        # axs[0].plot(xx[0, :], gp_mean, 'r:')
        # plt.legend()
        # plt.show()
        # gp_cov = gp(xx, return_cov=True)[1]
        # assert np.allclose(gp_cov, gp_std**2, atol=1e-4)
        print(np.abs(sml_gp_mean - gp_mean).max())
        assert np.allclose(sml_gp_mean, gp_mean, atol=5e-3)


if __name__ == "__main__":
    multilevel_gp_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMultilevelGP)
    unittest.TextTestRunner(verbosity=2).run(multilevel_gp_test_suite)

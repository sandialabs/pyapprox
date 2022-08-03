import unittest
import numpy as np
from functools import partial

from sklearn.gaussian_process.kernels import RBF

from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    GaussianProcess
)
from pyapprox.surrogates.gaussianprocess.gradient_enhanced_gp import (
    kernel_ff, get_gp_samples_kernel)
from pyapprox.surrogates.gaussianprocess.multilevel_gp import (
    MultilevelGPKernel, MultilevelGP, SequentialMultiLevelGP,
)

class TestMultilevelGP(unittest.TestCase):
    def test_multilevel_kernel(self):

        nsamples = int(1e6)
        XX1 = np.linspace(-1, 1, 5)[:, np.newaxis]
        shared_idx = [0, 2, 4]
        XX2 = XX1[shared_idx]

        length_scales = [1, 2]
        kernel1 = partial(kernel_ff, length_scale=length_scales[0])
        kernel2 = partial(kernel_ff, length_scale=length_scales[1])

        nvars1 = XX1.shape[0]
        nvars2 = XX2.shape[0]
        samples1 = np.random.normal(0, 1, (nvars1, nsamples))
        samples2 = np.random.normal(0, 1, (nvars2, nsamples))

        # y1 = f1(x1), x1 \subseteq x2
        # y2 = p12*f1(x2)+d2(x2)
        p12 = 2
        YY1 = np.linalg.cholesky(kernel1(XX1, XX1)).dot(samples1)
        # cannout use kernel1(XX2,XX2) here because this will generate different
        # samples to those used in YY1
        dsamples = np.linalg.cholesky(kernel2(XX2, XX2)).dot(samples2)
        YY2 = p12*np.linalg.cholesky(kernel1(XX1, XX1)
                                     ).dot(samples1)[shared_idx, :]+dsamples

        assert np.allclose(YY1[shared_idx], (YY2-dsamples)/p12)

        assert np.allclose(YY1.mean(axis=1), 0, atol=1e-2)
        assert np.allclose(YY2.mean(axis=1), 0, atol=1e-2)

        YY1_centered = YY1-YY1.mean(axis=1)[:, np.newaxis]
        YY2_centered = YY2-YY2.mean(axis=1)[:, np.newaxis]

        cov11 = np.cov(YY1)
        assert np.allclose(
            YY1_centered[shared_idx, :].dot(
                YY1_centered[shared_idx, :].T)/(nsamples-1),
            cov11[np.ix_(shared_idx, shared_idx)])
        assert np.allclose(cov11, kernel1(XX1, XX1), atol=1e-2)
        cov22 = np.cov(YY2)
        assert np.allclose(
            YY2_centered.dot(YY2.T)/(nsamples-1), cov22)
        # print(cov22-(kernel2(XX2,XX2)+p12**2*kernel1(XX2,XX2)))
        assert np.allclose(cov22, (kernel2(XX2, XX2)+p12 **
                           2*kernel1(XX2, XX2)), atol=2e-2)
        print('Ks1', kernel1(XX2, XX2))

        cov12 = YY1_centered[shared_idx, :].dot(YY2_centered.T)/(nsamples-1)
        # print(cov11-kernel1(XX1,XX1))
        # print(cov12-p12*kernel1(XX1[shared_idx,:],XX2))
        assert np.allclose(
            cov12, p12*kernel1(XX1[shared_idx, :], XX2), atol=1e-2)

        nvars, nmodels = 1, 2
        nsamples_per_model = [XX1.shape[0], XX2.shape[0]]
        length_scale = length_scales+[p12]
        print(length_scale)
        length_scale_bounds = [(1e-1, 10)] * \
            (nmodels*nvars) + [(1e-1, 1)]*(nmodels-1)
        mlgp_kernel = MultilevelGPKernel(
            nvars, nsamples_per_model, length_scale=length_scale,
            length_scale_bounds=length_scale_bounds)

        XX_train = np.vstack([XX1, XX2])
        np.set_printoptions(linewidth=500)
        K = mlgp_kernel(XX_train)
        assert np.allclose(K[np.ix_(shared_idx, shared_idx)],
                           kernel1(XX1, XX2)[shared_idx])
        assert np.allclose(K[XX1.shape[0]:, XX1.shape[0]:], cov22, atol=2e-2)
        assert np.allclose(K[shared_idx, XX1.shape[0]:], cov12, atol=2e-2)

        XX_train = np.vstack([XX1, XX2])
        K = mlgp_kernel(XX1, XX_train)
        assert np.allclose(K[:, :XX1.shape[0]], p12*kernel1(XX1, XX1))
        assert np.allclose(
            K[:, XX1.shape[0]:], p12**2*kernel1(XX1, XX2)+kernel2(XX1, XX2))
        print(K)

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
        print(x1)

        samples = [x1, x2]
        values = [f(x) for f, x in zip([f1, f2], samples)]
        nsamples_per_model = [s.shape[1] for s in samples]

        n_restarts_optimizer = 10

        rho = np.ones(nmodels-1)
        length_scale=[1]*(nmodels*(nvars))+list(rho);
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

        sml_gp = SequentialMultiLevelGP(sml_kernels)
        sml_gp.set_data(samples, values)
        sml_gp.fit(true_rho)

        print('ml', )
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

        # print('when n1=17,n2=9 Warning answer seems to be off by np.sqrt(5) on most of the domain. This changes depending on number of ')
        sml_gp_mean, sml_gp_std = sml_gp(xx)
        gp_mean, gp_std = gp(xx, return_std=True)
        # axs[0].plot(samples[1][0, :], values[1], 'ko')
        # axs[0].plot(xx[0, :], f2(xx), 'k-', label='f2')
        # axs[0].plot(xx[0, :], sml_gp_mean, 'b--')
        # axs[0].plot(xx[0, :], gp_mean, 'r:')
        # plt.legend()
        # plt.show()
        gp_cov = gp(xx, return_cov=True)[1]
        # assert np.allclose(gp_cov, gp_std**2, atol=1e-4)
        print(np.abs(sml_gp_mean - gp_mean).max())
        assert np.allclose(sml_gp_mean, gp_mean, atol=5e-3)


if __name__ == "__main__":
    multilevel_gp_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMultilevelGP)
    unittest.TextTestRunner(verbosity=2).run(multilevel_gp_test_suite)
    

import numpy as np
import unittest
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import _approx_fprime
from sklearn.gaussian_process import GaussianProcessRegressor
from pyapprox.surrogates.gaussianprocess.kernels import (
    WhiteKernel, RBF, DerivGPKernel, kernel_ff, kernel_dd, kernel_fd)

from pyapprox.util.utilities import cartesian_product
from pyapprox.surrogates.gaussianprocess.gradient_enhanced_gp import (
    GradientEnhancedGP, plot_gp_1d, predict_gpr_gradient
)


def gradient_enhanced_gp_example(num_vars, plot=False):
    np.set_printoptions(linewidth=200)
    if num_vars == 2:
        const = 2
    elif num_vars == 1:
        const = 4

    def function(xx): return np.cos(const*xx.sum(axis=1))

    def deriv(xx): return np.array(
        [-const*np.sin(const*xx.sum(axis=1)) for ii in range(num_vars)]).T

    noise_std = 0.

    num_XX_train_values_1d = 7
    num_XX_train_values = num_XX_train_values_1d**num_vars
    # num_XX_train_derivs = num_XX_train_values
    XX_candidates = cartesian_product(
        [np.linspace(-1, 1, num_XX_train_values_1d)]*num_vars, 1).T
    # I = np.random.permutation(
    #    np.arange(XX_candidates.shape[0]))[:num_XX_train_values]
    II = np.arange(XX_candidates.shape[0])
    XX_train_values = XX_candidates[II]
    # J = np.random.permutation(
    #    np.arange(XX_candidates.shape[0]))[:num_XX_train_derivs]
    J = np.arange(XX_candidates.shape[0])
    XX_train_derivs = XX_candidates[J]
    # XX_train_values = np.random.uniform(-1,1,(num_XX_train_values,num_vars))
    # XX_train_derivs = np.random.uniform(-1,1,(num_XX_train_derivs,num_vars))

    YY_train_values = function(XX_train_values)
    YY_train_values += np.random.normal(0, noise_std, XX_train_values.shape[0])
    YY_train_derivs = deriv(XX_train_derivs)
    YY_train_derivs += np.random.normal(0, noise_std, YY_train_derivs.shape)

    length_scale = [1]*num_vars
    noise_level = 0.02
    n_restarts_optimizer = 3

    gegp_kernel = DerivGPKernel(
        num_XX_train_values,
        length_scale=length_scale, length_scale_bounds=(1.e-2, 1e2))
    gegp_kernel += WhiteKernel(  # optimize gp noise
        noise_level=noise_level, noise_level_bounds=(1e-8, 1))
    gegp = GradientEnhancedGP(
        kernel=gegp_kernel, n_restarts_optimizer=n_restarts_optimizer,
        alpha=0.0)

    # The state of kernel below will not be edited. Instead interegate
    # gp.kernel_
    # having 1.*kernel will cause constant to be optimized
    # but answer is sensitive to order of 1*kernel and kernel*1
    # for small sample sizes
    gp_kernel = RBF(
        length_scale=length_scale, length_scale_bounds=(1e-1, 1e2))
    gp_kernel += WhiteKernel(  # optimize gp noise
        noise_level=noise_level, noise_level_bounds=(1e-5, 1))
    gp = GaussianProcessRegressor(
        kernel=gp_kernel, n_restarts_optimizer=n_restarts_optimizer,
        alpha=0.0)

    gegp.fit(XX_train_values, YY_train_values,
             XX_train_derivs, YY_train_derivs)
    gp.fit(XX_train_values, YY_train_values)

    # print(gp.kernel_)
    # print(gegp.kernel_)

    XX_test, YY_test = gegp.get_training_derivs_data()
    # print(gegp.predict_derivatives(XX_test)-YY_test)
    assert np.allclose(gegp.predict_derivatives(XX_test), YY_test, atol=1e-4)

    if plot and num_vars == 2:
        fig, axs = plt.subplots(2, 3, figsize=(3*8, 2*6))
        axs = axs.ravel()
        bounds = [-1, 1, -1, 1]
        num_XX_test_1d = 30
        plot_2d(function, num_XX_test_1d, bounds, XX_train_values, ax=axs[0])
        plot_2d(gp.predict, num_XX_test_1d, bounds, XX_train_values, ax=axs[1])
        plot_2d(gegp.predict, num_XX_test_1d,
                bounds, XX_train_values, ax=axs[2])

        def deriv0(XX): return deriv(XX)[:, 0]
        plot_2d(deriv0, num_XX_test_1d, bounds, XX_train_values, ax=axs[3])
        # plot_2d(gp.predict,num_XX_test_1d,bounds,XX_train_values,ax=axs[1])
        plot_2d(gegp.predict_derivatives, num_XX_test_1d, bounds,
                XX_train_values,
                ax=axs[5])

    if plot and num_vars == 1:
        num_stdev = 2
        num_XX_test = 101
        bounds = [-1, 1]
        fig, axs = plt.subplots(1, 4, figsize=(4*8, 6))
        plot_gp_1d(
            axs[0], gp.predict, num_XX_test, bounds, XX_train_values,
            YY_train_values, num_stdev, function,
            gp_label=r'$f_\mathrm{GP}(x)$', function_label=r'$f(x)$')
        gegp.plot_1d(num_XX_test, bounds, axs[1], 2, function,
                     r'$f_\mathrm{GP_{enh}}(x)$', r'$f(x)$')
        gegp.plot_1d(num_XX_test, bounds, axs[2], 2, deriv,
                     r'$f^\prime_\mathrm{GP_{enh}}(x)$', r'$f^\prime(x)$',
                     True)
        def gp_deriv(XX_test, return_cov): return predict_gpr_gradient(
            gp, XX_test, return_cov)
        plot_gp_1d(
            axs[3], gp_deriv,
            num_XX_test, bounds, XX_train_derivs, YY_train_derivs, num_stdev,
            deriv,
            gp_label=r'$f^\prime_\mathrm{GP}(x)$',
            function_label=r'$f^\prime(x)$')
        XX_test = np.linspace(bounds[0], bounds[1], num_XX_test)[:, np.newaxis]
        eps = 1e-8
        vals = gp.predict(XX_test)
        perturbed_vals = gp.predict(XX_test+eps)
        fd_derivs = (perturbed_vals-vals)/eps
        axs[3].plot(XX_test[:, 0], fd_derivs, 'g-',
                    label=r'$f^\prime_\mathrm{GP}(x)$ fd')

        for ii in range(len(axs)):
            axs[ii].set_xlim(-1, 1)
            axs[ii].legend()
        ylim = [min(axs[0].get_ylim()[0], axs[1].get_ylim()[0]),
                max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])]
        axs[0].set_ylim(ylim)
        axs[1].set_ylim(ylim)


class TestGradientEnhancedGP(unittest.TestCase):
    def test_fd_kernel_1d(self):
        num_pts = 4
        length_scale = 1
        Xf = np.linspace(0., 1., num_pts)
        Y = Xf.copy()
        K_ff = kernel_ff(Xf[:, np.newaxis], Xf[:, np.newaxis], length_scale)
        K_fd_fd = np.zeros_like(K_ff)
        for ii in range(num_pts):
            def f(x):
                Xf[ii] = x[0]
                return kernel_ff(Xf[:, np.newaxis], Y[:, np.newaxis],
                                 length_scale)

            length_scale_gradient = _approx_fprime([Xf[ii]], f, 1e-8)
            K_fd_fd += length_scale_gradient.reshape(K_ff.shape)
            # print length_scale_gradient.reshape(K_ff.shape)
        K_fd = kernel_fd(Xf[:, np.newaxis], Xf[:, np.newaxis], length_scale, 0)
        assert np.allclose(K_fd, K_fd_fd)

    def test_ff_kernel_1d(self):
        num_pts = 4
        length_scale = 1
        Xf = np.linspace(0., 1., num_pts)
        K_ff = kernel_ff(Xf[:, np.newaxis], Xf[:, np.newaxis], length_scale)
        K_ff_fd = np.zeros_like(K_ff)
        x = Xf.copy()
        y = Xf.copy()
        eps = 1e-4
        k = kernel_ff(x[:, np.newaxis], y[:, np.newaxis], length_scale)
        for ii in range(num_pts):
            for jj in range(ii+1, num_pts):
                x_eps = x.copy()
                x_eps[ii] += eps
                y_eps = y.copy()
                y_eps[jj] += eps
                k_eps = kernel_ff(
                    x_eps[:, np.newaxis], y_eps[:, np.newaxis], length_scale)
                K_ff_fd[ii, jj] += k_eps[ii, jj]/(4.*eps**2)
                x_eps = x.copy()
                x_eps[ii] -= eps
                y_eps = y.copy()
                y_eps[jj] += eps
                k_eps = kernel_ff(
                    x_eps[:, np.newaxis], y_eps[:, np.newaxis], length_scale)
                K_ff_fd[ii, jj] -= k_eps[ii, jj]/(4.*eps**2)
                x_eps = x.copy()
                x_eps[ii] += eps
                y_eps = y.copy()
                y_eps[jj] -= eps
                k_eps = kernel_ff(
                    x_eps[:, np.newaxis], y_eps[:, np.newaxis], length_scale)
                K_ff_fd[ii, jj] -= k_eps[ii, jj]/(4.*eps**2)
                x_eps = x.copy()
                x_eps[ii] -= eps
                y_eps = y.copy()
                y_eps[jj] -= eps
                k_eps = kernel_ff(
                    x_eps[:, np.newaxis], y_eps[:, np.newaxis], length_scale)
                K_ff_fd[ii, jj] += k_eps[ii, jj]/(4.*eps**2)
                K_ff_fd[jj, ii] = K_ff_fd[ii, jj]
            K_ff_fd[ii, ii] = 1.

        K_dd = kernel_dd(Xf[:, np.newaxis],
                         Xf[:, np.newaxis], length_scale, 0, 0)
        # print('K_dd',K_dd)
        # print('K_ff_fd',K_ff_fd)
        assert np.allclose(K_dd, K_ff_fd)

    def test_gradient_of_gp(self):
        num_pts = 4
        length_scale = 1
        XX_train = np.linspace(0., 1., num_pts)
        K_ff = kernel_ff(
            XX_train[:, np.newaxis], XX_train[:, np.newaxis], length_scale)
        coef = np.ones(num_pts)
        K_ff_inv = np.linalg.inv(K_ff)
        # Y_train = K_ff_inv.dot(coef)

        num_test_pts = 101
        X_test = np.linspace(0., 1., num_test_pts)
        K_fd = kernel_fd(
            X_test[:, np.newaxis], XX_train[:, np.newaxis], length_scale, 0)
        grads = K_fd.dot(K_ff_inv.dot(coef))

        def f(x):
            if np.isscalar(x):
                x = np.asarray([x])
            K_test = kernel_ff(
                x[:, np.newaxis], XX_train[:, np.newaxis], length_scale)
            return K_test.dot(K_ff_inv.dot(coef))

        eps = 1e-4
        fd_grads = np.empty_like(grads)
        for ii in range(num_test_pts):
            x = X_test[ii]
            x_eps = x + eps
            fd_grads[ii] = (f(x_eps)-f(x))/eps
        assert np.allclose(grads, fd_grads, atol=1e-4)

        # fig,axs = plt.subplots(1,2,figsize=(2*8,6))
        # axs[0].plot(X_test,f(X_test),'r-',label=r'f',lw=3)
        # axs[1].plot(X_test,grads,'k-',label=r'analytic',lw=3)
        # axs[1].plot(X_test,fd_grads,'b--',label=r'fd',lw=3)
        # plt.legend()
        # plt.show()

    def test_gradient_of_gp_examples(self):
        gradient_enhanced_gp_example(1)
        # plt.show()
        # gradient_enhanced_gp_example(2)
        # plt.show()


if __name__ == "__main__":
    gradient_enhanced_gp_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestGradientEnhancedGP))
    unittest.TextTestRunner(verbosity=2).run(gradient_enhanced_gp_test_suite)

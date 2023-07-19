from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.surrogates.gaussianprocess.kernels import (
    ConstantKernel, Product
)
from pyapprox.util.visualization import get_meshgrid_function_data
from pyapprox.surrogates.gaussianprocess.kernels import (
    combine_kernel_dd, combine_kernel_fd
)


def plot_gp_1d(ax, predict, num_XX_test, bounds, XX_train=None, YY_train=None,
               num_stdev=1, function=None, gp_label=None, function_label=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    XX_test = np.linspace(bounds[0], bounds[1], num_XX_test)[:, np.newaxis]
    # return_std=True does not work for gradient enhanced krigging
    # gp_mean, gp_std = predict(XX_test,return_std=True)
    gp_mean, gp_cov = predict(XX_test, return_cov=True)
    gp_std = np.sqrt(np.diag(gp_cov))
    ax.plot(XX_test, gp_mean, '--k', lw=3, zorder=11, label=gp_label)
    ax.fill_between(
        XX_test[:, 0], gp_mean - num_stdev*gp_std, gp_mean + num_stdev*gp_std,
        alpha=0.5, color='k')
    if function is not None:
        ax.plot(XX_test, function(XX_test), 'r', lw=3, zorder=9,
                label=function_label)
    if XX_train is not None:
        ax.scatter(XX_train, YY_train, c='r', s=50, zorder=10)
    return ax


def plot_2d(function, num_XX_test_1d, bounds, XX_train_values=None, ax=None):
    # gp_mean_func = lambda XX: gp_predict(gp,x_kernel,kernel_ff,XX.T)[0]
    def gp_mean_func(XX): return function(XX.T)
    num_contour_levels = 20
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    X, Y, Z = get_meshgrid_function_data(gp_mean_func, bounds, num_XX_test_1d)
    cset = ax.contourf(
        X, Y, Z, levels=np.linspace(Z.min(), Z.max(), num_contour_levels))
    if XX_train_values is not None:
        ax.plot(XX_train_values[:, 0], XX_train_values[:, 1], 'ko')
    plt.colorbar(cset, ax=ax)
    return ax


def get_gp_samples_kernel(gp):
    if not hasattr(gp.kernel_, 'k1'):
        return gp.kernel_

    if type(gp.kernel_.k1) == Product:
        if type(gp.kernel_.k1.k1) == ConstantKernel:
            return gp.kernel_.k1.k2
        else:
            return gp.kernel_.k1.k1
    elif type(gp.kernel_.k1) == ConstantKernel:
        return gp.kernel_.k2
    else:
        return gp.kernel_.k1


class GradientEnhancedGP(GaussianProcessRegressor):
    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(GradientEnhancedGP, self).__init__(
            kernel=kernel, alpha=alpha,
            optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y, copy_X_train=copy_X_train,
            random_state=random_state)
        self.num_training_values = 0

    def stack_XX_derivs(self, XX_derivs):
        num_vars = XX_derivs.shape[1]
        return np.tile(XX_derivs, (num_vars, 1))

    def fit(self, XX_train_values, YY_train_values,
            XX_train_derivs, YY_train_derivs):
        r"""
        XX_train_values : np.ndarray (num_XX_train_values,num_vars)
            The location at which function values are obtained

        XX_train_derivs : np.ndarray (num_XX_train_derivs,num_vars)
            The location at which derivatives are obtained

        YY_train_values : np.ndarray (num_XX_train_values,1)
            The function values at each point in XX_train_derivs

        YY_train_derivs : np.ndarray (num_XX_train_derivs,num_vars)
            The derivatives at each point in XX_train_derivs
        """
        assert YY_train_values.shape[0] == XX_train_values.shape[0]
        assert YY_train_derivs.shape[0] == XX_train_derivs.shape[0]

        XX_train = np.vstack(
            (XX_train_values, self.stack_XX_derivs(XX_train_derivs)))
        self.num_training_values = XX_train_values.shape[0]

        YY_train = np.hstack(
            (YY_train_values, YY_train_derivs.flatten(order='F')))

        print('total train points', XX_train.shape)
        print('values train points', XX_train_values.shape)
        print('unique derivs train points', XX_train_derivs.shape)

        assert YY_train.shape[0] == XX_train.shape[0]
        super(GradientEnhancedGP, self).fit(XX_train, YY_train)

    def predict(self, XX_test, return_std=False, return_cov=False):
        print('predict')
        deriv_gp_kernel = get_gp_samples_kernel(self)
        # deriv_gp_kernel.length_scale=deriv_gp_kernel.length_scale*0+0.689
        return_code = deriv_gp_kernel.return_code
        deriv_gp_kernel.return_code = 'values'
        predictions = super(GradientEnhancedGP, self).predict(
            XX_test, return_std, return_cov)
        deriv_gp_kernel.return_code = return_code
        return predictions

    def predict_derivatives(self, XX_test, return_std=False, return_cov=False):
        deriv_gp_kernel = get_gp_samples_kernel(self)
        return_code = deriv_gp_kernel.return_code
        deriv_gp_kernel.return_code = 'derivs'
        predictions = super(GradientEnhancedGP, self).predict(
            XX_test, return_std, return_cov)
        deriv_gp_kernel.return_code = return_code
        return predictions

    def get_training_values_data(self):
        XX = self.X_train_[:self.num_training_values, :]
        YY = self.y_train_[:self.num_training_values]
        return XX, YY

    def get_training_derivs_data(self):
        XX = self.X_train_[self.num_training_values:, :]
        YY = self.y_train_[self.num_training_values:]
        return XX, YY

    def plot_1d(self, num_XX_test, bounds, axs, num_stdev=2, function=None,
                gp_label=None, function_label=None, plot_deriv=False):
        num_vars = len(bounds)//2
        assert num_vars == 1

        if plot_deriv:
            XX_train, YY_train = self.get_training_derivs_data()
            predict = self.predict_derivatives
        else:
            XX_train, YY_train = self.get_training_values_data()
            predict = self.predict
        plot_gp_1d(
            axs, predict, num_XX_test, bounds, XX_train, YY_train,
            num_stdev, function, gp_label=gp_label,
            function_label=function_label)


def predict_gpr_gradient(gp, XX_test, return_cov):
    r"""
    Compute gradient of standard GPR.

    When K_trans = gp.kernel_(X, gp.X_train_) is called
    the noise kernel just returns zeros so can ignore it in this function

    This function assumes gp_kernel = ConstantKernel*gp_kernel + ...
    """
    from scipy.linalg import cho_solve
    gp_kernel = get_gp_samples_kernel(gp)
    length_scale = gp_kernel.length_scale
    K_trans = combine_kernel_fd(None, gp.X_train_, length_scale, XX_test)
    if type(gp.kernel_.k1) == Product:
        # assumes gp.kernel_ = ConstantKernel*gp_kernel + ...
        K_trans *= gp.kernel_.k1.k1.constant_value
    # K_trans = gp.kernel_(X, gp.X_train_)
    y_mean = K_trans.dot(gp.alpha_)  # Line 4 (y_mean = f_star)
    y_mean = gp._y_train_mean + y_mean  # undo normal.
    if return_cov:
        K_dd = combine_kernel_dd(XX_test, XX_test, length_scale)
        if type(gp.kernel_.k1) == Product:
            # assumes kernel = ConstantKernel*gp_kernel
            K_dd *= gp.kernel_.k1.k1.constant_value
        v = cho_solve((gp.L_, True), K_trans.T)  # Line 5
        y_cov = K_dd - K_trans.dot(v)  # Line 6
        return y_mean, y_cov
    return y_mean

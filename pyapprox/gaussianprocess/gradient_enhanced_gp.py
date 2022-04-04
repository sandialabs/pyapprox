from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    StationaryKernelMixin, NormalizedKernelMixin, Kernel,
    _check_length_scale, Hyperparameter, _approx_fprime, Product
)
from scipy.spatial.distance import cdist
from pyapprox.util.visualization import get_meshgrid_function_data


def kernel_dd(XX1, XX2, length_scale, var_num1, var_num2):
    r"""
    For Gaussian kernel
    K(x,x^*) = exp(-0.5*(\sum_{i=1}^d w_i(x_i-x_i^*)^2)

    Evaluate partial cross derivative
    d/dx_i d/dx_j K(x,x^*) = w_i(w_i*(x_i-x_i^*)^2-1)K(x,x^*) x_i=x_j
    d/dx_i d/dx_j K(x,x^*) = w_iw_j(x_i-x_i^*)(x_j-x_j^*)K(x,x^*) x_i\neq x_j

    Parameters
    ----------
    XX1 : np.ndarray (nsamples_x,nvars)
        Samples x

    XX2 : np.ndarray (nsamples_y,nvars)
        Samples x^*

    length_scale : double
        w = 1/length_scale**2

    var_num1 : integer
        The direction i of the first partial derivative

    var_num2 : integer
        The direction j of the second partial derivative
    """
    length_scale = np.atleast_1d(length_scale)
    w = 1./length_scale**2

    K = kernel_ff(XX1, XX2, length_scale)
    if var_num1 == var_num2:
        K *= w[var_num1]*(1-w[var_num1]*(
            XX1[:, var_num1][:, np.newaxis]-XX2[:, var_num1][np.newaxis, :])**2)
    else:
        K *= -w[var_num1]*w[var_num2]
        K *= (XX1[:, var_num1][:, np.newaxis]-XX2[:, var_num1][np.newaxis, :])
        K *= (XX1[:, var_num2][:, np.newaxis]-XX2[:, var_num2][np.newaxis, :])

    # assert np.allclose(K,K1)
    return K


def kernel_ff(XX1, XX2, length_scale):
    r"""
    Evaluate Gaussian Kernel
    K(x,x^*) = exp(-0.5*(\sum_{i=1}^d w_i(x_i-x_i^*)^2)

    Parameters
    ----------
    XX1 : np.ndarray (nsamples_x,nvars)
        Samples x

    XX2 : np.ndarray (nsamples_y,nvars)
        Samples x^*

    length_scale : double
        w = 1/length_scale**2
    """
    length_scale = np.atleast_1d(length_scale)
    dists = cdist(XX1 / length_scale, XX2 / length_scale,
                  metric='sqeuclidean')
    K = np.exp(-.5 * dists)
    return K


def kernel_fd(XX1, XX2, length_scale, var_num):
    r"""
    For Gaussian Kernel
    K(x,x^*) = exp(-0.5*(\sum_{i=1}^d w_i(x_i-x_i^*)^2)

    Evaluate first partial derivative
    d/dx_i K(x,x^*)=--w_i*(x_i-x_i^*)*K(x,x^*)

    Parameters
    ----------
    XX1 : np.ndarray (nsamples_x,nvars)
        Samples x

    XX2 : np.ndarray (nsamples_y,nvars)
        Samples x^*

    length_scale : double
        w = 1/length_scale**2

    var_num : integer
        The direction i of the partial derivative
    """
    length_scale = np.atleast_1d(length_scale)
    w = 1./length_scale**2

    K = kernel_ff(XX1, XX2, length_scale)
    K *= -w[var_num]*(XX1[:, var_num][:, np.newaxis] -
                      XX2[:, var_num][np.newaxis, :])

    return K


def function_values_kernel(X1, X2, length_scale, X3=None):
    r"""
    Evaluate kernel used to compute function values from GP
    """
    num_vars = X1.shape[1]
    assert X2.shape[0] % num_vars == 0
    num_deriv_samples = X2.shape[0]//num_vars

    if X3 is None:
        XA = X1
        XB = X1
        XC = X2
    else:
        XA = X3
        XB = X1
        XC = X2

    K = kernel_ff(XA, XB, length_scale)
    for ii in range(num_vars):
        idx1 = num_deriv_samples*ii
        idx2 = num_deriv_samples*(ii+1)
        K = np.hstack((K, -kernel_fd(XA, XC[idx1:idx2, :], length_scale, ii)))
    return K


def combine_kernel_dd(X1, X2, length_scale, X3=None):
    num_vars = X1.shape[1]
    assert X1.shape[0] % num_vars == 0
    if X3 is None:
        XA = X1
        XB = X1
        # XC = X2
    else:
        XA = X3
        XB = X1
         #XC = X2
    num_XA_deriv = XA.shape[0]//num_vars
    num_XB_deriv = XB.shape[0]//num_vars

    for ii in range(num_vars):
        # indexing assumes derivatives in each direction are at the same
        # set of points in X<letter>
        idxA1 = num_XA_deriv*ii
        idxA2 = num_XA_deriv*(ii+1)
        idxB1 = num_XB_deriv*ii
        idxB2 = num_XB_deriv*(ii+1)
        K_dd_ii = kernel_dd(
            XA[idxA1:idxA2], XB[idxB1:idxB2], length_scale, ii, 0)
        for jj in range(1, num_vars):
            K_dd_ii = np.hstack(
                (K_dd_ii, kernel_dd(XA[idxA1:idxA2], XB[idxB1:idxB2],
                                    length_scale, ii, jj)))
        if ii == 0:
            K_dd = K_dd_ii
        else:
            K_dd = np.vstack((K_dd, K_dd_ii))

    return K_dd


def combine_kernel_fd(X1, X2, length_scale, X3=None):
    if X3 is None:
        XA = X1
        XC = X2
    else:
        XA = X3
        XC = X2
    num_vars = XA.shape[1]
    assert XA.shape[0] % num_vars == 0
    num_XA_deriv = XA.shape[0]//num_vars

    idxA1 = 0
    idxA2 = num_XA_deriv
    K_fd = kernel_fd(XA[idxA1:idxA2], XC, length_scale, 0)
    for ii in range(1, num_vars):
        idxA1 = num_XA_deriv*ii
        idxA2 = num_XA_deriv*(ii+1)
        K_fd = np.vstack(
            (K_fd, kernel_fd(XA[idxA1:idxA2], XC, length_scale, ii)))

    return K_fd


def gradients_kernel(X1, X2, length_scale, X3=None):
    r"""
    Evaluate kernel used to compute function values from GP
    """
    K_dd = combine_kernel_dd(X1, X2, length_scale, X3)
    K_fd = combine_kernel_fd(X1, X2, length_scale, X3)
    K = np.hstack((K_fd, K_dd))
    return K


def full_kernel(XX1, length_scale, n_XX_func, XX2=None, return_code='full'):
    r"""
    return_code : string
        'full'   return full covariance matrix
        'values' return values covariance matrix
        'derivs' return derivs covariance matrix
    """
    if XX2 is None:
        if return_code == 'values':
            return kernel_ff(XX1, XX1, length_scale)
        elif return_code == 'derivs':
            return combine_kernel_dd(XX1, XX1, length_scale)
        XX_func = XX1[:n_XX_func]  # samples at which function values are known
        XX_deriv = XX1[n_XX_func:]  # samples at which derivatives are known
        K1 = function_values_kernel(XX_func, XX_deriv, length_scale)
        K2 = gradients_kernel(XX_deriv, XX_func, length_scale)
        K = np.vstack((K1, K2))
    else:
        XX_func = XX2[:n_XX_func]  # samples at which function values are known
        XX_deriv = XX2[n_XX_func:]  # samples at which derivatives are known
        if return_code != 'derivs':
            K1 = function_values_kernel(XX_func, XX_deriv, length_scale, XX1)
            if return_code == 'values':
                return K1
        K2 = gradients_kernel(XX_deriv, XX_func, length_scale, XX1)
        if return_code == 'derivs':
            return K2
        K = np.vstack((K1, K2))
    return K


class DerivGPKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, n_XX_func, length_scale=[1.0],
                 length_scale_bounds=(1e-5, 1e5)):
        r"""
        Parameters
        ----------
        n_XX_func : integer
            The number of training points at which function values are known
        """
        self.length_scale = length_scale
        self.length_scale = np.atleast_1d(self.length_scale)
        self.length_scale_bounds = length_scale_bounds
        self.n_XX_func = n_XX_func
        self.return_code = 'full'

    def __call__(self, XX1, XX2=None, eval_gradient=False):
        r"""Return the kernel k(XX1, XX2) and optionally its gradient.

        Parameters
        ----------
        XX1 : array, shape (n_samples_XX1, n_features)
            Left argument of the returned kernel k(XX1, XX2)

        XX2 : array, shape (n_samples_XX2, n_features), (optional, default=None)
            Right argument of the returned kernel k(XX1, XX2). If None,
            k(XX1, XX1) is evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when XX2 is None.

        Returns
        -------
        K : array, shape (n_samples_XX1, n_samples_XX2)
            Kernel k(XX1, XX2)

        K_gradient : array (opt.), shape (n_samples_XX1, n_samples_XX1, n_dims)
            The gradient of the kernel k(XX1, XX1) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        XX1 = np.atleast_2d(XX1)
        length_scale = _check_length_scale(XX1, self.length_scale)
        if XX2 is None:
            K = full_kernel(XX1, length_scale, self.n_XX_func,
                            return_code=self.return_code)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when XX2 is None.")
            K = full_kernel(XX1, length_scale, self.n_XX_func, XX2,
                            self.return_code)

        if not eval_gradient:
            return K

        if self.hyperparameter_length_scale.fixed:
            # Hyperparameter l kept fixed
            length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))
        else:
            # approximate gradient numerically
            def f(gamma):  # helper function
                return full_kernel(XX1, gamma, self.n_XX_func,
                                   return_code=self.return_code)
            length_scale = np.atleast_1d(length_scale)
            length_scale_gradient = _approx_fprime(length_scale, f, 1e-8)
        return K, length_scale_gradient

    def values_kernel(self, XX1, XX2):
        return function_values_kernel(XX1, XX2, self.length_scale, X3=None)

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)


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

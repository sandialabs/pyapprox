import numpy as np
from scipy.spatial.distance import cdist

from sklearn.gaussian_process.kernels import (
    Matern, Product, Sum, ConstantKernel, WhiteKernel, _check_length_scale,
    Hyperparameter, _approx_fprime
)
from sklearn.gaussian_process.kernels import RBF as SKL_RBF


class RBF(SKL_RBF):
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Unlike sklearn True is supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if not eval_gradient or Y is None:
            return super().__call__(X, Y, eval_gradient)
        K = super().__call__(X, Y)
        length_scale = _check_length_scale(X, self.length_scale)
        if self.hyperparameter_length_scale.fixed:
            return K, np.empty((X.shape[0], Y.shape[0], 0))
        if not self.anisotropic or length_scale.shape[0] == 1:
            dists = cdist(X, Y, metric="sqeuclidean")
            K_gradient = (K * dists[:, :]/length_scale**2)[:, :, np.newaxis]
        elif self.anisotropic:
            dists = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2 / (
                length_scale ** 2)
            K_gradient = dists
            K_gradient *= K[..., np.newaxis]

        return K, K_gradient


class MultilevelGPKernel(RBF):
    def __init__(self, nvars, nsamples_per_model, kernels, length_scale=[1.0],
                 length_scale_bounds=(1e-5, 1e5)):

        self.nmodels = len(nsamples_per_model)
        assert len(length_scale) == self.nmodels*nvars+self.nmodels-1
        super().__init__(length_scale, length_scale_bounds)
        self.nvars = nvars
        self.nsamples_per_model = nsamples_per_model
        self.return_code = 'full'
        for kernel in kernels:
            # todo add ability to compute gradients when Y is not None
            # to matern kernel
            if type(kernel) != RBF:
                raise ValueError("Only RBF Kernels are curently supported")
        assert len(kernels) == self.nmodels
        self.kernels = kernels

    @staticmethod
    def _sprod(scalings, ii, jj):
        if jj > len(scalings):
            raise RuntimeError()
        return np.prod(scalings[ii:jj+1])

    def _eval_kernel(self, kernel, XX1, XX2, eval_gradient):
        result = kernel(XX1, XX2, eval_gradient=eval_gradient)
        if type(result) == tuple:
            return result
        else:
            return result, np.nan

    def _diagonal_kernel_block(self, XX1, kernels, scalings, nmodels,
                               eval_gradient):
        nvars = XX1.shape[1]
        nhyperparams = nvars*nmodels+nmodels-1
        kk = len(kernels)
        K_block, K_grad = self._eval_kernel(
            kernels[kk-1], XX1, None, eval_gradient)
        if eval_gradient:
            K_block_grad = np.zeros(
                (K_grad.shape[0], K_grad.shape[1], nhyperparams))
            K_block_grad[..., (kk-1)*nvars:kk*nvars] = K_grad
            # derivative with respect to scalings for kk model is zero so
            # do nothing
        else:
            K_block_grad = np.nan
        for nn in range(kk-1):
            K, K_grad = self._eval_kernel(
                kernels[nn], XX1, None, eval_gradient)
            const = self._sprod(scalings, nn, kk-1)**2
            K_block += const*K
            if eval_gradient:
                # length_scale grad
                K_block_grad[..., nn*nvars:(nn+1)*nvars] += const*K_grad
                # scalings grad
                idx1 = nmodels*nvars+nn
                idx2 = nmodels*nvars+kk-1
                # tmp = 2*const/scalings[nn:kk]  # when NO using log of hyperparams
                tmp = 2*const  # when YES using log of hyperparams
                # print(nn, kk, scalings[nn:kk], const, tmp, idx1, idx2)
                K_block_grad[..., idx1:idx2] += K[..., None]*tmp
        return K_block, K_block_grad

    def _off_diagonal_kernel_block(self, Xkk, Xll, kernels, scalings, kk, ll,
                                   nmodels, eval_gradient):
        assert kk < ll
        nvars = Xkk.shape[1]
        nhyperparams = nvars*nmodels+nmodels-1
        K_block, K_block_grad = 0, np.nan
        if eval_gradient:
            K_block_grad = np.zeros((Xkk.shape[0], Xll.shape[0], nhyperparams))
        for nn in range(kk+1):
            const = (self._sprod(scalings, nn, kk-1) *
                     self._sprod(scalings, nn, ll-1))
            K, K_grad = self._eval_kernel(kernels[nn], Xkk, Xll, eval_gradient)
            K_block += const*K
            if eval_gradient:
                # length_scale grad
                K_block_grad[..., nn*nvars:(nn+1)*nvars] += const*K_grad
                # scalings grad
                idx1 = nmodels*nvars+nn
                idx2 = nmodels*nvars+kk
                if idx1 < idx2:
                    # products that have squared terms
                    tmp = 2*const
                    K_block_grad[..., idx1:idx2] += K[..., None]*tmp
                # products with just linear terms
                idx1 = nmodels*nvars+kk
                idx2 = nmodels*nvars+ll
                tmp = const
                K_block_grad[..., idx1:idx2] += K[..., None]*tmp
        return K_block, K_block_grad

    @staticmethod
    def _unpack_samples(XX, nsamples_per_model):
        samples = []
        lb, ub = 0, 0
        for nn in nsamples_per_model:
            lb = ub
            ub += nn
            samples.append(XX[lb:ub, :])
        return samples

    def _eval_K(self, XX1, nsamples_per_model, kernels, scalings,
                eval_gradient):
        nrows = XX1.shape[0]
        nmodels = len(nsamples_per_model)
        samples = self._unpack_samples(XX1, nsamples_per_model)
        K = np.zeros((nrows, nrows), dtype=float)
        K_grad = [[None for ll in range(nmodels)] for kk in range(nmodels)]
        lb1, ub1 = 0, 0
        for kk in range(nmodels):
            lb1 = ub1
            ub1 += nsamples_per_model[kk]
            lb2, ub2 = lb1, ub1
            K_block, K_block_grad = self._diagonal_kernel_block(
                samples[kk], kernels[:kk+1], scalings[:kk],
                nmodels, eval_gradient)
            K[lb1:ub1, lb2:ub2] = K_block
            K_grad[kk][kk] = K_block_grad
            for ll in range(kk+1, nmodels):
                lb2 = ub2
                ub2 += nsamples_per_model[ll]
                K[lb1:ub1, lb2:ub2], K_block_grad = (
                    self._off_diagonal_kernel_block(
                        samples[kk], samples[ll], kernels[:ll],
                        scalings[:ll], kk, ll, nmodels, eval_gradient))
                K[lb2:ub2, lb1:ub1] = K[lb1:ub1, lb2:ub2].T
                K_grad[kk][ll] = K_block_grad
                if eval_gradient:
                    K_grad[ll][kk] = np.transpose(K_block_grad, axes=[1, 0, 2])
        if eval_gradient:
            for kk in range(nmodels):
                K_grad[kk] = np.hstack(K_grad[kk])
            K_grad = np.vstack(K_grad)
        return K, K_grad

    # @staticmethod
    # def _shared_samples(samples1, samples2):
    #     # recall sklearn kernels defines samples transpose of pyapprox format
    #     shared_idx = []
    #     for ii in range(samples2.shape[0]):
    #         sample2 = samples2[ii, :]
    #         for jj in range(samples1.shape[0]):
    #             if np.allclose(sample2, samples1[jj, :], atol=1e-13):
    #                 shared_idx.append(jj)
    #     return shared_idx

    # def _all_shared_samples(self, samples):
    #     nmodels = len(samples)
    #     shared_idx_list = [None for nn in range(1, nmodels)]
    #     for nn in range(1, nmodels):
    #         shared_idx_list[nn-1] = self._shared_samples(
    #             samples[nn-1], samples[nn])
    #     return shared_idx_list

    def _eval_t(self, XX1, XX2, nsamples_per_model, kernels, scalings):
        nmodels = len(nsamples_per_model)
        assert np.sum(nsamples_per_model) == XX2.shape[0]
        samples = self._unpack_samples(XX2, nsamples_per_model)
        const = self._sprod(scalings, 0, nmodels-1)
        t_blocks = [const*self.kernels[0](XX1, samples[nn])
                    for nn in range(nmodels)]
        for nn in range(1, nmodels):
            const = self._sprod(scalings, nn, nmodels-1)
            t_blocks[nn:] = [
                const*self.kernels[nn](XX1, samples[kk]) +
                scalings[nn-1]*t_blocks[kk]
                for kk in range(nn, nmodels)]
        return np.hstack(t_blocks)

    def __call__(self, XX1, XX2=None, eval_gradient=False):
        XX1 = np.atleast_2d(XX1)
        hyperparams = np.squeeze(self.length_scale).astype(float)
        length_scales = np.asarray(hyperparams[:self.nmodels*self.nvars])
        assert hyperparams.ndim == 1
        scalings = np.asarray(hyperparams[self.nmodels*self.nvars:])
        for kk in range(self.nmodels):
            self.kernels[kk].length_scale = (
                length_scales[kk*self.nvars:(kk+1)*self.nvars])
        if XX2 is None:
            K, K_grad = self._eval_K(
                XX1, self.nsamples_per_model, self.kernels, scalings,
                eval_gradient)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when XX2 is None.")
            K = self._eval_t(
                XX1, XX2, self.nsamples_per_model, self.kernels, scalings)
        if not eval_gradient:
            return K
        else:
            return K, K_grad

    def diag(self, X):
        """
        TODO make this function more efficient by directly evaluating diagonal
        only
        """
        hyperparams = np.squeeze(self.length_scale).astype(float)
        assert hyperparams.ndim == 1
        scalings = np.asarray(hyperparams[self.nmodels*self.nvars:])
        # D1 = self._diagonal_kernel_block(X, self.kernels, scalings,
        #                                  self.nmodels, False)[0]
        # D1 = np.diag(D1).copy()

        # compute diag for high_fidelity model
        kk = self.nmodels
        D = self.kernels[kk-1].diag(X)
        for nn in range(kk-1):
            const = self._sprod(scalings, nn, kk-1)**2
            D += const*self.kernels[nn].diag(X)
        # assert np.allclose(D, D1)
        return D


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


class DerivGPKernel(SKL_RBF):
    def __init__(self, n_XX_func, length_scale=[1.0],
                 length_scale_bounds=(1e-5, 1e5)):
        r"""
        Parameters
        ----------
        n_XX_func : integer
            The number of training points at which function values are known
        """
        super().__init__(length_scale, length_scale_bounds)
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

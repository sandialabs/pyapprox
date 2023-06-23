import copy
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.gaussian_process.kernels import (
    Product, Sum, _check_length_scale, Hyperparameter, _approx_fprime,
    Matern, WhiteKernel) # so can be imported directly from this file
from sklearn.gaussian_process.kernels import RBF as SKL_RBF
from sklearn.gaussian_process.kernels import (
    ConstantKernel as SKL_ConstantKernel)
from sklearn.gaussian_process.kernels import (
    _num_samples as _SKL_num_samples)

from pyapprox.variables.transforms import AffineTransform
from pyapprox.surrogates.polychaos.gpc import (
    get_univariate_quadrature_rules_from_variable)
from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_indices, tensor_product_indices)
from pyapprox.surrogates.interp.monomial import monomial_basis_matrix


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

    def _length_scale_repr(self):
        if self.anisotropic:
            return "[{0}]".format(
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0:.3g}".format(
                np.ravel(self.length_scale)[0])

    def __repr__(self):
        return "PyA{0}(length_scale={1})".format(
            self.__class__.__name__, self._length_scale_repr())


class ConstantKernel(SKL_ConstantKernel):
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object, \
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Unlike sklearn, Y is None is supported.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is None:
            Y = X

        K = np.full(
            (_SKL_num_samples(X), _SKL_num_samples(Y)),
            self.constant_value,
            dtype=np.array(self.constant_value).dtype,
        )
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (
                    K,
                    np.full(
                        (_SKL_num_samples(X), _SKL_num_samples(Y), 1),
                        self.constant_value,
                        dtype=np.array(self.constant_value).dtype,
                    ),
                )
            else:
                return K, np.empty(
                    (_SKL_num_samples(X), _SKL_num_samples(Y), 0))
        else:
            return K


class MonomialScaling():
    def __init__(self, nvars, degree, log=True):
        self.nvars = nvars
        self.degree = degree
        self.indices = compute_hyperbolic_indices(self.nvars, self.degree)

        self.nhyperparams = self.indices.shape[1]
        self.coef = None
        self.log = log

    def set_params(self, coef):
        assert coef.shape[0] == self.indices.shape[1]
        assert coef.ndim == 2 and coef.shape[1] == 1
        self.coef = coef

    def __call__(self, XX, eval_gradient=False):
        basis_mat = monomial_basis_matrix(self.indices, XX.T, deriv_order=0)
        vals = basis_mat.dot(self.coef)
        if not eval_gradient:
            return vals, np.inf
        if self.log:
            # below is true when coef=exp(rho) and we need gradient with
            # respect to rho
            # equiv to basis_mat.dot(diag(coef))
            return vals, self.coef[:, 0]*basis_mat
        return vals, basis_mat

    def __repr__(self):
        return "{0}(degree={1})".format(
            self.__class__.__name__, self.degree)


class MultilevelKernel(RBF):
    def __init__(self, nvars, kernel_types, scalings,
                 length_scale=None, length_scale_bounds=(1e-5, 1e5),
                 rho=None, rho_bounds=(1e-5, 10),
                 sigma=None, sigma_bounds=(1e-5, 10),
                 nsamples_per_model=None):

        self.nmodels = len(kernel_types)
        self.nvars = nvars
        assert len(scalings) == self.nmodels-1
        self.scalings = scalings

        (length_scale, length_scale_bounds, self.nhyperparams_per_kernel,
         self.nkernel_hyperparams) = self._validate_length_scale(
             length_scale, length_scale_bounds)
        super().__init__(length_scale, length_scale_bounds)

        self.sigma, self.sigma_bounds, self.nsigma_hyperparams = (
            self._validate_sigma(sigma, sigma_bounds))
        self.kernel_types, self.kernels = self._validate_kernels(
            kernel_types, self.sigma, self.sigma_bounds, self.length_scale,
            self.length_scale_bounds)

        (self.rho, self.rho_bounds, self.nhyperparams_per_scaling,
         self.nscaling_hyperparams) = self._validate_rho(rho, rho_bounds)

        self.nhyperparams = (
            self.nkernel_hyperparams+self.nsigma_hyperparams +
            self.nscaling_hyperparams)

        # model_eval_id determines which fidelity to evaluate on test data
        self.model_eval_id = self.nmodels-1
        # nsamples_per_model must be an argument otherwise sklearn will
        # not copy kernel properly when calling gfit()
        self.nsamples_per_model = nsamples_per_model

    def _validate_length_scale(self, length_scale, length_scale_bounds):
        # theta only contains parameters that are not fixed
        nhyperparams_per_kernel = np.full(self.nmodels, self.nvars, dtype=int)
        if length_scale_bounds == "fixed":
            # how many parameters are optimized
            nkernel_hyperparams = 0
        else:
            nkernel_hyperparams = nhyperparams_per_kernel.sum()
        if length_scale is None:
            length_scale = np.ones(nkernel_hyperparams)
        if len(length_scale) != nhyperparams_per_kernel.sum():
            msg = "length_scale has incorrect shape. Should be "
            msg += f"{nhyperparams_per_kernel.sum()} but is "
            msg += f"{len(length_scale)}"
            raise ValueError(msg)
        if isinstance(length_scale_bounds, tuple):
            length_scale_bounds = (
                [length_scale_bounds]*(self.nmodels*self.nvars))
        if (length_scale_bounds != "fixed" and
                len(length_scale_bounds) != nkernel_hyperparams):
            raise ValueError("length_scale_bounds does not have correct shape")
        return (length_scale, length_scale_bounds, nhyperparams_per_kernel,
                nkernel_hyperparams)

    def _validate_sigma(self, sigma, sigma_bounds):
        if sigma is None:
            sigma = np.ones(self.nmodels, dtype=float)
        sigma = np.atleast_1d(sigma)
        if sigma_bounds == "fixed":
            nsigma_hyperparams = 0
        else:
            nsigma_hyperparams = self.nmodels
        if isinstance(sigma_bounds, tuple):
            sigma_bounds = [sigma_bounds]*self.nmodels
        if sigma_bounds != "fixed" and len(sigma_bounds) != self.nmodels:
            raise ValueError("sigma_bounds does not have correct shape")
        if len(sigma) != self.nmodels:
            msg = f"sigma {sigma.shape} does not have correct shape"
            msg += f" {self.nmodels}"
            raise ValueError(msg)
        return sigma, sigma_bounds, nsigma_hyperparams

    def _validate_kernels(self, kernel_types, sigma, sigma_bounds,
                          length_scale, length_scale_bounds):
        kernels = []
        if length_scale_bounds == "fixed":
            ls_bounds = ["fixed"]*self.nmodels
        else:
            ls_bounds = length_scale_bounds
        if sigma_bounds == "fixed":
            sigma_bounds = ["fixed"]*self.nmodels
        else:
            sigma_bounds = sigma_bounds
        for kk in range(self.nmodels):
            kernels.append(
                ConstantKernel(sigma[kk], sigma_bounds[kk]) *
                kernel_types[kk](
                    length_scale[kk*self.nvars:(kk+1)*self.nvars],
                    ls_bounds[kk]))
        return kernel_types, kernels

    def _validate_rho(self, rho, rho_bounds):
        nhyperparams_per_scaling = np.asarray(
            [s.nhyperparams for s in self.scalings])
        if rho is None:
            rho = np.full(nhyperparams_per_scaling.sum(), 0.5, dtype=float)
        rho = np.atleast_1d(rho)
        if rho_bounds == "fixed":
            # how many parameters are optimized
            nscaling_hyperparams = 0
        else:
            nscaling_hyperparams = nhyperparams_per_scaling.sum()
        if isinstance(rho_bounds, tuple):
            rho_bounds = [rho_bounds]*nscaling_hyperparams
        if rho_bounds != "fixed" and len(rho) != nhyperparams_per_scaling.sum():
            msg = f"rho {rho.shape} does not have correct shape"
            msg += f" {nhyperparams_per_scaling.sum()}"
            raise ValueError(msg)
        return rho, rho_bounds, nhyperparams_per_scaling, nscaling_hyperparams

    def set_nsamples_per_model(self, nsamples_per_model):
        if len(nsamples_per_model) != self.nmodels:
            raise ValueError("nsamples_per_model does not match self.nmodels")
        self.nsamples_per_model = np.asarray(nsamples_per_model, dtype=int)

    def _kernel_hyperparam_indices(self, kk):
        assert self.length_scale_bounds != "fixed"
        idx1 = self.nhyperparams_per_kernel[:kk].sum()
        idx2 = idx1 + self.nhyperparams_per_kernel[kk]
        return idx1, idx2

    def _scaling_indices(self, kk):
        assert self.rho_bounds != "fixed"
        idx1 = (
            self.nkernel_hyperparams+self.nhyperparams_per_scaling[:kk].sum())
        idx2 = idx1 + self.nhyperparams_per_scaling[kk]
        return idx1, idx2

    def _sigma_hyperparam_indices(self, kk):
        # the order stored in self.theta and thus indices for grad matrices
        # are dependent on order of _bounds in kwarg list to self.__init__
        assert self.sigma_bounds != "fixed"
        idx1 = self.nkernel_hyperparams+self.nscaling_hyperparams+kk
        idx2 = idx1+1  # self.kernels[kk].k1.n_dims
        return idx1, idx2

    @staticmethod
    def _sprod(scalings, ii, jj):
        if jj > len(scalings):
            # print(jj, len(scalings))
            raise RuntimeError("scalings is the wrong size")
        return np.prod(scalings[ii:jj+1])

    @staticmethod
    def _eval_kernel(kernel, XX1, XX2, eval_gradient):
        result = kernel(XX1, XX2, eval_gradient=eval_gradient)
        if type(result) == tuple:
            return result
        else:
            return result, np.nan

    def _diagonal_kernel_block(self, XX1, XX2, kernels, scalings,
                               nmodels, eval_gradient):
        # hack because only scalar scalings allowed
        scalings = [s[0][0] for s in scalings]
        kk = len(kernels)
        K_block, K_grad = self._eval_kernel(
            kernels[kk-1], XX1, XX2, eval_gradient)

        if eval_gradient:
            # n_dims defines number of optimized parameters
            # self.nhyperparams includes both optimized and fixed parameters
            K_block_grad = np.zeros(
                (K_grad.shape[0], K_grad.shape[1], self.n_dims))

            if self.sigma_bounds != "fixed":
                idx1, idx2 = self._sigma_hyperparam_indices(kk-1)
                K_block_grad[..., idx1:idx2] = K_grad[..., :idx2-idx1]
            else:
                idx1, idx2 = 0, 0
            if self.length_scale_bounds != "fixed":
                idx3, idx4 = self._kernel_hyperparam_indices(kk-1)
                K_block_grad[..., idx3:idx4] = K_grad[..., idx2-idx1:]
            # derivative with respect to scalings for kk model is zero so
            # do nothing
        else:
            K_block_grad = np.nan

        for nn in range(kk-1):
            K, K_grad = self._eval_kernel(
                kernels[nn], XX1, XX2, eval_gradient)
            const = self._sprod(scalings, nn, kk-1)**2
            K_block += const*K
            if eval_gradient:
                if self.sigma_bounds != "fixed":
                    idx1, idx2 = self._sigma_hyperparam_indices(nn)
                    K_block_grad[..., idx1:idx2] = (
                        const*K_grad[..., :idx2-idx1])
                else:
                    idx1, idx2 = 0, 0
                if self.length_scale_bounds != "fixed":
                    # length_scale grad
                    idx3, idx4 = self._kernel_hyperparam_indices(nn)
                    K_block_grad[..., idx3:idx4] = (
                        const*K_grad[..., idx2-idx1:])
                if self.rho_bounds != "fixed":
                    # scalings grad
                    idx1 = self.nkernel_hyperparams+nn
                    idx2 = self.nkernel_hyperparams+kk-1
                    # when NOT using log of hyperparams
                    # tmp = 2*const/scalings[nn:kk]
                    # when YES using log of hyperparams
                    tmp = 2*const
                    K_block_grad[..., idx1:idx2] += K[..., None]*tmp
        return K_block, K_block_grad

    def _off_diagonal_kernel_block(self, Xkk, Xll, kernels, scalings, kk, ll,
                                   nmodels, eval_gradient):
        assert kk < ll
        K_block, K_block_grad = 0, np.nan
        if eval_gradient:
            K_block_grad = np.zeros(
                (Xkk.shape[0], Xll.shape[0], self.n_dims))
        # hack because only scalar scalings allowed
        scalings = [s[0][0] for s in scalings]
        for nn in range(kk+1):
            const = (self._sprod(scalings, nn, kk-1) *
                     self._sprod(scalings, nn, ll-1))
            K, K_grad = self._eval_kernel(kernels[nn], Xkk, Xll, eval_gradient)
            K_block += const*K
            if eval_gradient:
                if self.sigma_bounds != "fixed":
                    idx1, idx2 = self._sigma_hyperparam_indices(nn)
                    K_block_grad[..., idx1:idx2] += (
                        const*K_grad[..., :idx2-idx1])
                else:
                    idx1, idx2 = 0, 0
                if self.length_scale_bounds != "fixed":
                    # length_scale grad
                    idx3, idx4 = self._kernel_hyperparam_indices(nn)
                    K_block_grad[..., idx3:idx4] += (
                        const*K_grad[..., idx2-idx1:])
                if self.rho_bounds != "fixed":
                    # scalings grad

                    idx1 = self.nkernel_hyperparams+nn
                    idx2 = self.nkernel_hyperparams+kk
                    if idx1 < idx2:
                        # products that have squared terms
                        tmp = 2*const
                        K_block_grad[..., idx1:idx2] += K[..., None]*tmp
                    # else: # products with just linear terms
                    idx1, _ = self._scaling_indices(kk)
                    _, idx2 = self._scaling_indices(ll-1)
                    #  idx1 = self.nkernel_hyperparams+kk
                    # idx2 = self.nkernel_hyperparams+ll
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

    @staticmethod
    def _unpack_kernel_scalings(kernel_scalings, nsamples_per_model, kk):
        # scalings for one kernel
        # kk : XX1 index, ll kernel index
        idx1 = np.sum(nsamples_per_model[:kk])
        idx2 = idx1 + nsamples_per_model[kk]
        return kernel_scalings[idx1:idx2]

    def diagonal_kernel_block(self, samples,  kernels, scalings1,
                              kk, nmodels, eval_gradient, samples2=None,
                              scalings=None):
        if np.any([s.degree > 0 for s in self.scalings]):
            raise ValueError("Only constant scalings supported")
        # TODO cannot handle scalings being a function of points
        # this is why only scalings 1 is passed to _diagonal_kernel_block
        if samples2 is None:
            return self._diagonal_kernel_block(
                samples[kk], None, kernels[:kk+1], scalings1[:kk],
                nmodels, eval_gradient)
        return self._diagonal_kernel_block(
            samples, samples2[kk], kernels[:kk+1], scalings1[:kk],
            nmodels, eval_gradient)

    def off_diagonal_kernel_block(self, XX1, XX2,  kernels, scalings, kk, ll,
                                  nmodels, eval_gradient, X_kk_is_test=False,
                                  scalings2=None):
        if np.any([s.degree > 0 for s in self.scalings]):
            raise ValueError("Only constant scalings supported")
        # TODO cannot handle scalings being a function of points
        return self._off_diagonal_kernel_block(
                XX1, XX2, kernels[:ll],
                scalings[:ll], kk, ll, nmodels, eval_gradient)

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
            K_block, K_block_grad = self.diagonal_kernel_block(
                samples, kernels, scalings, kk, nmodels, eval_gradient)
            K[lb1:ub1, lb2:ub2] = K_block
            K_grad[kk][kk] = K_block_grad
            for ll in range(kk+1, nmodels):
                lb2 = ub2
                ub2 += nsamples_per_model[ll]
                K[lb1:ub1, lb2:ub2], K_block_grad = (
                    self.off_diagonal_kernel_block(
                        samples[kk], samples[ll],
                        kernels, scalings, kk, ll, nmodels,
                        eval_gradient))
                K[lb2:ub2, lb1:ub1] = K[lb1:ub1, lb2:ub2].T
                K_grad[kk][ll] = K_block_grad
                if eval_gradient:
                    K_grad[ll][kk] = np.transpose(K_block_grad, axes=[1, 0, 2])
        if eval_gradient:
            for kk in range(nmodels):
                K_grad[kk] = np.hstack(K_grad[kk])
            K_grad = np.vstack(K_grad)
        return K, K_grad

    def _eval_t(self, XX1, XX2, nsamples_per_model, kernels, scalings_XX1,
                scalings_XX2, model_eval_id):
        nmodels = len(nsamples_per_model)
        assert np.sum(nsamples_per_model) == XX2.shape[0]
        samples = self._unpack_samples(XX2, nsamples_per_model)
        t_blocks = [None for nn in range(nmodels)]
        for ll in range(0, model_eval_id):
            t_blocks[ll] = (
                self.off_diagonal_kernel_block(
                    samples[ll], XX1, kernels,
                    scalings_XX2, ll, model_eval_id, nmodels, False,
                    X_kk_is_test=True, scalings2=scalings_XX1)[0].T)
        t_blocks[model_eval_id] = self.diagonal_kernel_block(
            XX1, kernels, scalings_XX1, model_eval_id, nmodels, False,
            samples, scalings_XX2)[0]
        for ll in range(model_eval_id+1, nmodels):
            t_blocks[ll] = (
                self.off_diagonal_kernel_block(
                    XX1, samples[ll], kernels, scalings_XX1, model_eval_id, ll,
                    nmodels, False, X_kk_is_test=True,
                    scalings2=scalings_XX2)[0])
        return np.hstack(t_blocks)

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                theta.append(params[hyperparameter.name])
        if len(theta) > 0:
            return np.log(np.hstack(theta))
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        ii = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue

            #TODO do not apply exp for scaling update theta property as well
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                params[hyperparameter.name] = np.exp(
                    theta[ii:ii + hyperparameter.n_elements]
                )
                ii += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = np.exp(theta[ii])
                ii += 1

        if ii != len(theta):
            raise ValueError(
                "theta has not the correct number of entries."
                " Should be %d; given are %d" % (ii, len(theta))
            )
        self.set_params(**params)

    def _set_kernel_hyperparameters(self):
        length_scales = np.atleast_1d(self.length_scale).astype(float)
        sigma = np.atleast_1d(self.sigma).astype(float)
        for kk in range(self.nmodels):
            self.kernels[kk].k2.length_scale = (
                length_scales[kk*self.nvars:(kk+1)*self.nvars])
            self.kernels[kk].k1.constant_value = sigma[kk]

    def _set_scaling_hyperparameters(self):
        rho = np.atleast_1d(self.rho)
        for kk in range(len(self.scalings)):
            idx1 = self.nhyperparams_per_scaling[:kk].sum()
            idx2 = idx1 + self.nhyperparams_per_scaling[kk]
            self.scalings[kk].set_params(rho[idx1:idx2, None])

    def _get_scalings(self, XX1, eval_gradient):
        result = []
        self._set_scaling_hyperparameters()
        for kk in range(len(self.scalings)):
            result.append(self.scalings[kk](XX1, eval_gradient))
        return result

    def __call__(self, XX1, XX2=None, eval_gradient=False):
        XX1 = np.atleast_2d(XX1)
        # take self.theta and update internal structures appropriately
        self._set_kernel_hyperparameters()
        scalings_XX1 = self._get_scalings(XX1, eval_gradient)
        if self.nsamples_per_model is None:
            raise ValueError("Must call self.set_nsamples_per_model")
        if XX2 is None:
            K, K_grad = self._eval_K(
                XX1, self.nsamples_per_model, self.kernels, scalings_XX1,
                eval_gradient)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when XX2 is None.")
            scalings_XX2 = self._get_scalings(XX2, eval_gradient)
            K = self._eval_t(
                XX1, XX2, self.nsamples_per_model, self.kernels, scalings_XX1,
                scalings_XX2, self.model_eval_id)
        if not eval_gradient:
            return K
        else:
            return K, K_grad

    def diag(self, X):
        scalings = np.atleast_1d(self.rho)
        # D1 = self._diagonal_kernel_block(X, self.kernels, scalings,
        #                                  self.nmodels, False)[0]
        # D1 = np.diag(D1).copy()

        # compute diag for high_fidelity model
        kk = self.model_eval_id+1
        D = self.kernels[kk-1].diag(X)
        for nn in range(kk-1):
            const = self._sprod(scalings, nn, kk-1)**2
            D += const*self.kernels[nn].diag(X)
            # assert np.allclose(D, D1)
        return D

    @property
    def hyperparameter_rho(self):
        if np.iterable(self.rho):
            return Hyperparameter(
                "rho", "numeric", self.rho_bounds, len(self.rho))
        return Hyperparameter(
            "rho", "numeric", self.rho_bounds)

    def _rho_repr(self):
        if np.iterable(self.rho):
            return "[{0}]".format(
                ", ".join(map("{0:.3g}".format, self.rho)),
            )
        else:
            return "{0:.3g}".format(np.ravel(self.rho)[0])

    @property
    def hyperparameter_sigma(self):
        if np.iterable(self.sigma):
            return Hyperparameter(
                "sigma", "numeric", self.sigma_bounds, len(self.sigma))
        return Hyperparameter(
            "sigma", "numeric", self.sigma_bounds)

    def _sigma_repr(self):
        if np.iterable(self.sigma):
            return "[{0}]".format(
                ", ".join(map("{0:.3g}".format, self.sigma)),
            )
        else:
            return "{0:.3g}".format(np.ravel(self.sigma)[0])

    def __repr__(self):
        return "{0}(rho={1}, sigma={2}, length_scale={3})".format(
            self.__class__.__name__, self._rho_repr(), self._sigma_repr(),
            self._length_scale_repr()
        )

    # def _integrate_tau_P_1d(self, xx_1d, ww_1d, xtr,
    #                         scalings_XX1, scalings_XX2):
    #     K = self._eval_t(xx_1d.T, xtr.T, self.nsamples_per_model, self.kernels,
    #                      scalings_XX1, scalings_XX2, self.model_eval_id)
    #     # K = self.__call__(xx_1d.T, xtr.T)
    #     tau = ww_1d.dot(K)
    #     P = K.T.dot(ww_1d[:, np.newaxis]*K)
    #     return tau, P

    # def _integrate_tau_P(self, variable, nquad_samples, X_train,
    #                      transform_quad_rules):
    #     var_trans = AffineTransform(variable)
    #     nvars = variable.num_vars()
    #     degrees = [nquad_samples]*nvars
    #     univariate_quad_rules = get_univariate_quadrature_rules_from_variable(
    #         variable, np.asarray(degrees)+1, True)

    #     kernels = [copy.deepcopy(k) for k in self.kernels]
    #     base_kernels = [
    #         extract_covariance_kernel(self.kernels[kk], [RBF], view=True)
    #         for kk in range(self.nmodels)]
    #     length_scales = [
    #         copy.deepcopy(base_kernels[kk].length_scale)
    #         for kk in range(self.nmodels)]

    #     tau_list, P_list = [], []
    #     for ii in range(nvars):
    #         # Get 1D quadrature rule
    #         xtr = X_train[ii:ii+1, :]
    #         xx_1d, ww_1d = univariate_quad_rules[ii](degrees[ii]+1)
    #         scalings_XX2 = self._get_scalings(xtr.T, False)
    #         scalings_XX1 = self._get_scalings(xx_1d[:, None], False)
    #         xx_1d = xx_1d[None, :]
    #         if transform_quad_rules:
    #             xx_1d = var_trans.map_from_canonical_1d(xx_1d, ii)
    #         for kk in range(self.nmodels):
    #             base_kernels[kk].length_scale = np.atleast_1d(length_scales[kk])[ii]
    #         tau_ii, P_ii = self._integrate_tau_P_1d(
    #             xx_1d, ww_1d, xtr, scalings_XX1, scalings_XX2)
    #         tau_list.append(tau_ii)
    #         P_list.append(P_ii)
    #     self.kernels = kernels
    #     for kk in range(self.nmodels):
    #         base_kernels[kk].length_scale = length_scales[kk]
    #     tau = np.prod(np.array(tau_list), axis=0)
    #     P = np.prod(np.array(P_list), axis=0)
    #     return tau, P


class MultifidelityPeerKernel(MultilevelKernel):
    """
    Highest fidelity model at index nmodels-1
    """
    def diagonal_kernel_block(self, samples1,  kernels, scalings1, kk,
                              nmodels, eval_gradient,
                              samples2=None, scalings2=None):
        if samples2 is None:
            return self._diagonal_kernel_block(
                samples1[kk], None, kernels, scalings1, None, kk,
                nmodels, eval_gradient)
        return self._diagonal_kernel_block(
            samples1, samples2[kk], kernels, scalings1, scalings2, kk,
            nmodels, eval_gradient)

    def _diagonal_kernel_block(self, XX1, XX2, kernels, scalings1, scalings2,
                               model_id, nmodels, eval_gradient):
        """
        XX1 and XX2 are samples only for kernel[model_id]
        """
        K_block, K_block_grad = 0, np.nan
        if eval_gradient:
            if XX2 is None:
                XX2_shape = XX1.shape[0]
            else:
                XX2_shape = XX2.shape[0]
            K_block_grad = np.zeros((XX1.shape[0], XX2_shape, self.n_dims))

        K_block, K_block_grad_nn = self._eval_kernel(
            kernels[model_id], XX1, XX2, eval_gradient)

        if eval_gradient:
            if self.sigma_bounds != "fixed":
                idx1, idx2 = self._sigma_hyperparam_indices(model_id)
                K_block_grad[..., idx1:idx2] = K_block_grad_nn[..., :idx2-idx1]
            else:
                idx1, idx2 = 0, 0
            if self.length_scale_bounds != "fixed":
                idx3, idx4 = self._kernel_hyperparam_indices(model_id)
                K_block_grad[..., idx3:idx4] = K_block_grad_nn[..., idx2-idx1:]

        if model_id < nmodels-1:
            return K_block, K_block_grad

        for nn in range(nmodels-1):
            K_block_nn, K_block_grad_nn = self._eval_kernel(
                kernels[nn], XX1, XX2, eval_gradient)
            # const = scalings[nn][0]**2
            if XX2 is not None:
                # assume XX1 is only for one model
                scaling1 = scalings1[nn][0]
                scaling2 = self._unpack_kernel_scalings(
                    scalings2[nn][0], self.nsamples_per_model, nmodels-1)
            else:
                scaling1 = self._unpack_kernel_scalings(
                    scalings1[nn][0], self.nsamples_per_model, nmodels-1)
                scaling2 = scaling1
            # const = scaling**2
            const = scaling1.dot(scaling2.T)
            K_block += const*K_block_nn
            if not eval_gradient:
                continue
            if self.sigma_bounds != "fixed":
                idx1, idx2 = self._sigma_hyperparam_indices(nn)
                K_block_grad[..., idx1:idx2] = (
                    const[..., None]*K_block_grad_nn[..., :idx2-idx1])
            else:
                idx1, idx2 = 0, 0
            if self.length_scale_bounds != "fixed":
                # length_scale grad
                idx3, idx4 = self._kernel_hyperparam_indices(nn)
                K_block_grad[..., idx3:idx4] = (
                    const[..., None]*K_block_grad_nn[..., idx2-idx1:])

            if self.rho_bounds == "fixed":
                continue
            # scalings grad
            assert XX2 is None
            idx1, idx2 = self._scaling_indices(nn)
            # d/dx exp(s_kk)*exp(s_kk) = d/dx exp(2*s_kk)
            # = 2*exp(2*s_kk)=2*const
            # const_grad = 2*const
            scaling_grad1 = self._unpack_kernel_scalings(
                scalings1[nn][1], self.nsamples_per_model, nmodels-1)
            const_grad = (
                np.einsum("ij,jkl->ikl", scaling1, scaling_grad1[None, :, :]) +
                np.einsum("ij,jkl->kil", scaling1, scaling_grad1[None, :, :]))
            # const_grad = np.stack(
            #     [scaling.dot(scaling_grad[None, :, kk]) +
            #      scaling_grad[:, None, kk].dot(scaling.T)
            #      for kk in range(scaling_grad.shape[-1])], axis=-1)
            # assert np.allclose(const_grad1, const_grad)
            K_block_grad[..., idx1:idx2] += (
                const_grad*K_block_nn[..., None])
        return K_block, K_block_grad

    def off_diagonal_kernel_block(self, XX1, XX2, kernels, scalings1, kk, ll,
                                  nmodels, eval_gradient, X_kk_is_test=False,
                                  scalings2=None):
        return self._off_diagonal_kernel_block(
            XX1, XX2, kernels, scalings1, kk, ll, nmodels, eval_gradient,
            X_kk_is_test, scalings2)

    def _off_diagonal_kernel_block(self, Xkk, Xll, kernels, scalings1, kk, ll,
                                   nmodels, eval_gradient, X_kk_is_test,
                                   scalings2):

        assert kk < ll
        if kk != nmodels-1 and ll != nmodels-1:
            return (np.zeros((Xkk.shape[0], Xll.shape[0])),
                    np.zeros((Xkk.shape[0], Xll.shape[0], self.n_dims)))

        K_block, K_block_grad = 0, np.nan
        if eval_gradient:
            K_block_grad = np.zeros(
                (Xkk.shape[0], Xll.shape[0], self.n_dims))

        K, K_grad = self._eval_kernel(kernels[kk], Xkk, Xll, eval_gradient)
        if not X_kk_is_test:
            assert kk < ll
            const = self._unpack_kernel_scalings(
                scalings1[kk][0], self.nsamples_per_model, ll)
            K_block = K*const[:, 0]
        else:
            if self.model_eval_id < ll:
                const = self._unpack_kernel_scalings(
                    scalings2[kk][0], self.nsamples_per_model, ll)
                K_block = K*const[:, 0]
            else:
                const = scalings2[kk][0]
                K_block = K*const[:, 0]
        if not eval_gradient:
            return K_block, K_block_grad

        if self.sigma_bounds != "fixed":
            idx1, idx2 = self._sigma_hyperparam_indices(kk)
            K_block_grad[..., idx1:idx2] = np.einsum(
                "j,ijk->ijk", const[:, 0], K_grad[..., :idx2-idx1])
        else:
            idx1, idx2 = 0, 0
        if self.length_scale_bounds != "fixed":
            # length_scale grad
            idx3, idx4 = self._kernel_hyperparam_indices(kk)
            K_block_grad[..., idx3:idx4] = np.einsum(
                "j,ijk->ijk", const[:, 0], K_grad[..., idx2-idx1:])

        if self.rho_bounds != "fixed":
            # scalings grad
            idx1, idx2 = self._scaling_indices(kk)
            # d/dx exp(s_kk) = exp(s_kk) = const
            scaling_grad = self._unpack_kernel_scalings(
                scalings1[kk][1], self.nsamples_per_model, nmodels-1)
            K_block_grad[..., idx1:idx2] += np.einsum(
                "ij,jk->ijk", K, scaling_grad)
        return K_block, K_block_grad

    def diag(self, X):
        scalings = self._get_scalings(X, False)
        # compute diag for model_eval_id
        D = self.kernels[self.model_eval_id].diag(X)
        if self.model_eval_id != self.nmodels-1:
            return D

        for nn in range(self.nmodels-1):
            const = scalings[nn][0][:, 0]**2
            D += const*self.kernels[nn].diag(X)
        return D


class MultiTaskKernel(MultifidelityPeerKernel):
    """
    Shared model at index 0
    """
    def diagonal_kernel_block(self, samples1,  kernels, scalings1, kk,
                              nmodels, eval_gradient,
                              samples2=None, scalings2=None):
        if samples2 is None:
            return self._diagonal_kernel_block(
                samples1[kk], None, kernels, scalings1, kk,
                nmodels, eval_gradient)
        return self._diagonal_kernel_block(
            samples1, samples2[kk], kernels, scalings1, kk,
            nmodels, eval_gradient)

    def off_diagonal_kernel_block(self, XX1, XX2, kernels, scalings1, kk, ll,
                                  nmodels, eval_gradient, X_kk_is_test=False,
                                  scalings2=None):
        return self._off_diagonal_kernel_block(
            XX1, XX2, kernels, scalings1, kk, ll, nmodels, eval_gradient)

    def _diagonal_kernel_block(self, XX1, XX2, kernels, scalings,
                               model_id, nmodels, eval_gradient):
        # hack because only scalar scalings allowed
        scalings = [s[0][0] for s in scalings]
        nvars = XX1.shape[1]
        nhyperparams = nvars*nmodels+nmodels-1
        K_block, K_block_grad = 0, np.nan
        if eval_gradient:
            if XX2 is None:
                XX2_shape = XX1.shape[0]
            else:
                XX2_shape = XX2.shape[0]
            K_block_grad = np.zeros((XX1.shape[0], XX2_shape, self.n_dims))

        K_block, K_block_grad_nn = self._eval_kernel(
            kernels[model_id], XX1, XX2, eval_gradient)
        if eval_gradient:
            if self.sigma_bounds != "fixed":
                idx1, idx2 = self._sigma_hyperparam_indices(model_id)
                K_block_grad[..., idx1:idx2] = K_block_grad_nn[..., :idx2-idx1]
            else:
                idx1, idx2 = 0, 0
            if self.length_scale_bounds != "fixed":
                idx3, idx4 = self._kernel_hyperparam_indices(model_id)
                K_block_grad[..., idx3:idx4] = K_block_grad_nn[..., idx2-idx1:]

        if model_id == 0:
            return K_block, K_block_grad

        const = scalings[model_id-1]**2
        K_block_nn, K_block_grad_nn = self._eval_kernel(
            kernels[0], XX1, XX2, eval_gradient)
        K_block += const*K_block_nn
        if eval_gradient:
            if self.sigma_bounds != "fixed":
                idx1, idx2 = self._sigma_hyperparam_indices(0)
                K_block_grad[..., idx1:idx2] += (
                    const*K_block_grad_nn[..., :idx2-idx1])
            else:
                idx1, idx2 = 0, 0
            if self.length_scale_bounds != "fixed":
                # length_scale grad
                idx3, idx4 = self._kernel_hyperparam_indices(0)
                K_block_grad[..., idx3:idx4] += (
                    const*K_block_grad_nn[..., idx2-idx1:])
            if self.rho_bounds != "fixed":
                # scalings grad
                idx5, idx6 = self._scaling_indices(model_id-1)
                # idx1 = self.nkernel_hyperparams+(model_id-1)
                # d/dx exp(s_kk)*exp(s_kk) = d/dx exp(2*s_kk)
                # = 2*exp(2*s_kk)=2*const
                K_block_grad[..., idx5:idx6] += (
                    2*const*K_block_nn[..., None])
        return K_block, K_block_grad

    def _off_diagonal_kernel_block(self, Xkk, Xll, kernels, scalings, kk, ll,
                                   nmodels, eval_gradient):
        # hack because only scalar scalings allowed
        scalings = [s[0][0] for s in scalings]
        assert kk < ll

        K_block, K_block_grad = 0, np.nan
        if eval_gradient:
            K_block_grad = np.zeros(
                (Xkk.shape[0], Xll.shape[0], self.n_dims))

        if kk == 0:
            const = scalings[ll-1]
        else:
            const = scalings[kk-1]*scalings[ll-1]
        K, K_grad = self._eval_kernel(kernels[0], Xkk, Xll, eval_gradient)
        K_block = const*K
        if not eval_gradient:
            return K_block, K_block_grad

        if self.sigma_bounds != "fixed":
            idx1, idx2 = self._sigma_hyperparam_indices(0)
            K_block_grad[..., idx1:idx2] += const*K_grad[..., :idx2-idx1]
        else:
            idx1, idx2 = 0, 0
        if self.length_scale_bounds != "fixed":
            idx3, idx4 = self._kernel_hyperparam_indices(0)
            K_block_grad[..., idx3:idx4] += const*K_grad[..., idx2-idx1:]
        if self.rho_bounds != "fixed":
            # scalings grad
            if kk == 0:
                # d/dx exp(s_ll) = exp(s_ll) = const
                idx = self.nkernel_hyperparams+(ll-1)
                K_block_grad[..., idx:idx+1] += const*K[..., None]
            else:
                # d/dx exp(s_kk)*exp(s_ll) = exp(s_kk+s_ll) = const
                idx = self.nkernel_hyperparams+(kk-1)
                K_block_grad[..., idx:idx+1] += const*K[..., None]
                idx = self.nkernel_hyperparams+(ll-1)
                K_block_grad[..., idx:idx+1] += const*K[..., None]
        return K_block, K_block_grad

    def diag(self, X):
        scalings = np.atleast_1d(self.rho)
        # compute diag for model_eval_id
        D = self.kernels[self.model_eval_id].diag(X)
        if self.model_eval_id == 0:
            return D

        const = scalings[self.model_eval_id-1]**2
        D += const*self.kernels[0].diag(X)
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


class SequentialMultilevelKernel(RBF):
    def __init__(self, kernels, fixed_rho, length_scale=[1.0],
                 length_scale_bounds=(1e-5, 1e5), rho=1.0,
                 rho_bounds=(1e-10, 10)):
        super().__init__(length_scale, length_scale_bounds)
        for ii in range(len(kernels)-1):
            if kernels[ii].length_scale_bounds != "fixed":
                msg = "Hyperparameters of all but last kernel must be fixed"
                raise ValueError(msg)
        self.kernels = kernels
        # rho of the fixed kernels
        self.fixed_rho = fixed_rho
        self.rho = rho
        self.rho_bounds = rho_bounds

    def __call__(self, XX1, XX2=None, eval_gradient=False):
        length_scale = np.atleast_1d(self.length_scale).astype(float)
        assert length_scale.shape[0] == XX1.shape[1]
        rho = np.hstack((self.fixed_rho, [self.rho]))
        self.kernels[-1].length_scale = length_scale
        vals = 0
        for ii in range(len(self.kernels)):
            const = MultilevelKernel._sprod(
                rho, ii, len(self.kernels)-1)**2
            K = self.kernels[ii](XX1, XX2)
            vals += const*K
        if not eval_gradient:
            return vals

        raise NotImplementedError()

    @property
    def hyperparameter_rho(self):
        return Hyperparameter("rho", "numeric", self.rho_bounds)

    def __repr__(self):
        return "{0}(rho={1:.3g}, length_scale={2:.3g})".format(
            self.__class__.__name__, self.rho, self.length_scale
        )


def is_covariance_kernel(kernel, kernel_types):
    return (type(kernel) in kernel_types)


def extract_covariance_kernel(kernel, kernel_types, view=False):
    cov_kernel = None
    if is_covariance_kernel(kernel, kernel_types):
        if not view:
            return copy.deepcopy(kernel)
        return kernel
    if type(kernel) == Product or type(kernel) == Sum:
        cov_kernel = extract_covariance_kernel(kernel.k1, kernel_types, view)
        if cov_kernel is None:
            cov_kernel = extract_covariance_kernel(
                kernel.k2, kernel_types, view)
    if not view:
        return copy.deepcopy(cov_kernel)
    return cov_kernel

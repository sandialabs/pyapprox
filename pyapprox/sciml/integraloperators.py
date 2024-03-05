from abc import ABC, abstractmethod

import numpy as np

from pyapprox.sciml.util._torch_wrappers import (
    empty, inf, hstack, flip, cos, arange, diag, zeros, pi, sqrt, cfloat, conj,
    fft, ifft, fftshift, ifftshift, meshgrid, ones)
from pyapprox.sciml.util.hyperparameter import (
    HyperParameter, HyperParameterList, IdentityHyperParameterTransform)
from pyapprox.sciml.util import fct


class IntegralOperator(ABC):
    @abstractmethod
    def _integrate(self, y_k_samples):
        raise NotImplementedError

    def __call__(self, y_k_samples):
        return self._integrate(y_k_samples)

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, self._hyp_list._short_repr())


class KernelIntegralOperator(IntegralOperator):
    def __init__(self, kernel, quad_rule_k, quad_rule_kp1):
        self._kernel = kernel
        self._hyp_list = self._kernel.hyp_list
        self._quad_rule_k = quad_rule_k
        self._quad_rule_kp1 = quad_rule_kp1

    def _integrate(self, y_k_samples):
        z_k_samples, z_k_weights = self._quad_rule_k.get_samples_weights()
        z_kp1_samples = self._quad_rule_kp1.get_samples_weights()[0]
        K_mat = self._kernel(z_kp1_samples, z_k_samples)
        WK_mat = K_mat * z_k_weights[:, 0]  # equivalent to K @ diag(w)
        u_samples = WK_mat.double() @ y_k_samples.double()
        return u_samples


class DenseAffineIntegralOperator(IntegralOperator):
    def __init__(self, ninputs: int, noutputs: int, v0=None):
        '''
        Implements the usual fully connected layer of an MLP:

            u_{k+1} = W_k y_k + b_k

        where W_k is a 2D array of shape (N_{k+1}, N_k), y_k is a 1D array of
        shape (N_k,), and b_k is a 1D array of shape (N_{k+1},)
        '''
        self._ninputs = ninputs
        self._noutputs = noutputs
        nvars_mat = self._noutputs * (self._ninputs+1)

        weights_biases = self._default_values(nvars_mat, v0)
        bounds = self._default_bounds(nvars_mat)
        self._weights_biases = HyperParameter(
            "weights_biases", nvars_mat, weights_biases,
            bounds, IdentityHyperParameterTransform())

        self._hyp_list = HyperParameterList([self._weights_biases])

    def _default_values(self, nvars_mat, v0):
        weights_biases = np.empty((nvars_mat,), dtype=float)
        weights_biases[:] = (
            np.random.normal(0, 1, nvars_mat) if v0 is None else v0)
        return weights_biases

    def _default_bounds(self, nvars_mat):
        return np.tile([-np.inf, np.inf], nvars_mat)

    def _integrate(self, y_k_samples):
        mat = self._weights_biases.get_values().reshape(
            self._noutputs, self._ninputs+1)
        W = mat[:, :-1]
        b = mat[:, -1:]
        return W @ y_k_samples + b


class DenseAffineIntegralOperatorFixedBias(DenseAffineIntegralOperator):
    def __init__(self, ninputs: int, noutputs: int, v0=None):
        super().__init__(ninputs, noutputs, v0)

    def _default_values(self, nvars_mat, v0):
        weights_biases = super()._default_values(nvars_mat, v0)
        weights_biases[self._ninputs::self._ninputs+1] = 0.
        return weights_biases

    def _default_bounds(self, nvars_mat):
        bounds = super()._default_bounds(nvars_mat).reshape((nvars_mat, 2))
        bounds[self._ninputs::self._ninputs+1, 0] = np.nan
        bounds[self._ninputs::self._ninputs+1, 1] = np.nan
        return bounds.flatten()


class FourierConvolutionOperator(IntegralOperator):
    def __init__(self, kmax, shape=None, v0=None):
        """
        Parameters
        ----------
        kmax : integer
            The maximum retained frequency

        v0 : float
            the initial entries of the tensor representing the fourier
            transform of the implicitly defined kernel
        """
        self._kmax = kmax
        if hasattr(shape, '__iter__'):
            self._shape = tuple(shape)
        elif shape is None:
            self._shape = None
        else:
            self._shape = (shape,)
        self._d = 1 if self._shape is None else len(self._shape)

        # Use symmetry since target is real-valued.
        # 1 entry for constant, 2 for each mode between 1 and kmax
        v = empty(((2*self._kmax+1) ** self._d,)).numpy()
        v[:] = 0.0 if v0 is None else v0
        self._R = HyperParameter(
            'Fourier_R', v.size, v, [-inf, inf],
            IdentityHyperParameterTransform())
        self._hyp_list = HyperParameterList([self._R])

    def _integrate(self, y_k_samples):
        if self._shape is None:
            self._shape = (y_k_samples.shape[0],)
        kmax_lim = min(self._shape) // 2
        if self._kmax > kmax_lim:
            raise ValueError(
                'Maximum retained frequency too high; kmax must be <= '
                f'{kmax_lim}')
        nyquist = [n // 2 for n in self._shape]
        ntrain = y_k_samples.shape[-1]

        # Project onto modes -kmax, ..., 0, ..., kmax
        fft_y = fft(y_k_samples.reshape((*self._shape, ntrain)),
                    axis=list(range(self._d)))
        fftshift_y = fftshift(fft_y)
        freq_slices = [slice(n-self._kmax, n+self._kmax+1) for n in nyquist]
        fftshift_y_proj = fftshift_y[freq_slices]

        # Use symmetry c_{-n} = c_n, 1 <= n <= kmax
        v_float = self._hyp_list.get_values()
        v = zeros(((v_float.shape[0]+1) // 2,), dtype=cfloat)
        # v[n] = c_n,   0 <= n <= kmax
        v.real = v_float[:v.shape[0]]
        v.imag[1:] = v_float[v.shape[0]:]
        # R[n] = c_n,   -kmax <= n <= kmax
        R = hstack([flip(conj(v[1:]), dims=[0]), v])

        # Do convolution and lift into original spatial resolution
        conv_shift = diag(R) @ fftshift_y_proj.reshape(R.shape[0], ntrain)
        conv_shift = conv_shift.reshape(fftshift_y_proj.shape)
        conv_shift_lift = zeros(fftshift_y.shape, dtype=cfloat)
        conv_shift_lift[freq_slices] = conv_shift
        conv_lift = ifftshift(conv_shift_lift)
        res = ifft(conv_lift, axis=list(range(self._d))).real
        return res.reshape(y_k_samples.shape)


class ChebyshevConvolutionOperator(IntegralOperator):
    def __init__(self, kmax, shape=None, v0=None):
        # maximum retained degree
        self._kmax = kmax

        if hasattr(shape, '__iter__'):
            self._shape = tuple(shape)
        elif shape is None:
            self._shape = None
        else:
            self._shape = (shape,)
        self._d = 1 if self._shape is None else len(self._shape)

        # 1 entry for each mode between 0 and kmax
        v = empty(((self._kmax+1) ** self._d,)).numpy()
        v[:] = 0.0 if v0 is None else v0
        self._R = HyperParameter(
            'Chebyshev_R', v.size, v, [-inf, inf],
            IdentityHyperParameterTransform())
        self._hyp_list = HyperParameterList([self._R])
        self._fct_y = None
        self._N_tot = None
        self._W_tot_R = None
        self._W_tot_ifct = None

    def _precompute_weights(self):
        w_arr = []
        w_arr_ifct = []
        N_tot = 1
        for s in self._shape:
            w = fct.make_weights(self._kmax+1)
            w[-1] += (self._kmax != s-1)    # adjust final element
            w_arr.append(w)

            w_ifct = fct.make_weights(s)
            w_arr_ifct.append(w_ifct)

            N_tot *= 2*(s-1)

        W = meshgrid(*w_arr, indexing='ij')
        W_ifct = meshgrid(*w_arr_ifct, indexing='ij')
        W_tot = ones(W[0].shape)
        W_tot_ifct = ones(W_ifct[0].shape)
        for k in range(self._d):
            W_tot *= W[k]
            W_tot_ifct *= W_ifct[k]
        W_tot_ifct = fct.even_periodic_extension(W_tot_ifct[..., None])

        self._N_tot = N_tot
        self._W_tot_R = W_tot.flatten()
        self._W_tot_ifct = W_tot_ifct.flatten()

    def _integrate(self, y_k_samples):
        if self._shape is None:
            self._shape = (y_k_samples.shape[0],)
        kmax_lim = min(self._shape)-1
        ntrain = y_k_samples.shape[-1]
        if self._kmax > kmax_lim:
            raise ValueError(
                'Maximum retained degree too high; kmax must be <= '
                f'{kmax_lim}')

        # Project onto T_0, ..., T_{kmax}
        if self._fct_y is None:
            self._fct_y = fct.fct(y_k_samples.reshape((*self._shape, ntrain)))
        deg_slices = [slice(self._kmax+1) for k in self._shape]
        fct_y_proj = self._fct_y[deg_slices]

        # Construct convolution factor R; keep books on weights
        if self._W_tot_R is None:
            self._precompute_weights()
        R = empty(((self._kmax+1) ** self._d,))
        R[:] = self._N_tot/self._W_tot_R * self._hyp_list.get_values()

        # Do convolution and lift into original spatial resolution
        r_conv_y = diag(R) @ fct_y_proj.reshape((R.shape[0], ntrain))
        r_conv_y = r_conv_y.reshape(fct_y_proj.shape)
        conv_lift = zeros(self._fct_y.shape)
        conv_lift[deg_slices] = r_conv_y
        res = fct.ifct(conv_lift, W_tot=self._W_tot_ifct)
        return res.reshape(y_k_samples.shape)


class ChebyshevIntegralOperator(IntegralOperator):
    def __init__(self, kmax, shape=None, v0=None, nonzero_inds=None,
                 chol=False):
        r'''
        Compute

            .. math:: \int_{-1}^1 K(x,z) y(z) dz

        where :math:`x \in [-1,1]`, :math:`K(x,z) = w(x) \phi(x)^T A \phi(z)
        w(z)`, and

            .. math:: \phi_i(x) = T_i(x), \qquad i = 0, ..., k_\mathrm{max}

        '''
        # maximum retained degree
        self._kmax = kmax

        # A must be symmetric since K(x,z) = K(z,x), so only store the upper
        # triangle
        if nonzero_inds is None:
            # Upper triangle of symmetric matrix (row-major order)
            v = empty(((self._kmax+1)*(self._kmax+2)//2, )).numpy()
        else:
            # Sparse symmetric matrix, nonzero entries of upper triangle
            v = empty((nonzero_inds.shape[0], )).numpy()
        if chol:
            v[:] = 1.0 if v0 is None else v0
        else:
            v[:] = 0.0 if v0 is None else v0
        self._A = HyperParameter(
            'Chebyshev_A', v.size, v, [-inf, inf],
            IdentityHyperParameterTransform())
        self._hyp_list = HyperParameterList([self._A])
        self._nonzero_inds = nonzero_inds
        self._chol = chol

    def _integrate(self, y_k_samples):
        # Build A
        v = self._hyp_list.get_values()
        if self._nonzero_inds is None:
            cheb_U = v
        else:
            cheb_U = zeros(((self._kmax+1)*(self._kmax+2)//2, ))
            for i in range(self._nonzero_inds.shape[0]):
                cheb_U[self._nonzero_inds[i]] = v[i]
        U = zeros((self._kmax+1, self._kmax+1))
        diag_idx = range(self._kmax+1)
        c = 0
        for k in diag_idx:
            U[k, k:] = cheb_U[c:c+self._kmax+1-k]
            c += self._kmax+1-k
        if not self._chol:
            A = U + U.T
            A[diag_idx, diag_idx] = U[diag_idx, diag_idx]

        n = y_k_samples.shape[0]
        z_k_samples = cos(pi*arange(n)/(n-1))
        Phi = fct.chebyshev_poly_basis(z_k_samples, self._kmax+1)

        # factor[n] = \int_{-1}^1 (T_n(x))^2 w(x) dx
        factor = zeros((self._kmax+1,))
        factor[0] = pi
        factor[1:] = pi/2
        fct_y = diag(factor) @ fct.fct(y_k_samples)[:self._kmax+1, :]

        # define weighting function w and avoid singularity
        w = 1.0 / (1e-14+sqrt(1-z_k_samples**2))
        w[0] = (w[1] + (z_k_samples[2] - z_k_samples[1]) / (z_k_samples[0]
                - z_k_samples[1]) * (w[2] - w[1]))
        w[-1] = w[0]
        if not self._chol:
            return diag(w) @ Phi.T @ (A @ fct_y)
        return diag(w) @ Phi.T @ (U.T @ (U @ fct_y))

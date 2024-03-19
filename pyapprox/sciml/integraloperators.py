from abc import ABC, abstractmethod
import numpy as np
from pyapprox.sciml.util._torch_wrappers import (
    empty, inf, vstack, flip, cos, arange, diag, zeros, pi, sqrt, cfloat, conj,
    fft, ifft, fftshift, ifftshift, meshgrid, ones, einsum)
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

    def _format_nx(self, nx):
        if hasattr(nx, '__iter__'):
            self._nx = tuple(nx)
        elif nx is None:
            self._nx = None
        elif type(nx) == int:
            self._nx = (nx,)
        else:
            raise ValueError('nx must be int, tuple of ints, or None')


class EmbeddingOperator(IntegralOperator):
    def __init__(self, dim_in: int, dim_out: int, nx=None, v0=None):
        nvars_mat = dim_in*dim_out
        bounds = np.tile([-np.inf, np.inf], nvars_mat)
        W = np.empty((nvars_mat,), dtype=float)
        W[:] = (np.random.normal(0, 1/nvars_mat, nvars_mat) if v0 is None
                else v0)
        self._W = HyperParameter(
            "W_embedding", nvars_mat, W, bounds,
            IdentityHyperParameterTransform())
        self._hyp_list = HyperParameterList([self._W])
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._format_nx(nx)

    def _integrate(self, y_k_samples):
        if y_k_samples.ndim < 3:
            raise ValueError('y_k_samples must have shape (n_x, d_c, n_train)')
        if self._nx is None:
            self._format_nx(y_k_samples.shape[:-2])
        W = self._hyp_list.get_values().reshape(self._dim_out, self._dim_in)
        # y_k_flat = y_k_samples.reshape(self._N, self._dim_in,
        #                                y_k_samples.shape[-1])
        return einsum('ij,...jl->...il', W, y_k_samples)

    def _format_nx(self, nx):
        if hasattr(nx, '__iter__'):
            self._nx = tuple(nx)
        elif type(nx) == int:
            self._nx = (nx,)
        elif nx is None:
            self._nx = None
        else:
            raise ValueError('nx must be int, tuple of ints, or None')
        self._N = 1
        for n in self._nx:
            self._N *= n


class AffineChannelOperator(EmbeddingOperator):
    def __init__(self, d_c=1, nx=None, v0=None):
        super().__init__(d_c, d_c, nx, v0)


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
    def __init__(self, kmax, nx=None, v0=None, channel_in=1, channel_out=1):
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
        self._format_nx(nx)
        self._d = 1 if self._nx is None else len(self._nx)
        self._channel_in = channel_in
        self._channel_out = channel_out

        # Use symmetry since target is real-valued.
        # 1 entry for constant, 2 for each mode between 1 and kmax
        v = empty((self._channel_in * self._channel_out *
                  (2*self._kmax+1)**self._d,)).numpy()
        v[:] = 0.0 if v0 is None else v0
        self._R = HyperParameter(
            'Fourier_R', v.size, v, [-inf, inf],
            IdentityHyperParameterTransform())
        self._hyp_list = HyperParameterList([self._R])

    def _integrate(self, y_k_samples):
        # If channel_in is not explicit in y_k_samples, then assume
        # channel_in = 1. Otherwise, raise error.
        channel_implicit = False
        if y_k_samples.shape[-2] != self._channel_in:
            if self._channel_in == 1:
                channel_implicit = True
                y_k_samples = y_k_samples[..., None, :]
            else:
                raise ValueError(
                    'Could not infer channel dimension. y_k_samples.shape[-2] '
                    'must be channel_in.')

        # Bookkeeping on shape in case channel_dim is squeezed
        if not channel_implicit:
            output_shape = (*y_k_samples.shape[:-2], self._channel_out,
                            y_k_samples.shape[-1])
        else:
            output_shape = (*y_k_samples.shape[:-2], y_k_samples.shape[-1])

        # If nx was not specified at initialization
        if self._nx is None:
            self._nx = (*y_k_samples.shape[:-2],)

        # Enforce limits on kmax
        kmax_lim = min(self._nx) // 2
        if self._kmax > kmax_lim:
            raise ValueError(
                'Maximum retained frequency too high; kmax must be <= '
                f'{kmax_lim}')
        nyquist = [n // 2 for n in self._nx]
        ntrain = y_k_samples.shape[-1]

        # Project onto modes -kmax, ..., 0, ..., kmax
        fft_y = fft(y_k_samples.reshape((*self._nx, self._channel_in, ntrain)),
                    axis=list(range(self._d)))

        fftshift_y = fftshift(fft_y, axis=list(range(self._d)))
        freq_slices = [slice(n-self._kmax, n+self._kmax+1) for n in nyquist]
        fftshift_y_proj = fftshift_y[freq_slices]

        # Use symmetry c_{-n} = c_n, 1 <= n <= kmax
        v_float = self._hyp_list.get_values()
        v = zeros(((1+(2*self._kmax+1)**self._d) // 2, self._channel_out,
                   self._channel_in), dtype=cfloat)

        # v[n] = c_n,   0 <= n <= kmax
        real_imag_cutoff = v.shape[0] * self._channel_in * self._channel_out
        v.real.flatten()[:] = v_float[:real_imag_cutoff]
        v.imag[1:, ...].flatten()[:] = v_float[real_imag_cutoff:]

        # R[n, d_c, d_c] = c_n,   -kmax <= n <= kmax
        R = vstack([flip(conj(v[1:, ...]), dims=[0]), v])

        # Do convolution and lift into original spatial resolution
        fftshift_y_proj_flat = fftshift_y_proj.reshape(R.shape[0],
                                                       self._channel_in,
                                                       ntrain)
        conv_shift = einsum('ijk,ikl->ijl', R, fftshift_y_proj_flat)
        conv_shift = conv_shift.reshape((*fftshift_y_proj.shape[:-2],
                                         self._channel_out,
                                         ntrain))
        conv_shift_lift = zeros((*fft_y.shape[:-2], self._channel_out, ntrain),
                                dtype=cfloat)
        conv_shift_lift[freq_slices] = conv_shift
        conv_lift = ifftshift(conv_shift_lift, axis=list(range(self._d)))
        res = ifft(conv_lift, axis=list(range(self._d))).real
        return res.reshape(output_shape)


class ChebyshevConvolutionOperator(IntegralOperator):
    def __init__(self, kmax, nx=None, v0=None, channel_in=1, channel_out=1):
        # maximum retained degree
        self._kmax = kmax
        self._format_nx(nx)
        self._d = 1 if self._nx is None else len(self._nx)
        self._channel_in = channel_in
        self._channel_out = channel_out

        # 1 entry for each mode between 0 and kmax
        v = empty((channel_in * channel_out *
                  (self._kmax+1)**self._d,)).numpy()
        v[:] = 0.0 if v0 is None else v0
        self._R = HyperParameter(
            'Chebyshev_R', v.size, v, [-inf, inf],
            IdentityHyperParameterTransform())
        self._hyp_list = HyperParameterList([self._R])
        self._N_tot = None
        self._W_tot_R = None
        self._W_tot_ifct = None

    def _precompute_weights(self):
        w_arr = []
        w_arr_ifct = []
        N_tot = 1
        for s in self._nx:
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

        self._N_tot = N_tot
        self._W_tot_R = W_tot.flatten()
        self._W_tot_ifct = W_tot_ifct.flatten()

    def _integrate(self, y_k_samples):
        # If channel_in is not explicit in y_k_samples, then assume
        # channel_in = 1. Otherwise, raise error.
        channel_implicit = False
        if y_k_samples.shape[-2] != self._channel_in:
            if self._channel_in == 1:
                channel_implicit = True
                y_k_samples = y_k_samples[..., None, :]
            else:
                raise ValueError(
                    'Could not infer channel dimension. y_k_samples.shape[-2] '
                    'must be channel_in.')

        # Bookkeeping on shape in case channel_dim is squeezed
        if not channel_implicit:
            output_shape = (*y_k_samples.shape[:-2], self._channel_out,
                            y_k_samples.shape[-1])
        else:
            output_shape = (*y_k_samples.shape[:-2], y_k_samples.shape[-1])

        # If nx was not specified at initialization
        if self._nx is None:
            self._nx = (*y_k_samples.shape[:-2],)

        # kmax <= \min_k nx[k]-1
        kmax_lim = min(self._nx)-1
        ntrain = y_k_samples.shape[-1]
        if self._kmax > kmax_lim:
            raise ValueError(
                'Maximum retained degree too high; kmax must be <= '
                f'{kmax_lim}')

        # Project onto T_0, ..., T_{kmax}
        fct_y = fct.fct(y_k_samples.reshape((*self._nx, self._channel_in,
                                             ntrain)))
        deg_slices = [slice(self._kmax+1) for k in self._nx]
        fct_y_proj = fct_y[deg_slices]

        # Construct convolution factor R; keep books on weights
        if self._W_tot_R is None:
            self._precompute_weights()
        P = diag(self._N_tot / self._W_tot_R)
        fct_y_proj_flat = fct_y_proj.reshape(P.shape[0], self._channel_in,
                                             ntrain)
        fct_y_proj_precond = einsum('ij,jkl->ikl', P, fct_y_proj_flat)
        R = self._hyp_list.get_values().reshape(P.shape[0], self._channel_out,
                                                self._channel_in)

        # Do convolution and lift into original spatial resolution
        r_conv_y = einsum('ijk,ikl->ijl', R, fct_y_proj_precond)
        r_conv_y = r_conv_y.reshape((*fct_y_proj.shape[:-2], self._channel_out,
                                     ntrain))
        conv_lift = zeros((*fct_y.shape[:-2], self._channel_out,
                           fct_y.shape[-1]))
        conv_lift[deg_slices] = r_conv_y
        res = fct.ifct(conv_lift, W_tot=self._W_tot_ifct)
        return res.reshape(output_shape)


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

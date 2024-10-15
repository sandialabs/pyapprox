from abc import ABC, abstractmethod
import numpy as np
from pyapprox.util.hyperparameter import (
    HyperParameter, HyperParameterList, HyperParameterTransform,
    IdentityHyperParameterTransform)
from pyapprox.sciml.util import LinAlgMixin, TorchLinAlgMixin, FCT


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
    def __init__(self, integralops, channel_in: int, channel_out: int,
                 nx=None):
        self._channel_in = channel_in
        self._channel_out = channel_out
        if (isinstance(integralops, list) and
            all(issubclass(op.__class__, IntegralOperator)
                for op in integralops)):
            self._integralops = integralops
        elif issubclass(integralops.__class__, IntegralOperator):
            self._integralops = self._channel_out*[integralops]
        else:
            raise ValueError(
                'integralops must be IntegralOperator, or list '
                'thereof')
        self._hyp_list = sum([iop._hyp_list for iop in self._integralops])
        self._bkd = self._hyp_list._bkd

        # ensure proper setup
        assert len(self._integralops) == self._channel_out
        for iop in self._integralops:
            assert iop._channel_in == self._channel_in
            assert iop._channel_out == 1    # decoupled channels for now
        self._format_nx(nx)

    def _integrate(self, y_k_samples):
        if y_k_samples.ndim < 3:
            raise ValueError('y_k_samples must have shape (n_x, d_c, n_train)')
        if self._nx is None:
            self._format_nx(y_k_samples.shape[:-2])

        out = self._bkd.zeros((*self._nx, self._channel_out,
                              y_k_samples.shape[-1]))
        for k in range(self._channel_out):
            out[..., k, :] = self._integralops[k](y_k_samples)[..., 0, :]
        return out


class AffineProjectionOperator(IntegralOperator):
    def __init__(self, channel_in: int, v0=None, nx=None,
                 transform: HyperParameterTransform
                 = IdentityHyperParameterTransform,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        self._channel_in = channel_in
        self._channel_out = 1
        self._format_nx(nx)
        self._nvars_mat = self._channel_in + 1
        self._bkd = backend
        affine_weights = np.ones(self._nvars_mat)
        if v0 is not None:
            affine_weights[:] = np.copy(v0)
        else:
            affine_weights[-1] = 0.0
        self._affine_weights = HyperParameter(
            'affine_weights', self._nvars_mat, affine_weights,
            np.tile([-np.inf, np.inf], self._nvars_mat),
            transform(backend=self._bkd))
        self._hyp_list = HyperParameterList([self._affine_weights])
        self._format_nx(nx)

    def _integrate(self, y_k_samples):
        if y_k_samples.ndim < 3:
            raise ValueError('y_k_samples must have shape (n_x, d_c, n_train)')
        if self._nx is None:
            self._format_nx(y_k_samples.shape[:-2])
        out = self._bkd.einsum('i,...ik->...k',
                               self._hyp_list.get_values()[:-1],
                               y_k_samples) + self._hyp_list.get_values()[-1]
        return out[..., None, :]


class KernelIntegralOperator(IntegralOperator):
    def __init__(self, kernels, quad_rule_k, quad_rule_kp1, channel_in=1,
                 channel_out=1, backend: LinAlgMixin = TorchLinAlgMixin):
        self._bkd = backend
        if not hasattr(kernels, '__iter__'):
            self._kernels = channel_in*[kernels]
            self._hyp_list = kernels.hyp_list
        elif len(kernels) != channel_in:
            raise ValueError('len(kernels) must equal channel_in')
        else:
            self._kernels = kernels
            self._hyp_list = sum([kernel.hyp_list for kernel in kernels])

        self._channel_in = channel_in
        self._channel_out = channel_out
        self._quad_rule_k = quad_rule_k
        self._quad_rule_kp1 = quad_rule_kp1

    def _integrate(self, y_k_samples):
        # Apply matvec to each channel in parallel
        z_k_samples, w_k = self._quad_rule_k()
        z_kp1_samples = self._quad_rule_kp1()[0]
        self._WK_mat = self._bkd.zeros((z_kp1_samples.shape[1],
                                        z_k_samples.shape[1],
                                        len(self._kernels)))
        for k in range(len(self._kernels)):
            self._WK_mat[..., k] = (
                self._kernels[k](z_kp1_samples, z_k_samples) * w_k[:, 0])

        u_samples = self._bkd.einsum('ijk,jk...->ik...', self._WK_mat.double(),
                                     y_k_samples.double())
        return u_samples


class DenseAffineIntegralOperator(IntegralOperator):
    def __init__(self, ninputs: int, noutputs: int, v0=None, channel_in=1,
                 channel_out=1, transform: HyperParameterTransform
                 = IdentityHyperParameterTransform,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        r"""
        Implements the usual fully connected layer of an MLP:

            u_{k+1} = W_k y_k + b_k         (single channel)

        where W_k is a 2D array of shape (N_{k+1}, N_k), y_k is a 1D array of
        shape (N_k,), and b_k is a 1D array of shape (N_{k+1},).

        In continuous form,

            u_{k+1}(z_{k+1}, c_{k+1}) = \int_{D_k} \int_{D'_k} K(z_{k+1}, z_k;
                c_{k+1}, c_k) y_k(z_k, c_k) d(c_k) d(z_k)

        where c is the channel variable.
        """
        self._ninputs = ninputs
        self._noutputs = noutputs
        self._channel_in = channel_in
        self._channel_out = channel_out
        self._b_size = self._noutputs*self._channel_out
        self._nvars_mat = (self._noutputs * self._channel_out * (
                           self._ninputs * self._channel_in + 1))
        self._bkd = backend
        weights_biases = self._default_values(v0)
        bounds = self._default_bounds()
        self._weights_biases = HyperParameter(
            "weights_biases", self._nvars_mat, weights_biases, bounds,
            transform(backend=self._bkd))

        self._hyp_list = HyperParameterList([self._weights_biases])

    def _default_values(self, v0):
        weights_biases = np.empty((self._nvars_mat,), dtype=float)
        weights_biases[:] = (
            np.random.normal(0, 1, self._nvars_mat) if v0 is None else
            np.copy(v0))
        return weights_biases

    def _default_bounds(self):
        return np.tile([-np.inf, np.inf], self._nvars_mat)

    def _integrate(self, y_k_samples):
        if y_k_samples.ndim < 3:
            y_k_samples = y_k_samples[..., None, :]
        if y_k_samples.shape[-2] != self._channel_in:
            if self._channel_in == 1:
                y_k_samples = y_k_samples[..., None, :]
            else:
                raise ValueError(
                    'Could not infer channel dimension. y_k_samples.shape[-2] '
                    'must be channel_in.')

        W = (self._weights_biases.get_values()[:-self._b_size].reshape(
             self._noutputs, self._ninputs, self._channel_out,
             self._channel_in))
        b = (self._weights_biases.get_values()[-self._b_size:].reshape(
             self._noutputs, self._channel_out))
        if self._channel_in > 1 or self._channel_out > 1:
            return (self._bkd.einsum('ijkl,jlm->ikm', W, y_k_samples)
                    + b[..., None])
        else:
            # handle separately for speed
            return (W[..., 0, 0] @ y_k_samples[..., 0, :] + b)[..., None, :]


class DenseAffineIntegralOperatorFixedBias(DenseAffineIntegralOperator):
    def __init__(self, ninputs: int, noutputs: int, v0=None, channel_in=1,
                 channel_out=1, transform: HyperParameterTransform
                 = IdentityHyperParameterTransform,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        super().__init__(ninputs, noutputs, v0, channel_in, channel_out,
                         transform, backend)

    def _default_values(self, v0):
        weights_biases = super()._default_values(v0)
        weights_biases[-self._b_size:] = 0.
        return weights_biases

    def _default_bounds(self):
        bounds = super()._default_bounds().reshape(self._nvars_mat, 2)
        bounds[-self._b_size:, 0] = np.nan
        bounds[-self._b_size:, 1] = np.nan
        return bounds.flatten()


class DenseAffinePointwiseOperator(IntegralOperator):
    def __init__(self, v0=None, channel_in=1, channel_out=1,
                 transform: HyperParameterTransform
                 = IdentityHyperParameterTransform,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        r"""
        Implements a pointwise lifting/projection:

            u_{k+1} = W_k y_k + b_k

        where W_k is a 2D array of shape (channel_out, channel_in), y_k is a 1D
        array of shape (channel_in,), and b_k is a 1D array of shape
        (channel_out,).

        In continuous form,

            u(z, c_{k+1}) = \int_{D'_k) K(c_{k+1}, c_k) y_k(z, c_k) d(c_k)

        where c is the channel variable. This is analogous to
        DenseAffineIntegralOperator, but with \delta(z_k-z_{k+1}) inserted in
        the integral.
        """
        self._channel_in = channel_in
        self._channel_out = channel_out
        self._b_size = self._channel_out
        self._nvars_mat = (self._channel_out * (self._channel_in + 1))
        self._bkd = backend

        weights_biases = self._default_values(v0)
        bounds = self._default_bounds()
        self._weights_biases = HyperParameter(
            "weights_biases_ptwise", self._nvars_mat, weights_biases, bounds,
            transform(backend=self._bkd))

        self._hyp_list = HyperParameterList([self._weights_biases])

    def _default_values(self, v0):
        weights_biases = np.empty((self._nvars_mat,), dtype=float)
        weights_biases[:] = (
            np.random.normal(0, 1, self._nvars_mat) if v0 is None else
            np.copy(v0))
        return weights_biases

    def _default_bounds(self):
        return np.tile([-np.inf, np.inf], self._nvars_mat)

    def _integrate(self, y_k_samples):
        if y_k_samples.ndim < 3:
            y_k_samples = y_k_samples[..., None, :]
        if y_k_samples.shape[-2] != self._channel_in:
            if self._channel_in == 1:
                y_k_samples = y_k_samples[..., None, :]
            else:
                raise ValueError(
                    'Could not infer channel dimension. y_k_samples.shape[-2] '
                    'must be channel_in.')
        W = (self._weights_biases.get_values()[:-self._b_size].reshape(
             self._channel_out, self._channel_in))
        b = self._weights_biases.get_values()[-self._b_size:]
        return (self._bkd.einsum('ij,...jk->...ik', W, y_k_samples) +
                b[None, ..., None])


class DenseAffinePointwiseOperatorFixedBias(DenseAffinePointwiseOperator):
    def __init__(self, v0=None, channel_in=1, channel_out=1,
                 transform: HyperParameterTransform
                 = IdentityHyperParameterTransform,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        super().__init__(v0, channel_in, channel_out, transform, backend)

    def _default_values(self, v0):
        weights_biases = super()._default_values(v0)
        weights_biases[-self._b_size:] = 0.
        return weights_biases

    def _default_bounds(self):
        bounds = super()._default_bounds().reshape(self._nvars_mat, 2)
        bounds[-self._b_size:, 0] = np.nan
        bounds[-self._b_size:, 1] = np.nan
        return bounds.flatten()


class Reshape(IntegralOperator):
    def __init__(self, output_shape, backend: LinAlgMixin = TorchLinAlgMixin):
        if not hasattr(output_shape, '__iter__'):
            raise ValueError('output_shape must be iterable')
        self._bkd = backend
        self._hyps = HyperParameter(
            "reshape", 0, np.asarray([]), np.asarray([np.nan, np.nan]),
            IdentityHyperParameterTransform(backend=self._bkd))
        self._hyp_list = HyperParameterList([self._hyps])
        self._output_shape = output_shape

    def _integrate(self, y_k_samples):
        nsamples = y_k_samples.shape[-1]
        return y_k_samples.reshape(*self._output_shape, nsamples)


class BaseFourierOperator(IntegralOperator):
    def __init__(self, kmax, nx=None, v0=None, channel_in=1, channel_out=1,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        self._kmax = kmax
        self._format_nx(nx)
        self._d = 1 if self._nx is None else len(self._nx)
        self._channel_in = channel_in
        self._channel_out = channel_out
        self._num_freqs = (self._kmax+1)**self._d
        self._num_coefs = (2*self._kmax+1)**self._d
        self._bkd = backend

    def _integrate(self, y_k_samples):
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
        fft_y = self._bkd.fft(y_k_samples.reshape((*self._nx, self._channel_in,
                                                   ntrain)))

        fftshift_y = self._bkd.fftshift(fft_y)
        freq_slices = [slice(n-self._kmax, n+self._kmax+1) for n in nyquist]
        fftshift_y_proj = self._bkd.get_slices(fftshift_y, freq_slices)

        R, summation_str = self._form_operator()

        # Do convolution and lift into original spatial resolution
        conv_shift = self._bkd.einsum(summation_str, R,
                                      fftshift_y_proj.reshape(self._num_coefs,
                                                              self._channel_in,
                                                              ntrain))
        conv_shift = conv_shift.reshape(*fftshift_y_proj.shape[:-2],
                                        self._channel_out, ntrain)
        conv_shift_lift = self._bkd.zeros((*fft_y.shape[:-2],
                                           self._channel_out, ntrain),
                                          dtype=self._bkd.cfloat())
        if self._bkd == TorchLinAlgMixin:
            conv_shift_lift[freq_slices] = conv_shift
        else:
            conv_shift_lift[*freq_slices] = conv_shift
        conv_lift = self._bkd.ifftshift(conv_shift_lift)
        res = self._bkd.ifft(conv_lift).real
        return res.reshape(output_shape)


class FourierHSOperator(BaseFourierOperator):
    def __init__(self, kmax, nx=None, v0=None, channel_in=1, channel_out=1,
                 channel_coupling='full',
                 transform: HyperParameterTransform
                 = IdentityHyperParameterTransform,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        """
        Dense coupling in space (non-radial kernel). Not tested for spatial
        dimension > 1

        Parameters
        ----------
        kmax : integer
            The maximum retained frequency

        nx : int or tuple of ints
            Spatial discretization

        v0 : array of floats
            The initial entries of the tensor representing the fourier
            transform of the implicitly defined kernel

        channel_in : int
            Channel dimension of inputs

        channel_out : int
            Channel dimension of outputs

        channel_coupling : str
            'full' : dense matrix (fully coupled channels)
            'diag' : diagonal matrix (fully decoupled channels)
        """

        super().__init__(kmax=kmax, nx=nx, v0=v0, channel_in=channel_in,
                         channel_out=channel_out, backend=backend)

        if channel_coupling.lower() not in ['full', 'diag']:
            raise ValueError("channel_coupling must be 'full' or 'diag'")
        self._channel_coupling = channel_coupling.lower()

        # Use conjugate symmetry since target is real-valued.
        # 1 entry for constant, 2 for each mode between 1 and kmax
        self._channel_factor = (self._channel_in*self._channel_out
                                if self._channel_coupling == 'full' else
                                self._channel_in)
        v = self._bkd.empty(((2*self._num_freqs**2-1) * self._channel_factor,))
        v = self._bkd.to_numpy(v)
        v[:] = 0.0 if v0 is None else np.copy(v0)
        self._R = HyperParameter(
            'FourierHS_Operator', v.size, v,
            [-self._bkd.inf(), self._bkd.inf()],
            transform(backend=self._bkd))
        self._hyp_list = HyperParameterList([self._R])

    def _form_operator(self):
        v_float = self._hyp_list.get_values()
        if self._channel_coupling == 'full':
            v = self._bkd.zeros((self._num_coefs, self._num_coefs,
                                 self._channel_out, self._channel_in),
                                dtype=self._bkd.cfloat())
        else:
            v = self._bkd.zeros((self._num_coefs, self._num_coefs,
                                 self._channel_out),
                                dtype=self._bkd.cfloat())

        # With channel_in = channel_out = 1, we need
        #
        #       u_i = \sum_{j=-kmax}^{kmax} R_{ij} y_j
        #
        # to be conjugate-symmetric about i=0, and we need off-diagonal
        # elements of R to be Hermitian so that
        #
        #       K(x, y) = K(y, x)           (in the real part).
        #
        # Pumping through the algebra yields the construction below. Compared
        # to learning all R_{ij} independently, this reduces the number of
        # trainable parameters by a factor of 4.

        start = 0
        for i in range(self._kmax+1):
            stride = (2*self._kmax+1 - 2*i)*self._channel_factor
            cols = slice(i, 2*self._kmax+1-i)
            v[i, cols, ...].real.flatten()[:] = v_float[start:start+stride]
            if i < self._kmax:
                v[i, cols, ...].imag.flatten()[:] = v_float[start + stride:
                                                            start + 2*stride]
            start += 2*stride

        # Take Hermitian transpose in first two dimensions; torch operates on
        # last two dimensions by default
        v = self._bkd.transpose(v)
        A = v + self._bkd.tril(v, k=-1).mH
        Atilde = self._bkd.tril(self._bkd.flip(A, axis=(-2,)), k=-1)
        Atilde = self._bkd.flip(Atilde, axis=(-1,)).conj()
        R = A + Atilde
        R = self._bkd.transpose(R)
        summation_str = ('ijkl,jlm->ikm' if self._channel_coupling == 'full'
                         else 'ijk,jkm->ikm')
        return (R, summation_str)


class FourierConvolutionOperator(BaseFourierOperator):
    def __init__(self, kmax, nx=None, v0=None, channel_in=1, channel_out=1,
                 channel_coupling='full',
                 transform: HyperParameterTransform
                 = IdentityHyperParameterTransform,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        """
        Diagonal coupling in space (radial/convolutional kernel).

        Parameters
        ----------
        kmax : integer
            The maximum retained frequency

        nx : int or tuple of ints
            Spatial discretization

        v0 : array of floats
            The initial entries of the tensor representing the fourier
            transform of the implicitly defined kernel

        channel_in : int
            Channel dimension of inputs

        channel_out : int
            Channel dimension of outputs

        channel_coupling : str
            'full' : dense matrix (fully coupled channels)
            'diag' : diagonal matrix (fully decoupled channels)
        """

        super().__init__(kmax=kmax, nx=nx, v0=v0, channel_in=channel_in,
                         channel_out=channel_out, backend=backend)

        if channel_coupling.lower() not in ['full', 'diag']:
            raise ValueError("channel_coupling must be 'full' or 'diag'")
        self._channel_coupling = channel_coupling.lower()

        # Use symmetry since target is real-valued.
        # 1 entry for constant, 2 for each mode between 1 and kmax
        self._channel_factor = (self._channel_in*self._channel_out
                                if self._channel_coupling == 'full' else
                                self._channel_in)
        v = self._bkd.zeros((self._num_coefs * self._channel_factor,))
        v = self._bkd.to_numpy(v)
        v[:] = 0.0 if v0 is None else np.copy(v0)
        self._R = HyperParameter(
            'FourierConv_Operator', v.size, v,
            [-self._bkd.inf(), self._bkd.inf()],
            transform(backend=self._bkd))
        self._hyp_list = HyperParameterList([self._R])

    def _form_operator(self):
        if self._channel_coupling == 'full':
            v = self._bkd.zeros(((1+self._num_coefs)//2, self._channel_out,
                                self._channel_in), dtype=self._bkd.cfloat())
        else:
            v = self._bkd.zeros(((1+self._num_coefs)//2, self._channel_out),
                                dtype=self._bkd.cfloat())

        # Use symmetry c_{-n} = c_n, 1 <= n <= kmax
        v_float = self._hyp_list.get_values()

        # v[n] = c_n,   0 <= n <= kmax
        real_imag_cutoff = v.shape[0] * self._channel_factor
        v.real.flatten()[:] = v_float[:real_imag_cutoff]
        v.imag[1:, ...].flatten()[:] = v_float[real_imag_cutoff:]

        # R[n, d_c, d_c] = c_n,   -kmax <= n <= kmax
        R = self._bkd.vstack([self._bkd.flip(v[1:, ...].conj(), axis=(0,)), v])
        summation_str = ('ikl,ilm->ikm' if self._channel_coupling == 'full'
                         else 'ik,ikm->ikm')
        return (R, summation_str)


class ChebyshevConvolutionOperator(IntegralOperator):
    def __init__(self, kmax, nx=None, v0=None, channel_in=1, channel_out=1,
                 transform: HyperParameterTransform
                 = IdentityHyperParameterTransform,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        # maximum retained degree
        self._kmax = kmax
        self._format_nx(nx)
        self._d = 1 if self._nx is None else len(self._nx)
        self._channel_in = channel_in
        self._channel_out = channel_out
        self._bkd = backend

        # 1 entry for each mode between 0 and kmax
        v = self._bkd.zeros((channel_in * channel_out *
                             (self._kmax+1)**self._d,))
        v = self._bkd.to_numpy(v)
        v[:] = 0.0 if v0 is None else np.copy(v0)
        self._R = HyperParameter(
            'Chebyshev_R', v.size, v, [-self._bkd.inf(), self._bkd.inf()],
            transform(backend=self._bkd))
        self._hyp_list = HyperParameterList([self._R])
        self._N_tot = None
        self._W_tot_R = None
        self._W_tot_ifct = None
        self._fct = FCT(backend=self._bkd)

    def _precompute_weights(self):
        w_arr = []
        w_arr_ifct = []
        N_tot = 1
        for s in self._nx:
            w = self._fct.make_weights(self._kmax+1)
            w[-1] += (self._kmax != s-1)    # adjust final element
            w_arr.append(w)

            w_ifct = self._fct.make_weights(s)
            w_arr_ifct.append(w_ifct)

            N_tot *= 2*(s-1)

        W = self._bkd.meshgrid(*w_arr, indexing='ij')
        W_ifct = self._bkd.meshgrid(*w_arr_ifct, indexing='ij')
        W_tot = self._bkd.ones(W[0].shape)
        W_tot_ifct = self._bkd.ones(W_ifct[0].shape)
        for k in range(self._d):
            W_tot *= W[k]
            W_tot_ifct *= W_ifct[k]

        self._N_tot = N_tot
        self._W_tot_R = W_tot
        self._W_tot_ifct = W_tot_ifct

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
        fct_y = self._fct.fct(y_k_samples.reshape((*self._nx, self._channel_in,
                                                   ntrain)))
        deg_slices = [slice(self._kmax+1) for k in self._nx]
        fct_y_proj = self._bkd.get_slices(fct_y, deg_slices)

        # Construct convolution factor R; keep books on weights
        if self._W_tot_R is None:
            self._precompute_weights()
        P = self._N_tot / self._W_tot_R
        fct_y_proj_precond = self._bkd.einsum('...,...jk->...jk', P,
                                              fct_y_proj)
        R = self._hyp_list.get_values().reshape(*fct_y_proj.shape[:-2],
                                                self._channel_out,
                                                self._channel_in)

        # Do convolution and lift into original spatial resolution
        r_conv_y = self._bkd.einsum('...jk,...kl->...jl', R,
                                    fct_y_proj_precond)
        conv_lift = self._bkd.zeros((*self._nx, self._channel_out,
                                     fct_y.shape[-1]))
        if self._bkd == TorchLinAlgMixin:
            conv_lift[deg_slices] = r_conv_y.real
        else:
            conv_lift[*deg_slices] = r_conv_y.real
        res = self._fct.ifct(conv_lift, W_tot=self._W_tot_ifct)
        return res.reshape(output_shape)


class ChebyshevIntegralOperator(IntegralOperator):
    def __init__(self, kmax, shape=None, v0=None, nonzero_inds=None,
                 chol=False,
                 transform: HyperParameterTransform
                 = IdentityHyperParameterTransform,
                 backend: LinAlgMixin = TorchLinAlgMixin):
        r"""
        Compute

            .. math:: \int_{-1}^1 K(x,z) y(z) dz

        where :math:`x \in [-1,1]`, :math:`K(x,z) = w(x) \phi(x)^T A \phi(z)
        w(z)`, and

            .. math:: \phi_i(x) = T_i(x), \qquad i = 0, ..., k_\mathrm{max}

        """
        # maximum retained degree
        self._kmax = kmax

        self._bkd = backend
        self._fct = FCT(backend=self._bkd)

        # A must be symmetric since K(x,z) = K(z,x), so only store the upper
        # triangle
        if nonzero_inds is None:
            # Upper triangle of symmetric matrix (row-major order)
            v = self._bkd.zeros(((self._kmax+1)*(self._kmax+2)//2, ))
            v = self._bkd.to_numpy(v)
        else:
            # Sparse symmetric matrix, nonzero entries of upper triangle
            v = self._bkd.zeros((nonzero_inds.shape[0], ))
            v = self._bkd.to_numpy(v)
        if chol:
            v[:] = 1.0 if v0 is None else np.copy(v0)
        else:
            v[:] = 0.0 if v0 is None else np.copy(v0)
        self._A = HyperParameter(
            'Chebyshev_A', v.size, v, [-self._bkd.inf(), self._bkd.inf()],
            transform(backend=self._bkd))
        self._hyp_list = HyperParameterList([self._A])
        self._nonzero_inds = nonzero_inds
        self._chol = chol

    def _integrate(self, y_k_samples):
        # Build A
        v = self._hyp_list.get_values()
        if self._nonzero_inds is None:
            cheb_U = v
        else:
            cheb_U = self._bkd.zeros(((self._kmax+1)*(self._kmax+2)//2, ))
            for i in range(self._nonzero_inds.shape[0]):
                cheb_U[self._nonzero_inds[i]] = v[i]
        U = self._bkd.zeros((self._kmax+1, self._kmax+1))
        diag_idx = range(self._kmax+1)
        c = 0
        for k in diag_idx:
            U[k, k:] = cheb_U[c:c+self._kmax+1-k]
            c += self._kmax+1-k
        if not self._chol:
            A = U + U.T
            A[diag_idx, diag_idx] = U[diag_idx, diag_idx]

        n = y_k_samples.shape[0]
        pi = self._bkd.arccos(-self._bkd.ones(1))[0]
        z_k_samples = self._bkd.cos(pi*self._bkd.arange(n)/(n-1))
        Phi = self._fct.chebyshev_poly_basis(z_k_samples, self._kmax+1)
        fct_y_proj = self._fct.fct(y_k_samples)[:self._kmax+1, 0, :]

        # factor[n] = \int_{-1}^1 (T_n(x))^2 w(x) dx
        factor = self._bkd.zeros((self._kmax+1,))
        factor[0] = pi
        factor[1:] = pi/2
        fct_y_precond = self._bkd.diag(factor) @ fct_y_proj

        # define weighting function w and avoid singularity
        w = 1.0 / (1e-14 + self._bkd.sqrt(1-z_k_samples**2))
        w[0] = (w[1] + (z_k_samples[2] - z_k_samples[1]) / (z_k_samples[0]
                - z_k_samples[1]) * (w[2] - w[1]))
        w[-1] = w[0]
        if not self._chol:
            out = self._bkd.diag(w) @ Phi.T @ (A @ fct_y_precond)
        else:
            out = self._bkd.diag(w) @ Phi.T @ (U.T @ (U @ fct_y_precond))
        return out.reshape(y_k_samples.shape)

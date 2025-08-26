from abc import abstractmethod
from pyapprox.util.backends.template import BackendMixin
from pyapprox.util.backends.torch import TorchMixin


class FCT:
    def __init__(self, backend: BackendMixin = TorchMixin):
        self._bkd = backend

    def even_periodic_extension(self, array):
        '''
        Make even periodic extension along first ndim-2 axes of `array`
        '''
        Z = self._bkd.copy(array)
        if Z.ndim == 1:
            Z = Z[:, None, None]
        elif Z.ndim == 2:
            Z = Z[:, None, :]
        for k in range(Z.ndim-2):
            Z_extension = self._bkd.flip(Z, axis=(k,))
            Z_extension_trim = self._bkd.delete(Z_extension, [0, -1], axis=k)
            Z = self._bkd.concatenate([Z, Z_extension_trim], axis=k)
        return Z

    def fct(self, values, W_tot=None):
        '''
        coefs = fct(values)
            Fast Chebyshev transform of `values` along all axes except -1

        INPUTS:
            values: (n1, ..., nd, Ntrain) array
            W_tot:  optional, (n1*...*nd,) of precomputed DCT weights

        OUTPUTS:
            Chebyshev transform with shape `values.shape`
        '''

        v = self._bkd.zeros(values.shape)
        v[:] = values[:]
        if v.ndim == 1:
            v = v[:, None, None]
        elif v.ndim == 2:
            v = v[:, None, :]
        transform_shape = v.shape[:-1]
        N_tot = v[..., 0].flatten().shape[0]
        ntrain = v.shape[-1]
        slices = [slice(d) for d in v.shape]
        values_ext = self.even_periodic_extension(v)
        uhat = self._bkd.get_slices(self._bkd.ifft(values_ext), slices)
        if W_tot is None:
            W = self._bkd.meshgrid(*[self.make_weights(d)
                                     for d in transform_shape], indexing='ij')
            W_tot = self._bkd.ones(W[0].shape)
            for w in W:
                W_tot *= w
        W_cfloat = self._bkd.zeros(W_tot.shape, dtype=self._bkd.cfloat())
        W_cfloat[:].real = W_tot
        uhat = self._bkd.diag(W_cfloat.flatten()) @ uhat.reshape(N_tot, ntrain)
        return uhat.reshape(values.shape).real

    def ifct(self, coefs, W_tot=None):
        '''
        values = ifct(coefs)
            Inverse fast Chebyshev transform of `coefs` along all axes but -1

        INPUTS:
            coefs:  (n1, ..., nd, Ntrain) array
            W_tot:  optional, ((2(n1-1))*...*(2(nd-1)),) array of precomputed
                    even extension of IDCT weights

        OUTPUTS:
            Inverse Chebyshev transform with shape `coefs.shape`
        '''
        c = self._bkd.copy(coefs)
        if c.ndim == 1:
            c = c[:, None, None]
        elif c.ndim == 2:
            # explicit channel dim if d_c=1
            c = c[:, None, :]
        transform_shape = c.shape[:-2]
        slices = [slice(d) for d in c.shape]
        nx = c[..., 0, 0].flatten().shape[0]
        d_c = c.shape[-2]
        ntrain = c.shape[-1]
        if W_tot is None:
            W = self._bkd.meshgrid(*[self.make_weights(d)
                                     for d in transform_shape], indexing='ij')
            W_tot = self._bkd.ones(W[0].shape)
            for w in W:
                W_tot *= w
        P = self._bkd.diag(1.0 / W_tot.flatten())
        c = self._bkd.einsum('ij,jkl->ikl', P,
                             c.reshape(nx, d_c, ntrain)).reshape(c.shape)
        c_per = self.even_periodic_extension(c)
        u = self._bkd.fft(c_per).real
        return self._bkd.get_slices(u, slices).reshape(coefs.shape)

    def circ_conv(self, x, y):
        r'''
        z = circ_conv(x, y)
            Circular (periodic) convolution of x and y:
                z[i] = \sum_{j=0}^{N-1} x[j]*y[(i-j) mod N]

            Implementation does not use the FFT.

        INPUTS:
            x, y:   size-N 1D arraylike
        OUTPUTS:
            z:      size-N 1D arraylike
        '''
        n = x.shape[0]
        z = self._bkd.zeros((n,))
        for i in range(n):
            for j in range(n):
                z[i] += x.flatten()[j] * y.flatten()[(i-j) % n]
        return z.reshape(x.shape)

    def make_weights(self, n):
        '''
        Generate length-N vector of Chebyshev weights:

            [1, 2, 2, ..., 2, 1]
        '''
        w = self._bkd.zeros((n,))
        w[0] = 1
        w[1:-1] = 2
        w[-1] = 1
        return w

    def chebyshev_poly_basis(self, x, N):
        r'''
        Use the three-term recurrence relation to construct a 1D Chebyshev
        basis of degree N-1


        Parameters
        ----------
        x : array, shape (D,)
            Evaluation points of basis

        N : int (> 0)
            Number of basis elements
        '''
        xx = x.flatten()
        res = self._bkd.ones((N, xx.shape[0]), dtype=x.dtype)
        if N == 1:
            return res
        res[1, :] = xx[:]
        for k in range(1, N-1):
            res[k+1, :] = 2*xx*res[k, :] - res[k-1, :]
        return res

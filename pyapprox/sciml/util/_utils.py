from abc import ABC


class BaseUtilitiesSciML(ABC):
    def __init__(self):
        pass

    def _sciml_even_extension(self, mat):
        '''
        Make even periodic extension along first ndim-2 axes of `array`
        '''
        Z = self._la_copy(mat)
        if Z.ndim == 1:
            Z = Z[:, None, None]
        elif Z.ndim == 2:
            Z = Z[:, None, :]
        for k in range(Z.ndim-2):
            Z_extension = self._la_flip(Z, axis=[k])
            Z_extension_trim = self._la_delete(Z_extension, [0, -1], axis=k)
            Z = self._la_concatenate([Z, Z_extension_trim], axis=k)
        return Z

    def _sciml_fct(self, values, W_tot=None):
        '''
        coefs = fct(values)
            Fast Chebyshev transform of `values` along all axes except -1 and
            -2

        INPUTS:
            values: (n1, ..., nd, dc, nsamples) array
            W_tot:  optional, (n1*...*nd,) of precomputed DCT weights

        OUTPUTS:
            Chebyshev transform with shape `values.shape`
        '''

        v = self._la_empty(values.shape)
        v[...] = values[...]
        if v.ndim == 1:
            v = v[:, None, None]
        elif v.ndim == 2:
            v = v[:, None, :]
        transform_shape = v.shape[:-2]
        N_tot = v[..., 0, 0].flatten().shape[0]
        d_c = v.shape[-2]
        ntrain = v.shape[-1]
        slices = [slice(d) for d in v.shape]
        values_ext = self._sciml_even_extension(v)
        uhat = self._la_ifft(values_ext).real[slices]
        if W_tot is None:
            W = self._la_meshgrid(*[self._sciml_make_weights(d)
                                    for d in transform_shape])
            W_tot = self._la_empty(W[0].shape)
            W_tot[...] = 1.0
            for w in W:
                W_tot *= w
        P = self._la_diag(W_tot.flatten())
        uhat = self._la_einsum('ij,jkl->ikl', P,
                               uhat.reshape(N_tot, d_c, ntrain))
        return uhat.reshape(values.shape)

    def _sciml_ifct(self, coefs, W_tot=None):
        '''
        values = ifct(coefs)
            Inverse fast Chebyshev transform of `coefs` along all axes except
            -1 and -2

        INPUTS:
            coefs:  (n1, ..., nd, dc, nsamples) array
            W_tot:  optional, ((2(n1-1))*...*(2(nd-1)),) array of precomputed
                    even extension of IDCT weights

        OUTPUTS:
            Inverse Chebyshev transform with shape `coefs.shape`
        '''
        c = coefs.clone()
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
            W = self._la_meshgrid(*[self._sciml_make_weights(d)
                                    for d in transform_shape])
            W_tot = self._la_empty(W[0].shape)
            W_tot[...] = 1.0
            for w in W:
                W_tot *= w
        P = self._la_diag(1.0 / W_tot.flatten())
        c = self._la_einsum('ij,jkl->ikl', P,
                            c.reshape(nx, d_c, ntrain)).reshape(c.shape)
        c_per = self._sciml_even_extension(c)
        u = self._la_fft(c_per).real
        return u[slices].reshape(coefs.shape)

    def _sciml_circ_conv(self, x, y):
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
        z = self._la_empty((n,))
        z[:] = 0.0
        for i in range(n):
            for j in range(n):
                z[i] += x[j] * y[(i-j) % n]
        return z

    def _sciml_make_weights(self, n):
        '''
        Generate length-N vector of Chebyshev weights:

            [1, 2, 2, ..., 2, 1]
        '''
        w = self._la_empty((n,))
        w[0] = 1
        w[1:-1] = 2
        w[-1] = 1
        return w

    def _sciml_chebyshev_poly_basis(self, x, N):
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
        res = self._la_empty((N, xx.shape[0]))
        res[0, :] = 1.0
        if N == 1:
            return res
        res[1, :] = xx[:]
        for k in range(1, N-1):
            res[k+1, :] = 2*xx*res[k, :] - res[k-1, :]
        return res

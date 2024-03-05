from pyapprox.sciml.util._torch_wrappers import (
    fft, ifft, zeros, flip, ones, delete, cat, diagflat, meshgrid)


def even_periodic_extension(array):
    '''
    Make even periodic extension along all dimensions of `array` before -1
    '''
    Z = array.clone()
    if Z.ndim == 1:
        Z = Z[:, None]
    for k in range(Z.ndim-1):
        Z_extension = flip(Z, dims=[k])
        Z_extension_trim = delete(Z_extension, [0, -1], dim=k)
        Z = cat([Z, Z_extension_trim], dim=k)
    return Z


def fct(values, W_tot=None):
    '''
    coefs = fct(values)
        Fast Chebyshev transform of `values` along all axes except -1

    INPUTS:
        values: (n1, ..., nd, Ntrain) array
        W_tot:  optional, (n1*...*nd,) of precomputed DCT weights

    OUTPUTS:
        Chebyshev transform with shape `values.shape`
    '''

    v = zeros(values.shape)
    v[:] = values[:]
    if v.ndim == 1:
        v = v[:, None]
    transform_shape = v.shape[:-1]
    N_tot = v[..., 0].flatten().shape[0]
    ntrain = v.shape[-1]
    slices = [slice(d) for d in v.shape]
    values_ext = even_periodic_extension(v)
    uhat = ifft(values_ext).real[slices]
    if W_tot is None:
        W = meshgrid(*[make_weights(d) for d in transform_shape],
                     indexing='ij')
        W_tot = ones(W[0].shape)
        for w in W:
            W_tot *= w
    uhat = diagflat(W_tot) @ uhat.reshape((N_tot, ntrain))
    return uhat.reshape(values.shape)


def ifct(coefs, W_tot=None):
    '''
    values = ifct(coefs)
        Inverse fast Chebyshev transform of `coefs` along all axes except -1

    INPUTS:
        coefs:  (n1, ..., nd, Ntrain) array
        W_tot:  optional, ((2(n1-1))*...*(2(nd-1)),) array of precomputed even
                extension of IDCT weights

    OUTPUTS:
        Inverse Chebyshev transform with shape `coefs.shape`
    '''
    c = coefs.clone()
    if c.ndim == 1:
        c = c[:, None]
    transform_shape = c.shape[:-1]
    slices = [slice(d) for d in c.shape]
    c_per = even_periodic_extension(c)
    c_per_vec = c_per.reshape((c_per[..., 0].flatten().shape[0],
                               c_per.shape[-1]))
    if W_tot is None:
        W = meshgrid(*[make_weights(d) for d in transform_shape],
                     indexing='ij')
        W_tot_restricted = ones(W[0].shape)
        for w in W:
            W_tot_restricted *= w
        W_tot = even_periodic_extension(W_tot_restricted[..., None])
    c_per_vec = diagflat(1.0 / W_tot) @ c_per_vec
    c_per = c_per_vec.reshape(c_per.shape)
    u = fft(c_per).real
    return u[slices].reshape(coefs.shape)


def circ_conv(x, y):
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
    z = zeros((n,))
    for i in range(n):
        for j in range(n):
            z[i] += x[j] * y[(i-j) % n]
    return z


def make_weights(n):
    '''
    Generate length-N vector of Chebyshev weights:

        [1, 2, 2, ..., 2, 1]
    '''
    w = zeros((n,))
    w[0] = 1
    w[1:-1] = 2
    w[-1] = 1
    return w


def chebyshev_poly_basis(x, N):
    r'''
    Use the three-term recurrence relation to construct a 1D Chebyshev basis of
    degree N-1


    Parameters
    ----------
    x : array, shape (D,)
        Evaluation points of basis

    N : int (> 0)
        Number of basis elements
    '''
    xx = x.flatten()
    res = ones((N, xx.shape[0]))
    if N == 1:
        return res
    res[1, :] = xx[:]
    for k in range(1, N-1):
        res[k+1, :] = 2*xx*res[k, :] - res[k-1, :]
    return res

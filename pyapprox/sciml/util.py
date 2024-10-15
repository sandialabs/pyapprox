from abc import abstractmethod
from pyapprox.util.linearalgebra import linalgbase, numpylinalg, torchlinalg
import numpy as np
import torch


class LinAlgMixin(linalgbase.LinAlgMixin):
    def __init__(self):
        raise NotImplementedError("Do not instantiate this class")

    @staticmethod
    @abstractmethod
    def fft(mat, axis=None, **kwargs):
        """Compute fast Fourier transform of mat."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ifft(mat, axis=None, **kwargs):
        """Compute inverse fast Fourier transform of mat."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fftshift(mat, axis=None, **kwargs):
        """Re-index FFT so that mode 0 is in the center"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ifftshift(mat, axis=None, **kwargs):
        """Re-index inverse FFT"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def cfloat():
        """Returns native complex dtype (64-bit for each part)"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def delete(inds, axis=None):
        """Deletes sub-arrays along specified axis."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def transpose(self, mat, axis=None):
        """Returns transpose of mat along axis"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def concatenate(arrays, axis=0):
        """Concatenates arrays along specified axis"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def size(mat):
        """Returns number of elements in mat"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def random_seed(val):
        """Set random seed"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tril(mat, k=0):
        """Returns lower triangle of mat"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def triu(mat, k=0):
        """Returns upper triangle of mat"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def normal(mean, stdev, size=(1,), **kwargs):
        """Compute realizations of a normal distribution."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def uniform(lb, ub, size=(1,), **kwargs):
        """Compute realizations of a uniform distribution."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def nan():
        """Return native representation of nan."""
        raise NotImplementedError

    @staticmethod
    def detach(mat):
        """Detach a matrix from the computational graph.
        Override for backends that support automatic differentiation."""
        return mat

    @staticmethod
    @abstractmethod
    def get_slices(mat, slices):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def randperm(mat):
        raise NotImplementedError


class NumpyLinAlgMixin(LinAlgMixin, numpylinalg.NumpyLinAlgMixin):
    @staticmethod
    def fft(mat, axis=None, **kwargs):
        if mat.ndim < 3:
            raise ValueError('mat must explicitly express channel and sample '
                             'dimensions')
        _axis = list(range(mat.ndim-2)) if axis is None else axis
        return np.fft.fftn(mat, axes=_axis, **kwargs)

    @staticmethod
    def ifft(mat, axis=None, **kwargs):
        if mat.ndim < 3:
            raise ValueError('mat must explicitly express channel and sample '
                             'dimensions')
        _axis = list(range(mat.ndim-2)) if axis is None else axis
        return np.fft.ifftn(mat, axes=_axis, **kwargs)

    @staticmethod
    def fftshift(mat, axis=None, **kwargs):
        if mat.ndim < 3:
            raise ValueError('mat must explicitly express channel and sample '
                             'dimensions')
        _axis = list(range(mat.ndim-2)) if axis is None else axis
        return np.fft.fftshift(mat, axes=_axis, **kwargs)

    @staticmethod
    def ifftshift(mat, axis=None, **kwargs):
        if mat.ndim < 3:
            raise ValueError('mat must explicitly express channel and sample '
                             'dimensions')
        _axis = list(range(mat.ndim-2)) if axis is None else axis
        return np.fft.ifftshift(mat, axes=_axis, **kwargs)

    @staticmethod
    def cfloat():
        return np.complex128

    @staticmethod
    def delete(mat, inds, axis=None):
        return np.delete(mat, inds, axis=axis)

    @staticmethod
    def transpose(mat, axis=None):
        return np.transpose(mat, axes=axis)

    @staticmethod
    def concatenate(arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    @staticmethod
    def size(mat):
        return mat.size

    @staticmethod
    def random_seed(val):
        np.random.seed(val)

    @staticmethod
    def tril(mat, k=0):
        return np.tril(mat, k=k)

    @staticmethod
    def triu(mat, k=0):
        return np.triu(mat, k=k)

    @staticmethod
    def normal(mean, stdev, size=(1,), dtype=float):
        return np.random.normal(mean, stdev, size=size).astype(dtype)

    @staticmethod
    def uniform(lb, ub, size=(1,), dtype=float):
        return np.random.uniform(lb, ub, size=size).astype(dtype)

    @staticmethod
    def nan():
        return np.nan

    @staticmethod
    def get_slices(mat, slices):
        return mat[*slices]

    @staticmethod
    def randperm(n):
        return np.random.permutation(n)


class TorchLinAlgMixin(LinAlgMixin, torchlinalg.TorchLinAlgMixin):
    @staticmethod
    def fft(mat, axis=None, **kwargs):
        if mat.ndim < 3:
            raise ValueError('mat must explicitly express channel and sample '
                             'dimensions')
        _axis = list(range(mat.ndim-2)) if axis is None else axis
        return torch.fft.fftn(mat, dim=_axis, **kwargs)

    @staticmethod
    def ifft(mat, axis=None, **kwargs):
        if mat.ndim < 3:
            raise ValueError('mat must explicitly express channel and sample '
                             'dimensions')
        _axis = list(range(mat.ndim-2)) if axis is None else axis
        return torch.fft.ifftn(mat, dim=_axis, **kwargs)

    @staticmethod
    def fftshift(mat, axis=None, **kwargs):
        if mat.ndim < 3:
            raise ValueError('mat must explicitly express channel and sample '
                             'dimensions')
        _axis = list(range(mat.ndim-2)) if axis is None else axis
        return torch.fft.fftshift(mat, dim=_axis, **kwargs)

    @staticmethod
    def ifftshift(mat, axis=None, **kwargs):
        if mat.ndim < 3:
            raise ValueError('mat must explicitly express channel and sample '
                             'dimensions')
        _axis = list(range(mat.ndim-2)) if axis is None else axis
        return torch.fft.ifftshift(mat, dim=_axis, **kwargs)

    @staticmethod
    def cfloat():
        return torch.complex128

    @staticmethod
    def delete(mat, inds, axis=None):
        if axis is None:
            _arr = mat.flatten()
        else:
            _arr = mat
        skip = [i.item() for i in torch.arange(_arr.size(axis))[inds]]
        retained = [i.item() for i in torch.arange(_arr.size(axis))
                    if i not in skip]
        indices = [slice(None) if i != axis else retained
                   for i in range(_arr.ndim)]
        return _arr[indices]

    @staticmethod
    def transpose(mat, axis=None):
        if axis is None:
            axis = list(range(mat.ndim-1, -1, -1))
        if not hasattr(axis, '__iter__'):
            axis = [axis]
        return torch.permute(mat, dims=axis)

    @staticmethod
    def concatenate(arrays, axis=0):
        return torch.cat(arrays, dim=axis)

    @staticmethod
    def size(mat):
        return torch.numel(mat)

    @staticmethod
    def random_seed(val):
        torch.manual_seed(val)

    @staticmethod
    def tril(mat, k=0):
        return torch.tril(mat, diagonal=k)

    @staticmethod
    def triu(mat, k=0):
        return torch.triu(mat, diagonal=k)

    @staticmethod
    def normal(mean, stdev, size=(1,), dtype=float):
        return torch.normal(mean, stdev, size=size, dtype=dtype)

    @staticmethod
    def uniform(lb, ub, size=(1,), dtype=float):
        return torch.empty(size, dtype=dtype).uniform_(lb, ub)

    @staticmethod
    def nan():
        return torch.nan

    @staticmethod
    def detach(mat: torch.Tensor) -> torch.Tensor:
        return mat.detach()

    @staticmethod
    def get_slices(mat, slices):
        return mat[slices]

    @staticmethod
    def randperm(n):
        return torch.randperm(n)


class FCT:
    def __init__(self, backend: LinAlgMixin = TorchLinAlgMixin):
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

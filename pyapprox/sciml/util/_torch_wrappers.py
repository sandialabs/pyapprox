import torch
import numpy as np

# create wrappers for array operations so np and torch can be exchanged
array = torch.tensor
array_type = torch.Tensor
inf = torch.inf
pi = torch.pi
cfloat = torch.complex128

torch.set_default_dtype(torch.double)


def empty(*args, dtype=None):
    if dtype is None:
        dtype = torch.double
    return torch.empty(*args, dtype=dtype)


def full(*args, dtype=None):
    if dtype is None:
        dtype = torch.double
    return torch.full(*args, dtype=dtype)


def exp(array):
    return torch.exp(array)


def sqrt(array):
    return torch.sqrt(array)


def cos(array):
    return torch.cos(array)


def arccos(array):
    return torch.arccos(array)


def sin(array):
    return torch.sin(array)


def log(array):
    """Apply log element wise"""
    return torch.log(array)


def multidot(arrays):
    return torch.linalg.multi_dot(arrays)


def prod(array_list):
    return torch.prod(array_list)


def atleast1d(array, dtype=None):
    if dtype is None:
        dtype = torch.double
    return torch.atleast_1d(
        torch.as_tensor(array, dtype=dtype))


def hstack(arrays):
    return torch.hstack(arrays)


def vstack(arrays):
    return torch.vstack(arrays)


def arange(*args):
    return torch.arange(*args)


def ndim(array):
    return array.ndim


def repeat(array, nreps):
    # makes deep copies of array
    return array.repeat(nreps)


def cdist(X, Y):
    # equivalent to
    # scipy.spatial.distance.cdist(X, Y, metric="euclidean"))
    return torch.cdist(X, Y, p=2)


def asarray(array, dtype=None, requires_grad=False):
    if dtype is None:
        dtype = torch.double
    if not requires_grad:
        return torch.as_tensor(array, dtype=dtype)
    if isinstance(array, np.ndarray):
        return torch.tensor(array, dtype=dtype, requires_grad=requires_grad)
    return array.clone().detach().requires_grad_(True)


def isnan(array):
    return torch.isnan(array)


def cholesky(mat):
    return torch.linalg.cholesky(mat)


def cholesky_solve(chol_factor, rhs):
    return torch.cholesky_solve(rhs, chol_factor)


def solve_triangular(mat, rhs, upper=False):
    return torch.linalg.solve_triangular(mat, rhs, upper=upper)


def diag(mat):
    return torch.diag(mat)


def diagflat(array):
    return torch.diagflat(array)


def einsum(*args):
    return torch.einsum(*args)


def to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    return array.detach().numpy()


def copy(array):
    return array.clone()


def inv(matrix):
    return torch.linalg.inv(matrix)


def eye(nn, dtype=None):
    if dtype is None:
        dtype = torch.double
    return torch.eye(nn, dtype=dtype)


def trace(matrix):
    return torch.trace(matrix)


def solve(matrix, vec):
    return torch.linalg.solve(matrix, vec)


def pinv(matrix):
    return torch.linalg.pinv(matrix)


def tanh(array):
    return torch.tanh(array)


def get_diagonal(mat):
    # returns a view
    return torch.diagonal(mat)


def linspace(*args):
    return torch.linspace(*args)


def norm(*args, **kwargs):
    return torch.linalg.norm(*args, **kwargs)


def fft(array, **kwargs):
    # by default, transform over all but final axis
    if 'axis' not in kwargs.keys():
        kwargs['axis'] = list(range(array.ndim-1)) if array.ndim > 1 else [0]
    return torch.fft.fftn(array, **kwargs)


def ifft(array, **kwargs):
    # by default, transform over all but final axis
    if 'axis' not in kwargs.keys():
        kwargs['axis'] = list(range(array.ndim-1)) if array.ndim > 1 else [0]
    return torch.fft.ifftn(array, **kwargs)


def fftshift(array, **kwargs):
    return torch.fft.fftshift(array, **kwargs)


def ifftshift(array, **kwargs):
    return torch.fft.ifftshift(array, **kwargs)


def flip(array, **kwargs):
    return torch.flip(array, **kwargs)


def conj(array):
    return torch.conj(array)


def zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)


def maximum(*args):
    return torch.maximum(*args)


def erf(array):
    return torch.special.erf(array)


def randperm(n):
    return torch.randperm(n)


def cumsum(array, **kwargs):
    return torch.cumsum(array, **kwargs)


def delete(array, inds, dim=None):
    '''
    Functionality of np.delete
    '''
    if isinstance(array, np.ndarray):
        return np.delete(array, inds, axis=dim)

    if dim is None:
        _arr = array.flatten()
    else:
        _arr = array

    skip = [i.item() for i in torch.arange(_arr.size(dim))[inds]]  # for -1
    retained = [i.item() for i in torch.arange(_arr.size(dim))
                if i not in skip]
    indices = [slice(None) if i != dim else retained for i in range(_arr.ndim)]
    return _arr[indices]


def cat(array, **kwargs):
    return torch.cat(array, **kwargs)


def meshgrid(*args, **kwargs):
    return torch.meshgrid(*args, **kwargs)

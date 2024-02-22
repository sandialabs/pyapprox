import torch
import numpy as np

# create wrappers for array operations so np and torch can be exchanged
array = torch.tensor
array_type = torch.Tensor
inf = torch.inf

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


def asarray(array, dtype=None):
    if dtype is None:
        dtype = torch.double
    return torch.as_tensor(array, dtype=dtype)


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


def einsum(*args):
    return torch.einsum(*args)


def to_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    return array.numpy()


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


def floor(matrix):
    return torch.floor(matrix)

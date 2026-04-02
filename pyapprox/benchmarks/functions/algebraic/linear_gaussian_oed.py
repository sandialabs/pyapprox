"""Forward map builders for linear Gaussian OED benchmarks.

Provides pure functions that construct FunctionProtocol objects for
observation and QoI models used in linear Gaussian benchmarks.
"""

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.util.backends.protocols import Array, Backend


def _build_vandermonde(
    locations: Array,
    min_degree: int,
    degree: int,
    bkd: Backend[Array],
) -> Array:
    """Build polynomial Vandermonde matrix.

    Parameters
    ----------
    locations : Array
        Evaluation points. Shape: (n,)
    min_degree : int
        Minimum polynomial degree.
    degree : int
        Maximum polynomial degree.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    A : Array
        Vandermonde matrix. Shape: (n, degree - min_degree + 1)
        A[i, j] = locations[i]^(j + min_degree)
    """
    n = locations.shape[0]
    powers = bkd.arange(min_degree, degree + 1)
    x_col = bkd.reshape(locations, (n, 1))
    powers_row = bkd.reshape(powers, (1, len(powers)))
    return x_col**powers_row


def build_linear_obs_map(
    obs_locations: Array,
    min_degree: int,
    degree: int,
    bkd: Backend[Array],
) -> FunctionFromCallable[Array]:
    """Build linear observation map: y = A @ theta.

    Parameters
    ----------
    obs_locations : Array
        Observation locations. Shape: (nobs,)
    min_degree : int
        Minimum polynomial degree.
    degree : int
        Maximum polynomial degree.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    obs_map : FunctionFromCallable
        Maps theta (nparams, nsamples) -> y (nobs, nsamples).
    """
    A = _build_vandermonde(obs_locations, min_degree, degree, bkd)
    nobs = A.shape[0]
    nparams = A.shape[1]

    def _obs_fun(samples: Array) -> Array:
        return bkd.dot(A, samples)

    return FunctionFromCallable(nobs, nparams, _obs_fun, bkd)


def build_linear_qoi_map(
    qoi_locations: Array,
    min_degree: int,
    degree: int,
    bkd: Backend[Array],
) -> FunctionFromCallable[Array]:
    """Build linear QoI map: qoi = B @ theta.

    Parameters
    ----------
    qoi_locations : Array
        QoI prediction locations. Shape: (npred,)
    min_degree : int
        Minimum polynomial degree.
    degree : int
        Maximum polynomial degree.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    qoi_map : FunctionFromCallable
        Maps theta (nparams, nsamples) -> qoi (npred, nsamples).
    """
    B = _build_vandermonde(qoi_locations, min_degree, degree, bkd)
    npred = B.shape[0]
    nparams = B.shape[1]

    def _qoi_fun(samples: Array) -> Array:
        return bkd.dot(B, samples)

    return FunctionFromCallable(npred, nparams, _qoi_fun, bkd)


def build_exp_qoi_map(
    qoi_locations: Array,
    min_degree: int,
    degree: int,
    bkd: Backend[Array],
) -> FunctionFromCallable[Array]:
    """Build exponential QoI map: qoi = exp(B @ theta).

    Parameters
    ----------
    qoi_locations : Array
        QoI prediction locations. Shape: (npred,)
    min_degree : int
        Minimum polynomial degree.
    degree : int
        Maximum polynomial degree.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    qoi_map : FunctionFromCallable
        Maps theta (nparams, nsamples) -> qoi (npred, nsamples).
    """
    B = _build_vandermonde(qoi_locations, min_degree, degree, bkd)
    npred = B.shape[0]
    nparams = B.shape[1]

    def _qoi_fun(samples: Array) -> Array:
        return bkd.exp(bkd.dot(B, samples))

    return FunctionFromCallable(npred, nparams, _qoi_fun, bkd)



"""Forward map builders for linear Gaussian OED benchmarks.

Provides pure functions that construct FunctionProtocol objects for
observation and QoI models used in linear Gaussian benchmarks.
"""

from pyapprox.expdesign.benchmarks.problems.inference_problem import (
    GaussianInferenceProblem,
)
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
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


def build_linear_gaussian_inference_problem(
    nobs: int,
    degree: int,
    noise_std: float,
    prior_std: float,
    bkd: Backend[Array],
    min_degree: int = 0,
) -> GaussianInferenceProblem[Array]:
    """Build a GaussianInferenceProblem for linear Gaussian regression.

    Constructs a polynomial regression observation model with isotropic
    Gaussian prior and noise.

    Parameters
    ----------
    nobs : int
        Number of observation locations (equally spaced in [-1, 1]).
    degree : int
        Maximum polynomial degree.
    noise_std : float
        Standard deviation of observation noise.
    prior_std : float
        Standard deviation of prior on coefficients.
    bkd : Backend[Array]
        Computational backend.
    min_degree : int
        Minimum polynomial degree (default 0).

    Returns
    -------
    problem : GaussianInferenceProblem
        Configured inference problem.
    """
    obs_locations = bkd.linspace(-1.0, 1.0, nobs)
    obs_map = build_linear_obs_map(obs_locations, min_degree, degree, bkd)
    nparams = obs_map.nvars()

    prior_mean = bkd.zeros((nparams, 1))
    prior_covariance = bkd.eye(nparams) * prior_std**2
    prior = DenseCholeskyMultivariateGaussian(
        prior_mean, prior_covariance, bkd
    )

    noise_variances = bkd.full((nobs,), noise_std**2)

    return GaussianInferenceProblem(
        obs_map=obs_map,
        prior=prior,
        noise_variances=noise_variances,
        bkd=bkd,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
    )

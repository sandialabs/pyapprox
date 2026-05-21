"""Shared control-variate math used by CV, ACV, and (later) GroupACV.

This is a leaf module: it imports only the backend protocol, never any
estimator module. All functions are pure (arrays + backend in, array out).
"""

from pyapprox.util.backends.protocols import Array, Backend


def optimal_cv_weights(bkd: Backend[Array], CF: Array, cf: Array) -> Array:
    """Compute optimal control-variate weights.

    Parameters
    ----------
    bkd : Backend
        Array backend.
    CF : Array
        Discrepancy cross-covariance matrix.
    cf : Array
        Discrepancy–HF cross-covariance vector.

    Returns
    -------
    Array
        Optimal weights: ``-solve(CF, cf.T).T``.
    """
    return -bkd.solve(CF, cf.T).T


def covariance_with_weights(
    bkd: Backend[Array],
    hf_est_covar: Array,
    weights: Array,
    CF: Array,
    cf: Array,
) -> Array:
    """Compute estimator covariance for arbitrary (non-optimal) weights.

    Implements the general expression (e.g. Equation 8 from Dixon 2024).

    Parameters
    ----------
    bkd : Backend
        Array backend.
    hf_est_covar : Array
        High-fidelity estimator covariance.
    weights : Array
        Control-variate weights (not necessarily optimal).
    CF : Array
        Discrepancy cross-covariance matrix.
    cf : Array
        Discrepancy–HF cross-covariance vector.

    Returns
    -------
    Array
        Estimator covariance.
    """
    return (
        hf_est_covar
        + bkd.multidot([weights, CF, weights.T])
        + bkd.multidot([cf, weights.T])
        + bkd.multidot([weights, cf.T])
    )

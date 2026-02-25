"""
Conversions between CPD (Conditional Probability Distribution) and canonical form.

For linear-Gaussian CPDs:
    x_child = A @ x_parent + b + noise
    noise ~ N(0, noise_cov)

This creates a joint Gaussian over (x_parent, x_child) in canonical form.
"""

from typing import List, Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.gaussian import GaussianCanonicalForm
from .factor import GaussianFactor


def convert_cpd_to_canonical(
    A: Array,
    b: Array,
    noise_cov: Array,
    parent_var_ids: List[int],
    parent_nvars_per_var: List[int],
    child_var_id: int,
    bkd: Backend[Array],
) -> GaussianFactor[Array]:
    """
    Convert a linear-Gaussian CPD to canonical form factor.

    For a linear CPD:
        x_child = A @ x_parent + b + noise
        noise ~ N(0, noise_cov)

    Creates a factor over (parent_vars..., child_var) representing
    the conditional p(x_child | x_parent).

    Parameters
    ----------
    A : Array
        Linear coefficient matrix. Shape: (n_child, n_parent_total)
    b : Array
        Offset/intercept. Shape: (n_child,) or (n_child, 1)
    noise_cov : Array
        Noise covariance. Shape: (n_child, n_child)
    parent_var_ids : List[int]
        Variable IDs for parent variables.
    parent_nvars_per_var : List[int]
        Dimensions for each parent variable.
    child_var_id : int
        Variable ID for child variable.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    GaussianFactor
        Factor representing p(x_child | x_parent) in canonical form.

    Notes
    -----
    The resulting factor has vacuous information about the parents
    (infinite variance, zero precision). It only encodes the conditional.

    The canonical form for a linear-Gaussian CPD is:
        K = [[A^T Q A, -A^T Q], [-Q A, Q]]
        h = [A^T Q b, -Q b]

    where Q = noise_cov^{-1}.
    """
    if b.ndim == 2:
        b = b.flatten()

    n_child = noise_cov.shape[0]
    n_parent = sum(parent_nvars_per_var)

    # Check dimensions
    if A.shape != (n_child, n_parent):
        raise ValueError(
            f"A has shape {A.shape}, expected ({n_child}, {n_parent})"
        )
    if len(b) != n_child:
        raise ValueError(f"b has length {len(b)}, expected {n_child}")

    # Noise precision Q = noise_cov^{-1}
    Q = bkd.inv(noise_cov)

    # Build precision matrix (block structure: [parents, child])
    n_total = n_parent + n_child
    precision = bkd.zeros((n_total, n_total))

    # Convert to numpy for block assignment
    prec_np = bkd.to_numpy(precision)
    A_np = bkd.to_numpy(A)
    Q_np = bkd.to_numpy(Q)

    # K11 = A^T Q A
    prec_np[:n_parent, :n_parent] = A_np.T @ Q_np @ A_np
    # K12 = -A^T Q
    prec_np[:n_parent, n_parent:] = -A_np.T @ Q_np
    # K21 = -Q A
    prec_np[n_parent:, :n_parent] = -Q_np @ A_np
    # K22 = Q
    prec_np[n_parent:, n_parent:] = Q_np

    precision = bkd.asarray(prec_np)

    # Build shift vector
    shift = bkd.zeros((n_total,))
    shift_np = bkd.to_numpy(shift)
    b_np = bkd.to_numpy(b)

    # h1 = A^T Q b
    shift_np[:n_parent] = A_np.T @ Q_np @ b_np
    # h2 = -Q b
    shift_np[n_parent:] = -Q_np @ b_np

    shift = bkd.asarray(shift_np)

    # Normalization for the CPD
    # g = -0.5 * b^T Q b - 0.5 * n_child * log(2*pi) + 0.5 * log|Q|
    sign, logdet_Q = bkd.slogdet(Q)
    normalization = 0.5 * (
        -float(b_np @ Q_np @ b_np)
        - n_child * np.log(2 * np.pi)
        + float(logdet_Q)
    )

    canonical = GaussianCanonicalForm(precision, shift, normalization, bkd)

    # Build variable tracking
    var_ids = parent_var_ids + [child_var_id]
    nvars_per_var = parent_nvars_per_var + [n_child]

    return GaussianFactor(canonical, var_ids, nvars_per_var, bkd)


def convert_prior_to_factor(
    mean: Array,
    covariance: Array,
    var_id: int,
    bkd: Backend[Array],
) -> GaussianFactor[Array]:
    """
    Convert a Gaussian prior to a factor.

    Parameters
    ----------
    mean : Array
        Prior mean. Shape: (nvars,) or (nvars, 1)
    covariance : Array
        Prior covariance. Shape: (nvars, nvars)
    var_id : int
        Variable ID for this prior.
    bkd : Backend[Array]
        Backend.

    Returns
    -------
    GaussianFactor
        Factor representing the prior p(x).
    """
    if mean.ndim == 2:
        mean = mean.flatten()

    nvars = len(mean)
    return GaussianFactor.from_moments(
        mean, covariance, [var_id], [nvars], bkd
    )


def convert_likelihood_to_factor(
    observation_matrix: Array,
    observation: Array,
    noise_covariance: Array,
    state_var_id: int,
    bkd: Backend[Array],
) -> GaussianFactor[Array]:
    """
    Convert a linear-Gaussian observation likelihood to a factor.

    For the observation model:
        y = H @ x + noise
        noise ~ N(0, R)

    This creates a factor that, when multiplied with a prior on x,
    gives the posterior.

    Parameters
    ----------
    observation_matrix : Array
        Observation matrix H. Shape: (nobs, nstate)
    observation : Array
        Observed values y. Shape: (nobs,) or (nobs, 1)
    noise_covariance : Array
        Noise covariance R. Shape: (nobs, nobs)
    state_var_id : int
        Variable ID for the state variable x.
    bkd : Backend[Array]
        Backend.

    Returns
    -------
    GaussianFactor
        Likelihood factor (unnormalized).

    Notes
    -----
    The likelihood factor has:
        K = H^T R^{-1} H
        h = H^T R^{-1} y
        g = -0.5 * y^T R^{-1} y - 0.5 * n * log(2*pi) + 0.5 * log|R^{-1}|
    """
    if observation.ndim == 2:
        observation = observation.flatten()

    nobs, nstate = observation_matrix.shape

    # R^{-1}
    R_inv = bkd.inv(noise_covariance)

    # K = H^T R^{-1} H
    precision = observation_matrix.T @ R_inv @ observation_matrix

    # h = H^T R^{-1} y
    shift = observation_matrix.T @ R_inv @ observation

    # Normalization
    sign, logdet_R_inv = bkd.slogdet(R_inv)
    y_Rinv_y = float(observation @ (R_inv @ observation))
    normalization = 0.5 * (
        -y_Rinv_y - nobs * np.log(2 * np.pi) + float(logdet_R_inv)
    )

    canonical = GaussianCanonicalForm(precision, shift, normalization, bkd)

    return GaussianFactor(canonical, [state_var_id], [nstate], bkd)

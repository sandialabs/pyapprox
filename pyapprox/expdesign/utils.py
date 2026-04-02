"""General OED utilities.

Provides functions that work with any GaussianInferenceProblemProtocol,
not tied to specific benchmark implementations.
"""

import numpy as np

from pyapprox.expdesign.benchmarks.protocols import (
    GaussianInferenceProblemProtocol,
)
from pyapprox.inverse.conjugate.gaussian import (
    DenseGaussianConjugatePosterior,
)
from pyapprox.util.backends.protocols import Array


def compute_exact_eig(
    problem: GaussianInferenceProblemProtocol[Array],
    weights: Array,
) -> float:
    """Compute exact EIG via conjugate Gaussian posterior.

    Works for any linear-Gaussian inference problem. Uses
    DenseGaussianConjugatePosterior with weighted noise covariance.
    Zero-weight sensors are excluded.

    Parameters
    ----------
    problem : GaussianInferenceProblemProtocol
        Gaussian inference problem with obs_map, prior, noise_variances,
        prior_mean, prior_covariance.
    weights : Array
        Design weights. Shape: (nobs, 1).

    Returns
    -------
    eig : float
        Exact expected information gain.
    """
    bkd = problem.bkd()
    nobs = problem.nobs()
    w = bkd.reshape(weights, (nobs,))

    # Identify active sensors (positive weight)
    w_np = bkd.to_numpy(w)
    active_mask = w_np > 0
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)

    if n_active == 0:
        return 0.0

    # Get the design matrix from the obs_map
    # For linear models, obs_map(theta) = A @ theta, so we extract A
    # by evaluating on identity columns
    nparams = problem.nparams()
    identity = bkd.eye(nparams)
    A = problem.obs_map()(identity)  # (nobs, nparams)

    # Extract active rows
    A_active = A[active_indices, :]

    # Build effective noise covariance: diag(noise_var_i / w_i)
    noise_vars = problem.noise_variances()
    noise_vars_np = bkd.to_numpy(noise_vars)
    w_active_np = w_np[active_mask]
    noise_diag_eff_np = noise_vars_np[active_mask] / w_active_np
    noise_cov_eff = bkd.diag(bkd.asarray(noise_diag_eff_np))

    # Compute EIG via conjugate posterior
    post = DenseGaussianConjugatePosterior(
        A_active,
        problem.prior_mean(),
        problem.prior_covariance(),
        noise_cov_eff,
        bkd,
    )
    # EIG is data-independent for linear Gaussian; use dummy observations
    dummy_obs = bkd.zeros((n_active, 1))
    post.compute(dummy_obs)
    return post.expected_kl_divergence()

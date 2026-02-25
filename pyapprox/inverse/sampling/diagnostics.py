"""
MCMC diagnostic functions for convergence assessment.

This module provides:
- autocorrelation: Compute autocorrelation at given lags
- integrated_autocorrelation_time: Estimate IAT using Sokal's windowing
- effective_sample_size: Compute ESS = n / IAT
- rhat: Compute Gelman-Rubin R-hat for multiple chains
- MCMCDiagnostics: Dataclass holding all diagnostic measures
- compute_diagnostics: Convenience function to compute all diagnostics
"""

from dataclasses import dataclass
from typing import Generic, List, Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def autocorrelation(
    samples: Array,
    max_lag: int,
    bkd: Backend[Array],
) -> Array:
    """
    Compute autocorrelation function for MCMC samples.

    Parameters
    ----------
    samples : Array
        MCMC samples. Shape: (nvars, nsamples) or (nsamples,)
    max_lag : int
        Maximum lag to compute autocorrelation for.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Autocorrelation values for lags 0 to max_lag.
        Shape: (nvars, max_lag + 1) or (max_lag + 1,)
    """
    samples_np = bkd.to_numpy(samples)
    was_1d = samples_np.ndim == 1
    if was_1d:
        samples_np = samples_np.reshape(1, -1)

    nvars, nsamples = samples_np.shape
    max_lag = min(max_lag, nsamples - 2)

    result = np.zeros((nvars, max_lag + 1))
    result[:, 0] = 1.0  # Lag 0 always has correlation 1

    for var_idx in range(nvars):
        series = samples_np[var_idx, :]
        mean = np.mean(series)
        var = np.var(series)
        if var < 1e-15:
            # Constant series, all autocorrelations are 1
            result[var_idx, :] = 1.0
            continue

        for lag in range(1, max_lag + 1):
            x_t = series[:-lag]
            x_tpk = series[lag:]
            cov = np.mean((x_t - mean) * (x_tpk - mean))
            result[var_idx, lag] = cov / var

    if was_1d:
        return bkd.asarray(result[0, :])
    return bkd.asarray(result)


def integrated_autocorrelation_time(
    samples: Array,
    bkd: Backend[Array],
    c: float = 5.0,
) -> Array:
    """
    Estimate integrated autocorrelation time using Sokal's windowing.

    The IAT is estimated as:
        IAT = 1 + 2 * sum_{k=1}^{M} rho_k

    where M is chosen automatically using Sokal's criterion:
    M is the first lag where M >= c * IAT_estimate.

    Parameters
    ----------
    samples : Array
        MCMC samples. Shape: (nvars, nsamples) or (nsamples,)
    bkd : Backend[Array]
        Computational backend.
    c : float, default=5.0
        Sokal's window parameter. Larger values give more conservative
        (larger) IAT estimates.

    Returns
    -------
    Array
        Integrated autocorrelation time per variable. Shape: (nvars,) or scalar.
    """
    samples_np = bkd.to_numpy(samples)
    was_1d = samples_np.ndim == 1
    if was_1d:
        samples_np = samples_np.reshape(1, -1)

    nvars, nsamples = samples_np.shape
    max_lag = nsamples // 2

    # Compute autocorrelation
    acf = bkd.to_numpy(autocorrelation(bkd.asarray(samples_np), max_lag, bkd))

    result = np.zeros(nvars)
    for var_idx in range(nvars):
        rho = acf[var_idx, :]

        # Sokal's automatic windowing
        iat = 1.0
        for m in range(1, max_lag + 1):
            iat += 2.0 * rho[m]
            # Sokal's stopping criterion
            if m >= c * iat:
                break

        result[var_idx] = max(1.0, iat)

    if was_1d:
        return bkd.asarray(result[0])
    return bkd.asarray(result)


def effective_sample_size(
    samples: Array,
    bkd: Backend[Array],
) -> Array:
    """
    Compute effective sample size (ESS) for MCMC samples.

    ESS = nsamples / IAT

    A larger ESS indicates better mixing. For independent samples,
    ESS ≈ nsamples.

    Parameters
    ----------
    samples : Array
        MCMC samples. Shape: (nvars, nsamples) or (nsamples,)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Effective sample size per variable. Shape: (nvars,) or scalar.
    """
    samples_np = bkd.to_numpy(samples)
    was_1d = samples_np.ndim == 1
    if was_1d:
        samples_np = samples_np.reshape(1, -1)

    nsamples = samples_np.shape[1]
    iat = bkd.to_numpy(
        integrated_autocorrelation_time(bkd.asarray(samples_np), bkd)
    )
    ess = nsamples / iat

    if was_1d:
        return bkd.asarray(ess[0])
    return bkd.asarray(ess)


def rhat(
    chains: List[Array],
    bkd: Backend[Array],
) -> Array:
    """
    Compute Gelman-Rubin R-hat diagnostic for multiple chains.

    R-hat measures convergence by comparing between-chain and within-chain
    variance. Values close to 1.0 indicate convergence; typically R-hat < 1.1
    is considered acceptable.

    R-hat = sqrt((n-1)/n + B/(n*W))

    where:
    - B = between-chain variance
    - W = within-chain variance
    - n = number of samples per chain

    Parameters
    ----------
    chains : List[Array]
        List of MCMC chains. Each has shape (nvars, nsamples).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        R-hat per variable. Shape: (nvars,)

    Raises
    ------
    ValueError
        If fewer than 2 chains provided or chains have different shapes.
    """
    if len(chains) < 2:
        raise ValueError("R-hat requires at least 2 chains")

    chains_np = [bkd.to_numpy(chain) for chain in chains]

    # Validate shapes
    nvars, nsamples = chains_np[0].shape
    for i, chain in enumerate(chains_np[1:], 1):
        if chain.shape != (nvars, nsamples):
            raise ValueError(
                f"Chain {i} has shape {chain.shape}, "
                f"expected ({nvars}, {nsamples})"
            )

    m = len(chains_np)  # number of chains
    n = nsamples

    # Chain means: (m, nvars)
    chain_means = np.array([np.mean(chain, axis=1) for chain in chains_np])

    # Grand mean: (nvars,)
    grand_mean = np.mean(chain_means, axis=0)

    # Between-chain variance: B = n/(m-1) * sum((chain_mean - grand_mean)^2)
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)

    # Within-chain variance: W = 1/m * sum(chain_var)
    chain_vars = np.array([np.var(chain, axis=1, ddof=1) for chain in chains_np])
    W = np.mean(chain_vars, axis=0)

    # Pooled variance estimate
    var_plus = ((n - 1) / n) * W + (1 / n) * B

    # R-hat
    rhat_vals = np.sqrt(var_plus / W)

    return bkd.asarray(rhat_vals)


@dataclass
class MCMCDiagnostics(Generic[Array]):
    """
    Container for MCMC diagnostic measures.

    Attributes
    ----------
    ess : Array
        Effective sample size per variable. Shape: (nvars,)
    iat : Array
        Integrated autocorrelation time per variable. Shape: (nvars,)
    autocorrelation : Array
        Autocorrelation function. Shape: (nvars, max_lag + 1)
    rhat : Optional[Array]
        Gelman-Rubin R-hat. Shape: (nvars,). None if only one chain.
    """

    ess: Array
    iat: Array
    autocorrelation: Array
    rhat: Optional[Array]


def compute_diagnostics(
    samples: Array,
    bkd: Backend[Array],
    max_lag: int = 100,
    other_chains: Optional[List[Array]] = None,
) -> MCMCDiagnostics[Array]:
    """
    Compute all MCMC diagnostics for a set of samples.

    Parameters
    ----------
    samples : Array
        MCMC samples. Shape: (nvars, nsamples)
    bkd : Backend[Array]
        Computational backend.
    max_lag : int, default=100
        Maximum lag for autocorrelation.
    other_chains : List[Array], optional
        Additional chains for R-hat computation.
        If provided, R-hat is computed using samples + other_chains.

    Returns
    -------
    MCMCDiagnostics
        Dataclass containing ESS, IAT, autocorrelation, and optionally R-hat.
    """
    ess = effective_sample_size(samples, bkd)
    iat = integrated_autocorrelation_time(samples, bkd)
    acf = autocorrelation(samples, max_lag, bkd)

    rhat_val = None
    if other_chains is not None and len(other_chains) > 0:
        all_chains = [samples] + other_chains
        rhat_val = rhat(all_chains, bkd)

    return MCMCDiagnostics(
        ess=ess,
        iat=iat,
        autocorrelation=acf,
        rhat=rhat_val,
    )

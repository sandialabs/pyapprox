"""Standalone PCE statistics functions.

This module provides functions for computing statistics from PCE
coefficients without requiring PCE class methods. These functions
work with any object implementing PCEStatisticsProtocol.
"""

from typing import List, Tuple

from pyapprox.surrogates.affine.protocols import PCEStatisticsProtocol
from pyapprox.util.backends.protocols import Array


def get_constant_index(pce: PCEStatisticsProtocol[Array]) -> int:
    """Find index of the constant basis function.

    Parameters
    ----------
    pce : PCEStatisticsProtocol[Array]
        PCE object.

    Returns
    -------
    int
        Index of constant term.
    """
    bkd = pce.bkd()
    indices = pce.get_indices()
    index_sums = bkd.sum(indices, axis=0)
    const_mask = index_sums == 0
    const_indices = bkd.nonzero(const_mask)
    if len(const_indices[0]) == 0:
        raise ValueError("Basis does not contain constant term")
    return int(const_indices[0][0])


def mean(pce: PCEStatisticsProtocol[Array]) -> Array:
    """Compute mean from PCE coefficients.

    For orthonormal polynomials, E[f] = c_0 (coefficient of constant term).

    Parameters
    ----------
    pce : PCEStatisticsProtocol[Array]
        PCE object.

    Returns
    -------
    Array
        Mean values. Shape: (nqoi,)
    """
    const_idx = get_constant_index(pce)
    return pce.get_coefficients()[const_idx, :]


def variance(pce: PCEStatisticsProtocol[Array]) -> Array:
    """Compute variance from PCE coefficients.

    For orthonormal polynomials: Var[f] = Σ_{i≠0} c_i²

    Parameters
    ----------
    pce : PCEStatisticsProtocol[Array]
        PCE object.

    Returns
    -------
    Array
        Variance values. Shape: (nqoi,)
    """
    bkd = pce.bkd()
    const_idx = get_constant_index(pce)
    coef = pce.get_coefficients()
    coef_sq = coef**2
    total = bkd.sum(coef_sq, axis=0)
    return total - coef_sq[const_idx, :]


def std(pce: PCEStatisticsProtocol[Array]) -> Array:
    """Compute standard deviation from PCE coefficients.

    Parameters
    ----------
    pce : PCEStatisticsProtocol[Array]
        PCE object.

    Returns
    -------
    Array
        Standard deviation values. Shape: (nqoi,)
    """
    return pce.bkd().sqrt(variance(pce))


def covariance(pce: PCEStatisticsProtocol[Array]) -> Array:
    """Compute covariance matrix between QoIs.

    For orthonormal polynomials: Cov[f_i, f_j] = Σ_{k≠0} c_{k,i} c_{k,j}

    Parameters
    ----------
    pce : PCEStatisticsProtocol[Array]
        PCE object.

    Returns
    -------
    Array
        Covariance matrix. Shape: (nqoi, nqoi)
    """
    bkd = pce.bkd()
    const_idx = get_constant_index(pce)
    coef = pce.get_coefficients()

    # Remove constant term
    coef_nonconstant = bkd.concatenate(
        [coef[:const_idx, :], coef[const_idx + 1 :, :]], axis=0
    )

    return bkd.dot(coef_nonconstant.T, coef_nonconstant)


def total_sobol_indices(pce: PCEStatisticsProtocol[Array]) -> Array:
    """Compute total Sobol sensitivity indices.

    The total Sobol index T_i measures the total contribution of
    variable i (including interactions) to the output variance.

    Parameters
    ----------
    pce : PCEStatisticsProtocol[Array]
        PCE object.

    Returns
    -------
    Array
        Total Sobol indices. Shape: (nvars, nqoi)
    """
    bkd = pce.bkd()
    indices = pce.get_indices()
    coef = pce.get_coefficients()
    var = variance(pce)
    nvars = pce.nvars()
    nqoi = pce.nqoi()

    # Avoid division by zero
    var_safe = bkd.where(var > 0, var, bkd.ones_like(var))
    total_indices = bkd.zeros((nvars, nqoi))

    for dd in range(nvars):
        depends_on_dd = indices[dd, :] > 0
        coef_sq = coef**2
        mask = bkd.asarray(depends_on_dd, dtype=bkd.default_dtype())
        contribution = bkd.sum(coef_sq * bkd.reshape(mask, (-1, 1)), axis=0)
        total_indices[dd, :] = contribution / var_safe

    # Set to zero where variance is zero
    total_indices = bkd.where(
        bkd.reshape(var, (1, -1)) > 0, total_indices, bkd.zeros_like(total_indices)
    )
    return total_indices


def main_effect_sobol_indices(pce: PCEStatisticsProtocol[Array]) -> Array:
    """Compute main effect (first-order) Sobol indices.

    The main effect S_i measures the contribution of variable i
    alone (no interactions) to the output variance.

    Parameters
    ----------
    pce : PCEStatisticsProtocol[Array]
        PCE object.

    Returns
    -------
    Array
        Main effect Sobol indices. Shape: (nvars, nqoi)
    """
    bkd = pce.bkd()
    indices = pce.get_indices()
    coef = pce.get_coefficients()
    var = variance(pce)
    nvars = pce.nvars()
    nqoi = pce.nqoi()

    # Avoid division by zero
    var_safe = bkd.where(var > 0, var, bkd.ones_like(var))
    main_indices = bkd.zeros((nvars, nqoi))

    for dd in range(nvars):
        # Terms that depend ONLY on variable dd (no interactions)
        depends_on_dd = indices[dd, :] > 0
        index_sum = bkd.sum(indices, axis=0)
        other_vars_zero = index_sum == indices[dd, :]
        main_effect_terms = depends_on_dd & other_vars_zero

        coef_sq = coef**2
        mask = bkd.asarray(main_effect_terms, dtype=bkd.default_dtype())
        contribution = bkd.sum(coef_sq * bkd.reshape(mask, (-1, 1)), axis=0)
        main_indices[dd, :] = contribution / var_safe

    # Set to zero where variance is zero
    main_indices = bkd.where(
        bkd.reshape(var, (1, -1)) > 0, main_indices, bkd.zeros_like(main_indices)
    )
    return main_indices


def interaction_sobol_indices(
    pce: PCEStatisticsProtocol[Array],
    variable_sets: List[Tuple[int, ...]],
) -> Array:
    """Compute interaction Sobol indices for specified variable sets.

    The interaction index S_{i,j,...} measures the contribution of the
    interaction between variables i, j, ... to the output variance.

    Parameters
    ----------
    pce : PCEStatisticsProtocol[Array]
        PCE object.
    variable_sets : List[Tuple[int, ...]]
        Variable index sets to compute interactions for.

    Returns
    -------
    Array
        Interaction Sobol indices. Shape: (len(variable_sets), nqoi)
    """
    bkd = pce.bkd()
    indices = pce.get_indices()
    coef = pce.get_coefficients()
    var = variance(pce)
    nvars_pce = pce.nvars()
    nqoi = pce.nqoi()
    nterms = indices.shape[1]

    # Avoid division by zero
    var_safe = bkd.where(var > 0, var, bkd.ones_like(var))
    interaction_indices = bkd.zeros((len(variable_sets), nqoi))

    for ii, var_set in enumerate(variable_sets):
        var_set_frozen = set(var_set)

        # Build mask for terms where exactly the specified variables are active
        # Start with all True
        active_mask = bkd.ones((nterms,), dtype=bkd.default_dtype()) > 0.5

        for dd in range(nvars_pce):
            if dd in var_set_frozen:
                # Variable must be active
                active_mask = active_mask & (indices[dd, :] > 0)
            else:
                # Variable must be inactive
                active_mask = active_mask & (indices[dd, :] == 0)

        coef_sq = coef**2
        mask = bkd.asarray(active_mask, dtype=bkd.default_dtype())
        contribution = bkd.sum(coef_sq * bkd.reshape(mask, (-1, 1)), axis=0)
        interaction_indices[ii, :] = contribution / var_safe

    # Set to zero where variance is zero
    interaction_indices = bkd.where(
        bkd.reshape(var, (1, -1)) > 0,
        interaction_indices,
        bkd.zeros_like(interaction_indices),
    )
    return interaction_indices

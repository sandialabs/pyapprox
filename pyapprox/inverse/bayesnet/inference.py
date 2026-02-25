"""
Variable elimination inference for Gaussian Bayesian networks.

This module provides sum-product variable elimination for exact inference
in linear-Gaussian graphical models.
"""

from typing import Dict, Generic, List, Optional, Set, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from .factor import GaussianFactor
from .network import GaussianNetwork


def sum_product_eliminate_variable(
    factors: List[GaussianFactor[Array]],
    var_id: int,
    bkd: Backend[Array],
) -> List[GaussianFactor[Array]]:
    """
    Eliminate a variable from a set of factors using sum-product.

    1. Find all factors containing var_id
    2. Multiply them together
    3. Marginalize out var_id
    4. Return remaining factors plus the new marginal

    Parameters
    ----------
    factors : List[GaussianFactor]
        Current set of factors.
    var_id : int
        Variable ID to eliminate.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    List[GaussianFactor]
        Updated factor list with var_id eliminated.
    """
    # Separate factors containing var_id from others
    involved: List[GaussianFactor[Array]] = []
    remaining: List[GaussianFactor[Array]] = []

    for factor in factors:
        if var_id in factor.var_ids():
            involved.append(factor)
        else:
            remaining.append(factor)

    if not involved:
        # Variable not in any factor - nothing to do
        return factors

    # Multiply all involved factors
    product = involved[0]
    for f in involved[1:]:
        product = product.multiply(f)

    # Marginalize out var_id
    marginal = product.marginalize_vars([var_id])

    # Return remaining factors plus the marginal
    remaining.append(marginal)
    return remaining


def sum_product_variable_elimination(
    factors: List[GaussianFactor[Array]],
    elim_order: List[int],
    bkd: Backend[Array],
) -> GaussianFactor[Array]:
    """
    Perform variable elimination in specified order.

    Parameters
    ----------
    factors : List[GaussianFactor]
        Initial factors from the network.
    elim_order : List[int]
        Order in which to eliminate variables.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    GaussianFactor
        Final factor after all eliminations.

    Raises
    ------
    ValueError
        If no factors remain after elimination.
    """
    current_factors = list(factors)

    for var_id in elim_order:
        current_factors = sum_product_eliminate_variable(
            current_factors, var_id, bkd
        )

    if not current_factors:
        raise ValueError("No factors remain after elimination")

    # Multiply remaining factors
    result = current_factors[0]
    for f in current_factors[1:]:
        result = result.multiply(f)

    return result


def cond_prob_variable_elimination(
    network: GaussianNetwork[Array],
    query_ids: List[int],
    evidence: Optional[Dict[int, Array]] = None,
) -> GaussianFactor[Array]:
    """
    Compute conditional probability using variable elimination.

    Computes p(query_vars | evidence) by:
    1. Converting network to factors
    2. Conditioning factors on evidence
    3. Eliminating hidden variables
    4. Normalizing result

    Parameters
    ----------
    network : GaussianNetwork
        The Bayesian network.
    query_ids : List[int]
        Variable IDs to query.
    evidence : Dict[int, Array], optional
        Dictionary mapping evidence variable IDs to observed values.

    Returns
    -------
    GaussianFactor
        Factor representing p(query_vars | evidence).
    """
    bkd = network.bkd()

    # Convert network to factors
    factors = network.convert_to_factors()

    # Condition on evidence
    if evidence:
        conditioned_factors: List[GaussianFactor[Array]] = []
        for factor in factors:
            # Check which evidence variables are in this factor's scope
            evidence_in_scope = [
                var_id for var_id in evidence.keys()
                if var_id in factor.var_ids()
            ]

            if evidence_in_scope:
                # Condition this factor on the evidence
                for var_id in evidence_in_scope:
                    value = evidence[var_id]
                    if value.ndim == 2:
                        value = value.flatten()
                    factor = factor.condition_vars([var_id], value)

            # Only keep factor if it has remaining variables
            if factor.scope_size() > 0:
                conditioned_factors.append(factor)

        factors = conditioned_factors

    # Determine elimination order (all variables except query)
    all_var_ids: Set[int] = set()
    for factor in factors:
        all_var_ids.update(factor.var_ids())

    query_set = set(query_ids)
    evidence_set = set(evidence.keys()) if evidence else set()

    # Variables to eliminate: all except query and evidence
    elim_vars = all_var_ids - query_set - evidence_set

    # Use reverse topological order for elimination (if available)
    # This is often a good heuristic
    try:
        topo_order = network.reverse_topological_order()
        elim_order = [v for v in topo_order if v in elim_vars]
    except Exception:
        elim_order = list(elim_vars)

    # Perform elimination
    if elim_order:
        result = sum_product_variable_elimination(factors, elim_order, bkd)
    else:
        # No elimination needed - multiply all factors
        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)

    # Normalize the result
    canonical = result.canonical().normalize()
    return GaussianFactor(
        canonical, result.var_ids(), result.nvars_per_var(), bkd
    )


def compute_marginal(
    network: GaussianNetwork[Array],
    var_ids: List[int],
) -> GaussianFactor[Array]:
    """
    Compute the marginal distribution over specified variables.

    This is equivalent to cond_prob_variable_elimination with no evidence.

    Parameters
    ----------
    network : GaussianNetwork
        The Bayesian network.
    var_ids : List[int]
        Variables to compute marginal for.

    Returns
    -------
    GaussianFactor
        Marginal distribution p(var_ids).
    """
    return cond_prob_variable_elimination(network, var_ids, evidence=None)


def compute_posterior(
    network: GaussianNetwork[Array],
    query_ids: List[int],
    evidence: Dict[int, Array],
) -> GaussianFactor[Array]:
    """
    Compute the posterior distribution given evidence.

    Parameters
    ----------
    network : GaussianNetwork
        The Bayesian network.
    query_ids : List[int]
        Variables to query.
    evidence : Dict[int, Array]
        Observed values for evidence variables.

    Returns
    -------
    GaussianFactor
        Posterior distribution p(query_ids | evidence).
    """
    return cond_prob_variable_elimination(network, query_ids, evidence)

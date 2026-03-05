"""
Factory for constructing a D-vine copula from a Gaussian Bayesian network.

Uses efficient local covariance propagation to compute partial correlations
in O(n * k^4) time, where n is the number of variables and k is the
precision bandwidth (maximum parent distance). This avoids forming the
full n x n precision or covariance matrix.
"""

import math
from typing import Dict, List, Tuple

import numpy as np

from pyapprox.inverse.bayesnet.network import GaussianNetwork
from pyapprox.probability.copula.bivariate.gaussian import (
    BivariateGaussianCopula,
)
from pyapprox.probability.copula.bivariate.protocols import (
    BivariateCopulaProtocol,
)
from pyapprox.probability.copula.vine.dvine import DVineCopula
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.cholesky_factor import CholeskyFactor


def _extract_chain_structure(
    network: GaussianNetwork[Array],
    bkd: Backend[Array],
) -> Tuple[List[int], List[List[Tuple[int, float]]], List[float], int]:
    """
    Extract CPD structure from the network in positional coordinates.

    Validates scalar nodes and computes bandwidth.

    Parameters
    ----------
    network : GaussianNetwork[Array]
        The Gaussian Bayesian network.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    topo_order : List[int]
        Node IDs in topological order.
    parent_pos_coeffs : List[List[Tuple[int, float]]]
        For each position i, list of (parent_position, coefficient) pairs.
    noise_variance : List[float]
        Noise/prior variance for each position.
    bandwidth : int
        Maximum parent distance in topological order.
    """
    topo_order = network.topological_order()
    n = len(topo_order)
    node_to_pos = {nid: pos for pos, nid in enumerate(topo_order)}

    for nid in topo_order:
        nvars = network.get_node_nvars(nid)
        if nvars != 1:
            raise ValueError(
                f"Node {nid} has nvars={nvars}; "
                f"dvine_from_gaussian_network requires scalar nodes (nvars=1)"
            )

    parent_pos_coeffs: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    noise_variance: List[float] = [0.0] * n
    bandwidth = 0

    for pos, nid in enumerate(topo_order):
        if network.is_root(nid):
            prior_cov = network.get_prior_cov(nid)
            noise_variance[pos] = bkd.to_float(prior_cov[0, 0])
        else:
            parents = network.get_parents(nid)
            coeffs = network.get_cpd_coefficients(nid)
            cpd_noise = network.get_cpd_noise_cov(nid)
            noise_variance[pos] = bkd.to_float(cpd_noise[0, 0])

            for pid, coeff in zip(parents, coeffs):
                p_pos = node_to_pos[pid]
                c_val = bkd.to_float(coeff[0, 0])
                parent_pos_coeffs[pos].append((p_pos, c_val))
                bandwidth = max(bandwidth, pos - p_pos)

    return topo_order, parent_pos_coeffs, noise_variance, bandwidth


def _propagate_local_covariances(
    n: int,
    bandwidth: int,
    parent_pos_coeffs: List[List[Tuple[int, float]]],
    noise_variance: List[float],
) -> Dict[Tuple[int, int], float]:
    """
    Forward-propagate local covariances through the chain.

    Computes Cov(X_i, X_j) for |i - j| <= bandwidth without forming the
    full covariance matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    bandwidth : int
        Maximum parent distance.
    parent_pos_coeffs : List[List[Tuple[int, float]]]
        Parent positions and coefficients for each node.
    noise_variance : List[float]
        Noise/prior variance for each node.

    Returns
    -------
    Dict[Tuple[int, int], float]
        Maps (i, d) to Cov(X_i, X_{i-d}) for d = 0, ..., min(i, k).
    """
    cov: Dict[Tuple[int, int], float] = {}

    # Root node
    cov[(0, 0)] = noise_variance[0]

    for i in range(1, n):
        pcs = parent_pos_coeffs[i]

        # Var(X_i) = sum_{j,l} A_j * A_l * Cov(X_j, X_l) + noise_var
        var_i = noise_variance[i]
        for j_pos, a_j in pcs:
            for l_pos, a_l in pcs:
                hi = max(j_pos, l_pos)
                d = abs(j_pos - l_pos)
                var_i += a_j * a_l * cov[(hi, d)]
        cov[(i, 0)] = var_i

        # Cov(X_i, X_m) for m in [max(0, i-k), i-1]
        for m in range(max(0, i - bandwidth), i):
            cov_im = 0.0
            for j_pos, a_j in pcs:
                hi = max(j_pos, m)
                d = abs(j_pos - m)
                cov_im += a_j * cov[(hi, d)]
            cov[(i, i - m)] = cov_im

    return cov


def _extract_partial_correlations(
    n: int,
    bandwidth: int,
    cov: Dict[Tuple[int, int], float],
    bkd: Backend[Array],
) -> Dict[int, List[float]]:
    """
    Extract partial correlations from local covariances.

    For each tree level t and edge e, builds a (t+1)x(t+1) local
    correlation submatrix and computes the partial correlation via
    Cholesky-based inversion.

    Parameters
    ----------
    n : int
        Number of nodes.
    bandwidth : int
        Truncation level (= precision bandwidth).
    cov : Dict[Tuple[int, int], float]
        Local covariance entries from forward propagation.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Dict[int, List[float]]
        Maps tree level (1-indexed) to list of partial correlations.
    """
    std = [math.sqrt(cov[(i, 0)]) for i in range(n)]

    partial_corrs: Dict[int, List[float]] = {}
    for t in range(1, bandwidth + 1):
        partial_corrs[t] = []
        for e in range(n - t):
            if t == 1:
                rho = cov[(e + 1, 1)] / (std[e] * std[e + 1])
                partial_corrs[t].append(rho)
            else:
                size = t + 1
                R_np = np.eye(size, dtype=np.float64)
                for ii in range(size):
                    for jj in range(ii + 1, size):
                        d = jj - ii
                        cov_val = cov[(e + jj, d)]
                        r = cov_val / (std[e + ii] * std[e + jj])
                        R_np[ii, jj] = r
                        R_np[jj, ii] = r
                R_sub = bkd.asarray(R_np)
                L = bkd.cholesky(R_sub)
                chol = CholeskyFactor(L, bkd)
                P = chol.matrix_inverse()
                rho_partial = -P[0, t] / bkd.sqrt(P[0, 0] * P[t, t])
                partial_corrs[t].append(bkd.to_float(rho_partial))

    return partial_corrs


def dvine_from_gaussian_network(
    network: GaussianNetwork[Array],
    bkd: Backend[Array],
) -> DVineCopula[Array]:
    """
    Construct a D-vine copula from a Gaussian Bayesian network.

    Uses efficient local covariance propagation to compute partial
    correlations without forming the full precision or covariance matrix.
    Complexity is O(n * k^4) where k is the precision bandwidth.

    The network must have scalar nodes (nvars=1 per node). Any DAG
    structure is accepted; the D-vine ordering follows the topological
    order of the network.

    Parameters
    ----------
    network : GaussianNetwork[Array]
        A Gaussian Bayesian network with scalar nodes.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    DVineCopula[Array]
        D-vine copula with BivariateGaussianCopula pair copulas.

    Raises
    ------
    ValueError
        If any node has nvars != 1.
    """
    topo_order, parent_pos_coeffs, noise_variance, bandwidth = _extract_chain_structure(
        network, bkd
    )
    n = len(topo_order)

    if n < 2 or bandwidth == 0:
        return DVineCopula({}, n, 0, bkd)

    cov = _propagate_local_covariances(n, bandwidth, parent_pos_coeffs, noise_variance)

    partial_corrs = _extract_partial_correlations(n, bandwidth, cov, bkd)

    pair_copulas: Dict[int, List[BivariateCopulaProtocol[Array]]] = {}
    for t in range(1, bandwidth + 1):
        pair_copulas[t] = [
            BivariateGaussianCopula(rho, bkd) for rho in partial_corrs[t]
        ]

    return DVineCopula(pair_copulas, n, bandwidth, bkd)

"""Barycentric interpolation data for Chebyshev nodes: the Gauss-Lobatto
and Gauss node sets on :math:`[-1, 1]` and their barycentric weights,
all in closed form.

Nodes are increasing, for any count. The weights are the closed forms,
equal to :math:`1 / \\prod_{k \\ne j} (x_j - x_k)` up to a common factor,
which is all barycentric formulas built on weight ratios need
(:func:`~pyapprox.polynomials.univariate.lagrange.
lagrange_differentiation_matrix`).
"""

import math

from pyapprox.util.backends.protocols import Array, Backend


def chebyshev_gauss_lobatto_nodes(npts: int, bkd: Backend[Array]) -> Array:
    """The extrema of :math:`T_{n-1}`, :math:`-\\cos(\\pi j / (n - 1))`,
    both ends included. Shape: (npts,)

    Raises
    ------
    ValueError
        If ``npts`` is below 2.
    """
    if npts < 2:
        raise ValueError(
            f"Chebyshev-Gauss-Lobatto nodes need at least 2 nodes, got {npts}"
        )
    j = bkd.arange(npts)
    return -bkd.cos(math.pi * j / (npts - 1))


def chebyshev_gauss_lobatto_weights(npts: int, bkd: Backend[Array]) -> Array:
    """Barycentric weights of :func:`chebyshev_gauss_lobatto_nodes`,
    :math:`(-1)^j`, halved at both ends. Shape: (npts,)

    Raises
    ------
    ValueError
        If ``npts`` is below 2.
    """
    if npts < 2:
        raise ValueError(
            f"Chebyshev-Gauss-Lobatto nodes need at least 2 nodes, got {npts}"
        )
    return bkd.asarray(
        [(-1.0) ** j * (0.5 if j in (0, npts - 1) else 1.0) for j in range(npts)]
    )


def chebyshev_gauss_nodes(npts: int, bkd: Backend[Array]) -> Array:
    """The roots of :math:`T_n`, :math:`-\\cos(\\pi (2 j + 1) / (2 n))`,
    strictly inside :math:`(-1, 1)`. Shape: (npts,)

    Raises
    ------
    ValueError
        If ``npts`` is below 1.
    """
    if npts < 1:
        raise ValueError(f"Chebyshev-Gauss nodes need at least 1 node, got {npts}")
    j = bkd.arange(npts)
    return -bkd.cos(math.pi * (2 * j + 1) / (2 * npts))


def chebyshev_gauss_weights(npts: int, bkd: Backend[Array]) -> Array:
    """Barycentric weights of :func:`chebyshev_gauss_nodes`,
    :math:`(-1)^j \\sin(\\pi (2 j + 1) / (2 n))`. Shape: (npts,)

    Raises
    ------
    ValueError
        If ``npts`` is below 1.
    """
    if npts < 1:
        raise ValueError(f"Chebyshev-Gauss nodes need at least 1 node, got {npts}")
    j = bkd.arange(npts)
    signs = bkd.asarray([(-1.0) ** k for k in range(npts)])
    return signs * bkd.sin(math.pi * (2 * j + 1) / (2 * npts))

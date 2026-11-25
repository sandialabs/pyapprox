"""Closed-form Chebyshev nodes and barycentric weights.

The Gauss-Lobatto nodes are the extrema of T_{n-1} with both ends, the
Gauss nodes the roots of T_n; both increasing. Each closed-form weight
set is the product-formula weights of its nodes up to one common factor.
"""

import pytest

from pyapprox.polynomials.univariate.chebyshev_barycentric import (
    chebyshev_gauss_lobatto_nodes,
    chebyshev_gauss_lobatto_weights,
    chebyshev_gauss_nodes,
    chebyshev_gauss_weights,
)
from pyapprox.polynomials.univariate.lagrange import compute_barycentric_weights
from pyapprox.util.backends.protocols import Array, Backend


@pytest.mark.parametrize("npts", [2, 3, 6, 11])
def test_gauss_lobatto(bkd: Backend[Array], npts: int) -> None:
    nodes = chebyshev_gauss_lobatto_nodes(npts, bkd)
    bkd.assert_allclose(nodes[0], bkd.asarray(-1.0))
    bkd.assert_allclose(nodes[-1], bkd.asarray(1.0))
    assert bkd.to_bool(bkd.all(bkd.diff(nodes) > 0))
    # The interior nodes are the zeros of T_{n-1}' = (n - 1) U_{n-2}
    theta = bkd.arccos(nodes[1:-1])
    bkd.assert_allclose(bkd.sin((npts - 1) * theta), bkd.zeros((npts - 2,)), atol=1e-14)
    _check_proportional(
        bkd,
        chebyshev_gauss_lobatto_weights(npts, bkd),
        compute_barycentric_weights(nodes, bkd),
    )


def test_nodes_and_weights_take_the_configured_precision(
    config_bkd: Backend[Array],
) -> None:
    # The node formulas multiply an integer range by pi; each must convert
    # it to the configured real dtype rather than take a library default.
    bkd = config_bkd
    for result in (
        chebyshev_gauss_lobatto_nodes(5, bkd),
        chebyshev_gauss_lobatto_weights(5, bkd),
        chebyshev_gauss_nodes(5, bkd),
        chebyshev_gauss_weights(5, bkd),
    ):
        assert result.dtype == bkd.real_dtype()


@pytest.mark.parametrize("npts", [1, 2, 5, 12])
def test_gauss(bkd: Backend[Array], npts: int) -> None:
    nodes = chebyshev_gauss_nodes(npts, bkd)
    bkd.assert_allclose(
        bkd.cos(npts * bkd.arccos(nodes)), bkd.zeros((npts,)), atol=1e-14
    )
    assert bkd.to_bool(bkd.all(bkd.abs(nodes) < 1.0))
    assert bkd.to_bool(bkd.all(bkd.diff(nodes) > 0))
    _check_proportional(
        bkd,
        chebyshev_gauss_weights(npts, bkd),
        compute_barycentric_weights(nodes, bkd),
    )


def _check_proportional(bkd: Backend[Array], closed: Array, product: Array) -> None:
    """``closed`` is ``product`` times one common factor."""
    ratio = product / closed
    bkd.assert_allclose(ratio, bkd.full(ratio.shape, bkd.to_float(ratio[0])))


def test_refusals(bkd: Backend[Array]) -> None:
    with pytest.raises(ValueError, match="at least 2 nodes"):
        chebyshev_gauss_lobatto_nodes(1, bkd)
    with pytest.raises(ValueError, match="at least 2 nodes"):
        chebyshev_gauss_lobatto_weights(1, bkd)
    with pytest.raises(ValueError, match="at least 1 node"):
        chebyshev_gauss_nodes(0, bkd)
    with pytest.raises(ValueError, match="at least 1 node"):
        chebyshev_gauss_weights(0, bkd)

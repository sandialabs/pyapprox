"""The Lagrange differentiation matrix on its own nodes.

On Legendre-Gauss nodes, with weights from the nodes: it equals the
first derivatives of the Lagrange basis evaluated at the nodes, is exact
on monomials up to degree n - 1 and has zero row sums. Weights scaled by
a common factor give the same matrix; one node gives zero.
"""

import pytest

from pyapprox.polynomials.univariate import LegendrePolynomial1D
from pyapprox.polynomials.univariate.lagrange import (
    compute_barycentric_weights,
    lagrange_differentiation_matrix,
    univariate_lagrange_first_derivative,
)
from pyapprox.util.backends.protocols import Array, Backend


def _legendre_gauss_nodes(bkd: Backend[Array], npts: int) -> Array:
    poly = LegendrePolynomial1D(bkd)
    poly.set_nterms(npts)
    return bkd.flatten(poly.gauss_quadrature_rule(npts)[0])


@pytest.mark.parametrize("npts", [2, 4, 7])
def test_matches_the_basis_derivatives_and_is_exact(
    bkd: Backend[Array], npts: int
) -> None:
    nodes = _legendre_gauss_nodes(bkd, npts)
    D = lagrange_differentiation_matrix(nodes, bkd)
    bkd.assert_allclose(
        D, univariate_lagrange_first_derivative(nodes, nodes, bkd), atol=1e-12
    )
    bkd.assert_allclose(bkd.sum(D, axis=1), bkd.zeros((npts,)), atol=1e-12)
    for degree in range(1, npts):
        bkd.assert_allclose(
            D @ nodes**degree, degree * nodes ** (degree - 1), atol=1e-11
        )


def test_weights_up_to_a_common_factor(bkd: Backend[Array]) -> None:
    nodes = _legendre_gauss_nodes(bkd, 6)
    weights = compute_barycentric_weights(nodes, bkd)
    bkd.assert_allclose(
        lagrange_differentiation_matrix(nodes, bkd, -3.5 * weights),
        lagrange_differentiation_matrix(nodes, bkd),
        rtol=1e-14,
        atol=1e-14,
    )


def test_one_node(bkd: Backend[Array]) -> None:
    bkd.assert_allclose(
        lagrange_differentiation_matrix(bkd.asarray([0.3]), bkd), bkd.zeros((1, 1))
    )

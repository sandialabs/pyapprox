"""The Chebyshev 1D collocation bases: Gauss-Lobatto and Gauss.

Each makes its own nodes and differentiates and interpolates on them.
Cases, on both families: the nodes against their closed forms;
derivative matrices exact on monomials up to degree n - 1 (and their
square on second derivatives), zero row sums and spectral convergence on
sin(pi x); interpolation exact on polynomials of degree n - 1, from
either grid to the other; one Gauss node; the refusals.
"""

from typing import Callable

import numpy as np
import pytest

from pyapprox.pde.collocation.basis.chebyshev.nodal import (
    ChebyshevGauss1D,
    ChebyshevGaussLobatto1D,
)
from pyapprox.pde.collocation.protocols.basis_1d import CollocationBasis1DProtocol
from pyapprox.util.backends.protocols import Array, Backend

_FAMILIES = (ChebyshevGaussLobatto1D, ChebyshevGauss1D)
_IDS = ["gauss-lobatto", "gauss"]


def test_satisfy_the_protocol() -> None:
    for family in _FAMILIES:
        assert isinstance(family(), CollocationBasis1DProtocol)


def test_nodes_are_the_closed_forms(bkd: Backend[Array]) -> None:
    """Gauss-Lobatto nodes are the extrema of T_{n-1} with both ends;
    Gauss nodes are the roots of T_n, strictly inside (-1, 1); both
    increasing and symmetric."""
    lobatto = ChebyshevGaussLobatto1D[Array]().nodes(7, bkd)
    bkd.assert_allclose(lobatto[0], bkd.asarray(-1.0))
    bkd.assert_allclose(lobatto[-1], bkd.asarray(1.0))
    # T_6' vanishes at the interior extrema: T_6'(x) = 6 U_5(x)
    theta = bkd.arccos(lobatto[1:-1])
    bkd.assert_allclose(bkd.sin(6 * theta), bkd.zeros((5,)), atol=1e-14)
    for npts in (1, 2, 5, 8):
        gauss = ChebyshevGauss1D[Array]().nodes(npts, bkd)
        bkd.assert_allclose(
            bkd.cos(npts * bkd.arccos(gauss)), bkd.zeros((npts,)), atol=1e-14
        )
        assert bkd.to_bool(bkd.all(bkd.abs(gauss) < 1.0))
        bkd.assert_allclose(gauss, -bkd.flip(gauss), atol=1e-15)
    for nodes in (lobatto, ChebyshevGauss1D[Array]().nodes(8, bkd)):
        assert bkd.to_bool(bkd.all(bkd.diff(nodes) > 0))


@pytest.mark.parametrize("family", _FAMILIES, ids=_IDS)
@pytest.mark.parametrize("npts", [2, 5, 8])
def test_derivative_is_exact_to_degree_n_minus_1(
    bkd: Backend[Array],
    family: Callable[[], CollocationBasis1DProtocol[Array]],
    npts: int,
) -> None:
    """D x^k = k x^(k-1) for k < n, and the rows sum to zero."""
    basis = family()
    nodes = basis.nodes(npts, bkd)
    D = basis.derivative_matrix(npts, bkd)
    assert D.shape == (npts, npts)
    bkd.assert_allclose(bkd.sum(D, axis=1), bkd.zeros((npts,)), atol=1e-12)
    for degree in range(npts):
        expected = degree * nodes ** (degree - 1) if degree > 0 else bkd.zeros((npts,))
        bkd.assert_allclose(D @ nodes**degree, expected, atol=1e-11)


@pytest.mark.parametrize("family", _FAMILIES, ids=_IDS)
def test_second_derivative(
    bkd: Backend[Array], family: Callable[[], CollocationBasis1DProtocol[Array]]
) -> None:
    """D @ D x^3 = 6 x on 10 nodes."""
    basis = family()
    nodes = basis.nodes(10, bkd)
    D = basis.derivative_matrix(10, bkd)
    bkd.assert_allclose(D @ (D @ nodes**3), 6.0 * nodes, atol=1e-9)


@pytest.mark.parametrize("family", _FAMILIES, ids=_IDS)
def test_spectral_convergence(
    bkd: Backend[Array], family: Callable[[], CollocationBasis1DProtocol[Array]]
) -> None:
    """The derivative of sin(pi x): errors at 10, 20, 40 nodes fall by
    more than 100 times, to below 1e-10."""
    basis = family()
    errors = []
    for npts in (10, 20, 40):
        nodes = basis.nodes(npts, bkd)
        D = basis.derivative_matrix(npts, bkd)
        error = D @ bkd.sin(np.pi * nodes) - np.pi * bkd.cos(np.pi * nodes)
        errors.append(bkd.to_float(bkd.max(bkd.abs(error))))
    assert errors[-1] < 0.01 * errors[0]
    assert errors[-1] < 1e-10


@pytest.mark.parametrize("family", _FAMILIES, ids=_IDS)
def test_interpolation_is_exact_to_degree_n_minus_1(
    bkd: Backend[Array], family: Callable[[], CollocationBasis1DProtocol[Array]]
) -> None:
    """On 6 nodes, a quintic at 9 points; the matrix is the identity at
    the nodes themselves."""
    basis = family()
    nodes = basis.nodes(6, bkd)
    points = bkd.linspace(-1.0, 1.0, 9)
    matrix = basis.interpolation_matrix(6, points, bkd)
    assert matrix.shape == (9, 6)

    def quintic(x: Array) -> Array:
        return 1.0 - 2.0 * x + x**3 - 0.5 * x**5

    bkd.assert_allclose(matrix @ quintic(nodes), quintic(points), atol=1e-13)
    bkd.assert_allclose(
        basis.interpolation_matrix(6, nodes, bkd), bkd.eye(6), atol=1e-15
    )


def test_transfer_between_the_staggered_grids(bkd: Backend[Array]) -> None:
    """A velocity grid of 7 Gauss-Lobatto nodes and a pressure grid of 5
    Gauss nodes: a quartic moves exactly from pressure to velocity nodes,
    and a quartic sampled on the velocity nodes moves exactly to the
    pressure nodes."""
    lobatto, gauss = ChebyshevGaussLobatto1D[Array](), ChebyshevGauss1D[Array]()
    velocity, pressure = lobatto.nodes(7, bkd), gauss.nodes(5, bkd)

    def quartic(x: Array) -> Array:
        return 0.3 + x - 2.0 * x**2 + x**4

    bkd.assert_allclose(
        gauss.interpolation_matrix(5, velocity, bkd) @ quartic(pressure),
        quartic(velocity),
        atol=1e-13,
    )
    bkd.assert_allclose(
        lobatto.interpolation_matrix(7, pressure, bkd) @ quartic(velocity),
        quartic(pressure),
        atol=1e-13,
    )


def test_one_gauss_node(bkd: Backend[Array]) -> None:
    """One Gauss node is the midpoint; its interpolant is a constant, with
    zero derivative."""
    gauss = ChebyshevGauss1D[Array]()
    bkd.assert_allclose(gauss.nodes(1, bkd), bkd.zeros((1,)), atol=1e-15)
    bkd.assert_allclose(gauss.derivative_matrix(1, bkd), bkd.zeros((1, 1)))
    bkd.assert_allclose(
        gauss.interpolation_matrix(1, bkd.asarray([-0.5, 0.7]), bkd),
        bkd.ones((2, 1)),
    )


def test_refusals(bkd: Backend[Array]) -> None:
    """Gauss-Lobatto needs 2 nodes, Gauss 1."""
    with pytest.raises(ValueError, match="Lobatto nodes need at least 2 nodes"):
        ChebyshevGaussLobatto1D[Array]().derivative_matrix(1, bkd)
    with pytest.raises(ValueError, match="Gauss nodes need at least 1 node"):
        ChebyshevGauss1D[Array]().nodes(0, bkd)

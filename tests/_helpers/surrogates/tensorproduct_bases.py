"""Univariate bases for testing the tensor-product evaluation kernels."""

from typing import List

from pyapprox.polynomials.univariate import LagrangeBasis1D, LegendrePolynomial1D
from pyapprox.util.backends.protocols import Array, Backend


def make_lagrange_bases(
    bkd: Backend[Array], nvars: int
) -> List[LagrangeBasis1D[Array]]:
    """Separate LagrangeBasis1D instances per dimension, each on the Gauss
    points of a degree-9 Legendre polynomial."""
    bases = []
    for _ in range(nvars):
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(10)
        bases.append(LagrangeBasis1D(bkd, poly.gauss_quadrature_rule))
    return bases

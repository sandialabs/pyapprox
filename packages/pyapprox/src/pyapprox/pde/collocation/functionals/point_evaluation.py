"""Interpolation weights for evaluating a 1D collocation state at
arbitrary points, through the Lagrange interpolant on the collocation
nodes.
"""
# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module

from typing import Sequence

from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.surrogates.affine.univariate.lagrange import (
    univariate_lagrange_polynomial,
)
from pyapprox.util.backends.protocols import Array, Backend


def point_interpolation_weights(
    basis: ChebyshevBasis1D[Array], points: Sequence[float], bkd: Backend[Array]
) -> Array:
    r"""The weights :math:`W` whose column :math:`k` interpolates a nodal
    state at :math:`x_k`: :math:`w_k^\top u = u(x_k)`, the Lagrange
    interpolant through the collocation nodes.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    points : sequence of float
        The points :math:`x_k`, in physical coordinates, each in the mesh's
        closed domain.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        :math:`W`, shape ``(npts, npoints)``.

    Raises
    ------
    ValueError
        If ``points`` is empty or a point lies outside the domain, where
        the interpolant would extrapolate.
    """
    points = list(points)
    if not points:
        raise ValueError("an interpolation needs at least one point")
    physical = basis.mesh().points()[0]
    lower = bkd.to_float(bkd.min(physical))
    upper = bkd.to_float(bkd.max(physical))
    slack = 1e-12 * (upper - lower)
    outside = [x for x in points if x < lower - slack or x > upper + slack]
    if outside:
        raise ValueError(f"points {outside} lie outside the domain [{lower}, {upper}]")
    targets = bkd.asarray([points])  # shape (1, npoints)
    targets = basis.mesh().transform().map_to_reference(targets)
    rows = univariate_lagrange_polynomial(basis.nodes(), targets[0], bkd)
    return rows.T

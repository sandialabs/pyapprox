r"""Quadrature weights at the collocation nodes of one side of a mesh.

A side of a tensor-product collocation mesh is where one reference
coordinate :math:`\xi_k` is :math:`\pm 1`. Its nodes are the tensor
product of the 1D Chebyshev-Gauss-Lobatto nodes along the other
directions, so a reference weight is the product of the 1D collocation
quadrature weights along those directions. The physical surface element
follows from Nanson's formula, :math:`dS = \det J\, \lVert J^{-\top}
e_k \rVert\, d\xi_\perp`, which holds for any smooth transform, so the
weights integrate over curved sides too.

:func:`boundary_side_quadrature` integrates on a side by any tensor
rule (:class:`SideRuleProtocol`), with the matrix interpolating values at
the side's nodes to the rule's points. Its weights are what integrate the
side's nodal interpolant of :math:`f`; the rule decides how exactly.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, runtime_checkable

from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.protocols.mesh import MeshWithTransformProtocol
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)
from pyapprox.polynomials.univariate.globalpoly.jacobi import LegendrePolynomial1D
from pyapprox.polynomials.univariate.lagrange import univariate_lagrange_polynomial
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SideRuleProtocol(Protocol):
    """A 1D quadrature rule on :math:`[-1, 1]` for one tangent direction
    of a mesh side, chosen from that direction's collocation nodes."""

    def rule(self, nodes: Array, bkd: Backend[Array]) -> tuple[Array, Array]:
        """Points and weights for a direction whose nodes are ``nodes``,
        shape (nnodes,); the weights sum to 2. Shapes: (npoints,),
        (npoints,)."""
        ...


class CollocationNodeRule:
    r"""The direction's own nodes with their Clenshaw–Curtis weights.

    The interpolation to its points is the identity, so the side's
    integral of :math:`f` is :math:`\sum_i w_i f_i`: lumped. Exact for
    polynomials of degree below the number of nodes, so the integral of a
    product of two nodal interpolants is not exact.
    """

    def rule(self, nodes: Array, bkd: Backend[Array]) -> tuple[Array, Array]:
        line = ChebyshevBasis1D(TransformedMesh1D(int(nodes.shape[0]), bkd), bkd)
        return nodes, CollocationQuadrature1D(line, bkd).full_domain_weights()


class GaussLegendreSideRule:
    r"""Gauss–Legendre with ``extra`` more points than the direction has
    nodes.

    With ``extra`` 0, exact on an affine side for the product of two of
    the side's nodal interpolants (degree :math:`2n - 2` per direction):
    the consistent mass. ``extra`` adds points for a curved side, whose
    surface element is not polynomial.

    Raises
    ------
    ValueError
        If ``extra`` is negative.
    """

    def __init__(self, extra: int = 0) -> None:
        if extra < 0:
            raise ValueError(f"extra must be >= 0, got {extra}")
        self._extra = extra

    def rule(self, nodes: Array, bkd: Backend[Array]) -> tuple[Array, Array]:
        npoints = int(nodes.shape[0]) + self._extra
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(max(npoints, 3))
        points, weights = poly.gauss_quadrature_rule(npoints)
        weights = bkd.flatten(weights)
        # The Legendre rule integrates the probability measure; rescale to
        # the length of [-1, 1].
        return bkd.flatten(points), 2.0 * weights / bkd.sum(weights)


@dataclass(frozen=True)
class SideQuadrature(Generic[Array]):
    """A quadrature on one side of a mesh, and the interpolation of values
    at the side's nodes to its points.

    ``interpolation`` has shape (npoints, nboundary), columns in
    ``boundary_indices`` order; ``points`` (ndim, npoints), physical;
    ``weights`` (npoints,), in the physical surface measure.
    """

    interpolation: Array
    points: Array
    weights: Array


def _tangent_tensor(
    mesh: MeshWithTransformProtocol[Array],
    boundary_id: int,
    rule: SideRuleProtocol,
) -> tuple[Array, Array, Array]:
    """The tensor rule over a side's tangent directions, the first
    fastest as the side's nodes are: the interpolation from the nodes,
    the reference points, and the reference weights."""
    bkd, ndim = mesh.bkd(), mesh.ndim()
    normal_dim = boundary_id // 2
    interpolation = bkd.ones((1, 1))
    weights = bkd.ones((1,))
    # Reference coordinates per direction, over the points so far.
    coords: dict[int, Array] = {}
    for dim in range(ndim):
        if dim == normal_dim:
            continue
        nodes = mesh.reference_nodes_1d(dim)
        points, along = rule.rule(nodes, bkd)
        lagrange = univariate_lagrange_polynomial(nodes, points, bkd)
        interpolation = bkd.kron(lagrange, interpolation)
        inner = int(weights.shape[0])
        coords = {
            d: bkd.kron(bkd.ones((points.shape[0],)), c) for d, c in coords.items()
        }
        coords[dim] = bkd.kron(points, bkd.ones((inner,)))
        weights = bkd.kron(along, weights)
    npoints = int(weights.shape[0])
    end = 1.0 if boundary_id % 2 == 1 else -1.0
    rows = [
        end * bkd.ones((npoints,)) if dim == normal_dim else coords[dim]
        for dim in range(ndim)
    ]
    return interpolation, bkd.stack(rows, axis=0), weights


def boundary_side_quadrature(
    mesh: MeshWithTransformProtocol[Array],
    boundary_id: int,
    rule: SideRuleProtocol,
) -> SideQuadrature[Array]:
    r"""A quadrature on a side of ``mesh`` by the tensor product of
    ``rule`` over its tangent directions, mapped by the mesh's transform.

    With ``S`` the interpolation and ``w`` the weights,
    :math:`w \cdot (S f)` integrates the side's nodal interpolant of ``f``
    (given at the side's nodes in ``mesh.boundary_indices(boundary_id)``
    order), exactly when the rule integrates it. The weights carry the
    surface element of Nanson's formula at the rule's points. A side of a
    1D mesh is its node, with weight 1.

    Parameters
    ----------
    mesh
        A tensor-product collocation mesh.
    boundary_id
        The side, in the meshes' numbering (:func:`boundary_quadrature_weights`).
    rule
        The 1D rule per tangent direction.

    Raises
    ------
    TypeError
        If ``mesh`` or ``rule`` does not satisfy its protocol.
    ValueError
        If ``boundary_id`` is not a side of ``mesh``.
    """
    _check_side(mesh, boundary_id)
    if not isinstance(rule, SideRuleProtocol):
        raise TypeError(
            f"rule must satisfy SideRuleProtocol, got {type(rule).__name__}"
        )
    bkd = mesh.bkd()
    interpolation, reference, weights = _tangent_tensor(mesh, boundary_id, rule)
    transform = mesh.transform()
    normal_dim = boundary_id // 2
    # J^{-T} e_k is row k of J^{-1}.
    inverse_row = bkd.inv(transform.jacobian_matrix(reference))[:, normal_dim, :]
    surface = transform.jacobian_determinant(reference) * bkd.sqrt(
        bkd.sum(inverse_row**2, axis=1)
    )
    return SideQuadrature(
        interpolation,
        transform.map_to_physical(reference),
        weights * bkd.abs(surface),
    )


def _check_side(mesh: MeshWithTransformProtocol[Array], boundary_id: int) -> None:
    if not isinstance(mesh, MeshWithTransformProtocol):
        raise TypeError(
            f"mesh must satisfy MeshWithTransformProtocol, got {type(mesh).__name__}"
        )
    ndim = mesh.ndim()
    if not 0 <= boundary_id < 2 * ndim:
        raise ValueError(
            f"boundary_id must lie in [0, {2 * ndim}) for a {ndim}D mesh, "
            f"got {boundary_id}"
        )


def boundary_quadrature_weights(
    mesh: MeshWithTransformProtocol[Array], boundary_id: int
) -> Array:
    r"""Weights :math:`w` with :math:`w \cdot f` approximating the
    integral of :math:`f` over a side of ``mesh``, given ``f`` at the
    side's nodes in ``mesh.boundary_indices(boundary_id)`` order.

    Exact, on an affine mesh, for polynomials of degree below the number
    of nodes along each direction of the side. A side of a 1D mesh is a
    point, with weight 1.

    Parameters
    ----------
    mesh
        A tensor-product collocation mesh.
    boundary_id
        The side: ``2 k`` is :math:`\xi_k = -1` and ``2 k + 1`` is
        :math:`\xi_k = +1`, the meshes' boundary numbering (left, right,
        bottom, top, front, back).

    Returns
    -------
    Array
        Shape: (nboundary,)

    Raises
    ------
    TypeError
        If ``mesh`` does not satisfy ``MeshWithTransformProtocol``.
    ValueError
        If ``boundary_id`` is not a side of ``mesh``.
    """
    _check_side(mesh, boundary_id)
    bkd = mesh.bkd()
    normal_dim = boundary_id // 2
    # The reference weights on every node, constant along the normal
    # direction; the first direction varies fastest, as the mesh's nodes.
    reference = bkd.ones((1,))
    for dim, npts in enumerate(mesh.npts_per_dim()):
        if dim == normal_dim:
            along = bkd.ones((npts,))
        else:
            line = ChebyshevBasis1D(TransformedMesh1D(npts, bkd), bkd)
            along = CollocationQuadrature1D(line, bkd).full_domain_weights()
        reference = bkd.kron(along, reference)
    nodes = mesh.boundary_indices(boundary_id)
    # J^{-T} e_k is row k of J^{-1}.
    inverse_row = bkd.inv(mesh.jacobian_matrix()[nodes])[:, normal_dim, :]
    surface = mesh.jacobian_determinant()[nodes] * bkd.sqrt(
        bkd.sum(inverse_row**2, axis=1)
    )
    return reference[nodes] * bkd.abs(surface)

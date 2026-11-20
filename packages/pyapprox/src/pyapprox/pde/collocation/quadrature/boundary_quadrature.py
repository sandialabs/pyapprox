r"""Quadrature weights at the collocation nodes of one side of a mesh.

A side of a tensor-product collocation mesh is where one reference
coordinate :math:`\xi_k` is :math:`\pm 1`. Its nodes are the tensor
product of the 1D Chebyshev-Gauss-Lobatto nodes along the other
directions, so a reference weight is the product of the 1D collocation
quadrature weights along those directions. The physical surface element
follows from Nanson's formula, :math:`dS = \det J\, \lVert J^{-\top}
e_k \rVert\, d\xi_\perp`, which holds for any smooth transform, so the
weights integrate over curved sides too.
"""

from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.protocols.mesh import MeshWithTransformProtocol
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)
from pyapprox.util.backends.protocols import Array


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

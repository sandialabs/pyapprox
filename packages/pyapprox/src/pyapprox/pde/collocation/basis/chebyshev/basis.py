"""Chebyshev collocation basis in any number of directions."""

from pyapprox.pde.collocation.basis.tensor_product import TensorProductBasis
from pyapprox.pde.collocation.protocols.mesh import MeshWithTransformProtocol
from pyapprox.util.backends.protocols import Array, Backend


class ChebyshevBasis(TensorProductBasis[Array]):
    """Chebyshev collocation basis on a tensor-product mesh.

    The derivative matrices of the mesh's 1D bases (Chebyshev-Gauss-Lobatto
    by default; Chebyshev-Gauss on a staggered grid), assembled by
    Kronecker products; the number of directions is the mesh's.

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        Mesh providing reference nodes, physical nodes and gradient
        factors, e.g. ``TransformedMesh``.
    bkd : Backend
        Computational backend.
    """

    def __init__(self, mesh: MeshWithTransformProtocol[Array], bkd: Backend[Array]):
        super().__init__(mesh, bkd)


def check_ndim(mesh: MeshWithTransformProtocol[Array], ndim: int) -> None:
    """Refuse a mesh whose number of directions is not ``ndim``.

    Raises
    ------
    ValueError
        If ``mesh.ndim() != ndim``.
    """
    if mesh.ndim() != ndim:
        raise ValueError(f"a {ndim}D basis needs a {ndim}D mesh, got {mesh.ndim()}D")

"""1D Chebyshev collocation basis."""

from pyapprox.pde.collocation.basis.chebyshev.basis import ChebyshevBasis, check_ndim
from pyapprox.pde.collocation.protocols.mesh import MeshWithTransformProtocol
from pyapprox.util.backends.protocols import Array, Backend


class ChebyshevBasis1D(ChebyshevBasis[Array]):
    """:class:`ChebyshevBasis` of an interval, with defaults for its one
    direction.

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        A 1D mesh, e.g. ``TransformedMesh1D`` with an optional transform.
    bkd : Backend
        Computational backend.

    Raises
    ------
    ValueError
        If the mesh is not 1D.

    Examples
    --------
    >>> transform = AffineTransform1D((0.0, 1.0), bkd)
    >>> mesh = TransformedMesh1D(10, bkd, transform)
    >>> basis = ChebyshevBasis1D(mesh, bkd)
    """

    def __init__(self, mesh: MeshWithTransformProtocol[Array], bkd: Backend[Array]):
        check_ndim(mesh, 1)
        super().__init__(mesh, bkd)

    def nodes(self) -> Array:
        """Return the reference Chebyshev-Gauss-Lobatto nodes. Shape:
        ``(npts,)``."""
        return self.nodes_1d(0)

    def derivative_matrix(self, order: int = 1, dim: int = 0) -> Array:
        """Return the physical derivative matrix of ``order``. Shape:
        ``(npts, npts)``.

        Raises
        ------
        ValueError
            If ``dim`` is not 0.
        """
        if dim != 0:
            raise ValueError(f"dim must be 0 for 1D basis, got {dim}")
        return super().derivative_matrix(order, 0)

    def reference_derivative_matrix(self, order: int = 1, dim: int = 0) -> Array:
        """Return the reference derivative matrix of ``order``. Shape:
        ``(npts, npts)``.

        Raises
        ------
        ValueError
            If ``dim`` is not 0.
        """
        if dim != 0:
            raise ValueError(f"dim must be 0 for 1D basis, got {dim}")
        return super().reference_derivative_matrix(order, 0)

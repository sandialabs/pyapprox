"""3D Chebyshev collocation basis."""

from pyapprox.pde.collocation.basis.chebyshev.basis import ChebyshevBasis, check_ndim
from pyapprox.pde.collocation.protocols.mesh import MeshWithTransformProtocol
from pyapprox.util.backends.protocols import Array, Backend


class ChebyshevBasis3D(ChebyshevBasis[Array]):
    """:class:`ChebyshevBasis` in three directions, with x/y/z accessors.

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        A 3D mesh, e.g. ``TransformedMesh3D`` with an optional transform.
    bkd : Backend
        Computational backend.

    Raises
    ------
    ValueError
        If the mesh is not 3D.

    Examples
    --------
    >>> transform = AffineTransform3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), bkd)
    >>> mesh = TransformedMesh3D(10, 10, 10, bkd, transform)
    >>> basis = ChebyshevBasis3D(mesh, bkd)
    """

    def __init__(self, mesh: MeshWithTransformProtocol[Array], bkd: Backend[Array]):
        check_ndim(mesh, 3)
        super().__init__(mesh, bkd)

    def nodes_x(self) -> Array:
        """Return the reference nodes in x. Shape: ``(npts_x,)``."""
        return self.nodes_1d(0)

    def nodes_y(self) -> Array:
        """Return the reference nodes in y. Shape: ``(npts_y,)``."""
        return self.nodes_1d(1)

    def nodes_z(self) -> Array:
        """Return the reference nodes in z. Shape: ``(npts_z,)``."""
        return self.nodes_1d(2)

    def derivative_matrix_x(self, order: int = 1) -> Array:
        """Return the physical x-derivative matrix of ``order``."""
        return self.derivative_matrix(order, 0)

    def derivative_matrix_y(self, order: int = 1) -> Array:
        """Return the physical y-derivative matrix of ``order``."""
        return self.derivative_matrix(order, 1)

    def derivative_matrix_z(self, order: int = 1) -> Array:
        """Return the physical z-derivative matrix of ``order``."""
        return self.derivative_matrix(order, 2)

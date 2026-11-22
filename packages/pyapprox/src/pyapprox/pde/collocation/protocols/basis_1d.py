"""What a collocation mesh and its tensor-product basis read from the 1D
basis of each direction.

A 1D basis owns its node family: where its nodes lie, how it
differentiates on them and how it interpolates from them. The mesh takes
its nodes from it and the tensor-product basis its derivative matrix, so
a derivative formula is never applied to another family's nodes. Each
basis computes these as suits its family (a closed form, or barycentric
weights from the nodes).
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class CollocationBasis1DProtocol(Protocol, Generic[Array]):
    """A 1D nodal basis on the reference interval :math:`[-1, 1]`."""

    def nodes(self, npts: int, bkd: Backend[Array]) -> Array:
        """The ``npts`` nodes, increasing. A first node of exactly -1 (a
        last node of exactly 1) gives the direction a lower (upper)
        boundary. Shape: ``(npts,)``"""
        ...

    def derivative_matrix(self, npts: int, bkd: Backend[Array]) -> Array:
        """:math:`D_{ij} = \\ell_j'(x_i)` on the ``npts`` nodes, the
        derivative of the nodal interpolant. Shape: ``(npts, npts)``"""
        ...

    def interpolation_matrix(
        self, npts: int, points: Array, bkd: Backend[Array]
    ) -> Array:
        """:math:`\\ell_j(p_i)`: values on the ``npts`` nodes to the
        interpolant at reference ``points``, shape ``(npoints,)``.
        Shape: ``(npoints, npts)``"""
        ...

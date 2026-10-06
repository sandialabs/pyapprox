"""DOF orderings of a vector field's components.

A vector field with ``ncomp`` components at ``npts`` points is stored as a
flat vector of length ``ncomp * npts``, and discretizations order it
differently: a Galerkin vector Lagrange basis interleaves the components at
each node, while component-major storage lists all of one component first.
A field map that produces vector fields computes the components as an array
``(ncomp, npts, ...)`` and lets an injected layout flatten them, so the map
knows nothing of any discretization. The discretization supplies its layout.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class VectorFieldLayoutProtocol(Protocol, Generic[Array]):
    """Flattens per-component values into a discretization's DOF order."""

    def flatten(self, components: Array) -> Array:
        """Flatten the leading two axes.

        Parameters
        ----------
        components : Array
            Shape: ``(ncomp, npts, ...)``; trailing axes (e.g. Jacobian
            columns) are carried through.

        Returns
        -------
        Array
            Shape: ``(ncomp * npts, ...)``.
        """
        ...


class InterleavedLayout(Generic[Array]):
    """Components interleaved per point: ``(x_0, y_0, x_1, y_1, ...)``.

    The layout of a Galerkin ``VectorLagrangeBasis``.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def flatten(self, components: Array) -> Array:
        bkd = self._bkd
        ncomp, npts = int(components.shape[0]), int(components.shape[1])
        trailing = tuple(int(n) for n in components.shape[2:])
        axes = (1, 0) + tuple(range(2, components.ndim))
        return bkd.reshape(
            bkd.transpose(components, axes), (ncomp * npts,) + trailing
        )

    def __repr__(self) -> str:
        return "InterleavedLayout()"


class BlockedLayout(Generic[Array]):
    """Components one after another: ``(x_0, x_1, ..., y_0, y_1, ...)``."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def flatten(self, components: Array) -> Array:
        ncomp, npts = int(components.shape[0]), int(components.shape[1])
        trailing = tuple(int(n) for n in components.shape[2:])
        return self._bkd.reshape(components, (ncomp * npts,) + trailing)

    def __repr__(self) -> str:
        return "BlockedLayout()"

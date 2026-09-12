r"""Where a field's values live, and what "error" means there.

Two things a fixed-basis operator surrogate may want from the space its
fields inhabit, and they are **independent**:

- an inner product, which decides what a best approximation is, which
  basis is orthonormal, and how large a residual is, and
- a rule for evaluating tabulated values away from the points they were
  tabulated at.

Building a POD basis needs the first and not the second. Evaluating a
fitted basis somewhere new needs the second and not the first. Fusing
them would make every implementation answer questions its consumer never
asks, and would foreclose the mixed-space case -- velocity and pressure
on different grids sharing a metric -- by admitting only one set of
sample points. So they are two protocols, and a consumer depends on the
one it uses.

**Deliberately not named for meshes or solvers.** Operator learning is
not a PDE method. A pixel grid, a time axis and a sensor network are as
valid here as a finite element mesh, and nothing in this package imports
a solver to say so.

:class:`OffGridEvaluatorProtocol` exists ahead of any implementation in
this package on purpose. It is the seam a PDE package hooks its own
evaluator into -- a spectral basis, an FE interpolant -- and because it
is structural, that package needs no import from here and this one needs
none from it. Declaring it now fixes the shape those implementations
must meet; guessing at *implementations* before a consumer exists is a
different thing, and the ones here are limited to what phases 2, 3 and 6
actually call.
"""

from __future__ import annotations

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import InnerProductProtocol


@runtime_checkable
class MetricSpaceProtocol(Protocol, Generic[Array]):
    r"""A space of field values with an inner product.

    The whole requirement for building an empirical basis and for
    measuring a field-space error: :math:`\langle f, g \rangle_M` says
    which subspace is optimal, which basis is orthonormal, and how large
    a residual is.

    Says nothing about *where* the values sit. A POD basis is determined
    by snapshots and a metric; the coordinates of the points those
    snapshots were sampled at never enter the eigenproblem. A domain
    that does know its coordinates is free to expose them -- both
    implementations here do -- but a consumer that only projects should
    depend on this and not on that.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def inner_product(self) -> InnerProductProtocol[Array]:
        """Return the metric :math:`M` on field values."""
        ...


@runtime_checkable
class OffGridEvaluatorProtocol(Protocol, Generic[Array]):
    """Tabulated values, and a rule for reading them off the table.

    What a fixed-basis surrogate needs to answer at a point it was not
    fitted at, and the seam a solver package implements to supply its
    own rule -- a tensor-product Lagrange interpolant, an FE basis
    evaluation -- without an import in either direction.

    Whether off-site evaluation is supported is answered by
    ``isinstance`` against this protocol rather than by a boolean on a
    larger one: an object that cannot interpolate does not implement it,
    instead of implementing it in order to decline. A grid-bound domain
    is then a fact about which protocols an object satisfies.

    No implementation in this package satisfies it yet. Nothing in the
    fixed-basis path evaluates off-site -- that is the capability a
    learned trunk buys -- so the implementations are left to the
    consumers that need them.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def ndim(self) -> int:
        """Spatial dimension of the sample points."""
        ...

    def nsites(self) -> int:
        """Number of points values are tabulated at."""
        ...

    def sample_points(self) -> Array:
        """Return the tabulation points. Shape: (ndim, nsites)."""
        ...

    def interpolate(self, site_values: Array, query_points: Array) -> Array:
        """Evaluate tabulated values at new points.

        Parameters
        ----------
        site_values : Array
            Values at :meth:`sample_points`. Shape: (nsites, nfields)
        query_points : Array
            Where to evaluate. Shape: (ndim, nquery)

        Returns
        -------
        Array
            Shape: (nquery, nfields)

        Notes
        -----
        Values are points-down-rows while ``query_points`` is
        coordinates-down-rows, crossing this package's
        samples-are-columns convention in one place. It is deliberate:
        an interpolant is built over *points*, so the axis it reduces
        over is the site axis. Stated here because an implementer will
        otherwise "fix" it.
        """
        ...

r"""Residuals evaluated on a subset of elements, before assembly sums them.

An assembled residual is a sum of element contributions,
:math:`F = \sum_e L_e^T F_e(u_e)`, where :math:`L_e` gathers element
:math:`e`'s DOFs. Summing into shared DOFs cannot be undone, so anything
that works on part of the mesh needs the operator to evaluate that part
itself:

- hyper-reduction re-weights the contributions of a few elements;
- parallel assembly gives each worker the elements of its partition and
  sums the partial results across partitions;
- workset assembly bounds memory by evaluating elements in batches.

The protocols here are that capability, independent of the assembly
library, the backend, the discretization (finite-volume cells serve as
elements as well as finite elements do) and the kind of parallelism.

:class:`RestrictedElementOperatorProtocol` is the primary contract: an
operator over one set of elements, set up once and then evaluated many
times from element-local states, numbering its DOFs locally and
reporting the local-to-global map. Consumers (hyper-reduction, partial
assembly) accept it and never require a whole-mesh operator, because on
a distributed-memory system none exists on any one process: each rank
builds the restricted operator of its own partition from its local mesh.
:class:`ElementwiseOperatorProtocol` is the convenience for when a
whole-mesh operator does exist (one process, or shared memory), which
can restrict itself to any subset.

Nothing global-length crosses these interfaces, and no single address
space is assumed. Which rank owns which DOFs, ghost exchange and the
reduction of partial results across ranks belong to the backend or
communicator, which the local-to-global map serves. Pickling is one way
to move a restricted operator between processes (multiprocessing), not a
requirement: under MPI each rank constructs its own.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


class ElementSet(Generic[Array]):
    """A subset of an operator's elements, in a fixed order.

    Parameters
    ----------
    indices : Array
        Distinct integer element indices. Shape: (nselected,).
    bkd : Backend
        Computational backend.

    Raises
    ------
    ValueError
        If ``indices`` is not 1D or repeats an element.
    """

    def __init__(self, indices: Array, bkd: Backend[Array]) -> None:
        if indices.ndim != 1:
            raise ValueError(f"indices must be 1D, got ndim={indices.ndim}")
        as_list = [int(index) for index in bkd.to_numpy(indices)]
        if len(set(as_list)) != len(as_list):
            raise ValueError("indices must not repeat an element")
        self._indices = indices
        self._bkd = bkd

    @classmethod
    def all(cls, nelements: int, bkd: Backend[Array]) -> "ElementSet[Array]":
        """Every element, in order."""
        return cls(bkd.arange(nelements, dtype=bkd.int64_dtype()), bkd)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def indices(self) -> Array:
        """Return the element indices. Shape: (nselected,)."""
        return self._indices

    def nselected(self) -> int:
        """Return the number of elements in the set."""
        return int(self._indices.shape[0])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nselected={self.nselected()})"


@runtime_checkable
class RestrictedElementOperatorProtocol(Protocol, Generic[Array]):
    r"""An operator over one element set, evaluated per element.

    Implementations precompute what the set needs when they are built.
    DOFs are numbered locally, ``0 .. nlocal_dofs - 1`` over every DOF
    the set's elements touch; :meth:`global_dofs` maps them to global
    DOFs. Every quantity is local to the set.

    The rows an element contributes to and the states it reads are
    reported separately. For finite elements they coincide (an element's
    nodes); for a finite-volume cell they do not (one row, a stencil of
    states), and for a face flux they do again (two cells each).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def elements(self) -> ElementSet[Array]:
        """Return the element set this operator is over."""
        ...

    def global_dofs(self) -> Array:
        """Return the global DOF of each local DOF. Shape: (nlocal_dofs,)."""
        ...

    def element_row_dofs(self) -> Array:
        """Return the local DOFs each element's residual rows belong to.

        Shape: (nselected, nrow); entries index :meth:`global_dofs`.
        """
        ...

    def element_state_dofs(self) -> Array:
        """Return the local DOFs whose states each element reads.

        Shape: (nselected, nstate); entries index :meth:`global_dofs`.
        """
        ...

    def element_residuals(self, local_states: Array, time: float) -> Array:
        r"""Return :math:`F_e(u_e, t)`. Shape: (nselected, nrow).

        ``local_states`` has shape (nselected, nstate), ordered as
        :meth:`element_state_dofs`.
        """
        ...

    def element_jacobians(self, local_states: Array, time: float) -> Array:
        r"""Return :math:`\partial F_e/\partial u_e`.

        Shape: (nselected, nrow, nstate); entry ``[e, i, j]`` is the
        derivative of row ``i`` with respect to state ``j``.
        """
        ...


@runtime_checkable
class ElementwiseOperatorProtocol(Protocol, Generic[Array]):
    r"""An operator that can be restricted to a subset of its elements.

    Conformance: restricted to :meth:`ElementSet.all`, scattering
    ``element_residuals`` to global rows through ``element_row_dofs``
    then ``global_dofs`` must give the operator's assembled residual,
    and likewise for the Jacobian. Over a partition of the elements the
    restricted operators' scattered contributions must sum to the same.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nelements(self) -> int:
        """Return the total number of elements."""
        ...

    def restrict(
        self, elements: ElementSet[Array]
    ) -> RestrictedElementOperatorProtocol[Array]:
        """Return the operator restricted to ``elements``, set up once."""
        ...

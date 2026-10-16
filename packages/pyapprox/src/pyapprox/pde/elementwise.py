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

    Only indices: whether the elements are homogeneous enough to batch
    is the operator's knowledge, so ``restrict`` refuses a mixed set and
    :func:`check_restricted_operator` checks the result.

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

    The element Jacobian is supplied as forward and transpose actions on
    a block of directions, never as a required matrix: a matrix-free or
    partially assembled implementation has only actions, a consumer
    reducing onto ``k`` basis vectors needs only ``k`` of them, and the
    block itself is the action on the identity
    (:func:`element_jacobian_blocks`).

    **The element set is homogeneous:** every element has the same
    ``nrow`` and ``nstate``, so per-element quantities batch as
    rectangular arrays, as element libraries batch them (libCEED and
    MFEM element restrictions, Phalanx worksets). A mesh mixing element
    types, orders or stencil sizes is several restricted operators, one
    per homogeneous group. Rows are never padded: every DOF entry is a
    valid local DOF and no element lists one twice. Consumers check this
    at their public boundary with :func:`check_restricted_operator`.

    Shapes are logical: entry ``[e, i, k]`` means element ``e``, local
    DOF ``i``, direction ``k``. How an implementation stores them (for
    instance element index last, for coalesced GPU access) is its own
    choice behind those indices.
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

    def element_jacobian_actions(
        self, local_states: Array, time: float, directions: Array
    ) -> Array:
        r"""Return :math:`J_e d_e`, with :math:`J_e = \partial F_e/\partial u_e`.

        ``directions`` has shape (nselected, nstate, k): ``k`` directions
        per element, ordered as :meth:`element_state_dofs`. Returns shape
        (nselected, nrow, k).

        Raises
        ------
        ValueError
            If ``directions`` is not 3D or its first two dimensions are
            not (nselected, nstate).
        """
        ...

    def element_jacobian_transpose_actions(
        self, local_states: Array, time: float, adjoints: Array
    ) -> Array:
        r"""Return :math:`J_e^T a_e`, the plain transpose.

        ``adjoints`` has shape (nselected, nrow, k), ordered as
        :meth:`element_row_dofs`. Returns shape (nselected, nstate, k).

        Raises
        ------
        ValueError
            If ``adjoints`` is not 3D or its first two dimensions are not
            (nselected, nrow).
        """
        ...


@runtime_checkable
class ElementwiseOperatorProtocol(Protocol, Generic[Array]):
    r"""An operator that can be restricted to a subset of its elements.

    Conformance: restricted to :meth:`ElementSet.all`, scattering
    ``element_residuals`` to global rows through ``element_row_dofs``
    then ``global_dofs`` must give the operator's assembled residual,
    and likewise for the Jacobian's actions on gathered directions. Over
    a partition of the elements the
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
        """Return the operator restricted to ``elements``, set up once.

        Raises
        ------
        ValueError
            If ``elements`` mixes elements of different local sizes (see
            ``RestrictedElementOperatorProtocol``): an operator over a
            mixed mesh refuses the set rather than pad it.
        """
        ...


def check_element_block(block: Array, nselected: int, nlocal: int, name: str) -> None:
    """Check a per-element block of vectors has shape (nselected, nlocal, k).

    For implementations of the action methods to validate their input.

    Raises
    ------
    ValueError
        If ``block`` is not 3D or its first two dimensions differ.
    """
    if block.ndim != 3:
        raise ValueError(f"{name} must be 3D (nselected, n, k), got ndim={block.ndim}")
    if int(block.shape[0]) != nselected or int(block.shape[1]) != nlocal:
        raise ValueError(
            f"{name} must have leading shape ({nselected}, {nlocal}), got "
            f"{tuple(block.shape[:2])}"
        )


def _check_dofs(
    dofs: Array, nselected: int, nlocal: int, name: str, bkd: Backend[Array]
) -> None:
    if dofs.ndim != 2 or int(dofs.shape[0]) != nselected:
        raise ValueError(
            f"{name} must have shape ({nselected}, n), got {tuple(dofs.shape)}"
        )
    if int(dofs.shape[1]) == 0:
        raise ValueError(f"{name} must list at least one DOF per element")
    if nselected == 0:
        return
    if bkd.any_bool(dofs < 0) or bkd.any_bool(dofs >= nlocal):
        raise ValueError(
            f"{name} entries must be local DOFs in [0, {nlocal}); padded rows "
            "are not allowed"
        )
    # After sorting each row, a repeated DOF sits next to itself.
    sorted_rows = bkd.sort(dofs, axis=1)
    if bkd.any_bool(bkd.equal(sorted_rows[:, 1:], sorted_rows[:, :-1])):
        raise ValueError(
            f"{name} lists a DOF twice in one element; padded rows are not allowed"
        )


def check_restricted_operator(
    restricted: RestrictedElementOperatorProtocol[Array],
) -> None:
    """Check a restricted operator is over a homogeneous, unpadded set.

    Called once by consumers at their public boundary. Equal local sizes
    follow from the DOF arrays being rectangular; this rejects the ways
    a mixed set could be forced into them: entries outside the local
    DOFs (a sentinel such as -1) and a DOF repeated within an element.

    Raises
    ------
    TypeError
        If ``restricted`` does not satisfy the protocol.
    ValueError
        If the DOF arrays are malformed or padded.
    """
    if not isinstance(restricted, RestrictedElementOperatorProtocol):
        raise TypeError(
            "restricted must satisfy RestrictedElementOperatorProtocol, got "
            f"{type(restricted).__name__}"
        )
    bkd = restricted.bkd()
    global_dofs = restricted.global_dofs()
    if global_dofs.ndim != 1:
        raise ValueError(f"global_dofs must be 1D, got ndim={global_dofs.ndim}")
    nlocal = int(global_dofs.shape[0])
    nselected = restricted.elements().nselected()
    _check_dofs(
        restricted.element_row_dofs(), nselected, nlocal, "element_row_dofs", bkd
    )
    _check_dofs(
        restricted.element_state_dofs(), nselected, nlocal, "element_state_dofs", bkd
    )


def element_jacobian_blocks(
    restricted: RestrictedElementOperatorProtocol[Array],
    local_states: Array,
    time: float,
) -> Array:
    r"""Return each element's Jacobian, :math:`J_e`, as a dense block.

    The forward action on the identity, so any implementation of the
    actions yields its blocks, at the order of cost of forming them
    directly. For consumers that need entries (sparse assembly, element
    storage); one reducing onto a few directions should apply the
    actions to those directions instead.

    Returns
    -------
    Array
        Shape: (nselected, nrow, nstate); entry ``[e, i, j]`` is the
        derivative of row ``i`` with respect to state ``j``.
    """
    bkd = restricted.bkd()
    state_dofs = restricted.element_state_dofs()
    nselected, nstate = int(state_dofs.shape[0]), int(state_dofs.shape[1])
    identity = bkd.tile(bkd.eye(nstate)[None, :, :], (nselected, 1, 1))
    return restricted.element_jacobian_actions(local_states, time, identity)

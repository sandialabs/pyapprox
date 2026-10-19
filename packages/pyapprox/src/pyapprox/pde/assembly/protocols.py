r"""Assembly into a linear-algebra stack.

An assembled PDE operator is a sum of element contributions,
:math:`A = \sum_e L_e^\top w_e A_e R_e`, where :math:`L_e` and
:math:`R_e` gather element :math:`e`'s row and column DOFs and
:math:`w_e` is an optional element weight (1 for full assembly, a
reduced-quadrature weight for hyper-reduction). An *assembly stack* owns
where those sums go: it hands the assembler a *sink* whose sparsity
pattern is fixed once from the element DOF maps, accepts batched element
blocks, and produces an operator of the stack's own type.

The assembler never sees the stack's storage. It sees:

- sinks for matrices and vectors, rectangular in general (rows and
  columns may belong to different spaces);
- a declared bulk capability, for a single-process assembler that
  already produces a whole sparse matrix, ``None`` on a distributed
  stack;
- the prolongation :math:`P` from the solve space (the true DOFs a
  solver works on) to the assembly space (the local DOFs elements
  write). The solved operator is :math:`P^\top A P`. Today :math:`P = I`
  on every mesh; hanging-node constraints and shared parallel DOFs make
  it nontrivial, without changing any assembler.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from scipy.sparse import spmatrix

from pyapprox.util.backends.protocols import Array_contra
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    OperatorProperties,
    RowSetProtocol,
    SpaceDescriptor,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F = TypeVar("F")
V_co = TypeVar("V_co", covariant=True)
F_co = TypeVar("F_co", covariant=True)
# Element data only flows in to a sink, so its array types are
# contravariant: I for DOF index arrays, L for element values.
I_contra = TypeVar("I_contra", contravariant=True)
L_contra = TypeVar("L_contra", contravariant=True)
# A DOF map only flows out of its supplier.
I_co = TypeVar("I_co", covariant=True)


@runtime_checkable
class ElementDofMapProtocol(Protocol, Generic[I_co]):
    """Supplies each element's DOFs, from which a sink's pattern is built.

    ``I_co`` is the index-array type, the stack's choice: a supplier and
    a stack pair only when they agree on it.
    """

    def element_dofs(self) -> I_co:
        """Each element's DOFs, one element per row, in assembly-space
        numbering. Shape: (nelems, ndofs_per_element)."""
        ...


@runtime_checkable
class BulkMatrixSinkProtocol(Protocol):
    """Accepts a whole assembled sparse matrix, on one process."""

    def add_assembled(self, matrix: spmatrix) -> None:
        """Add ``matrix``, whose entries must lie in the sink's pattern.

        Raises
        ------
        ValueError
            If the shape differs from the sink's, or an entry lies outside
            the pattern.
        """
        ...


@runtime_checkable
class MatrixSinkProtocol(Protocol, Generic[V, F_co, I_contra, L_contra]):
    """Accumulates element matrices into an operator of fixed pattern.

    Created for one pair of spaces and one sparsity pattern; reused for
    every assembly with that pattern (``zero``, then adds, then
    ``assemble``). ``I_contra`` is the array type of DOF maps and
    ``L_contra`` that of element values and weights.
    """

    def add_element_matrices(
        self,
        row_dofs: I_contra,
        col_dofs: I_contra,
        values: L_contra,
        weights: Optional[L_contra] = None,
    ) -> None:
        """Add ``weights[e] * values[e]`` at ``(row_dofs[e], col_dofs[e])``.

        Shapes: ``row_dofs`` (nsel, nrow), ``col_dofs`` (nsel, ncol),
        ``values`` (nsel, nrow, ncol), ``weights`` (nsel,). DOFs index the
        range and domain assembly spaces.

        Raises
        ------
        ValueError
            If a shape disagrees or an entry lies outside the pattern.
        """
        ...

    def bulk(self) -> Optional[BulkMatrixSinkProtocol]:
        """The bulk capability, or ``None`` where it does not exist."""
        ...

    def zero(self) -> None:
        """Reset every accumulated value, keeping the pattern."""
        ...

    def assemble(
        self, properties: Optional[OperatorProperties] = None
    ) -> ActionOperatorProtocol[V, F_co]:
        """Return the accumulated operator, independent of later adds."""
        ...


@runtime_checkable
class BulkVectorSinkProtocol(Protocol, Generic[L_contra]):
    """Accepts a whole assembled vector, on one process.

    ``L_contra`` is the sink's element-value array type: a whole vector
    is a 1-D array of the same kind.
    """

    def add_assembled(self, vector: L_contra) -> None:
        """Add a 1-D array of the sink's size."""
        ...


@runtime_checkable
class VectorSinkProtocol(Protocol, Generic[V_co, I_contra, L_contra]):
    """Accumulates element vectors into a single vector."""

    def add_element_vectors(
        self, row_dofs: I_contra, values: L_contra, weights: Optional[L_contra] = None
    ) -> None:
        """Add ``weights[e] * values[e]`` at ``row_dofs[e]``.

        Shapes: ``row_dofs`` and ``values`` (nsel, nrow), ``weights``
        (nsel,).
        """
        ...

    def bulk(self) -> Optional[BulkVectorSinkProtocol[L_contra]]:
        """The bulk capability, or ``None`` where it does not exist."""
        ...

    def zero(self) -> None:
        """Reset the accumulated vector."""
        ...

    def assemble(self) -> V_co:
        """Return the accumulated vector, a single column, independent of
        later adds."""
        ...


@runtime_checkable
class AssemblyStackProtocol(Protocol, Generic[V, F, I_contra, L_contra]):
    """Where assembled operators and vectors are built.

    ``V`` is the stack's vector type, ``F`` the format of the operators
    its sinks produce, ``I_contra`` the array type of element DOF maps and
    ``L_contra`` that of element values.
    """

    def space(self, descriptor: SpaceDescriptor) -> VectorSpaceProtocol[V]:
        """The vector space of ``descriptor``."""
        ...

    def matrix_sink(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: I_contra,
        col_dofs: I_contra,
    ) -> MatrixSinkProtocol[V, F, I_contra, L_contra]:
        """A sink whose pattern covers every ``(row_dofs[e], col_dofs[e])``
        block. Both spaces are assembly spaces."""
        ...

    def vector_sink(
        self, descriptor: SpaceDescriptor
    ) -> VectorSinkProtocol[V, I_contra, L_contra]:
        """A sink for vectors of the assembly space ``descriptor``."""
        ...

    def row_set(self, descriptor: SpaceDescriptor, rows: I_contra) -> RowSetProtocol[V]:
        """The stack's own selection of ``rows`` of ``descriptor``.

        ``rows`` is a 1-D index array in the assembly-space numbering that
        element DOF maps use; the stack maps it to its own representation
        and numbering.
        """
        ...

    def prolongation(self, assembly: SpaceDescriptor) -> ActionOperatorProtocol[V, F]:
        """:math:`P`, from the solve space of ``assembly`` to ``assembly``."""
        ...

    def solve_operator(
        self, op: ActionOperatorProtocol[V, F]
    ) -> ActionOperatorProtocol[V, F]:
        r"""The stored :math:`P_r^\top A P_d` of an assembled operator,
        between the solve spaces of its range and domain."""
        ...


@runtime_checkable
class BulkCapableMatrixSinkProtocol(
    MatrixSinkProtocol[V, F_co, I_contra, L_contra],
    Protocol[V, F_co, I_contra, L_contra],
):
    """A matrix sink that always accepts whole assembled matrices."""

    def bulk(self) -> BulkMatrixSinkProtocol:
        """The bulk capability, never ``None``."""
        ...


@runtime_checkable
class BulkCapableAssemblyStackProtocol(
    AssemblyStackProtocol[V, F, I_contra, L_contra],
    Protocol[V, F, I_contra, L_contra],
):
    """A stack whose matrix sinks accept whole assembled matrices.

    An assembler that produces a whole sparse matrix on one process
    needs this, not just :class:`AssemblyStackProtocol`, so pairing it
    with a stack that lacks the capability (a distributed one) is a type
    error rather than a silent, unscalable fallback. Such stacks take
    element-wise assembly instead.
    """

    def matrix_sink(
        self,
        range: SpaceDescriptor,
        domain: SpaceDescriptor,
        row_dofs: I_contra,
        col_dofs: I_contra,
    ) -> BulkCapableMatrixSinkProtocol[V, F, I_contra, L_contra]:
        """A sink as in :class:`AssemblyStackProtocol`, with bulk adds."""
        ...


@runtime_checkable
class JacobianAssemblerProtocol(Protocol[Array_contra, V, F_co]):
    """Assembles a state Jacobian into a stack operator.

    How the entries reach the stack (a whole matrix through a bulk sink,
    or element blocks) is the assembler's own; it states what it needs
    of the stack in its constructor's types.
    """

    def assemble_jacobian(
        self, state: Array_contra, time: float
    ) -> ActionOperatorProtocol[V, F_co]:
        """The Jacobian at ``(state, time)``, independent of later calls."""
        ...

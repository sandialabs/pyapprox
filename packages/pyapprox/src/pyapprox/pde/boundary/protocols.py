"""Solver-neutral boundary-condition role protocols.

Splits boundary conditions into two disjoint roles so dispatch never
depends on implementation types:

- ``WeakFormBCProtocol``: natural BCs (Neumann, Robin) assembled into
  the variational form. They contribute to the load vector, stiffness
  matrix, residual, and Jacobian but never replace rows.
- ``EssentialBCProtocol``: strong constraints on DOF values
  (Dirichlet). They expose constrained DOFs and values only; row
  replacement is performed by a constraint set, not by the BC itself.

``ConstraintSetProtocol`` is the interface consumed by solvers and
time-integration wrappers; ``DirichletConstraintSet`` (constraint_set.py)
is the default implementation.
"""

from typing import Generic, Protocol, Union, runtime_checkable

from scipy.sparse import spmatrix

from pyapprox.pde.boundary.classification import BCDofClassification
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class WeakFormBCProtocol(Protocol, Generic[Array]):
    """Natural BCs assembled into the variational form (Neumann, Robin).

    A weak-form BC has no row-replacement methods to misuse: it only
    adds boundary-integral contributions to assembled operators.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def apply_to_load(self, load: Array, time: float) -> Array:
        """Add the boundary-integral contribution to the load vector.

        Parameters
        ----------
        load : Array
            Load vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified load vector. Shape: (nstates,)
        """
        ...

    def apply_to_stiffness(
        self, stiffness: Union[spmatrix, Array], time: float
    ) -> Union[spmatrix, Array]:
        """Add the boundary-integral contribution to the stiffness matrix.

        Robin BCs add ``alpha * integral_{Gamma} u * phi ds``; a pure
        Neumann BC returns the matrix unchanged.

        Parameters
        ----------
        stiffness : sparse matrix or Array
            Stiffness matrix. Shape: (nstates, nstates)
        time : float
            Current time.

        Returns
        -------
        sparse matrix or Array
            Modified stiffness matrix (same type as input).
        """
        ...

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Add the boundary-integral contribution to the residual.

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified residual. Shape: (nstates,)
        """
        ...

    def apply_to_jacobian(
        self, jacobian: Union[spmatrix, Array], state: Array, time: float
    ) -> Union[spmatrix, Array]:
        """Add the boundary-integral contribution to the Jacobian.

        Parameters
        ----------
        jacobian : sparse matrix or Array
            Jacobian matrix. Shape: (nstates, nstates)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        sparse matrix or Array
            Modified Jacobian (same type as input).
        """
        ...


@runtime_checkable
class EssentialBCProtocol(Protocol, Generic[Array]):
    """Strong constraints on DOF values (Dirichlet).

    An essential BC has no assembly methods; it only reports which DOFs
    it constrains and their prescribed values. Constrained DOF LOCATIONS
    must be time-invariant (values may vary with time).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def constrained_dofs(self) -> Array:
        """Return the constrained global DOF indices.

        Returns
        -------
        Array
            Integer DOF indices. Shape: (ndofs,)
        """
        ...

    def constrained_values(self, time: float) -> Array:
        """Return the prescribed values at the constrained DOFs.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Prescribed values. Shape: (ndofs,)
        """
        ...


@runtime_checkable
class ConstraintSetProtocol(Protocol, Generic[Array]):
    """Aggregated essential constraints applied to assembled systems.

    Consumed by steady solvers and BC-enforcing time residual wrappers;
    ``DirichletConstraintSet`` is the default implementation. All
    methods must be exact no-ops / identity passthroughs when the set
    is empty.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def dofs(self) -> Array:
        """Return all constrained DOF indices. Shape: (ndofs,)"""
        ...

    def values(self, time: float) -> Array:
        """Return all prescribed values at ``time``. Shape: (ndofs,)"""
        ...

    def ndofs(self) -> int:
        """Return the number of constrained DOFs."""
        ...

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Replace constrained rows with ``state[d] - g(time)``."""
        ...

    def apply_to_jacobian(
        self, jacobian: Union[spmatrix, Array]
    ) -> Union[spmatrix, Array]:
        """Replace constrained rows with identity rows ``e_d``."""
        ...

    def apply_to_mass(
        self, mass: Union[spmatrix, Array]
    ) -> Union[spmatrix, Array]:
        """Return the mass matrix with identity rows at constrained DOFs.

        The result is cached per input matrix object.
        """
        ...

    def zero_rows(
        self, matrix: Union[spmatrix, Array]
    ) -> Union[spmatrix, Array]:
        """Zero constrained rows (e.g. dR/dp, same-step HVP outputs)."""
        ...

    def zero_cols(
        self, matrix: Union[spmatrix, Array]
    ) -> Union[spmatrix, Array]:
        """Zero constrained columns (transposed off-diagonal blocks)."""
        ...

    def zero_entries(self, vec: Array) -> Array:
        """Zero constrained entries of a vector (adjoint RHS)."""
        ...

    def inject(self, state: Array, time: float) -> Array:
        """Return ``state`` with ``state[d] = g(time)`` (initial guesses)."""
        ...

    def classification(self) -> BCDofClassification:
        """Return the DOF classification (essential == row_replaced)."""
        ...

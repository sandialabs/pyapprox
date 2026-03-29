"""Base physics class for spectral collocation methods.

Provides common functionality for PDE physics implementations.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Tuple

from pyapprox.pde.collocation.protocols import (
    BasisProtocol,
    BCDofClassification,
    BoundaryConditionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class AbstractPhysics(ABC, Generic[Array]):
    """Abstract base class for PDE physics.

    Provides common infrastructure for implementing PDE physics with
    boundary conditions. Subclasses implement specific PDEs.

    Parameters
    ----------
    basis : BasisProtocol
        Collocation basis (provides nodes, derivative matrices).
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        basis: BasisProtocol[Array],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._bkd = bkd
        self._boundary_conditions: List[BoundaryConditionProtocol[Array]] = []

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> BasisProtocol[Array]:
        """Return the collocation basis."""
        return self._basis

    def npts(self) -> int:
        """Return number of collocation points."""
        return int(self._basis.npts())

    @abstractmethod
    def ncomponents(self) -> int:
        """Return number of solution components.

        E.g., 1 for scalar PDE, 2-3 for coupled systems.
        """
        ...

    def nstates(self) -> int:
        """Return total number of states (ncomponents * npts)."""
        return self.ncomponents() * self.npts()

    def set_boundary_conditions(
        self, bcs: List[BoundaryConditionProtocol[Array]]
    ) -> None:
        """Set boundary conditions.

        Parameters
        ----------
        bcs : List[BoundaryConditionProtocol]
            List of boundary conditions to apply.
        """
        self._boundary_conditions = bcs

    def boundary_conditions(self) -> List[BoundaryConditionProtocol[Array]]:
        """Return list of boundary conditions."""
        return self._boundary_conditions

    @abstractmethod
    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        For steady problems: solve residual(u) = 0
        For transient problems: du/dt = residual(u, t)

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (nstates,)
        """
        ...

    @abstractmethod
    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/du.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        ...

    def apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array, time: float = 0.0
    ) -> Tuple[Array, Array]:
        """Apply boundary conditions to residual and Jacobian.

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        jacobian : Array
            Jacobian matrix. Shape: (nstates, nstates)
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Tuple[Array, Array]
            Modified (residual, jacobian).
        """
        for bc in self._boundary_conditions:
            residual = bc.apply_to_residual(residual, state, time)
            jacobian = bc.apply_to_jacobian(jacobian, state, time)
        return residual, jacobian

    def bc_dof_classification(self) -> BCDofClassification:
        """Classify boundary DOFs for adjoint operations.

        For collocation: all BCs replace residual rows, so row_replaced
        contains all BC DOFs. Essential contains only Dirichlet/periodic
        DOFs (where is_essential() returns True).

        For a future Galerkin solver, this method would classify
        differently: only strongly-enforced Dirichlet DOFs would appear
        in row_replaced, while weakly-enforced natural BCs would not
        appear in either list.

        Returns
        -------
        BCDofClassification
            Classification with essential and row_replaced index lists.
        """
        essential, row_replaced = [], []
        for bc in self._boundary_conditions:
            bc_idx = bc.boundary_indices()
            for ii in range(bc_idx.shape[0]):
                idx = self._bkd.to_int(bc_idx[ii])
                row_replaced.append(idx)
                if bc.is_essential():
                    essential.append(idx)
        return BCDofClassification(essential, row_replaced)

    def apply_bc_to_mass(self, mass: Array) -> Array:
        """Apply BC enforcement to mass matrix for adjoint at t=0.

        At t=0, the adjoint system uses M instead of A_n. Essential
        (Dirichlet) DOFs need identity rows/columns because the forward
        equation at t=0 is y_0[b] = g(0), which has identity Jacobian
        w.r.t. y_0. Natural (Robin) DOFs keep their normal mass row.

        Parameters
        ----------
        mass : Array
            The mass matrix M. Shape: (nstates, nstates).

        Returns
        -------
        Array
            Modified mass matrix with identity at essential BC DOFs.
        """
        mass = self._bkd.copy(mass)
        for idx in self.bc_dof_classification().essential:
            mass[idx, :] = 0.0
            mass[:, idx] = 0.0
            mass[idx, idx] = 1.0
        return mass

    def mass_matrix(self) -> Array:
        """Return mass matrix for time integration.

        For standard ODEs, this is the identity matrix.
        For DAEs, this may be singular.

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        return self._bkd.eye(self.nstates())

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix to a vector.

        Default is identity (returns vec unchanged). Subclasses with
        non-identity mass matrices should override for efficiency.

        Parameters
        ----------
        vec : Array
            Vector to multiply. Shape: (nstates,)

        Returns
        -------
        Array
            M @ vec. Shape: (nstates,)
        """
        return vec


class AbstractScalarPhysics(AbstractPhysics[Array]):
    """Abstract base class for scalar (single component) PDE physics."""

    def ncomponents(self) -> int:
        """Return 1 for scalar physics."""
        return 1


class AbstractVectorPhysics(AbstractPhysics[Array]):
    """Abstract base class for vector (multi-component) PDE physics.

    Parameters
    ----------
    basis : BasisProtocol
        Collocation basis.
    bkd : Backend
        Computational backend.
    ncomponents : int
        Number of solution components.
    """

    def __init__(
        self,
        basis: BasisProtocol[Array],
        bkd: Backend[Array],
        ncomponents: int,
    ):
        super().__init__(basis, bkd)
        self._ncomponents = ncomponents

    def ncomponents(self) -> int:
        """Return number of solution components."""
        return self._ncomponents

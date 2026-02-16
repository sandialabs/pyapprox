"""Base physics class for Galerkin finite element methods.

Provides common infrastructure for PDE physics implementations.
"""

from typing import Generic, Tuple, List, Optional, Callable, Union
from abc import ABC, abstractmethod

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.sparse_utils import solve_maybe_sparse
from pyapprox.typing.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.typing.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
    DirichletBCProtocol,
    NeumannBCProtocol,
    RobinBCProtocol,
)

# Import skfem for assembly
try:
    from skfem import asm, LinearForm, BilinearForm
    from skfem.models.poisson import mass
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class AbstractGalerkinPhysics(ABC, Generic[Array]):
    """Abstract base class for Galerkin PDE physics.

    Provides common infrastructure for:
    - Mass matrix assembly
    - Boundary condition application
    - Residual and Jacobian computation

    Subclasses must implement:
    - _assemble_stiffness: Assemble stiffness matrix
    - _assemble_load: Assemble load vector
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
    ):
        self._basis = basis
        self._bkd = basis.bkd()
        self._boundary_conditions = boundary_conditions or []

        # Cache mass matrix (constant for linear elements)
        self._mass_matrix_cached: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> GalerkinBasisProtocol[Array]:
        """Return the finite element basis."""
        return self._basis

    def nstates(self) -> int:
        """Return total number of DOFs."""
        return self._basis.ndofs()

    def mass_matrix(self):
        """Return the mass matrix from weak form.

        For Galerkin FEM with Lagrange elements:
            M_ij = integral(phi_i * phi_j)

        Returns
        -------
        sparse matrix
            Mass matrix in CSR format. Shape: (nstates, nstates)
        """
        if self._mass_matrix_cached is None:
            skfem_basis = self._basis.skfem_basis()
            self._mass_matrix_cached = asm(mass, skfem_basis)
        return self._mass_matrix_cached

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x.

        This default implementation uses a direct solve. Override in subclasses
        to exploit structure (e.g., diagonal/lumped mass matrix).

        Parameters
        ----------
        rhs : Array
            Right-hand side vector. Shape: (nstates,) or (nstates, ncols)

        Returns
        -------
        Array
            Solution x = M^{-1} * rhs. Same shape as rhs.
        """
        return solve_maybe_sparse(self._bkd, self.mass_matrix(), rhs)

    @abstractmethod
    def _assemble_stiffness(self, state: Array, time: float) -> Array:
        """Assemble stiffness matrix K(u, t).

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Stiffness matrix. Shape: (nstates, nstates)
        """
        ...

    @abstractmethod
    def _assemble_load(self, state: Array, time: float) -> Array:
        """Assemble load vector b(u, t).

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Load vector. Shape: (nstates,)
        """
        ...

    def _apply_bc_to_stiffness(self, stiffness: Array, time: float) -> Array:
        """Apply boundary conditions that modify stiffness matrix.

        Robin BCs add boundary mass terms to the stiffness matrix.

        Parameters
        ----------
        stiffness : Array
            Stiffness matrix. Shape: (nstates, nstates)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified stiffness matrix.
        """
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                stiffness = bc.apply_to_stiffness(stiffness, time)
        return stiffness

    def _apply_bc_to_load(self, load: Array, time: float) -> Array:
        """Apply boundary conditions that modify load vector.

        Neumann and Robin BCs add boundary integral contributions to load.

        Parameters
        ----------
        load : Array
            Load vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified load vector.
        """
        for bc in self._boundary_conditions:
            if isinstance(bc, NeumannBCProtocol):
                load = bc.apply_to_load(load, time)
            elif isinstance(bc, RobinBCProtocol):
                load = bc.apply_to_load(load, time)
        return load

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du without Dirichlet enforcement.

        Default implementation returns -K with Robin BC stiffness
        contributions but no Dirichlet row replacement. Override in
        subclasses with nonlinear terms (e.g., reaction Jacobian).

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian dF/du. Shape: (nstates, nstates)
        """
        stiffness = self._assemble_stiffness(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        return -stiffness

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual without Dirichlet enforcement.

        Computes F = b - K*u where:
        - K includes Robin BC contributions to stiffness
        - b includes Neumann and Robin BC contributions to load
        - Dirichlet BCs are NOT applied (no row replacement)

        This is used by explicit time steppers that handle Dirichlet
        DOFs separately to avoid M^{-1} contamination.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        stiffness = self._assemble_stiffness(state, time)
        load = self._assemble_load(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        load = self._apply_bc_to_load(load, time)
        # Use @ operator (not bkd methods) because stiffness may be sparse
        return load - stiffness @ state

    def dirichlet_dof_info(self, time: float) -> Tuple[Array, Array]:
        """Return Dirichlet DOF indices and their exact values.

        Collects information from all Dirichlet boundary conditions
        (excluding Robin BCs which also satisfy DirichletBCProtocol).

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Tuple[Array, Array]
            dof_indices : Array
                Global DOF indices for all Dirichlet BCs. Shape: (ndirichlet,)
            dof_values : Array
                Exact Dirichlet values at those DOFs. Shape: (ndirichlet,)
        """
        all_dofs = []
        all_vals = []
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                dofs_np = self._bkd.to_numpy(bc.boundary_dofs())
                vals_np = self._bkd.to_numpy(bc.boundary_values(time))
                all_dofs.append(dofs_np)
                all_vals.append(vals_np)
        if all_dofs:
            return (
                self._bkd.asarray(
                    np.concatenate(all_dofs).astype(np.int64)
                ),
                self._bkd.asarray(
                    np.concatenate(all_vals).astype(np.float64)
                ),
            )
        return (
            self._bkd.asarray(np.array([], dtype=np.int64)),
            self._bkd.asarray(np.array([], dtype=np.float64)),
        )

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual F(u, t) with Dirichlet BCs applied.

        For transient problems: M * du/dt = F(u, t)
        The residual is: F = b - K*u

        Boundary conditions are applied:
        - Robin/Neumann BCs modify stiffness and load (via spatial_residual)
        - Dirichlet BCs replace residual rows with constraint violation

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
        residual = self.spatial_residual(state, time)

        # Apply Dirichlet BCs: replace residual rows with constraint violation
        # Check Robin first since RobinBC also satisfies DirichletBCProtocol
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                residual = bc.apply_to_residual(residual, state, time)

        return residual

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du.

        For linear problems: dF/du = -K
        For nonlinear problems: includes derivative of K w.r.t. u

        Boundary conditions are applied:
        - Robin BCs modify stiffness (add boundary mass)
        - Dirichlet BCs replace Jacobian rows with identity

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
        stiffness = self._assemble_stiffness(state, time)

        # Apply Robin BC contributions to stiffness
        stiffness = self._apply_bc_to_stiffness(stiffness, time)

        # For linear problems, Jacobian is -K
        jacobian = -stiffness

        # Apply Dirichlet BCs: replace Jacobian rows with identity
        # Skip Robin BCs since they also satisfy DirichletBCProtocol
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                jacobian = bc.apply_to_jacobian(jacobian, state, time)

        return jacobian

    def apply_boundary_conditions(
        self,
        residual: Optional[Array],
        jacobian: Optional[Array],
        state: Array,
        time: float = 0.0,
    ) -> Tuple[Optional[Array], Optional[Array]]:
        """Apply boundary conditions to residual and/or Jacobian.

        This method applies all boundary conditions in the correct order:
        1. Robin BCs (modify interior of matrices)
        2. Dirichlet BCs (replace rows)

        Note: This method is provided for explicit control. The residual()
        and jacobian() methods already apply BCs automatically.

        Parameters
        ----------
        residual : Array or None
            Residual vector. Shape: (nstates,). None to skip.
        jacobian : Array or None
            Jacobian matrix. Shape: (nstates, nstates). None to skip.
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Tuple[Optional[Array], Optional[Array]]
            Modified (residual, jacobian).
        """
        res = residual
        jac = jacobian

        # Apply BCs using the BC objects' methods
        for bc in self._boundary_conditions:
            # Robin BCs (modify interior first)
            if isinstance(bc, RobinBCProtocol):
                if res is not None:
                    res = bc.apply_to_residual(res, state, time)
                if jac is not None:
                    jac = bc.apply_to_jacobian(jac, state, time)

        for bc in self._boundary_conditions:
            # Dirichlet BCs (replace rows last)
            # Skip Robin BCs since they also satisfy DirichletBCProtocol
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                if res is not None:
                    res = bc.apply_to_residual(res, state, time)
                if jac is not None:
                    jac = bc.apply_to_jacobian(jac, state, time)

        return res, jac

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self.nstates()})"

"""Base physics class for Galerkin finite element methods.

Provides common infrastructure for PDE physics implementations.
"""

from typing import Generic, Tuple, List, Optional, Callable
from abc import ABC, abstractmethod

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.typing.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
    DirichletBCProtocol,
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

    def mass_matrix(self) -> Array:
        """Return the mass matrix from weak form.

        For Galerkin FEM with Lagrange elements:
            M_ij = integral(phi_i * phi_j)

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        if self._mass_matrix_cached is None:
            skfem_basis = self._basis.skfem_basis()
            mass_np = asm(mass, skfem_basis).toarray()
            self._mass_matrix_cached = self._bkd.asarray(
                mass_np.astype(np.float64)
            )
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
        M = self.mass_matrix()
        return self._bkd.solve(M, rhs)

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

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual F(u, t).

        For transient problems: M * du/dt = F(u, t)
        The residual is: F = b - K*u

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
        stiffness = self._assemble_stiffness(state, time)
        load = self._assemble_load(state, time)

        # F = b - K*u
        residual = load - stiffness @ state

        return residual

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du.

        For linear problems: dF/du = -K
        For nonlinear problems: includes derivative of K w.r.t. u

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
        # For linear problems, Jacobian is -K
        stiffness = self._assemble_stiffness(state, time)
        return -stiffness

    def apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array
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

        Returns
        -------
        Tuple[Array, Array]
            Modified (residual, jacobian).
        """
        # Convert to numpy for modification
        res_np = self._bkd.to_numpy(residual).copy()
        jac_np = self._bkd.to_numpy(jacobian).copy()
        state_np = self._bkd.to_numpy(state)

        for bc in self._boundary_conditions:
            bc_dofs = self._bkd.to_numpy(bc.boundary_dofs())

            # For Dirichlet BCs, enforce u = g
            if isinstance(bc, DirichletBCProtocol):
                # This is a simplified implementation
                # Full implementation would handle time-dependent BCs
                for dof in bc_dofs:
                    # Set residual row to enforce constraint
                    res_np[dof] = 0.0  # Will be set properly during solve
                    # Set Jacobian row to identity
                    jac_np[dof, :] = 0.0
                    jac_np[dof, dof] = 1.0

        return (
            self._bkd.asarray(res_np),
            self._bkd.asarray(jac_np),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self.nstates()})"

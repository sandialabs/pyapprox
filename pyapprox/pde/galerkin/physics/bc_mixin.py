"""Mixin providing boundary condition dispatch logic for Galerkin physics.

All dispatch loops use Robin-first ordering to handle the fact that
RobinBC structurally satisfies DirichletBCProtocol.
"""

from typing import Generic, Optional, Tuple

import numpy as np

from pyapprox.pde.galerkin.protocols.boundary import (
    DirichletBCProtocol,
    NeumannBCProtocol,
    RobinBCProtocol,
)
from pyapprox.util.backends.protocols import Array


class GalerkinBCMixin(Generic[Array]):
    """Mixin providing BC dispatch logic for Galerkin physics classes.

    Pure method provider — no ``__init__``. Using classes must set
    ``_bkd`` (Backend) and ``_boundary_conditions``
    (list of BoundaryConditionProtocol) before calling mixin methods.
    ``GalerkinPhysicsBase.__init__`` handles this for most classes;
    ``EulerBernoulliBeamFEM`` and ``StokesPhysics`` set them directly.
    """

    def _apply_bc_to_stiffness(self, stiffness: Array, time: float) -> Array:
        """Apply Robin BC contributions to stiffness matrix.

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
        """Apply Neumann and Robin BC contributions to load vector.

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
                Global DOF indices. Shape: (ndirichlet,)
            dof_values : Array
                Exact Dirichlet values. Shape: (ndirichlet,)
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
                self._bkd.asarray(np.concatenate(all_dofs).astype(np.int64)),
                self._bkd.asarray(np.concatenate(all_vals).astype(np.float64)),
            )
        return (
            self._bkd.asarray(np.array([], dtype=np.int64)),
            self._bkd.asarray(np.array([], dtype=np.float64)),
        )

    def _apply_dirichlet_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Apply Dirichlet row replacement to residual.

        Skips Robin BCs (which also satisfy DirichletBCProtocol).

        Parameters
        ----------
        residual : Array
            Spatial residual (without Dirichlet enforcement).
        state : Array
            Current state vector.
        time : float
            Current time.

        Returns
        -------
        Array
            Residual with Dirichlet rows replaced.
        """
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                residual = bc.apply_to_residual(residual, state, time)
        return residual

    def _apply_dirichlet_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply Dirichlet row replacement to Jacobian.

        Skips Robin BCs (which also satisfy DirichletBCProtocol).

        Parameters
        ----------
        jacobian : Array
            Spatial Jacobian (without Dirichlet enforcement).
        state : Array
            Current state vector.
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian with Dirichlet rows replaced by identity.
        """
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                jacobian = bc.apply_to_jacobian(jacobian, state, time)
        return jacobian

    def _apply_dirichlet_to_param_jacobian(
        self, pjac: Array, state: Array, time: float
    ) -> Array:
        """Apply Dirichlet row replacement to parameter Jacobian.

        Uses ``hasattr(bc, "apply_to_param_jacobian")`` since not all
        Dirichlet BCs support this operation.

        Parameters
        ----------
        pjac : Array
            Parameter Jacobian (without Dirichlet enforcement).
        state : Array
            Current state vector.
        time : float
            Current time.

        Returns
        -------
        Array
            Parameter Jacobian with Dirichlet rows zeroed.
        """
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                if hasattr(bc, "apply_to_param_jacobian"):
                    pjac = bc.apply_to_param_jacobian(pjac, state, time)
        return pjac

    def apply_boundary_conditions(
        self,
        residual: Optional[Array],
        jacobian: Optional[Array],
        state: Array,
        time: float = 0.0,
    ) -> Tuple[Optional[Array], Optional[Array]]:
        """Apply all boundary conditions in the correct order.

        1. Robin BCs (modify interior of matrices)
        2. Dirichlet BCs (replace rows)

        Parameters
        ----------
        residual : Array or None
            Residual vector. None to skip.
        jacobian : Array or None
            Jacobian matrix. None to skip.
        state : Array
            Current state.
        time : float
            Current time.

        Returns
        -------
        Tuple[Optional[Array], Optional[Array]]
            Modified (residual, jacobian).
        """
        res = residual
        jac = jacobian

        # Robin BCs first (modify interior)
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                if res is not None:
                    res = bc.apply_to_residual(res, state, time)
                if jac is not None:
                    jac = bc.apply_to_jacobian(jac, state, time)

        # Dirichlet BCs last (replace rows)
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                if res is not None:
                    res = bc.apply_to_residual(res, state, time)
                if jac is not None:
                    jac = bc.apply_to_jacobian(jac, state, time)

        return res, jac

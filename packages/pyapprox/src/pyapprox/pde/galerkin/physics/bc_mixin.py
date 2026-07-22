"""Mixin providing boundary condition dispatch logic for Galerkin physics.

Dispatch uses the disjoint solver-neutral BC roles from
``pyapprox.pde.boundary``: weak-form BCs (Neumann, Robin) are applied
to assembled operators, essential (Dirichlet) constraints are applied
through a single cached ``DirichletConstraintSet``.
"""

from typing import Any, Callable, Generic, List, Optional, Tuple

from pyapprox.pde.boundary import (
    DirichletConstraintSet,
    EssentialBCProtocol,
    WeakFormBCProtocol,
)
from pyapprox.pde.galerkin.protocols.boundary import RobinBCProtocol
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinBCMixin(Generic[Array]):
    """Mixin providing BC dispatch logic for Galerkin physics classes.

    Pure method provider — no ``__init__``. Using classes must set
    ``_bkd`` (Backend) and ``_boundary_conditions``
    (list of role-protocol BCs) before calling mixin methods, and must
    provide ``nstates()``. ``GalerkinPhysicsBase.__init__`` handles the
    attributes for most classes; ``EulerBernoulliBeamFEM`` and
    ``StokesPhysics`` set them directly.
    """

    _bkd: Backend[Array]
    _boundary_conditions: List[Any]
    nstates: Callable[[], int]
    _constraint_set: Optional[DirichletConstraintSet[Array]] = None

    def weak_form_bcs(self) -> List[WeakFormBCProtocol[Array]]:
        """Return the natural (Neumann/Robin) BCs, in list order."""
        return [
            bc
            for bc in self._boundary_conditions
            if isinstance(bc, WeakFormBCProtocol)
        ]

    def essential_bcs(self) -> List[EssentialBCProtocol[Array]]:
        """Return the essential (Dirichlet) BCs, in list order."""
        return [
            bc
            for bc in self._boundary_conditions
            if isinstance(bc, EssentialBCProtocol)
        ]

    def constraint_set(self) -> DirichletConstraintSet[Array]:
        """Return the cached essential-constraint set for this physics.

        Built lazily on first call; constrained DOF locations are fixed
        at construction so the cache never invalidates.
        """
        if self._constraint_set is None:
            self._constraint_set = DirichletConstraintSet(
                self.essential_bcs(), self.nstates(), self._bkd
            )
        return self._constraint_set

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
        for bc in self.weak_form_bcs():
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
        for bc in self.weak_form_bcs():
            load = bc.apply_to_load(load, time)
        return load

    def dirichlet_dof_info(self, time: float) -> Tuple[Array, Array]:
        """Return Dirichlet DOF indices and their exact values.

        Deprecated delegating shim: use ``constraint_set()`` directly.
        DOFs are unique and sorted; DOFs shared by several BCs take the
        last BC's value.

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
        constraint_set = self.constraint_set()
        return constraint_set.dofs(), constraint_set.values(time)

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
        return self.constraint_set().apply_to_residual(residual, state, time)

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
        return self.constraint_set().apply_to_jacobian(jacobian)

    def _apply_dirichlet_to_param_jacobian(
        self, pjac: Array, state: Array, time: float
    ) -> Array:
        """Zero essential-BC rows of a parameter Jacobian.

        Essential constraints do not depend on PDE parameters, so all
        constrained rows of dR/dp are zeroed. The ``state`` and ``time``
        arguments are kept for backward compatibility; row zeroing
        depends on neither.

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
            Parameter Jacobian with constrained rows zeroed.
        """
        return self.constraint_set().zero_rows(pjac)

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

        # Robin BCs first (modify interior). Deliberately Robin-only,
        # not all weak-form BCs: this legacy path predates
        # NeumannBC.apply_to_residual and callers pass residuals whose
        # load already contains the Neumann contribution.
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                if res is not None:
                    res = bc.apply_to_residual(res, state, time)
                if jac is not None:
                    jac = bc.apply_to_jacobian(jac, state, time)

        # Essential constraints last (replace rows)
        constraint_set = self.constraint_set()
        if res is not None:
            res = constraint_set.apply_to_residual(res, state, time)
        if jac is not None:
            jac = constraint_set.apply_to_jacobian(jac)

        return res, jac

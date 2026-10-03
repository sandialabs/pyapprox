"""Mixin holding a Galerkin physics' boundary conditions, split by role.

Splits the BC list into the disjoint solver-neutral roles from
``pyapprox.pde.boundary`` — natural terms (composed into ``F`` by
``ComposedSpatialOperator``) and essential constraints (a cached
``DirichletConstraintSet``) — and answers ownership. It applies
nothing: the composed operator adds the terms, and constraint rows are
applied by the steady view or the time-stepping wrappers.
"""

from typing import Any, Callable, Generic, List, Optional, Tuple

from pyapprox.pde.boundary import (
    BCRoles,
    DirichletConstraintSet,
    EssentialBCProtocol,
    WeakFormBCProtocol,
    split_by_role,
)
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinBCMixin(Generic[Array]):
    """Mixin holding a Galerkin physics' BCs, split by role.

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
    _bc_roles: Optional[BCRoles[Array]] = None

    def _roles(self) -> BCRoles[Array]:
        """Return the BC list split by role (cached).

        Raises on a BC with no role or both, rather than dropping it.
        """
        if self._bc_roles is None:
            self._bc_roles = split_by_role(self._boundary_conditions)
        return self._bc_roles

    def owns(self, target: object) -> bool:
        """Whether ``target`` is this physics or one of its BCs.

        By identity; the BCs are what a BC-data parameterization targets.
        """
        return target is self or any(
            target is bc for bc in self._boundary_conditions
        )

    def weak_form_bcs(self) -> List[WeakFormBCProtocol[Array]]:
        """Return the natural (Neumann/Robin) BCs, in list order."""
        return self._roles().terms()

    def essential_bcs(self) -> List[EssentialBCProtocol[Array]]:
        """Return the essential (Dirichlet) BCs, in list order."""
        return self._roles().essentials()

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


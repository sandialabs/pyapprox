"""Universal base class for Galerkin physics.

Inherits GalerkinBCMixin for BC dispatch and provides constructor,
accessors, the composed spatial operator and Dirichlet-wrapped
residual/jacobian. Subclasses implement the abstract interior_residual()
and interior_jacobian(); the natural-BC terms are added here, once.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.boundary import BoundaryConditionRole, NaturalBCOperator
from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.spatial_operator import ComposedSpatialOperator
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinPhysicsBase(GalerkinBCMixin[Array], ABC, Generic[Array]):
    """Base class for Galerkin physics with a single basis.

    Provides:
    - Constructor setting ``_basis``, ``_bkd``, ``_boundary_conditions``
    - Accessors: ``bkd()``, ``basis()``, ``nstates()``
    - ``spatial_residual()`` and ``spatial_jacobian()``: the interior
      operator plus the natural-BC terms, ``F = F_Omega + F_Gamma``,
      composed here so no physics adds a boundary term itself
    - ``residual()`` and ``jacobian()`` wrapping those with Dirichlet
      row replacement from the mixin

    Subclasses must implement the abstract methods:
    - ``interior_residual(state, time) -> Array``
    - ``interior_jacobian(state, time) -> Array``

    Classes that don't fit this pattern (e.g., mixed-formulation Stokes
    with two bases, or EulerBernoulliBeamFEM with a raw skfem Basis)
    should use ``GalerkinBCMixin`` directly.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        bkd: Backend[Array],
        boundary_conditions: Optional[List[BoundaryConditionRole[Array]]] = None,
    ):
        self._basis = basis
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions or []
        # Split now so a BC with no role fails at construction.
        self._roles()
        self._spatial_operator: Optional[ComposedSpatialOperator[Array]] = None
        self._system: Optional[GalerkinSystem[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> GalerkinBasisProtocol[Array]:
        """Return the finite element basis."""
        return self._basis

    def nstates(self) -> int:
        """Return total number of DOFs."""
        return self._basis.ndofs()

    @abstractmethod
    def interior_residual(self, state: Array, time: float) -> Array:
        """Compute the interior residual ``F_Omega``, with no BC in it.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Interior residual. Shape: (nstates,)
        """

    @abstractmethod
    def interior_jacobian(self, state: Array, time: float) -> Array:
        """Compute ``dF_Omega/du``, with no BC in it.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Interior Jacobian. Shape: (nstates, nstates)
        """

    @abstractmethod
    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Return the optional second state derivatives of ``F_Omega``.

        Declared by every physics: ``StateDerivatives.linear`` when the
        interior is linear in u, ``StateDerivatives.none()`` when its
        curvature is not supplied.
        """

    @abstractmethod
    def interior_is_time_invariant(self) -> bool:
        """Whether every coefficient and forcing of ``F_Omega`` is
        declared time-independent."""

    @abstractmethod
    def mass_matrix(self) -> Array:
        """Return the mass matrix ``M``. Shape: (nstates, nstates)."""

    def state_derivatives(self) -> StateDerivatives[Array]:
        """Return the second state derivatives of the composed ``F``."""
        return self.spatial_operator().state_derivatives()

    def is_time_invariant(self) -> bool:
        """Whether the composed ``F`` is declared time-independent."""
        return self.spatial_operator().is_time_invariant()

    def natural_bc_operator(self) -> NaturalBCOperator[Array]:
        """Return the natural-BC operator built from this physics' BCs."""
        return self.spatial_operator().natural_bcs()

    def spatial_operator(self) -> ComposedSpatialOperator[Array]:
        """Return ``F = F_Omega + F_Gamma``: this physics' interior composed
        with the natural-BC terms of its BC list.

        A convenience while physics still receive their BCs; the
        composition itself lives in ``ComposedSpatialOperator``, which any
        interior operator can use without this base class.
        """
        if self._spatial_operator is None:
            self._spatial_operator = ComposedSpatialOperator(
                self, NaturalBCOperator(self.weak_form_bcs())
            )
        return self._spatial_operator

    def system(self) -> GalerkinSystem[Array]:
        """Return the composed system: ``F``, constraints, and mass.

        What models and solvers consume. The mass is read from this
        physics on every call, so a parameterized mass is never stale.
        """
        if self._system is None:
            self._system = GalerkinSystem(
                self.spatial_operator(), self.constraint_set(), self
            )
        return self._system

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute ``F = F_Omega + F_Gamma`` without Dirichlet enforcement.

        Delegates to ``spatial_operator()``.

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
        return self.spatial_operator().spatial_residual(state, time)

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute ``dF/du = dF_Omega/du + dF_Gamma/du`` without Dirichlet
        enforcement.

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
        return self.spatial_operator().spatial_jacobian(state, time)

    def residual(self, state: Array, time: float) -> Array:
        """Compute residual F(u, t) with Dirichlet BCs applied.

        A convenience: the steady view of this physics at ``time``
        (``system().steady_snapshot(time).residual``), which owns the
        constraint application.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual with Dirichlet rows replaced. Shape: (nstates,)
        """
        return self.system().steady_snapshot(time).steady_residual(state)

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute Jacobian dF/du with Dirichlet BCs applied.

        A convenience: ``system().steady_snapshot(time).jacobian``.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian with Dirichlet rows replaced. Shape: (nstates, nstates)
        """
        return self.system().steady_snapshot(time).steady_jacobian(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self.nstates()})"

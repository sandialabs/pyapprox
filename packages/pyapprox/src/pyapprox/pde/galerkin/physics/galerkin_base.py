"""Universal base class for Galerkin physics.

Inherits GalerkinBCMixin for BC dispatch and provides constructor,
accessors, the composed spatial operator and Dirichlet-wrapped
residual/jacobian. Subclasses implement the abstract interior_residual()
and interior_jacobian(); the natural-BC terms are added here, once.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional

from pyapprox.pde.boundary import NaturalBCOperator
from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.pde.galerkin.spatial_operator import ComposedSpatialOperator
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
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
    ):
        self._basis = basis
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions or []
        self._spatial_operator: Optional[ComposedSpatialOperator[Array]] = None

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

        Wraps ``spatial_residual()`` with Dirichlet row replacement.

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
        return self._apply_dirichlet_to_residual(
            self.spatial_residual(state, time), state, time
        )

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute Jacobian dF/du with Dirichlet BCs applied.

        Wraps ``spatial_jacobian()`` with Dirichlet row replacement.

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
        return self._apply_dirichlet_to_jacobian(
            self.spatial_jacobian(state, time), state, time
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nstates={self.nstates()})"

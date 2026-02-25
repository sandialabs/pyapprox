"""Universal base class for Galerkin physics.

Inherits GalerkinBCMixin for BC dispatch and provides constructor,
accessors, and Dirichlet-wrapped residual/jacobian. Subclasses implement
spatial_residual() and spatial_jacobian().
"""

from typing import Generic, List, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin


class GalerkinPhysicsBase(GalerkinBCMixin[Array], Generic[Array]):
    """Base class for Galerkin physics with a single basis.

    Provides:
    - Constructor setting ``_basis``, ``_bkd``, ``_boundary_conditions``
    - Accessors: ``bkd()``, ``basis()``, ``nstates()``
    - ``residual()`` and ``jacobian()`` wrapping subclass-provided
      ``spatial_residual()`` and ``spatial_jacobian()`` with Dirichlet
      row replacement from the mixin

    Subclasses must implement:
    - ``spatial_residual(state, time) -> Array``
    - ``spatial_jacobian(state, time) -> Array``

    Classes that don't fit this pattern (e.g., mixed-formulation Stokes
    with two bases, or EulerBernoulliBeamFEM with a raw skfem Basis)
    should use ``GalerkinBCMixin`` directly.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        bkd: Backend[Array],
        boundary_conditions: Optional[
            List[BoundaryConditionProtocol[Array]]
        ] = None,
    ):
        self._basis = basis
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions or []

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> GalerkinBasisProtocol[Array]:
        """Return the finite element basis."""
        return self._basis

    def nstates(self) -> int:
        """Return total number of DOFs."""
        return self._basis.ndofs()

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

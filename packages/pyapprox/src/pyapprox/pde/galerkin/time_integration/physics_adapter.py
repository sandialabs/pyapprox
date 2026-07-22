"""Adapter to use Galerkin physics with time integration from pyapprox.ode.

The time module expects ODEResidualProtocol: M * dy/dt = f(y, t)
Galerkin physics provides: M * du/dt = F(u, t)

The adapter returns raw (unmodified) quantities:
  f(y, t) = spatial_residual(y, t)   (no Dirichlet row zeroing)
  jacobian = spatial_jacobian(y, t)  (no Dirichlet row replacement)
  mass_matrix() = MassMatrixProtocol wrapping raw FEM mass matrix

Dirichlet BCs are enforced by the BC-enforcing time residual wrapper,
which wraps the stepper and applies R[d] = y[d] - g(t), J[d,:] = e_d
after the stepper assembles the full Newton system.

The parameterized adapter tiers and the capability-selecting factory
live in ``pyapprox.pde.models.galerkin.physics_adapter`` — the models
layer owns everything that requires both a physics and a
parameterization.
"""

from typing import Generic, Tuple

from scipy.sparse import issparse

from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    SparseMatrixOperator,
)
from pyapprox.ode.mass_matrix import MassMatrixProtocol, create_mass_matrix
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinPhysicsToODEResidualAdapter(
    DefaultNewtonJacobianMixin[Array], Generic[Array]
):
    """Adapter from GalerkinPhysics to ODEResidualProtocol (base tier).

    Returns raw M, F, J_F -- no BC modifications:
    - f(y) = spatial_residual(y, t) (unmodified)
    - jacobian(y) = spatial_jacobian(y, t) (unmodified)
    - mass_matrix() = MassMatrixProtocol wrapping M

    Dirichlet BCs are applied externally by the BC-enforcing time
    residual wrapper.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        The Galerkin physics to adapt. Must have spatial_residual(),
        spatial_jacobian(), and dirichlet_dof_info() methods.

    Examples
    --------
    >>> ode_residual = GalerkinPhysicsToODEResidualAdapter(physics)
    >>> time_stepper = BackwardEulerHVP(ode_residual)
    """

    def __init__(self, physics: GalerkinPhysicsProtocol[Array]) -> None:
        if not isinstance(physics, GalerkinPhysicsProtocol):
            raise TypeError(
                f"physics must satisfy GalerkinPhysicsProtocol, "
                f"got {type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = physics.bkd()
        self._time: float = 0.0
        # Cache mass matrix as value-object (handles sparse via splu)
        self._mass = create_mass_matrix(physics.mass_matrix(), self._bkd)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def physics(self) -> GalerkinPhysicsProtocol[Array]:
        """Return the wrapped physics object."""
        return self._physics

    def set_time(self, time: float) -> None:
        """Set the current time for evaluation.

        Parameters
        ----------
        time : float
            Current time.
        """
        self._time = time

    def __call__(self, state: Array) -> Array:
        """Evaluate spatial residual F(y, t) (unmodified).

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        return self._physics.spatial_residual(state, self._time)

    def jacobian(self, state: Array) -> Array:
        """Compute spatial Jacobian dF/du (unmodified).

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian dF/du. Shape: (nstates, nstates)
        """
        return self._physics.spatial_jacobian(state, self._time)

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the FEM mass matrix as a value-object."""
        return self._mass

    def newton_jacobian(
        self, state: Array, coefficient: float
    ) -> LinearOperatorProtocol[Array]:
        """Return M - coefficient * dF/du as a linear operator.

        Sparse FEM systems get a ``SparseMatrixOperator`` whose
        ``as_matrix()`` returns the SPARSE Newton matrix, keeping
        sparsity flowing to the implicit steppers and the BC-enforcing
        wrapper (which applies constraint rows sparsely). Dense systems
        (e.g. the torch backend) fall back to the default dense
        operator.
        """
        jacobian = self.jacobian(state)
        mass = self._mass.as_matrix()
        if (
            issparse(jacobian)
            and issparse(mass)
            and isinstance(self._bkd, NumpyBkd)
        ):
            return SparseMatrixOperator(
                mass - coefficient * jacobian, self.bkd()
            )
        return super().newton_jacobian(state, coefficient)

    def dirichlet_dof_info(self, time: float) -> Tuple[Array, Array]:
        """Return Dirichlet DOF indices and values at given time.

        Parameters
        ----------
        time : float
            Time at which to evaluate Dirichlet BCs.

        Returns
        -------
        Tuple[Array, Array]
            dof_indices : Array
                Global DOF indices. Shape: (ndirichlet,)
            dof_values : Array
                Exact Dirichlet values. Shape: (ndirichlet,)
        """
        return self._physics.dirichlet_dof_info(time)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"physics={type(self._physics).__name__})"
        )

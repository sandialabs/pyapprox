"""Physics protocols for spectral collocation methods.

Defines a 3-level protocol hierarchy for PDE physics:
1. PhysicsProtocol - basic residual and Jacobian
2. PhysicsWithParamJacobianProtocol - adds parameter sensitivity (for adjoint)
3. PhysicsWithHVPProtocol - adds Hessian-vector products (for second-order)
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.pde.collocation.protocols.basis import BasisProtocol
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class PhysicsProtocol(Protocol, Generic[Array]):
    """Protocol for PDE physics (Level 1).

    Defines spatial discretization of a PDE system with state Jacobian.
    This is the minimum interface for forward solve.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def basis(self) -> BasisProtocol[Array]:
        """Return the collocation basis."""
        ...

    def nstates(self) -> int:
        """Return total number of states (ncomponents * npts)."""
        ...

    def ncomponents(self) -> int:
        """Return number of solution components.

        E.g., 1 for scalar PDE, 2-3 for coupled systems.
        """
        ...

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
        self, residual: Array, jacobian: Array, state: Array
    ) -> Tuple[Array, Array]:
        """Apply boundary conditions to residual and Jacobian.

        Modifies rows corresponding to boundary points to enforce BCs.

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
        ...

    def mass_matrix(self) -> Array:
        """Return mass matrix for time integration.

        For standard ODEs, this is the identity matrix.
        For DAEs, this may be singular.

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        ...

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix to a vector.

        Default is identity (returns vec unchanged). Overridden for
        non-identity mass matrices (e.g., split physics).

        Parameters
        ----------
        vec : Array
            Vector to multiply. Shape: (nstates,)

        Returns
        -------
        Array
            M @ vec. Shape: (nstates,)
        """
        ...


@runtime_checkable
class PhysicsWithParamJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for physics with parameter sensitivity (Level 2).

    Extends PhysicsProtocol with parameter Jacobian for adjoint
    sensitivity analysis.
    """

    # --- All PhysicsProtocol methods ---
    def bkd(self) -> Backend[Array]: ...
    def basis(self) -> BasisProtocol[Array]: ...
    def nstates(self) -> int: ...
    def ncomponents(self) -> int: ...
    def residual(self, state: Array, time: float) -> Array: ...
    def jacobian(self, state: Array, time: float) -> Array: ...
    def apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array
    ) -> Tuple[Array, Array]: ...
    def mass_matrix(self) -> Array: ...

    # --- Additional parameter methods ---
    def nparams(self) -> int:
        """Return number of parameters."""
        ...

    def set_param(self, param: Array) -> None:
        """Set parameter values.

        Parameters
        ----------
        param : Array
            Parameter vector. Shape: (nparams,) or (nparams, 1)
        """
        ...

    def param_jacobian(self, state: Array, time: float) -> Array:
        """Compute parameter Jacobian df/dp.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        ...

    def initial_param_jacobian(self) -> Array:
        """Compute initial condition parameter Jacobian d(u_0)/dp.

        Returns
        -------
        Array
            Initial condition Jacobian. Shape: (nstates, nparams)
        """
        ...


@runtime_checkable
class PhysicsWithHVPProtocol(Protocol, Generic[Array]):
    """Protocol for physics with Hessian-vector products (Level 3).

    Extends PhysicsWithParamJacobianProtocol with HVP methods for
    second-order optimization (Newton methods, Gauss-Newton, etc.).
    """

    # --- All PhysicsWithParamJacobianProtocol methods ---
    def bkd(self) -> Backend[Array]: ...
    def basis(self) -> BasisProtocol[Array]: ...
    def nstates(self) -> int: ...
    def ncomponents(self) -> int: ...
    def residual(self, state: Array, time: float) -> Array: ...
    def jacobian(self, state: Array, time: float) -> Array: ...
    def apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array
    ) -> Tuple[Array, Array]: ...
    def mass_matrix(self) -> Array: ...
    def nparams(self) -> int: ...
    def set_param(self, param: Array) -> None: ...
    def param_jacobian(self, state: Array, time: float) -> Array: ...
    def initial_param_jacobian(self) -> Array: ...

    # --- HVP methods ---
    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        """Compute lambda^T * (d^2f/du^2) * w.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        adj_state : Array
            Adjoint variable (Lagrange multiplier). Shape: (nstates,)
        wvec : Array
            Direction vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        ...

    def state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array, time: float
    ) -> Array:
        """Compute lambda^T * (d^2f/dudp) * v.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        adj_state : Array
            Adjoint variable. Shape: (nstates,)
        vvec : Array
            Parameter direction. Shape: (nparams,)
        time : float
            Current time.

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        ...

    def param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        """Compute lambda^T * (d^2f/dpdu) * w.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        adj_state : Array
            Adjoint variable. Shape: (nstates,)
        wvec : Array
            State direction. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        ...

    def param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array, time: float
    ) -> Array:
        """Compute lambda^T * (d^2f/dp^2) * v.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        adj_state : Array
            Adjoint variable. Shape: (nstates,)
        vvec : Array
            Parameter direction. Shape: (nparams,)
        time : float
            Current time.

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        ...

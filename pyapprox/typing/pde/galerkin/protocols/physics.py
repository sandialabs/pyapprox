"""Physics protocols for Galerkin finite element methods.

Defines a 3-level protocol hierarchy for PDE physics:
1. GalerkinPhysicsProtocol - basic residual and Jacobian
2. GalerkinPhysicsWithParamJacobianProtocol - adds parameter sensitivity
3. GalerkinPhysicsWithHVPProtocol - adds Hessian-vector products

The key difference from collocation is that Galerkin uses weak formulation
with mass matrices: M*du/dt = F(u,t) instead of du/dt = f(u,t).
"""

from typing import Protocol, Generic, runtime_checkable, Tuple, Any

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class GalerkinPhysicsProtocol(Protocol, Generic[Array]):
    """Protocol for Galerkin PDE physics (Level 1).

    Defines weak form discretization of a PDE system.
    This is the minimum interface for forward solve.

    The Galerkin formulation produces:
        M * du/dt = F(u, t)
    where M is the mass matrix from the weak form.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def basis(self) -> Any:
        """Return the finite element basis."""
        ...

    def nstates(self) -> int:
        """Return total number of DOFs."""
        ...

    def mass_matrix(self) -> Array:
        """Return the mass matrix from weak form.

        For Galerkin FEM, this is typically:
            M_ij = integral(phi_i * phi_j)

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        ...

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x.

        This method can be overridden to exploit structure in the mass matrix.
        For example, with a lumped (diagonal) mass matrix, this becomes a
        simple element-wise division.

        Parameters
        ----------
        rhs : Array
            Right-hand side vector. Shape: (nstates,) or (nstates, ncols)

        Returns
        -------
        Array
            Solution x = M^{-1} * rhs. Same shape as rhs.
        """
        ...

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual F(u, t).

        For transient problems: M * du/dt = residual(u, t)
        For steady problems: solve residual(u) = 0

        This is the right-hand side of the weak form after
        assembly: F = b - K*u where K is the stiffness matrix.

        Parameters
        ----------
        state : Array
            Solution state (DOF coefficients). Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (nstates,)
        """
        ...

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du without Dirichlet enforcement.

        Returns the Jacobian with Robin/Neumann BC contributions but
        no Dirichlet row replacement.

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
        ...

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du.

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

        Modifies rows corresponding to boundary DOFs to enforce BCs.

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


@runtime_checkable
class GalerkinPhysicsWithParamJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for Galerkin physics with parameter sensitivity (Level 2).

    Extends GalerkinPhysicsProtocol with parameter Jacobian for adjoint
    sensitivity analysis.
    """

    # --- All GalerkinPhysicsProtocol methods ---
    def bkd(self) -> Backend[Array]: ...
    def basis(self) -> Any: ...
    def nstates(self) -> int: ...
    def mass_matrix(self) -> Array: ...
    def mass_solve(self, rhs: Array) -> Array: ...
    def residual(self, state: Array, time: float) -> Array: ...
    def spatial_jacobian(self, state: Array, time: float) -> Array: ...
    def jacobian(self, state: Array, time: float) -> Array: ...
    def apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array
    ) -> Tuple[Array, Array]: ...

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
        """Compute parameter Jacobian dF/dp.

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
class GalerkinPhysicsWithHVPProtocol(Protocol, Generic[Array]):
    """Protocol for Galerkin physics with Hessian-vector products (Level 3).

    Extends GalerkinPhysicsWithParamJacobianProtocol with HVP methods for
    second-order optimization (Newton methods, Gauss-Newton, etc.).
    """

    # --- All GalerkinPhysicsWithParamJacobianProtocol methods ---
    def bkd(self) -> Backend[Array]: ...
    def basis(self) -> Any: ...
    def nstates(self) -> int: ...
    def mass_matrix(self) -> Array: ...
    def mass_solve(self, rhs: Array) -> Array: ...
    def residual(self, state: Array, time: float) -> Array: ...
    def spatial_jacobian(self, state: Array, time: float) -> Array: ...
    def jacobian(self, state: Array, time: float) -> Array: ...
    def apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array
    ) -> Tuple[Array, Array]: ...
    def nparams(self) -> int: ...
    def set_param(self, param: Array) -> None: ...
    def param_jacobian(self, state: Array, time: float) -> Array: ...
    def initial_param_jacobian(self) -> Array: ...

    # --- HVP methods ---
    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        """Compute lambda^T * (d^2F/du^2) * w.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        adj_state : Array
            Adjoint variable. Shape: (nstates,)
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
        """Compute lambda^T * (d^2F/dudp) * v.

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
        """Compute lambda^T * (d^2F/dpdu) * w.

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
        """Compute lambda^T * (d^2F/dp^2) * v.

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

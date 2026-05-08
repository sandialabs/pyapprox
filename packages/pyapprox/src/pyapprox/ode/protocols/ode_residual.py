"""
Protocols for user-defined ODE residuals.

These protocols define the interface for ODE systems dy/dt = f(y, t; p)
at different capability levels.

Protocol Hierarchy
------------------
ODEResidualProtocol
    Basic ODE with state Jacobian.
ODEResidualWithParamJacobianProtocol
    Adds parameter Jacobian for gradient computation.
ODEResidualWithHVPProtocol
    Adds HVP methods for Hessian-vector products.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ODEResidualProtocol(Protocol, Generic[Array]):
    """
    Base protocol for ODE residuals: dy/dt = f(y, t; p).

    This is the user-defined ODE system. The residual evaluates f(y, t).

    Methods
    -------
    bkd()
        Get the computational backend.
    __call__(state)
        Evaluate f(y, t) at current time (set via set_time).
    set_time(time)
        Set the current time for evaluation.
    jacobian(state)
        Compute df/dy at current state and time.
    mass_matrix(nstates)
        Return the mass matrix M (identity for standard ODEs).
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the ODE residual f(y, t) at current time.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)

        Returns
        -------
        Array
            Residual f(y, t). Shape: (nstates,)
        """
        ...

    def set_time(self, time: float) -> None:
        """
        Set the current time for evaluation.

        Parameters
        ----------
        time : float
            Current time.
        """
        ...

    def jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian df/dy at current state and time.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian df/dy. Shape: (nstates, nstates)
        """
        ...

    def mass_matrix(self, nstates: int) -> Array:
        """
        Return the mass matrix M.

        For standard ODEs, this is the identity matrix.
        For DAEs, this may be singular.

        Parameters
        ----------
        nstates : int
            Number of states.

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        ...

    def apply_mass_matrix(self, vec: Array) -> Array:
        """
        Apply mass matrix to a vector: M @ vec.

        Default implementations should return mass_matrix(len(vec)) @ vec.
        Override for custom mass matrix application (e.g., sparse,
        block-diagonal, or singular mass matrices).

        Parameters
        ----------
        vec : Array
            Vector to multiply. Shape: (nstates,)

        Returns
        -------
        Array
            Result M @ vec. Shape: (nstates,)
        """
        ...


@runtime_checkable
class ODEResidualWithParamJacobianProtocol(Protocol, Generic[Array]):
    """
    ODE residual with parameter sensitivity: dy/dt = f(y, t; p).

    Extends ODEResidualProtocol with parameter Jacobian for adjoint
    sensitivity computations.

    Additional Methods
    ------------------
    nparams()
        Number of parameters.
    set_param(param)
        Set the parameter values.
    param_jacobian(state)
        Compute df/dp at current state and time.
    initial_param_jacobian()
        Jacobian of initial condition with respect to parameters.
    """

    def bkd(self) -> Backend[Array]: ...

    def __call__(self, state: Array) -> Array: ...

    def set_time(self, time: float) -> None: ...

    def jacobian(self, state: Array) -> Array: ...

    def mass_matrix(self, nstates: int) -> Array: ...

    def apply_mass_matrix(self, vec: Array) -> Array: ...

    def nparams(self) -> int:
        """
        Get the number of parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        ...

    def set_param(self, param: Array) -> None:
        """
        Set the parameter values.

        Parameters
        ----------
        param : Array
            Parameter values. Shape: (nparams,)
        """
        ...

    def param_jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian df/dp at current state and time.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian df/dp. Shape: (nstates, nparams)
        """
        ...

    def initial_param_jacobian(self) -> Array:
        """
        Jacobian of initial condition with respect to parameters.

        For y(0) = g(p), this returns dg/dp.
        Returns zeros if initial condition doesn't depend on parameters.

        Returns
        -------
        Array
            dy0/dp. Shape: (nstates, nparams)
        """
        ...


@runtime_checkable
class ODEResidualWithHVPProtocol(Protocol, Generic[Array]):
    """
    ODE residual with HVP support for second-order adjoint computations.

    Extends ODEResidualWithParamJacobianProtocol with four HVP methods
    for computing Hessian-vector products via the adjoint method.

    The HVP methods compute products of second derivatives with vectors,
    contracted with the adjoint state lambda for efficiency.

    Additional Methods
    ------------------
    state_state_hvp(state, adj_state, wvec)
        (d^2f/dy^2)w contracted with lambda
    state_param_hvp(state, adj_state, vvec)
        (d^2f/dydp)v contracted with lambda
    param_state_hvp(state, adj_state, wvec)
        (d^2f/dpdy)w contracted with lambda
    param_param_hvp(state, adj_state, vvec)
        (d^2f/dp^2)v contracted with lambda
    """

    def bkd(self) -> Backend[Array]: ...

    def __call__(self, state: Array) -> Array: ...

    def set_time(self, time: float) -> None: ...

    def jacobian(self, state: Array) -> Array: ...

    def mass_matrix(self, nstates: int) -> Array: ...

    def apply_mass_matrix(self, vec: Array) -> Array: ...

    def nparams(self) -> int: ...

    def set_param(self, param: Array) -> None: ...

    def param_jacobian(self, state: Array) -> Array: ...

    def initial_param_jacobian(self) -> Array: ...

    def state_state_hvp(self, state: Array, adj_state: Array, wvec: Array) -> Array:
        """
        Compute (d^2f/dy^2)w contracted with adjoint state.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)
        adj_state : Array
            Adjoint state lambda. Shape: (nstates,)
        wvec : Array
            Direction vector w. Shape: (nstates,)

        Returns
        -------
        Array
            lambda^T (d^2f/dy^2) w. Shape: (nstates,)
        """
        ...

    def state_param_hvp(self, state: Array, adj_state: Array, vvec: Array) -> Array:
        """
        Compute (d^2f/dydp)v contracted with adjoint state.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)
        adj_state : Array
            Adjoint state lambda. Shape: (nstates,)
        vvec : Array
            Direction vector v. Shape: (nparams,)

        Returns
        -------
        Array
            lambda^T (d^2f/dydp) v. Shape: (nstates,)
        """
        ...

    def param_state_hvp(self, state: Array, adj_state: Array, wvec: Array) -> Array:
        """
        Compute (d^2f/dpdy)w contracted with adjoint state.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)
        adj_state : Array
            Adjoint state lambda. Shape: (nstates,)
        wvec : Array
            Direction vector w. Shape: (nstates,)

        Returns
        -------
        Array
            lambda^T (d^2f/dpdy) w. Shape: (nparams,)
        """
        ...

    def param_param_hvp(self, state: Array, adj_state: Array, vvec: Array) -> Array:
        """
        Compute (d^2f/dp^2)v contracted with adjoint state.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)
        adj_state : Array
            Adjoint state lambda. Shape: (nstates,)
        vvec : Array
            Direction vector v. Shape: (nparams,)

        Returns
        -------
        Array
            lambda^T (d^2f/dp^2) v. Shape: (nparams,)
        """
        ...

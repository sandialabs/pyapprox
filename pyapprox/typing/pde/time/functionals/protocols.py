"""
Protocols for transient problem functionals.

Functionals define the quantity of interest (QoI) Q(y, p) for time-dependent
problems, along with derivatives needed for adjoint-based gradient computation.

The design follows the implicitfunction functionals pattern but adapted for
time-dependent problems where:
- State is a trajectory: (nstates, ntimes) instead of (nstates, 1)
- Quadrature weights are needed for path-integrated functionals
- State Jacobian returns (nstates, ntimes) for adjoint accumulation
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class TransientFunctionalWithJacobianProtocol(Protocol, Generic[Array]):
    """
    Protocol for transient functionals with Jacobian support.

    This protocol defines the interface for computing Q(y(t), p) and its
    derivatives for adjoint-based gradient computation.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend used for computations."""
        ...

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        ...

    def nstates(self) -> int:
        """Return the number of state variables."""
        ...

    def nparams(self) -> int:
        """Return the total number of parameters."""
        ...

    def nunique_params(self) -> int:
        """Return the number of parameters unique to the functional."""
        ...

    def __call__(self, sol: Array, param: Array) -> Array:
        """
        Evaluate the functional.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            QoI values. Shape: (nqoi, 1)
        """
        ...

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dy for the solution trajectory.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            State Jacobian. Shape: (nstates, ntimes)
        """
        ...

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """
        Compute dQ/dp.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nqoi, nparams)
        """
        ...


@runtime_checkable
class TransientFunctionalWithJacobianAndHVPProtocol(Protocol, Generic[Array]):
    """
    Protocol for transient functionals with Jacobian and HVP support.

    Extends TransientFunctionalWithJacobianProtocol with second-order
    derivative methods for Hessian-vector products.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def nqoi(self) -> int:
        """Return the number of QoI outputs."""
        ...

    def nstates(self) -> int:
        """Return the number of state variables."""
        ...

    def nparams(self) -> int:
        """Return the total number of parameters."""
        ...

    def nunique_params(self) -> int:
        """Return number of parameters unique to the functional."""
        ...

    def __call__(self, sol: Array, param: Array) -> Array:
        """Evaluate the functional."""
        ...

    def state_jacobian(self, sol: Array, param: Array) -> Array:
        """Compute dQ/dy."""
        ...

    def param_jacobian(self, sol: Array, param: Array) -> Array:
        """Compute dQ/dp."""
        ...

    def state_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dy^2)·w at a specific time.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        time_idx : int
            Time index.
        wvec : Array
            Direction vector. Shape: (nstates, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nstates, 1)
        """
        ...

    def state_param_hvp(
        self, sol: Array, param: Array, time_idx: int, vvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dy dp)·v at a specific time.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        time_idx : int
            Time index.
        vvec : Array
            Direction vector. Shape: (nparams, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nstates, 1)
        """
        ...

    def param_state_hvp(
        self, sol: Array, param: Array, time_idx: int, wvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dp dy)·w at a specific time.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        time_idx : int
            Time index.
        wvec : Array
            Direction vector. Shape: (nstates, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nparams, 1)
        """
        ...

    def param_param_hvp(
        self, sol: Array, param: Array, vvec: Array
    ) -> Array:
        """
        Compute (d^2Q/dp^2)·v.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        param : Array
            Parameters. Shape: (nparams, 1)
        vvec : Array
            Direction vector. Shape: (nparams, 1)

        Returns
        -------
        Array
            HVP result. Shape: (nparams, 1)
        """
        ...

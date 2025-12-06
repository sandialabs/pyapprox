from typing import Protocol, Generic, runtime_checkable, Tuple
from abc import ABC, abstractmethod

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class ImplicitODEResidualProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def __call__(self, iterate: Array) -> Array: ...

    def set_time(self, time: float) -> None: ...

    def jacobian(self, state: Array) -> Array: ...

    def mass_matrix(self, nstates: int) -> Array: ...


@runtime_checkable
class ImplicitTimeSteppingResidualProtocol(Protocol, Generic[Array]):
    """
    Protocol for residuals used in time integration.

    This protocol defines the required methods for a residual to be compatible
    with the TimeIntegrator class.
    """

    def bkd(self) -> Backend[Array]:
        """
        Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations.
        """
        ...

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
        """
        Set the current time, time step size, and state for the residual.

        Parameters
        ----------
        time : float
            Current time.
        deltat : float
            Time step size.
        prev_state : Array
            Previous state
        """
        ...

    def jacobian(self, state: Array) -> Array: ...

    def linsolve(self, state: Array, prev_residual: Array) -> Array: ...


@runtime_checkable
class AdjointEnabledImplicitTimeSteppingResidualProtocol(
    Protocol, Generic[Array]
):
    """
    Protocol for residuals used in time integration that are capable of
    computing the gradient of a functional of the state with respect to
    parameters.

    This protocol defines the required methods for a residual to be compatible
    with the TimeIntegrator class.
    """

    def bkd(self) -> Backend[Array]:
        """
        Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations.
        """
        ...

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
        """
        Set the current time, time step size, and state for the residual.

        Parameters
        ----------
        time : float
            Current time.
        deltat : float
            Time step size.
        prev_state : Array
            Previous state
        """
        ...

    def jacobian(self, state: Array) -> Array: ...

    def linsolve(self, state: Array, prev_residual: Array) -> Array: ...

    def jacobian_wrt_parameters(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """
        Compute the Jacobian of the residual with respect to parameters.

        Parameters
        ----------
        fsol_nm1 : Array
            Solution at the previous time step.
        fsol_n : Array
            Solution at the current time step.

        Returns
        -------
        Array
            Parameter Jacobian matrix.
        """
        ...

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        """
        Compute quadrature samples and weights for the time integration.

        Parameters
        ----------
        times : Array
            Time points.

        Returns
        -------
        Tuple[Array, Array]
            Quadrature samples and weights.
        """
        ...

    def adjoint_diagonal_jacobian(self, fsol_n: Array) -> Array:
        """
        Compute the diagonal Jacobian for adjoint computations.

        Parameters
        ----------
        fsol_n : Array
            Solution at the current time step.

        Returns
        -------
        Array
            Diagonal Jacobian matrix for adjoint computations.
        """
        ...

    def adjoint_off_diagonal_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """
        Compute the off-diagonal Jacobian for adjoint computations.

        Parameters
        ----------
        fsol_n : Array
            Solution at the current time step.
        deltat_np1 : float
            Time step size for the next time step.

        Returns
        -------
        Array
            Off-diagonal Jacobian matrix for adjoint computations.
        """
        ...


class ImplicitTimeSteppingResidualBase(ABC, Generic[Array]):
    def __init__(self, residual: ImplicitODEResidualProtocol[Array]):
        self._residual = residual
        self._bkd = residual.bkd()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
        self._time = time
        self._deltat = deltat
        self._prev_state = prev_state

    @abstractmethod
    def __call__(self, state: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, state: Array) -> Array:
        raise NotImplementedError

    def linsolve(self, state: Array, residual: Array) -> Array:
        return self._bkd.solve(self.jacobian(state), residual)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the ImplicitODEResidualBase
        object.

        Returns
        -------
        str
            String representation of the object.
        """
        return (
            f"{self.__class__.__name__}(\n"
            f"  residual={type(self._residual).__name__},\n"
            f"  backend={type(self._bkd).__name__},\n"
            f"  time={getattr(self, '_time', None)},\n"
            f"  deltat={getattr(self, '_deltat', None)},\n"
            ")"
        )

"""ODE QoI function and functional classes."""

from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ODEFunctionalProtocol(Protocol[Array]):
    """Lightweight protocol for ODE QoI extraction functionals.

    Functionals extract a quantity of interest from an ODE solution
    trajectory. Unlike TransientFunctionalWithJacobianProtocol, this
    protocol requires only evaluation and nqoi — no Jacobians or HVPs.
    """

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        ...

    def __call__(self, sol: Array, times: Array) -> Array:
        """Evaluate the functional.

        Parameters
        ----------
        sol : Array
            Solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)

        Returns
        -------
        Array
            QoI values. Shape: (nqoi,)
        """
        ...


class AllStatesEndpointODEFunctional(Generic[Array]):
    """Return all states at the final time.

    Parameters
    ----------
    nstates : int
        Number of state variables.
    """

    def __init__(self, nstates: int) -> None:
        self._nstates = nstates

    def nqoi(self) -> int:
        return self._nstates

    def __call__(self, sol: Array, times: Array) -> Array:
        return sol[:, -1]


class SingleStateEndpointODEFunctional(Generic[Array]):
    """Return a single state at the final time.

    Parameters
    ----------
    state_idx : int
        Index of the state variable to extract.
    """

    def __init__(self, state_idx: int) -> None:
        self._state_idx = state_idx

    def nqoi(self) -> int:
        return 1

    def __call__(self, sol: Array, times: Array) -> Array:
        return sol[self._state_idx : self._state_idx + 1, -1]


class MaxODEFunctional(Generic[Array]):
    """Return the maximum of each state over time.

    Parameters
    ----------
    nstates : int
        Number of state variables.
    bkd : Backend[Array]
        Backend for array operations.
    """

    def __init__(self, nstates: int, bkd: Backend[Array]) -> None:
        self._nstates = nstates
        self._bkd = bkd

    def nqoi(self) -> int:
        return self._nstates

    def __call__(self, sol: Array, times: Array) -> Array:
        return self._bkd.max(sol, axis=1)


def _create_functional_from_string(
    name: str,
    nstates: int,
    bkd: Backend[Array],
) -> ODEFunctionalProtocol[Array]:
    """Create a functional object from a string name.

    Parameters
    ----------
    name : str
        Functional name: "endpoint", "endpoint_0", "endpoint_1", ..., "max".
    nstates : int
        Number of ODE state variables.
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    ODEFunctionalProtocol[Array]
        The functional object.
    """
    if name == "endpoint":
        return AllStatesEndpointODEFunctional(nstates)
    elif name.startswith("endpoint_"):
        idx = int(name.split("_")[1])
        return SingleStateEndpointODEFunctional(idx)
    elif name == "max":
        return MaxODEFunctional(nstates, bkd)
    else:
        raise ValueError(f"Unknown functional name: {name}")


class ODEQoIFunction(Generic[Array]):
    """Callable QoI function wrapping an ODE residual.

    Integrates the ODE for each parameter sample and extracts a quantity
    of interest from the solution trajectory via a functional.

    Parameters
    ----------
    residual : Any
        ODE residual implementing ODEResidualWithParamJacobianProtocol.
    initial_condition : Array
        Initial condition. Shape: (nstates, 1) or (nstates,).
    time_config : ODETimeConfig
        Time integration configuration.
    nparams : int
        Number of parameters.
    functional : ODEFunctionalProtocol
        Functional that extracts QoI from the solution trajectory.
    bkd : Backend[Array]
        Computational backend.
    stepper : str, optional
        Time stepping method: "backward_euler" (default), "forward_euler",
        "heun", "crank_nicolson"
    """

    def __init__(
        self,
        residual: Any,
        initial_condition: Array,
        time_config: "ODETimeConfig",
        nparams: int,
        functional: ODEFunctionalProtocol[Array],
        bkd: Backend[Array],
        stepper: str = "backward_euler",
    ) -> None:
        self._residual = residual
        self._functional = functional
        self._stepper_type = stepper
        self._bkd = bkd
        self._time_config = time_config
        self._nparams = nparams

        # Flatten initial_condition from (nstates, 1) to (nstates,) for integrator
        if initial_condition.ndim == 2:
            self._init_state = self._bkd.flatten(initial_condition)
        else:
            self._init_state = initial_condition
        self._nstates = self._init_state.shape[0]

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables (parameters)."""
        return self._nparams

    def nqoi(self) -> int:
        """Return number of QoI outputs."""
        return self._functional.nqoi()

    def __call__(self, samples: Array) -> Array:
        """Evaluate QoI for each parameter sample.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nparams, nsamples)

        Returns
        -------
        Array
            QoI values. Shape: (nqoi, nsamples)
        """
        nsamples = samples.shape[1]
        qoi_values = []

        for ii in range(nsamples):
            param = samples[:, ii]
            qoi = self._evaluate_single(param)
            qoi_values.append(qoi)

        return self._bkd.stack(qoi_values, axis=1)

    def solve_trajectory(self, param: Array) -> Tuple[Array, Array]:
        """Solve the ODE and return full trajectory.

        Parameters
        ----------
        param : Array
            Single parameter sample. Shape: (nparams, 1) or (nparams,)

        Returns
        -------
        solutions : Array
            Solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        """
        if param.ndim == 2:
            param = param[:, 0]

        self._residual.set_param(param)

        time_residual = self._create_time_residual(self._residual)

        from pyapprox.ode.implicit_steppers.integrator import (
            TimeIntegrator,
        )
        from pyapprox.util.rootfinding.newton import NewtonSolver

        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=self._time_config.init_time,
            final_time=self._time_config.final_time,
            deltat=self._time_config.deltat,
            newton_solver=newton_solver,
        )

        return integrator.solve(self._init_state)

    def _evaluate_single(self, param: Array) -> Array:
        """Evaluate QoI for a single parameter sample.

        Parameters
        ----------
        param : Array
            Single parameter sample. Shape: (nparams,)

        Returns
        -------
        Array
            QoI values. Shape: (nqoi,)
        """
        solutions, times = self.solve_trajectory(param)
        return self._functional(solutions, times)

    @staticmethod
    def _create_time_residual_for_stepper(
        residual: Any, stepper_type: str,
    ) -> Any:
        """Create time stepping residual based on stepper type."""
        if stepper_type == "backward_euler":
            from pyapprox.ode.implicit_steppers.backward_euler import (
                BackwardEulerHVP,
            )

            return BackwardEulerHVP(residual)
        elif stepper_type == "forward_euler":
            from pyapprox.ode.explicit_steppers.forward_euler import (
                ForwardEulerHVP,
            )

            return ForwardEulerHVP(residual)
        elif stepper_type == "heun":
            from pyapprox.ode.explicit_steppers.heun import (
                HeunHVP,
            )

            return HeunHVP(residual)
        elif stepper_type == "crank_nicolson":
            from pyapprox.ode.implicit_steppers.crank_nicolson import (
                CrankNicolsonHVP,
            )

            return CrankNicolsonHVP(residual)
        else:
            raise ValueError(f"Unknown stepper type: {stepper_type}")

    def _create_time_residual(self, residual: Any) -> Any:
        """Create time stepping residual based on stepper type."""
        return self._create_time_residual_for_stepper(
            residual, self._stepper_type,
        )


@dataclass
class ODETimeConfig:
    """Time configuration for ODE integration.

    Attributes
    ----------
    init_time : float
        Initial time for integration.
    final_time : float
        Final time for integration.
    deltat : float
        Time step for integration.
    """

    init_time: float
    final_time: float
    deltat: float

    def ntimes(self) -> int:
        """Return number of time steps."""
        return int((self.final_time - self.init_time) / self.deltat) + 1



"""ODE benchmark wrapper class.

Wraps ODE residuals from pyapprox.pde.time.benchmarks and integrates
them with the benchmark registry system.
"""

from dataclasses import dataclass
from typing import (
    Generic, Optional, TypeVar, Any, List, Protocol, Tuple, Union,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.benchmarks.benchmark import BoxDomain
from pyapprox.benchmarks.ground_truth import ODEGroundTruth

GT = TypeVar("GT")


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
    name: str, nstates: int, bkd: Backend[Array],
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
    """Wrapper that turns an ODE benchmark into a callable QoI function.

    This class integrates the ODE for each parameter sample and extracts
    a quantity of interest from the solution trajectory via a functional.

    Parameters
    ----------
    benchmark : ODEBenchmark
        The ODE benchmark to wrap.
    functional : ODEFunctionalProtocol
        Functional that extracts QoI from the solution trajectory.
    stepper : str, optional
        Time stepping method: "backward_euler" (default), "forward_euler",
        "heun", "crank_nicolson"

    Examples
    --------
    >>> benchmark = lotka_volterra_3species(bkd)
    >>> functional = AllStatesEndpointODEFunctional(nstates=3)
    >>> qoi_func = ODEQoIFunction(benchmark, functional)
    >>> samples = bkd.array([[0.5]*12]).T  # Single sample, shape (12, 1)
    >>> qoi_values = qoi_func(samples)  # Shape (3, 1) - 3 states at final time
    """

    def __init__(
        self,
        benchmark: "ODEBenchmark[Array, Any]",
        functional: ODEFunctionalProtocol[Array],
        stepper: str = "backward_euler",
    ) -> None:
        self._benchmark = benchmark
        self._functional = functional
        self._stepper_type = stepper
        self._bkd = benchmark._domain.bkd()

        # Get ground truth for initial conditions and time config
        gt = benchmark.ground_truth()
        if not isinstance(gt, ODEGroundTruth):
            raise ValueError("ODEQoIFunction requires ODEGroundTruth")

        # Flatten initial_condition from (nstates, 1) to (nstates,) for integrator
        self._init_state = self._bkd.flatten(gt.initial_condition)
        self._time_config = benchmark.time_config()
        self._nstates = gt.nstates
        self._nparams = gt.nparams

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

        residual = self._benchmark.residual()
        residual.set_param(param)

        time_residual = self._create_time_residual(residual)

        from pyapprox.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )

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

    def _create_time_residual(self, residual: Any) -> Any:
        """Create time stepping residual based on stepper type."""
        if self._stepper_type == "backward_euler":
            from pyapprox.pde.time.implicit_steppers.backward_euler import (
                BackwardEulerResidual,
            )
            return BackwardEulerResidual(residual)
        elif self._stepper_type == "forward_euler":
            from pyapprox.pde.time.explicit_steppers.forward_euler import (
                ForwardEulerResidual,
            )
            return ForwardEulerResidual(residual)
        elif self._stepper_type == "heun":
            from pyapprox.pde.time.explicit_steppers.heun import (
                HeunResidual,
            )
            return HeunResidual(residual)
        elif self._stepper_type == "crank_nicolson":
            from pyapprox.pde.time.implicit_steppers.crank_nicolson import (
                CrankNicolsonResidual,
            )
            return CrankNicolsonResidual(residual)
        else:
            raise ValueError(f"Unknown stepper type: {self._stepper_type}")


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


@dataclass
class ODEBenchmark(Generic[Array, GT]):
    """Benchmark for ODE systems.

    Unlike standard benchmarks, ODE benchmarks wrap an ODE residual
    rather than a function. The residual defines the right-hand side
    of the ODE: dy/dt = f(y, t; p).

    Attributes
    ----------
    _name : str
        Benchmark identifier.
    _residual : Any
        The ODE residual implementing ODEResidualWithParamJacobianProtocol.
    _domain : BoxDomain[Array]
        Parameter domain (bounds for each parameter).
    _ground_truth : GT
        Ground truth values (ODEGroundTruth).
    _time_config : ODETimeConfig
        Time integration configuration.
    _prior : Any, optional
        Prior distribution over parameters for UQ applications.
    _description : str
        Human-readable description.
    _reference : str
        Literature reference.
    """

    _name: str
    _residual: Any  # ODEResidualWithParamJacobianProtocol[Array]
    _domain: BoxDomain[Array]
    _ground_truth: GT
    _time_config: ODETimeConfig
    _prior: Optional[Any] = None  # DistributionProtocol[Array]
    _description: str = ""
    _reference: str = ""

    def name(self) -> str:
        """Return benchmark name."""
        return self._name

    def residual(self) -> Any:
        """Return the ODE residual.

        The residual implements the ODE right-hand side f(y, t; p) and
        supports methods like jacobian(), param_jacobian(), etc.
        """
        return self._residual

    def domain(self) -> BoxDomain[Array]:
        """Return parameter domain."""
        return self._domain

    def ground_truth(self) -> GT:
        """Return ground truth."""
        return self._ground_truth

    def time_config(self) -> ODETimeConfig:
        """Return time integration configuration."""
        return self._time_config

    def prior(self) -> Optional[Any]:
        """Return prior distribution over parameters, if available."""
        return self._prior

    def description(self) -> str:
        """Return description."""
        return self._description

    def reference(self) -> str:
        """Return literature reference."""
        return self._reference

    def nstates(self) -> int:
        """Return number of state variables."""
        gt = self._ground_truth
        if isinstance(gt, ODEGroundTruth):
            return gt.nstates
        raise ValueError("Ground truth does not contain nstates")

    def nparams(self) -> int:
        """Return number of parameters."""
        gt = self._ground_truth
        if isinstance(gt, ODEGroundTruth):
            return gt.nparams
        raise ValueError("Ground truth does not contain nparams")

    def integrator(
        self,
        stepper: str = "backward_euler",
    ) -> Any:
        """Create a time integrator for this ODE benchmark.

        This returns a configured TimeIntegrator ready to solve the ODE.
        Before calling solve(), set parameters on the residual using
        `benchmark.residual().set_param(params)`.

        Parameters
        ----------
        stepper : str, optional
            Time stepping method: "backward_euler" (default), "forward_euler",
            "heun", "crank_nicolson"

        Returns
        -------
        TimeIntegrator
            Configured time integrator. Call `integrator.solve(init_state)`
            to solve the ODE.

        Examples
        --------
        >>> benchmark = lotka_volterra_3species(bkd)
        >>> gt = benchmark.ground_truth()
        >>> residual = benchmark.residual()
        >>> residual.set_param(gt.nominal_parameters)
        >>> integrator = benchmark.integrator()
        >>> solutions, times = integrator.solve(bkd.flatten(gt.initial_condition))
        """
        from pyapprox.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )

        # Create time stepping residual
        time_residual = self._create_time_residual(stepper)

        # Create Newton solver and integrator
        newton_solver = NewtonSolver(time_residual)
        return TimeIntegrator(
            init_time=self._time_config.init_time,
            final_time=self._time_config.final_time,
            deltat=self._time_config.deltat,
            newton_solver=newton_solver,
        )

    def _create_time_residual(self, stepper: str) -> Any:
        """Create time stepping residual based on stepper type."""
        if stepper == "backward_euler":
            from pyapprox.pde.time.implicit_steppers.backward_euler import (
                BackwardEulerResidual,
            )
            return BackwardEulerResidual(self._residual)
        elif stepper == "forward_euler":
            from pyapprox.pde.time.explicit_steppers.forward_euler import (
                ForwardEulerResidual,
            )
            return ForwardEulerResidual(self._residual)
        elif stepper == "heun":
            from pyapprox.pde.time.explicit_steppers.heun import (
                HeunResidual,
            )
            return HeunResidual(self._residual)
        elif stepper == "crank_nicolson":
            from pyapprox.pde.time.implicit_steppers.crank_nicolson import (
                CrankNicolsonResidual,
            )
            return CrankNicolsonResidual(self._residual)
        else:
            raise ValueError(f"Unknown stepper type: {stepper}")

    def qoi_function(
        self,
        functional: Union[str, ODEFunctionalProtocol[Array]] = "endpoint",
        stepper: str = "backward_euler",
    ) -> ODEQoIFunction[Array]:
        """Create a callable QoI function from this benchmark.

        This is a convenience method that accepts either a string shorthand
        or a functional object. Strings are converted to the corresponding
        built-in functional class.

        Parameters
        ----------
        functional : str or ODEFunctionalProtocol, optional
            How to extract QoI from the solution trajectory. Either a
            string shorthand ("endpoint", "endpoint_0", "max") or an
            object satisfying ODEFunctionalProtocol.
        stepper : str, optional
            Time stepping method: "backward_euler" (default), "forward_euler",
            "heun", "crank_nicolson"

        Returns
        -------
        ODEQoIFunction
            Callable function with signature: (samples: Array) -> Array
            where samples has shape (nparams, nsamples) and returns
            shape (nqoi, nsamples).

        Examples
        --------
        >>> benchmark = lotka_volterra_3species(bkd)
        >>> qoi_func = benchmark.qoi_function()
        >>> samples = prior.rvs(100, bkd)  # Shape: (12, 100)
        >>> qoi_values = qoi_func(samples)  # Shape: (3, 100)

        >>> # Get only the first state at final time
        >>> qoi_func = benchmark.qoi_function(functional="endpoint_0")
        >>> qoi_values = qoi_func(samples)  # Shape: (1, 100)
        """
        if isinstance(functional, str):
            functional = _create_functional_from_string(
                functional, self.nstates(), self._domain.bkd(),
            )
        return ODEQoIFunction(self, functional=functional, stepper=stepper)

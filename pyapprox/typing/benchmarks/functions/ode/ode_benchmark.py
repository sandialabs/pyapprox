"""ODE benchmark wrapper class.

Wraps ODE residuals from pyapprox.typing.pde.time.benchmarks and integrates
them with the benchmark registry system.
"""

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Any, Callable, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import BoxDomain
from pyapprox.typing.benchmarks.ground_truth import ODEGroundTruth

GT = TypeVar("GT")


class ODEQoIFunction(Generic[Array]):
    """Wrapper that turns an ODE benchmark into a callable QoI function.

    This class integrates the ODE for each parameter sample and extracts
    a quantity of interest from the solution trajectory.

    Parameters
    ----------
    benchmark : ODEBenchmark
        The ODE benchmark to wrap.
    functional : str or Callable, optional
        How to extract QoI from the solution trajectory:
        - "endpoint": Returns state at final time (default)
        - "endpoint_0", "endpoint_1", etc.: Returns specific state at final time
        - "max": Returns maximum of all states over time
        - Callable: Custom functional (sol: Array, times: Array) -> Array
    stepper : str, optional
        Time stepping method: "backward_euler" (default), "forward_euler",
        "heun", "crank_nicolson"

    Examples
    --------
    >>> benchmark = lotka_volterra_3species(bkd)
    >>> qoi_func = ODEQoIFunction(benchmark)
    >>> samples = bkd.array([[0.5]*12]).T  # Single sample, shape (12, 1)
    >>> qoi_values = qoi_func(samples)  # Shape (3, 1) - 3 states at final time
    """

    def __init__(
        self,
        benchmark: "ODEBenchmark[Array, Any]",
        functional: str = "endpoint",
        stepper: str = "backward_euler",
    ) -> None:
        self._benchmark = benchmark
        self._functional_type = functional
        self._stepper_type = stepper
        self._bkd = benchmark._domain.bkd()

        # Get ground truth for initial conditions and time config
        gt = benchmark.ground_truth()
        if not isinstance(gt, ODEGroundTruth):
            raise ValueError("ODEQoIFunction requires ODEGroundTruth")

        self._init_state = self._bkd.asarray(gt.initial_condition)
        self._time_config = benchmark.time_config()
        self._nstates = gt.nstates
        self._nparams = gt.nparams

        # Parse functional type
        if functional == "endpoint":
            self._qoi_indices: Optional[List[int]] = None  # All states
        elif functional.startswith("endpoint_"):
            idx = int(functional.split("_")[1])
            self._qoi_indices = [idx]
        elif functional == "max":
            self._qoi_indices = None
            self._functional_type = "max"
        elif callable(functional):
            self._custom_functional = functional
            self._functional_type = "custom"
        else:
            raise ValueError(f"Unknown functional type: {functional}")

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables (parameters)."""
        return self._nparams

    def nqoi(self) -> int:
        """Return number of QoI outputs."""
        if self._functional_type == "max":
            return self._nstates
        elif self._qoi_indices is not None:
            return len(self._qoi_indices)
        else:
            return self._nstates

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
        # Set parameters on residual
        residual = self._benchmark.residual()
        residual.set_param(param)

        # Create time stepper
        time_residual = self._create_time_residual(residual)

        # Create Newton solver and integrator
        from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
        from pyapprox.typing.pde.time.implicit_steppers.integrator import (
            TimeIntegrator,
        )

        newton_solver = NewtonSolver(time_residual)
        integrator = TimeIntegrator(
            init_time=self._time_config.init_time,
            final_time=self._time_config.final_time,
            deltat=self._time_config.deltat,
            newton_solver=newton_solver,
        )

        # Solve forward problem
        solutions, times = integrator.solve(self._init_state)

        # Extract QoI
        return self._extract_qoi(solutions, times)

    def _create_time_residual(self, residual: Any) -> Any:
        """Create time stepping residual based on stepper type."""
        if self._stepper_type == "backward_euler":
            from pyapprox.typing.pde.time.implicit_steppers.backward_euler import (
                BackwardEulerResidual,
            )
            return BackwardEulerResidual(residual)
        elif self._stepper_type == "forward_euler":
            from pyapprox.typing.pde.time.explicit_steppers.forward_euler import (
                ForwardEulerResidual,
            )
            return ForwardEulerResidual(residual)
        elif self._stepper_type == "heun":
            from pyapprox.typing.pde.time.explicit_steppers.heun import (
                HeunResidual,
            )
            return HeunResidual(residual)
        elif self._stepper_type == "crank_nicolson":
            from pyapprox.typing.pde.time.implicit_steppers.crank_nicolson import (
                CrankNicolsonResidual,
            )
            return CrankNicolsonResidual(residual)
        else:
            raise ValueError(f"Unknown stepper type: {self._stepper_type}")

    def _extract_qoi(self, solutions: Array, times: Array) -> Array:
        """Extract QoI from solution trajectory.

        Parameters
        ----------
        solutions : Array
            Solution trajectory. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)

        Returns
        -------
        Array
            QoI values. Shape: (nqoi,)
        """
        if self._functional_type == "custom":
            return self._custom_functional(solutions, times)
        elif self._functional_type == "max":
            # Return max value of each state over time
            return self._bkd.max(solutions, axis=1)
        elif self._qoi_indices is not None:
            # Return specified states at final time
            return solutions[self._qoi_indices, -1]
        else:
            # Return all states at final time
            return solutions[:, -1]


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

    def qoi_function(
        self,
        functional: str = "endpoint",
        stepper: str = "backward_euler",
    ) -> ODEQoIFunction[Array]:
        """Create a callable QoI function from this benchmark.

        This returns a function that takes parameter samples and returns
        QoI values, handling all the ODE integration internally.

        Parameters
        ----------
        functional : str, optional
            How to extract QoI from the solution trajectory:
            - "endpoint": Returns all states at final time (default)
            - "endpoint_0", "endpoint_1", etc.: Returns specific state at final time
            - "max": Returns maximum of each state over time
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
        return ODEQoIFunction(self, functional=functional, stepper=stepper)

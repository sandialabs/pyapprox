"""ODE forward UQ problem.

Parameterized ODE system with prior distribution for forward UQ.
All ODE instances have priors, so this is a forward UQ problem.
No ground truth — these are Problems, not Benchmarks.
"""

from __future__ import annotations

from typing import Generic, Union

from pyapprox_benchmarks.functions.ode.ode_qoi import (
    ODEFunctionalProtocol,
    ODEQoIFunction,
    ODETimeConfig,
    _create_functional_from_string,
)
from pyapprox_benchmarks.protocols import DomainProtocol
from pyapprox.ode.protocols.ode_residual import (
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class ODEForwardUQProblem(Generic[Array]):
    """Parameterized ODE forward UQ problem.

    Bundles an ODE residual with prior distribution, parameter domain,
    time integration configuration, initial condition, and nominal
    parameter values.

    Parameters
    ----------
    name : str
        Problem name.
    residual : ODEResidualWithParamJacobianProtocol[Array]
        ODE residual defining dy/dt = f(y, t; p).
    prior : DistributionProtocol[Array]
        Prior distribution over parameters.
    domain : DomainProtocol[Array]
        Parameter domain (bounds).
    time_config : ODETimeConfig
        Time integration configuration.
    nstates : int
        Number of state variables.
    initial_condition : Array
        Default initial condition. Shape: (nstates, 1).
    nominal_parameters : Array
        Nominal parameter values. Shape: (nparams, 1).
    bkd : Backend[Array]
        Computational backend.
    description : str
        Human-readable description.
    reference : str
        Literature reference.
    estimated_evaluation_cost : float
        Estimated cost per evaluation in seconds.
    """

    def __init__(
        self,
        name: str,
        residual: ODEResidualWithParamJacobianProtocol[Array],
        prior: DistributionProtocol[Array],
        domain: DomainProtocol[Array],
        time_config: ODETimeConfig,
        nstates: int,
        initial_condition: Array,
        nominal_parameters: Array,
        bkd: Backend[Array],
        description: str = "",
        reference: str = "",
        estimated_evaluation_cost: float = 0.0,
    ) -> None:
        self._name = name
        self._residual = residual
        self._prior = prior
        self._domain = domain
        self._time_config = time_config
        self._nstates = nstates
        self._initial_condition = initial_condition
        self._nominal_parameters = nominal_parameters
        self._bkd = bkd
        self._description = description
        self._reference = reference
        self._estimated_evaluation_cost = estimated_evaluation_cost

    def name(self) -> str:
        """Return the problem name."""
        return self._name

    def residual(self) -> ODEResidualWithParamJacobianProtocol[Array]:
        """Return the ODE residual."""
        return self._residual

    def prior(self) -> DistributionProtocol[Array]:
        """Return prior distribution over parameters."""
        return self._prior

    def domain(self) -> DomainProtocol[Array]:
        """Return parameter domain."""
        return self._domain

    def time_config(self) -> ODETimeConfig:
        """Return time integration configuration."""
        return self._time_config

    def nstates(self) -> int:
        """Return number of state variables."""
        return self._nstates

    def nparams(self) -> int:
        """Return number of parameters."""
        return self._residual.nparams()

    def initial_condition(self) -> Array:
        """Return initial condition. Shape: (nstates, 1)."""
        return self._initial_condition

    def nominal_parameters(self) -> Array:
        """Return nominal parameter values. Shape: (nparams, 1)."""
        return self._nominal_parameters

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def description(self) -> str:
        """Return human-readable description."""
        return self._description

    def reference(self) -> str:
        """Return literature reference."""
        return self._reference

    def estimated_evaluation_cost(self) -> float:
        """Return estimated cost per evaluation in seconds."""
        return self._estimated_evaluation_cost

    def function(
        self,
        functional: Union[str, ODEFunctionalProtocol[Array]] = "endpoint",
        stepper: str = "backward_euler",
    ) -> ODEQoIFunction[Array]:
        """Build a callable QoI function from this problem.

        Parameters
        ----------
        functional : str or ODEFunctionalProtocol, optional
            How to extract QoI from the solution trajectory. String
            shorthands: "endpoint", "endpoint_0", "max". Default
            "endpoint".
        stepper : str, optional
            Time stepping method. Default "backward_euler".

        Returns
        -------
        ODEQoIFunction
            Callable with signature (samples: Array) -> Array.
        """
        resolved_functional: ODEFunctionalProtocol[Array]
        if isinstance(functional, str):
            resolved_functional = _create_functional_from_string(
                functional, self._nstates, self._bkd,
            )
        else:
            resolved_functional = functional
        return ODEQoIFunction(
            residual=self._residual,
            initial_condition=self._initial_condition,
            time_config=self._time_config,
            nparams=self.nparams(),
            functional=resolved_functional,
            bkd=self._bkd,
            stepper=stepper,
        )

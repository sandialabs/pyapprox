"""Lotka-Volterra 3-species ODE benchmark instance."""

from __future__ import annotations

from typing import Any, Generic, Optional, Union

from pyapprox.benchmarks.benchmark import BoxDomain
from pyapprox.benchmarks.functions.ode import (
    ODEBenchmark,
    ODEQoIFunction,
    ODETimeConfig,
)
from pyapprox.benchmarks.ground_truth import ODEGroundTruth
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.functions.ode.lotka_volterra import (
    LotkaVolterraResidual,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


class ODEBenchmarkWrapper(Generic[Array]):
    """ODE benchmark wrapper.

    Satisfies: HasResidual, HasPrior, HasEstimatedEvaluationCost.
    """

    def __init__(
        self,
        inner: ODEBenchmark[Array, ODEGroundTruth[Array]],
        estimated_cost: float,
    ) -> None:
        self._inner = inner
        self._estimated_cost = estimated_cost

    def name(self) -> str:
        return self._inner.name()

    def residual(self) -> Any:
        return self._inner.residual()

    def domain(self) -> BoxDomain[Array]:
        return self._inner.domain()

    def prior(self) -> Optional[Any]:
        return self._inner.prior()

    def ground_truth(self) -> ODEGroundTruth[Array]:
        return self._inner.ground_truth()

    def time_config(self) -> ODETimeConfig:
        return self._inner.time_config()

    def nstates(self) -> int:
        return self._inner.nstates()

    def nparams(self) -> int:
        return self._inner.nparams()

    def description(self) -> str:
        return self._inner.description()

    def reference(self) -> str:
        return self._inner.reference()

    def qoi_function(
        self,
        functional: Union[str, Any] = "endpoint",
        stepper: str = "backward_euler",
    ) -> ODEQoIFunction[Array]:
        return self._inner.qoi_function(functional=functional, stepper=stepper)

    def estimated_evaluation_cost(self) -> float:
        return self._estimated_cost


def lotka_volterra_3species(
    bkd: Backend[Array],
    final_time: float = 10.0,
    deltat: float = 1.0,
) -> ODEBenchmarkWrapper[Array]:
    """Create the 3-species competitive Lotka-Volterra benchmark.

    Standard 3-species competitive Lotka-Volterra benchmark with uniform
    prior U[0.3, 0.7]^12 for all parameters.

    The system is governed by:
        dx_i/dt = r_i * x_i * (1 - sum_j a_ij * x_j), i = 1, 2, 3

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    final_time : float, optional
        Final simulation time. Default 10.0.
    deltat : float, optional
        Time step. Default 1.0.

    Returns
    -------
    ODEBenchmarkWrapper
        The Lotka-Volterra benchmark instance.

    Notes
    -----
    Parameters are organized as:
        p = [r_0, r_1, r_2, a_00, a_01, ..., a_22]

    Total: 12 parameters (3 growth rates + 9 competition coefficients).

    References
    ----------
    TODO: Find specific paper reference for this configuration.
    Original Lotka-Volterra: Lotka (1925), Volterra (1926).
    """
    nspecies = 3
    nparams = nspecies + nspecies * nspecies  # 3 + 9 = 12

    # Create residual
    residual = LotkaVolterraResidual(nspecies, bkd)

    # Prior: U[0.3, 0.7]^12 for all parameters
    # Note: scipy.stats.uniform(0.3, 0.4) means U[0.3, 0.3+0.4] = U[0.3, 0.7]
    prior = IndependentJoint(
        [UniformMarginal(0.3, 0.7, bkd) for _ in range(nparams)],
        bkd,
    )

    # Domain bounds
    bounds = bkd.array([[0.3, 0.7]] * nparams)

    # Initial condition (from legacy code) - shape (nstates, 1)
    initial_condition = bkd.array([[0.3], [0.4], [0.3]])

    # Nominal parameters (center of prior) - shape (nparams, 1)
    nominal_parameters = bkd.full((nparams, 1), 0.5)

    inner = ODEBenchmark(
        _name="lotka_volterra_3species",
        _residual=residual,
        _domain=BoxDomain(_bounds=bounds, _bkd=bkd),
        _ground_truth=ODEGroundTruth(
            nstates=nspecies,
            nparams=nparams,
            initial_condition=initial_condition,
            nominal_parameters=nominal_parameters,
            init_time=0.0,
            final_time=final_time,
            deltat=deltat,
        ),
        _time_config=ODETimeConfig(
            init_time=0.0,
            final_time=final_time,
            deltat=deltat,
        ),
        _prior=prior,
        _description="3-species competitive Lotka-Volterra",
        _reference="Lotka (1925), Volterra (1926)",  # TODO: find specific paper
    )

    return ODEBenchmarkWrapper(inner, estimated_cost=3.7e-05)


@BenchmarkRegistry.register(
    "lotka_volterra_3species",
    category="ode",
    description="3-species competitive Lotka-Volterra ODE system",
)
def _lotka_volterra_3species_factory(
    bkd: Backend[Array],
) -> ODEBenchmarkWrapper[Array]:
    return lotka_volterra_3species(bkd)

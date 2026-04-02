"""Lotka-Volterra OED benchmark for prediction.

Provides a benchmark coupling a 3-species competitive Lotka-Volterra ODE
with observation and prediction models for OED. The observation model
records species 0 and 2 at all time points; the prediction model
records species 1 at every other time point.
"""

from typing import Generic, Tuple

from pyapprox.benchmarks.functions.ode import (
    ODEFunctionalProtocol,
    ODEQoIFunction,
)
from pyapprox.benchmarks.instances.ode.lotka_volterra import (
    lotka_volterra_3species,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.problems.inverse import BayesianInferenceProblem
from pyapprox.util.backends.protocols import Array, Backend


class ObservationFunctional(Generic[Array]):
    """Extract species 0 and 2 at all time points.

    Parameters
    ----------
    ntimes : int
        Number of time points in the solution trajectory.
    bkd : Backend[Array]
        Backend for array operations.
    """

    def __init__(self, ntimes: int, bkd: Backend[Array]) -> None:
        self._ntimes = ntimes
        self._bkd = bkd

    def nqoi(self) -> int:
        return 2 * self._ntimes

    def __call__(self, sol: Array, times: Array) -> Array:
        return self._bkd.concatenate([sol[0, :], sol[2, :]])


class PredictionFunctional(Generic[Array]):
    """Extract species 1 at every other time point (odd indices).

    Parameters
    ----------
    ntimes : int
        Number of time points in the solution trajectory.
    """

    def __init__(self, ntimes: int) -> None:
        self._pred_indices = list(range(1, ntimes, 2))

    def nqoi(self) -> int:
        return len(self._pred_indices)

    def __call__(self, sol: Array, times: Array) -> Array:
        return sol[1, self._pred_indices]


class LotkaVolterraOEDBenchmark(Generic[Array]):
    """Lotka-Volterra 3-species OED benchmark for prediction.

    Uses the 3-species competitive Lotka-Volterra system with 12
    parameters (3 growth rates + 9 competition coefficients).

    Observation model: species 0 and 2 at all time points.
    Prediction model: species 1 at every other time point.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    noise_std : float, optional
        Observation noise standard deviation. Default 0.1.
    final_time : float, optional
        Final simulation time. Default 10.0.
    deltat : float, optional
        Time step. Default 1.0.
    stepper : str, optional
        Time stepping method: "backward_euler" (default), "forward_euler",
        "heun", "crank_nicolson".
    """

    def __init__(
        self,
        bkd: Backend[Array],
        noise_std: float = 0.1,
        final_time: float = 10.0,
        deltat: float = 1.0,
        stepper: str = "backward_euler",
    ) -> None:
        self._bkd = bkd
        self._wrapper = lotka_volterra_3species(
            bkd,
            final_time=final_time,
            deltat=deltat,
        )
        inner = self._wrapper._inner

        tc = inner.time_config()
        ntimes = tc.ntimes()

        obs_functional: ODEFunctionalProtocol[Array] = ObservationFunctional(
            ntimes,
            bkd,
        )
        pred_functional: ODEFunctionalProtocol[Array] = PredictionFunctional(
            ntimes,
        )

        self._obs_model = ODEQoIFunction(
            inner,
            functional=obs_functional,
            stepper=stepper,
        )
        self._pred_model = ODEQoIFunction(
            inner,
            functional=pred_functional,
            stepper=stepper,
        )

        # Create BayesianInferenceProblem
        noise_variances = bkd.full(
            (self._obs_model.nqoi(),), noise_std**2,
        )
        self._problem = BayesianInferenceProblem(
            obs_map=self._obs_model,
            prior=self._wrapper.prior(),
            noise_variances=noise_variances,
            bkd=bkd,
        )

        # Precompute time arrays
        self._solution_times = bkd.linspace(
            tc.init_time,
            tc.final_time,
            ntimes,
        )
        self._observation_times = self._solution_times
        pred_indices = list(range(1, ntimes, 2))
        self._prediction_times = self._solution_times[pred_indices]

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def problem(self) -> BayesianInferenceProblem[Array]:
        """Get the inference problem (obs_map + prior + noise)."""
        return self._problem

    def prior(self):
        """Return the prior distribution over parameters."""
        return self._wrapper.prior()

    def obs_map(self) -> ODEQoIFunction[Array]:
        """Return the observation map (species 0 and 2 at all times)."""
        return self._obs_model

    def qoi_map(self) -> ODEQoIFunction[Array]:
        """Return the QoI map (species 1 at odd times)."""
        return self._pred_model

    def solution_times(self) -> Array:
        """Return all solution time points. Shape: (ntimes,)"""
        return self._solution_times

    def observation_times(self) -> Array:
        """Return observation time points. Shape: (ntimes,)"""
        return self._observation_times

    def prediction_times(self) -> Array:
        """Return prediction time points. Shape: (npred_times,)"""
        return self._prediction_times

    def evaluate_both(self, samples: Array) -> Tuple[Array, Array]:
        """Evaluate observation and prediction models with a single solve.

        Solves the ODE once per sample and extracts both observation
        and prediction quantities from the same trajectory.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nparams, nsamples)

        Returns
        -------
        obs_values : Array
            Observation values. Shape: (nobs, nsamples)
        pred_values : Array
            Prediction values. Shape: (npred, nsamples)
        """
        nsamples = samples.shape[1]
        obs_results = []
        pred_results = []

        for ii in range(nsamples):
            param = samples[:, ii : ii + 1]
            solutions, times = self._obs_model.solve_trajectory(param)
            obs_results.append(self._obs_model._functional(solutions, times))
            pred_results.append(self._pred_model._functional(solutions, times))

        return (
            self._bkd.stack(obs_results, axis=1),
            self._bkd.stack(pred_results, axis=1),
        )


@BenchmarkRegistry.register(
    "lotka_volterra_oed",
    category="oed",
    description="Lotka-Volterra 3-species OED benchmark",
)
def _lotka_volterra_oed_factory(
    bkd: Backend[Array],
) -> LotkaVolterraOEDBenchmark[Array]:
    return LotkaVolterraOEDBenchmark(bkd, noise_std=0.1)

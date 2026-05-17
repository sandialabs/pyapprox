"""Trajectory matching fitter for dynamical systems surrogates.

Fits a learned function F_eta by minimizing trajectory MSE:
    L(eta) = (1/2sigma^2) sum_k sum_i ||y_k(t_i; eta) - obs_k(t_i)||^2

Uses a bindable optimizer (BindableOptimizerProtocol) following the same
pattern as GPMaximumLikelihoodFitter and MSEFitter.
"""

from typing import Generic, List, Optional, Tuple

import numpy as np

from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.result_protocol import (
    OptimizerResultProtocol,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.losses.trajectory_matching import (
    TrajectoryMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.protocols import (
    LearnedFunctionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class TrajectoryMatchingFitResult(Generic[Array]):
    """Result of trajectory matching fitting.

    Parameters
    ----------
    surrogate : LearnedFunctionProtocol[Array]
        Fitted surrogate with optimized parameters.
    optimizer_result : OptimizerResultProtocol[Array]
        Raw optimizer result.
    final_loss : float
        Loss value at optimized parameters.
    """

    def __init__(
        self,
        surrogate: LearnedFunctionProtocol[Array],
        optimizer_result: OptimizerResultProtocol[Array],
        final_loss: float,
    ) -> None:
        self._surrogate = surrogate
        self._optimizer_result = optimizer_result
        self._final_loss = final_loss

    def surrogate(self) -> LearnedFunctionProtocol[Array]:
        return self._surrogate

    def optimizer_result(self) -> OptimizerResultProtocol[Array]:
        return self._optimizer_result

    def final_loss(self) -> float:
        return self._final_loss


class TrajectoryMatchingFitter(Generic[Array]):
    """Fits a learned function by minimizing trajectory MSE via adjoint gradients.

    Accepts a pre-configured wrapper and integrator. The user controls the
    ODE pipeline (stepper choice, Newton options, time grid); the fitter
    builds the loss and runs the optimizer.

    Parameters
    ----------
    wrapper : BatchedBoundODEResidual[Array]
        ODE residual wrapping the learned function.
    integrator : TimeIntegrator[Array]
        Pre-configured time integrator with adjoint support.
    noise_std : float
        Observation noise standard deviation for MSE scaling.

    Examples
    --------
    >>> wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu)
    >>> stepper = ForwardEulerAdjoint(wrapper)
    >>> newton = NewtonSolver(stepper)
    >>> integrator = TimeIntegrator(0.0, 0.3, 0.01, newton)
    >>> fitter = TrajectoryMatchingFitter(wrapper, integrator)
    >>> fitter.set_optimizer(ScipyTrustConstrOptimizer(maxiter=200, gtol=1e-10))
    >>> result = fitter.fit(init_states, observations)
    >>> fitted = result.surrogate()
    """

    def __init__(
        self,
        wrapper: BatchedBoundODEResidual[Array],
        integrator: TimeIntegrator[Array],
        noise_std: float = 1.0,
    ) -> None:
        self._wrapper = wrapper
        self._integrator = integrator
        self._noise_std = noise_std
        self._bkd = wrapper.bkd()
        self._optimizer: Optional[BindableOptimizerProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_optimizer(self, optimizer: BindableOptimizerProtocol[Array]) -> None:
        """Set custom optimizer.

        Parameters
        ----------
        optimizer : BindableOptimizerProtocol[Array]
            Configured optimizer (e.g., ScipyTrustConstrOptimizer).
            Cloned during fit() to avoid shared state.
        """
        self._optimizer = optimizer

    def optimizer(self) -> Optional[BindableOptimizerProtocol[Array]]:
        return self._optimizer

    def fit(
        self,
        init_states: Array,
        observations: Array,
        obs_tuples: Optional[List[Tuple[int, Array]]] = None,
        bounds: Optional[Array] = None,
    ) -> TrajectoryMatchingFitResult[Array]:
        """Fit learned function to trajectory observations.

        Parameters
        ----------
        init_states : Array
            Initial states for batched trajectories. Shape: (n_dynamic*k,)
        observations : Array
            Target observations. Shape: (nobs,). If obs_tuples is None,
            assumes all states observed at all times and nobs = nstates*ntimes.
        obs_tuples : Optional[List[Tuple[int, Array]]]
            Observation specification: list of (state_idx, time_indices).
            If None, observes all states at all integrator time steps.
        bounds : Optional[Array]
            Parameter bounds. Shape: (nparams, 2). If None, unbounded.

        Returns
        -------
        TrajectoryMatchingFitResult[Array]
            Result containing fitted surrogate and optimization metadata.
        """
        nstates = init_states.shape[0]
        if obs_tuples is None:
            ntimes = self._integrator.ntimes()
            obs_time_indices = self._bkd.arange(ntimes)
            obs_tuples = [(i, obs_time_indices) for i in range(nstates)]

        loss = TrajectoryMatchingLoss(
            wrapper=self._wrapper,
            integrator=self._integrator,
            init_states=init_states,
            obs_tuples=obs_tuples,
            observations=observations,
            noise_std=self._noise_std,
        )

        nparams = self._wrapper.nparams()

        if bounds is None:
            bounds = self._bkd.hstack([
                self._bkd.full((nparams, 1), -np.inf),
                self._bkd.full((nparams, 1), np.inf),
            ])

        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=1000)

        optimizer.bind(loss, bounds)

        init_guess = self._bkd.reshape(
            self._wrapper.get_param(), (nparams, 1)
        )

        result = optimizer.minimize(init_guess)

        optimal_params = result.optima()
        if optimal_params.ndim == 2:
            optimal_params = optimal_params[:, 0]

        self._wrapper.set_param(optimal_params)

        return TrajectoryMatchingFitResult(
            surrogate=self._wrapper.learned_function(),
            optimizer_result=result,
            final_loss=result.fun(),
        )

"""Trajectory matching loss: forward ODE solve + MSE functional with adjoint gradient.

Wraps BatchedBoundODEResidual + TimeIntegrator + TransientMSEFunctional
as a FunctionWithJacobianProtocol for use with DerivativeChecker and optimizers.
"""

from typing import Generic, List, Tuple

from pyapprox.ode.functionals.mse import TransientMSEFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.util.backends.protocols import Array, Backend


class TrajectoryMatchingLoss(Generic[Array]):
    """MSE loss over ODE trajectories with adjoint-based gradient.

    Satisfies FunctionWithJacobianProtocol:
      - input: surrogate parameters eta, shape (nparams, 1)
      - output: scalar loss Q, shape (1, 1)
      - jacobian: dQ/d_eta via discrete adjoint, shape (1, nparams)

    Parameters
    ----------
    wrapper : BatchedBoundODEResidual[Array]
        ODE residual wrapping the learned function.
    integrator : TimeIntegrator[Array]
        Time integrator with adjoint support.
    init_states : Array
        Initial states for batched trajectories. Shape: (n_dynamic*k,)
    obs_tuples : List[Tuple[int, Array]]
        Observation specification for TransientMSEFunctional.
    observations : Array
        Flat observation vector. Shape: (nobs,)
    noise_std : float
        Noise standard deviation for the MSE functional.
    """

    def __init__(
        self,
        wrapper: BatchedBoundODEResidual[Array],
        integrator: TimeIntegrator[Array],
        init_states: Array,
        obs_tuples: List[Tuple[int, Array]],
        observations: Array,
        noise_std: float = 1.0,
    ) -> None:
        self._wrapper = wrapper
        self._integrator = integrator
        self._init_states = init_states
        self._bkd = wrapper.bkd()

        nstates = init_states.shape[0]
        functional = TransientMSEFunctional(
            nstates=nstates,
            nresidual_params=wrapper.nparams(),
            obs_tuples=obs_tuples,
            bkd=self._bkd,
            noise_std=noise_std,
        )
        functional.set_observations(observations)
        self._functional = functional
        integrator.set_functional(functional)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._wrapper.nparams()

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate loss at parameter values.

        Parameters
        ----------
        samples : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Scalar loss. Shape: (1, 1)
        """
        eta = samples[:, 0]
        self._wrapper.set_param(eta)
        fwd_sols, times = self._integrator.solve(self._init_states)
        return self._functional(fwd_sols, samples)

    def jacobian(self, sample: Array) -> Array:
        """Compute gradient dQ/d_eta via discrete adjoint.

        Parameters
        ----------
        sample : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Gradient. Shape: (1, nparams)
        """
        eta = sample[:, 0]
        self._wrapper.set_param(eta)
        fwd_sols, times = self._integrator.solve(self._init_states)
        return self._integrator.gradient(fwd_sols, times, sample)

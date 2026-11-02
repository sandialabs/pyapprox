"""Trajectory matching loss: forward ODE solve + MSE functional with adjoint gradient.

Wraps a :class:`BatchedBoundODEResidual`, a stepping method bound to its
system, and a :class:`TransientMSEFunctional`, for use with
``DerivativeChecker`` and optimizers. The gradient is the discrete adjoint
of the method's steps
(:class:`~pyapprox.ode.trajectory_derivatives.TrajectoryDerivatives`).
"""

from typing import Generic, List, Sequence, Tuple

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.functionals.mse import TransientMSEFunctional
from pyapprox.ode.integration import Integrator
from pyapprox.ode.protocols import StepperProtocol
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.stepping_methods import SteppingMethodProtocol
from pyapprox.ode.trajectory_derivatives import (
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_functional import from_transient_functional
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore, trajectory_array
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.util.backends.protocols import Array, Backend


class TrajectoryMatchingLoss(Generic[Array]):
    """MSE loss over ODE trajectories with adjoint-based gradient.

    Satisfies ``ObjectiveProtocol`` (jacobian also carried by the
    ``derivatives()`` bundle):
      - input: surrogate parameters eta, shape (nparams, 1)
      - output: scalar loss Q, shape (1, 1)
      - jacobian: dQ/d_eta via discrete adjoint, shape (1, nparams)

    Parameters
    ----------
    wrapper : BatchedBoundODEResidual[Array]
        ODE residual wrapping the learned function.
    method : SteppingMethodProtocol
        The time-stepping method, bound to ``wrapper``'s system.
    times : sequence of float
        The step boundaries :math:`t_0 < \\dots < t_N`, the observations'
        stored times; they must determine their step sizes exactly (see
        :meth:`~pyapprox.ode.step_plan.GivenSizes.from_times`).
    init_states : Array
        Initial states for batched trajectories. Shape: (n_dynamic*k,)
    obs_tuples : List[Tuple[int, Array]]
        Observation specification for TransientMSEFunctional, by state
        index and stored-time index.
    observations : Array
        Flat observation vector. Shape: (nobs,)
    noise_std : float
        Noise standard deviation for the MSE functional.

    Raises
    ------
    TypeError
        If ``method`` does not satisfy its protocol.
    """

    def __init__(
        self,
        wrapper: BatchedBoundODEResidual[Array],
        method: SteppingMethodProtocol[Array, Array],
        times: Sequence[float],
        init_states: Array,
        obs_tuples: List[Tuple[int, Array]],
        observations: Array,
        noise_std: float = 1.0,
    ) -> None:
        if not isinstance(method, SteppingMethodProtocol):
            raise TypeError(
                "method must satisfy SteppingMethodProtocol, got "
                f"{type(method).__name__}"
            )
        self._wrapper = wrapper
        self._bkd = wrapper.bkd()
        self._steps = GivenSizes.from_times(times)
        # The steppers take states as single columns.
        self._init_states = self._bkd.reshape(init_states, (-1, 1))
        self._stepper: StepperProtocol[Array, Array] = method.stepper(
            wrapper.mass_form_system()
        )
        # The initial states are supplied, not parameters: the residual
        # states their derivative zero.
        self._sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            self._stepper,
            None,
            ResidualInitialState(wrapper, BackendColumnForm(self._bkd)),
        )

        functional = TransientMSEFunctional(
            nstates=self._init_states.shape[0],
            nresidual_params=wrapper.nparams(),
            obs_tuples=obs_tuples,
            bkd=self._bkd,
            noise_std=noise_std,
        )
        functional.set_observations(observations)
        self._functional = functional
        # Adjoint-based jacobian is unconditional.
        self._derivs: Derivatives[Array] = Derivatives.first_order(
            jacobian=self.jacobian
        )

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative bundle."""
        return self._derivs

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._wrapper.nparams()

    def nqoi(self) -> int:
        return 1

    def _solve(self, sample: Array) -> InMemoryTrajectoryStore[Array]:
        """The trajectory at parameters ``sample`` (shape (nparams, 1))."""
        self._wrapper.set_param(sample[:, 0])
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(self._stepper).solve(self._init_states, self._steps, store)
        return store

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
        store = self._solve(samples)
        value: Array = self._functional(trajectory_array(store, self._bkd), samples)
        return value

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
        store = self._solve(sample)
        functional = from_transient_functional(self._functional, store, sample)
        return self._bkd.transpose(self._sweeps.gradient(store, functional))

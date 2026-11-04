"""Trajectory matching loss: forward ODE solve + misfit with adjoint gradient.

Wraps a :class:`BatchedBoundODEResidual`, a stepping method bound to its
system, and the misfit of observed states
(:class:`~pyapprox.ode.observation_terms.MisfitTerm`), for use with
``DerivativeChecker`` and optimizers. The value is the misfit summed as
the trajectory is computed; the gradient is the discrete adjoint of the
method's steps
(:class:`~pyapprox.ode.trajectory_derivatives.TrajectoryDerivatives`).
"""

from typing import Generic, List, Sequence, Tuple

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.functional_sink import FunctionalSink
from pyapprox.ode.integration import Integrator
from pyapprox.ode.observation_terms import MisfitTerm
from pyapprox.ode.protocols import StepperProtocol
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.stepping_methods import SteppingMethodProtocol
from pyapprox.ode.trajectory_derivatives import (
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectorySinkProtocol,
)
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
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
        The observations, by state entry and the stored-time indices
        (non-negative) it is observed at (see
        :meth:`~pyapprox.ode.observation_terms.MisfitTerm.from_state_times`).
    observations : Array
        Flat observation vector, each pair's data in order. Shape: (nobs,)
    noise_std : float
        Noise standard deviation :math:`\\sigma` of the misfit
        :math:`\\sum \\|y - d\\|^2 / (2\\sigma^2)`.

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

        bkd = self._bkd
        observed = [
            (entry, [bkd.to_int(index) for index in indices])
            for entry, indices in obs_tuples
        ]
        misfit: MisfitTerm[Array, Array] = MisfitTerm.from_state_times(
            bkd, self._init_states.shape[0], observed, observations, noise_std
        )
        self._functional: TrajectoryFunctional[Array, Array] = (
            TrajectoryFunctional.from_terms(bkd, 1, state_terms=(misfit,))
        )
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

    def _solve(
        self, sample: Array, sink: TrajectorySinkProtocol[Array]
    ) -> None:
        """Solve at parameters ``sample`` (shape (nparams, 1)) into
        ``sink``."""
        self._wrapper.set_param(sample[:, 0])
        Integrator(self._stepper).solve(self._init_states, self._steps, sink)

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
        functional = self._functional
        sink: FunctionalSink[Array, Array] = FunctionalSink(
            functional.zero, functional.parameter_terms, functional.state_terms
        )
        self._solve(samples, sink)
        return sink.value()

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
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        self._solve(sample, store)
        return self._bkd.transpose(self._sweeps.gradient(store, self._functional))

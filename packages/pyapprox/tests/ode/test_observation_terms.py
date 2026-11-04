r"""The native observation, misfit, endpoint and Tikhonov terms.

Lotka-Volterra with a mass from (10, 5), parameters (1, 0.1, 0.075, 1.5),
ten steps of 0.05 by Heun.

- Conformance: the misfit of prey at stored states 2, 5, 10 and predator
  at 3, 10 (noise 0.5), the endpoint of both states (two entries, under
  weights (0.7, -1.3)), and the Tikhonov term with weight
  ``[[2, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 3]]`` and
  alpha 0.3 pass the term checker everywhere they read, first and second
  derivatives.
- Against the old functionals on the same trajectory: the misfit's value
  summed over the states it reads equals the old MSE's, and its state
  gradient at each stored state the old state Jacobian's column, to
  1e-13; the Tikhonov value is (alpha/2) p^T W p.
- On the mock stack the misfit of one entry, observed through the space's
  inner product, passes the checker for backward Euler, first and state
  second derivatives (the stepper has no parameters): the term reaches
  states only through its observation.
- Refusals: empty or out-of-range entries; an empty misfit, a negative
  index, data of the wrong shape, a non-operator, a non-positive noise;
  values of the wrong length; a non-square or asymmetric weight, an unset
  or non-column parameter; an endpoint of a non-operator.
"""

from typing import Any, Generic

import pytest
from numpy.typing import NDArray

from pyapprox.ode.functionals.mse import TransientMSEFunctional
from pyapprox.ode.integration import Integrator
from pyapprox.ode.observation_terms import (
    EndpointTerm,
    MisfitTerm,
    ObservationOperatorProtocol,
    StateEntries,
    TikhonovTerm,
)
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.tableau import backward_euler, heun
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.trajectory_terms_check import TrajectoryTermChecker
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators.protocols import InnerProductSpaceProtocol
from tests._helpers.linalg.mock_stack import MockVector
from tests._helpers.ode.runge_kutta_cases import (
    LV_PARAM,
    backend_sample,
    mock_sample,
    mock_stepper,
)
from tests._helpers.ode.trajectory_cases import TrajectoryProblem

_PREY, _PREDATOR = (2, 5, 10), (3, 10)
_DATA = [9.0, 8.0, 7.5, 6.0, 5.5]
_WEIGHT = [
    [2.0, 0.5, 0.0, 0.0],
    [0.5, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 3.0],
]


def _misfit(bkd: Backend[Array]) -> MisfitTerm[Array, Array]:
    return MisfitTerm.from_state_times(
        bkd, 2, [(0, _PREY), (1, _PREDATOR)], bkd.asarray(_DATA), noise_std=0.5
    )


class _Case(Generic[Array]):
    """A recorded trajectory and a checker setting the Tikhonov term's
    parameters (the other terms read none)."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self.bkd = bkd
        self.param = bkd.asarray(LV_PARAM)
        problem: TrajectoryProblem[Array] = TrajectoryProblem(bkd, heun())
        self.store = problem.solve(self.param)
        self.tikhonov = TikhonovTerm(bkd, 0.3, bkd.asarray(_WEIGHT))
        self.tikhonov.set_param(self.param)
        self.checker: TrajectoryTermChecker[Array, Array] = TrajectoryTermChecker(
            problem.stepper(), bkd, backend_sample(bkd), self.tikhonov.set_param
        )


class TestConformance:
    def test_misfit_and_tikhonov(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        functional = TrajectoryFunctional.from_terms(
            bkd, 1, parameter_terms=(case.tikhonov,), state_terms=(_misfit(bkd),)
        )
        report = case.checker.check(functional, case.store, param=case.param)
        assert report.failures() == []
        for index in (*_PREY, *_PREDATOR):
            assert f"state_terms[0]@{index}.state_state" in report.fd_checks
        assert "state_terms[0]@4.wrt_state" not in report.fd_checks
        assert "parameter_terms[0].param_param" in report.fd_checks

    def test_a_two_entry_endpoint(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        endpoint = EndpointTerm(StateEntries(bkd, 2, [0, 1]))
        functional = TrajectoryFunctional.from_terms(bkd, 2, state_terms=(endpoint,))
        report = case.checker.check(
            functional, case.store, bkd.asarray([[0.7], [-1.3]]), case.param
        )
        assert report.failures() == []
        last = case.store.ntimes() - 1
        assert set(report.fd_checks) == {
            f"state_terms[0]@{last}.{name}"
            for name in (
                "wrt_state",
                "wrt_param",
                "state_state",
                "state_param",
                "param_state",
                "param_param",
            )
        }
        bkd.assert_allclose(
            endpoint.value(last, case.store.time(last), case.store.state(last)),
            case.store.state(last),
            rtol=1e-15,
        )


class TestAgainstTheOldFunctionals:
    def test_misfit_value_and_state_gradient(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        store, misfit = case.store, _misfit(bkd)
        old = TransientMSEFunctional(
            2,
            4,
            [
                (0, bkd.asarray(list(_PREY), dtype=int)),
                (1, bkd.asarray(list(_PREDATOR), dtype=int)),
            ],
            bkd,
            noise_std=0.5,
        )
        old.set_observations(bkd.asarray(_DATA))
        trajectory = bkd.hstack([store.state(n) for n in range(store.ntimes())])
        ntimes = store.ntimes()
        wrt_state = misfit.derivatives().wrt_state
        assert wrt_state is not None
        total = bkd.zeros((1, 1))
        old_gradient = old.state_jacobian(trajectory, case.param)
        for index in range(ntimes):
            state, time = store.state(index), store.time(index)
            if not misfit.reads(index, index == ntimes - 1):
                bkd.assert_allclose(
                    old_gradient[:, index : index + 1], bkd.zeros((2, 1)), atol=0.0
                )
                continue
            total = total + misfit.value(index, time, state)
            gradient = wrt_state(index, time, state).apply_transpose(bkd.ones((1, 1)))
            bkd.assert_allclose(
                gradient, old_gradient[:, index : index + 1], rtol=1e-13
            )
        bkd.assert_allclose(total, old(trajectory, case.param), rtol=1e-13)

    def test_tikhonov_value(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd)
        weight = bkd.asarray(_WEIGHT)
        expected = 0.15 * bkd.dot(case.param.T, bkd.dot(weight, case.param))
        bkd.assert_allclose(case.tikhonov.value(), expected, rtol=1e-14)


_MockSpace = InnerProductSpaceProtocol[MockVector, NDArray[Any]]


class _MockEntry:
    r"""The observation :math:`e^\top y` through the space's inner
    product, for a fixed vector :math:`e`."""

    def __init__(self, space: _MockSpace, direction: MockVector) -> None:
        self._space, self._direction = space, direction

    def nobs(self) -> int:
        return 1

    def apply(self, state: MockVector) -> NDArray[Any]:
        return self._space.dot(self._direction, state)

    def apply_transpose(self, values: NDArray[Any]) -> MockVector:
        return self._space.scale(float(values[0, 0]), self._direction)


class TestOnAnOpaqueStack:
    def test_the_misfit_reaches_states_only_through_its_observation(self) -> None:
        bkd = NumpyBkd()
        stepper = mock_stepper(backward_euler())
        space = stepper.system().space
        store: InMemoryTrajectoryStore[MockVector] = InMemoryTrajectoryStore()
        Integrator(stepper).solve(
            mock_sample(space.descriptor(), 3), GivenSizes(0.0, [0.1] * 4), store
        )
        observation = _MockEntry(space, mock_sample(space.descriptor(), 5))
        assert isinstance(observation, ObservationOperatorProtocol)
        data = {1: bkd.asarray([[0.4]]), 4: bkd.asarray([[-0.2]])}
        misfit: MisfitTerm[MockVector, NDArray[Any]] = MisfitTerm(
            bkd, {index: (observation, datum) for index, datum in data.items()}
        )
        functional = TrajectoryFunctional.from_terms(bkd, 1, state_terms=(misfit,))
        checker: TrajectoryTermChecker[MockVector, NDArray[Any]] = (
            TrajectoryTermChecker(stepper, bkd, mock_sample)
        )
        report = checker.check(functional, store)
        assert report.failures() == []
        assert set(report.fd_checks) == {
            f"state_terms[0]@{index}.{name}"
            for index in (1, 4)
            for name in ("wrt_state", "state_state")
        }


class TestRefusals:
    def test_state_entries(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            StateEntries(bkd, 2, [])
        with pytest.raises(ValueError, match="not all in"):
            StateEntries(bkd, 2, [0, 2])

    def test_misfit(self, bkd: Backend[Array]) -> None:
        entries = StateEntries(bkd, 2, [0])
        datum = bkd.asarray([[1.0]])
        with pytest.raises(ValueError, match="at least one observed state"):
            MisfitTerm(bkd, {})
        with pytest.raises(ValueError, match="non-negative"):
            MisfitTerm(bkd, {-1: (entries, datum)})
        with pytest.raises(ValueError, match="must have shape"):
            MisfitTerm(bkd, {0: (entries, bkd.asarray([[1.0], [2.0]]))})
        with pytest.raises(TypeError, match="ObservationOperatorProtocol"):
            MisfitTerm(bkd, {0: (object(), datum)})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive"):
            MisfitTerm(bkd, {0: (entries, datum)}, noise_std=0.0)
        with pytest.raises(ValueError, match="one datum per observation|shape"):
            MisfitTerm.from_state_times(bkd, 2, [(0, (1, 2))], bkd.asarray([1.0]))

    def test_tikhonov(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="square"):
            TikhonovTerm(bkd, 1.0, bkd.ones((2, 3)))
        with pytest.raises(ValueError, match="symmetric"):
            TikhonovTerm(bkd, 1.0, bkd.asarray([[1.0, 2.0], [0.0, 1.0]]))
        term = TikhonovTerm(bkd, 1.0)
        with pytest.raises(ValueError, match="set_param"):
            term.value()
        with pytest.raises(ValueError, match="column"):
            term.set_param(bkd.ones((2,)))

    def test_endpoint(self) -> None:
        with pytest.raises(TypeError, match="ObservationOperatorProtocol"):
            EndpointTerm(object())  # type: ignore[arg-type]

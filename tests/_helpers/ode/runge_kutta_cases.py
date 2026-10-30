r"""Shared cases for testing the tableau stepper and its derivatives.

- ``TABLEAUX``: every library tableau with its order and structure.
- :class:`LinearMassModel`: :math:`M \dot y = K y` on the backend (3
  states, a nontrivial mass, a dense Jacobian), whose steps have closed
  forms.
- :func:`lotka_volterra_with_mass`: the Lotka-Volterra test model with a
  nontrivial mass and second derivatives.
- :func:`rk_stepper`: a stepper with a tight Newton when the tableau has
  implicit stages.
- :func:`lotka_volterra_step` and :func:`backend_checker`: a converged
  step with a random step adjoint, and the stepper checker for it;
  :func:`backend_sample` and :func:`mock_sample` give the checker its
  random vectors.
- :class:`MockLinearRhs` and :func:`mock_linear_system`: :math:`\dot y =
  K y` on the mock stack, with a residual taking stack vectors (the
  identity residual form).
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.mass_matrix import ConstantDenseMassMatrix
from pyapprox.ode.pinned_rows import RowPinningProtocol
from pyapprox.ode.residual_derivatives import (
    ODEResidualDerivatives,
    backend_state_space,
    dense_jacobian,
)
from pyapprox.ode.residual_form import IdentityForm
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.stepper_derivatives_check import StepperBundleChecker
from pyapprox.ode.tableau import (
    ButcherTableau,
    alexander_sdirk3,
    backward_euler,
    classical_rk4,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import (
    BlockVector,
    LinearizationPoint,
    LinearizedOperator,
    SpaceDescriptor,
)
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.linalg.mock_stack import (
    MockFormat,
    MockLU,
    MockOperator,
    MockVector,
    MockVectorSpace,
    wrap,
)
from tests._helpers.ode_models import LotkaVolterra

TABLEAUX: List[tuple[Callable[[], ButcherTableau], int, bool, bool, bool]] = [
    (forward_euler, 1, True, True, False),
    (backward_euler, 1, False, False, True),
    (crank_nicolson, 2, False, True, True),
    (implicit_midpoint, 2, False, False, False),
    (heun, 2, True, True, False),
    (classical_rk4, 4, True, True, False),
    (alexander_sdirk3, 3, False, False, True),
]
"""``(tableau, order, explicit, trivial first stage, stiffly accurate)``."""

TABLEAU_IDS = [make().name for make, *_ in TABLEAUX]

LINEAR_MASS = [[2.0, 0.5, 0.0], [0.5, 1.5, 0.2], [0.0, 0.2, 1.0]]
LINEAR_K = [[-1.0, 0.3, 0.0], [0.2, -0.8, 0.1], [0.0, 0.4, -1.2]]


class LinearMassModel(Generic[Array]):
    r""":math:`M \dot y = K y` on the backend, with a dense Jacobian and no
    parameters (``LINEAR_MASS``, ``LINEAR_K``)."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._k = bkd.asarray(LINEAR_K)
        self._mass = ConstantDenseMassMatrix(bkd.asarray(LINEAR_MASS), bkd)
        states = backend_state_space(bkd, 3)
        self._derivatives: ODEResidualDerivatives[Array, DenseFormat[Array]] = (
            ODEResidualDerivatives.state_only(
                dense_jacobian(self.jacobian, states, bkd)
            )
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, state: Array, time: float) -> Array:
        return self._bkd.dot(self._k, state)

    def jacobian(self, state: Array, time: float) -> Array:
        return self._k

    def mass_matrix(self) -> ConstantDenseMassMatrix[Array]:
        return self._mass

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return self._derivatives


def linear_step_map(tableau: ButcherTableau, deltat: float) -> NDArray[np.float64]:
    r"""The matrix :math:`\Phi` of one step of :math:`M \dot y = K y`,
    :math:`y_{n+1} = \Phi y_n`, from the tableau alone:
    :math:`\Phi = I + \Delta t (b^\top \otimes Z)(I - \Delta t A \otimes
    Z)^{-1}(\mathbf{1} \otimes I)` with :math:`Z = M^{-1} K`."""
    a, b = np.array(tableau.a), np.array(tableau.b)
    z = np.linalg.solve(np.array(LINEAR_MASS), np.array(LINEAR_K))
    n, s = z.shape[0], len(b)
    stages = np.linalg.solve(
        np.eye(s * n) - deltat * np.kron(a, z), np.kron(np.ones((s, 1)), np.eye(n))
    )
    phi: NDArray[np.float64] = np.eye(n) + deltat * np.kron(b, z) @ stages
    return phi


def lotka_volterra_with_mass(bkd: Backend[Array]) -> LotkaVolterra[Array]:
    """Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]``."""
    return LotkaVolterra(bkd, mass=bkd.asarray([[2.0, 0.5], [0.5, 1.0]]))


def rk_stepper(
    tableau: ButcherTableau,
    system: MassFormSystem[Array, Array, DenseFormat[Array]],
    row_pinning: Optional[RowPinningProtocol[Array]] = None,
) -> RungeKuttaStepper[Array, Array, DenseFormat[Array]]:
    """A backend stepper, with Newton (atol 1e-13) for implicit stages and
    the given row pinning."""
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
    return RungeKuttaStepper(
        tableau, system, None if tableau.is_explicit() else newton, row_pinning
    )


def new_state(
    stepper: RungeKuttaStepper[Array, Array, DenseFormat[Array]],
    unknown: BlockVector[Array],
) -> Array:
    """:math:`y_{n+1}` read out of a step's unknown."""
    return stepper.readout().apply(unknown).blocks[0]


MOCK_STATE = SpaceDescriptor("state", 3, "float64")


class MockLinearRhs:
    r""":math:`\dot y = K y` (``LINEAR_K``) taking the mock stack's vectors
    directly, with a Jacobian and no parameters."""

    def __init__(self) -> None:
        self._k = MockOperator.from_dense(np.array(LINEAR_K), MOCK_STATE)
        self._derivatives: ODEResidualDerivatives[MockVector, MockFormat] = (
            ODEResidualDerivatives.state_only(self._jacobian)
        )

    def __call__(self, state: MockVector, time: float) -> MockVector:
        return self._k.apply(state)

    def _jacobian(
        self, state: MockVector, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[MockVector, MockFormat]:
        return LinearizedOperator(self._k, point)

    def derivatives(self) -> ODEResidualDerivatives[MockVector, MockFormat]:
        return self._derivatives


def mock_linear_system() -> MassFormSystem[MockVector, NDArray[Any], MockFormat]:
    """:class:`MockLinearRhs` with the identity mass, dense LU and the
    identity residual form."""
    rhs = MockLinearRhs()
    jacobian = rhs.derivatives().jacobian
    if jacobian is None:
        raise AssertionError("MockLinearRhs declares its Jacobian")
    return MassFormSystem(
        rhs=rhs,
        mass=MockOperator.identity(MOCK_STATE),
        jacobian=jacobian,
        space=MockVectorSpace(MOCK_STATE),
        solver=MockLU(),
        form=IdentityForm(),
    )


def backend_sample(bkd: Backend[Array]) -> Callable[[SpaceDescriptor, int], Array]:
    """Random single columns of backend spaces, seeded, for the checker."""

    def sample(descriptor: SpaceDescriptor, seed: int) -> Array:
        rng = np.random.default_rng(seed)
        return bkd.asarray(rng.normal(size=(descriptor.size, 1)))

    return sample


def mock_sample(descriptor: SpaceDescriptor, seed: int) -> MockVector:
    """A random single column of a mock-stack space, seeded."""
    rng = np.random.default_rng(seed)
    return wrap(rng.normal(size=(descriptor.size, 1)), descriptor)


LV_PARAM = [[1.0], [0.1], [0.075], [1.5]]
"""Lotka-Volterra's parameters (a, b, c, d) as a column."""


@dataclass(frozen=True)
class StepCase(Generic[Array]):
    """A converged Lotka-Volterra step from (10, 5) at t = 0.2, dt = 0.05,
    with a random step adjoint (seed 99) and the parameters as a column."""

    model: LotkaVolterra[Array]
    stepper: RungeKuttaStepper[Array, Array, DenseFormat[Array]]
    window: StepWindow[Array]
    adj: BlockVector[Array]
    param: Array


def lotka_volterra_step(
    bkd: Backend[Array], tableau: ButcherTableau
) -> StepCase[Array]:
    """The :class:`StepCase` of ``tableau`` on Lotka-Volterra with a mass."""
    model = lotka_volterra_with_mass(bkd)
    param = bkd.asarray(LV_PARAM)
    model.set_param(param)
    stepper = rk_stepper(tableau, dense_mass_form_system(model))
    state = bkd.asarray([[10.0], [5.0]])
    unknown = stepper.step((state,), (0.25, 0.2))
    window = StepWindow((0.25, 0.2), (state,), unknown, model.parameters_version())
    residual = stepper.residual_space()
    sample = backend_sample(bkd)
    adj = BlockVector(
        tuple(
            sample(residual.component(i).descriptor(), 99 + i)
            for i in range(residual.nblocks())
        )
    )
    return StepCase(model, stepper, window, adj, param)


def backend_checker(
    case: StepCase[Array], bkd: Backend[Array]
) -> StepperBundleChecker[Array, Array]:
    """The stepper checker for ``case``, its parameters set on the model."""
    return StepperBundleChecker(
        case.stepper, backend_sample(bkd), set_param=case.model.set_param
    )


def mock_stepper(
    tableau: ButcherTableau,
) -> RungeKuttaStepper[MockVector, NDArray[Any], MockFormat]:
    """A mock-stack stepper for :func:`mock_linear_system`."""
    newton: Newton[MockVector, NDArray[Any]] = Newton(atol=1e-13, rtol=0.0)
    return RungeKuttaStepper(
        tableau, mock_linear_system(), None if tableau.is_explicit() else newton
    )

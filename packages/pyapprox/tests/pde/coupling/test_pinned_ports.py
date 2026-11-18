r"""Port rows pinned to the port data, the data as parameters.

- The placement :math:`B` (state rows (5, 2) of a 6-row space, in that
  port order) satisfies the dot identity and places a port block on its
  rows, on dense NumPy and Torch and the opaque mock stack.
- On the mock P1 operator of ``tests._helpers.mock_elementwise`` (8 cells
  of [0, 1], the :math:`u^2` reaction, u(0) = 1 essential, node 8 the
  port), pinned with :func:`pin_ports`:

  - the essential and port rows read :math:`u - g`, the others equal the
    essentially pinned residual, at u = linspace(0.5, 1.5);
  - the parameter Jacobian in :math:`\lambda` passes central finite
    differences (smallest relative error below 1e-6), and setting the
    port data changes the parameter version;
  - solved by Newton (atol 1e-12, dense LU, from zero) at
    :math:`\lambda = 1.7`, the state equals the single-domain solve with
    u(1) = 1.7 essential, to 1e-12: an independent reference built with
    no port code.
- Refusals: a port row that is an essential row, a residual with its own
  parameters, mismatched row counts, and port rows of the wrong type.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, Union

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.boundary import DirichletConstraintSet
from pyapprox.pde.coupling.pinned_ports import (
    PortParameterResidual,
    PortPinnedData,
    PortPlacement,
    pin_ports,
)
from pyapprox.pde.coupling.ports import BackendPortRows, PortRowsProtocol
from pyapprox.pde.galerkin.system import GalerkinSteadySystem
from pyapprox.pde.galerkin.time_integration.dense_discretization import (
    DenseSteadyDiscretization,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.linalg.operators import LinearizationPoint, SpaceDescriptor
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling import MockPortRows
from tests._helpers.linalg.mock_stack import MockFormat, MockVector
from tests._helpers.linalg.stacks import (
    A,
    F,
    TypedStackCase,
    V,
    dense_case,
    mock_case,
)
from tests._helpers.mock_elementwise import (
    MockP1Operator,
    ModalCoefficientParameterization,
)
from tests._helpers.rom_problem import FixedValue

_NCELLS = 8
_PORT = _NCELLS
_LAMBDA = 1.7
_POINT = LinearizationPoint(0, ("test",))


@dataclass
class _Stack(Generic[V, F, A]):
    case: TypedStackCase[V, F, A]
    ports: Callable[[SpaceDescriptor, Sequence[int]], PortRowsProtocol[V, A]]


def _dense_stack(bkd: Backend[Array]) -> _Stack[Array, DenseFormat[Array], Array]:
    def ports(
        descriptor: SpaceDescriptor, dofs: Sequence[int]
    ) -> BackendPortRows[Array]:
        return BackendPortRows(descriptor, dofs, bkd)

    return _Stack(dense_case(bkd), ports)


_AnyStack = Union[
    _Stack[MockVector, MockFormat, NDArray[Any]],
    _Stack[ArrayProtocol, DenseFormat[ArrayProtocol], ArrayProtocol],
]


@pytest.fixture(params=["dense-numpy", "dense-torch", "mock"])
def stack(request: pytest.FixtureRequest) -> _AnyStack:
    if request.param == "mock":
        return _Stack(mock_case(), MockPortRows)
    fixture = "numpy_bkd" if request.param == "dense-numpy" else "torch_bkd"
    return _dense_stack(request.getfixturevalue(fixture))


def test_placement_on_every_stack(stack: _Stack[V, F, A]) -> None:
    case = stack.case
    states_d, params_d = SpaceDescriptor("states", 6), SpaceDescriptor("ports", 2)
    states, params = case.space(states_d), case.space(params_d)
    placement = PortPlacement(
        stack.ports(states_d, [5, 2]), stack.ports(params_d, [0, 1]), states, params
    )
    case.assert_dot_identity(placement, seed=3)
    lam = case.vec(params_d, 1, seed=4)
    expected = np.zeros((6, 2))
    expected[5, 0] = expected[2, 1] = 1.0
    case.assert_vector(placement.apply(lam), case.times(case.dense(expected), lam))


class _Subdomain(Generic[Array]):
    """The P1 problem with u(0) = 1 essential and node 8 the port."""

    def __init__(self, bkd: Backend[Array], port_essential: bool = False) -> None:
        self.bkd = bkd
        self.n = _NCELLS + 1
        self.operator: MockP1Operator[Array] = MockP1Operator(_NCELLS, True, bkd)
        fixed = [FixedValue(0, 1.0, bkd)]
        if port_essential:
            fixed.append(FixedValue(_PORT, _LAMBDA, bkd))
        system = GalerkinSteadySystem(
            self.operator, DirichletConstraintSet(fixed, self.n, bkd)
        )
        self.discretization: DenseSteadyDiscretization[Array] = (
            DenseSteadyDiscretization(system)
        )
        self.pinned = galerkin_steady_residual(self.discretization)

    def ports(self, dofs: Sequence[int] = (_PORT,)) -> PortPinnedData[Array, Array]:
        disc, bkd = self.discretization, self.bkd
        states = disc.space()
        params = disc.parameter_space(len(dofs))
        return PortPinnedData(
            BackendRowSet(states.descriptor(), dofs, bkd),
            BackendPortRows(states.descriptor(), dofs, bkd),
            BackendPortRows(params.descriptor(), range(len(dofs)), bkd),
            states,
            params,
        )

    def solve(self, residual: PinnedResidual[Array, DenseFormat[Array]]) -> Array:
        system = SteadySystem(
            residual,
            self.discretization.space(),
            RecomputeAlways(BackendDenseLU(self.bkd)),
        )
        newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=30)
        point = system.solve(newton, self.bkd.zeros((self.n, 1)))
        assert point.converged()
        return point.state()


class TestTheGalerkinSubdomain:
    def test_rows_and_version(self, bkd: Backend[Array]) -> None:
        sub = _Subdomain(bkd)
        ports = sub.ports()
        residual = pin_ports(sub.pinned, ports)
        version = residual.parameters_version()
        ports.set_port_data(bkd.asarray([[_LAMBDA]]))
        assert residual.parameters_version() == version + 1
        state = bkd.linspace(0.5, 1.5, sub.n)[:, None]
        value, below = residual(state), sub.pinned(state)
        bkd.assert_allclose(value[1:-1], below[1:-1], atol=1e-14)
        bkd.assert_allclose(value[0, 0], state[0, 0] - 1.0, atol=1e-14)
        bkd.assert_allclose(value[-1, 0], state[-1, 0] - _LAMBDA, atol=1e-14)

    def test_port_data_jacobian(self, bkd: Backend[Array]) -> None:
        sub = _Subdomain(bkd)
        ports = sub.ports()
        residual = pin_ports(sub.pinned, ports)
        param = residual.derivatives().param
        assert param is not None
        state = bkd.linspace(0.5, 1.5, sub.n)[:, None]

        def value(lam: Array) -> Array:
            ports.set_port_data(lam)
            return residual(state)

        def jacobian(lam: Array) -> Array:
            ports.set_port_data(lam)
            return param(state).apply(bkd.eye(1))

        errors = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=sub.n, nvars=1, fun=value, jacobian=jacobian, bkd=bkd
            )
        ).check_derivatives(bkd.asarray([[_LAMBDA]]), relative=True, central=True)
        assert bkd.to_float(bkd.min(errors[0])) < 1e-6

    def test_nested_jacobian_is_the_union_pin(self, bkd: Backend[Array]) -> None:
        r"""The nested pin's state Jacobian equals that of one pin of the
        essential and port rows together, built with no port code."""
        sub = _Subdomain(bkd)
        nested = pin_ports(sub.pinned, sub.ports()).derivatives().jacobian
        union = _Subdomain(bkd, port_essential=True).pinned.derivatives().jacobian
        assert nested is not None and union is not None
        state = bkd.linspace(0.5, 1.5, sub.n)[:, None]
        eye = bkd.eye(sub.n)
        expected = union(state, _POINT).apply(eye)
        bkd.assert_allclose(nested(state, _POINT).apply(eye), expected, atol=1e-14)

    def test_solve_equals_the_essential_problem(self, bkd: Backend[Array]) -> None:
        sub = _Subdomain(bkd)
        ports = sub.ports()
        ports.set_port_data(bkd.asarray([[_LAMBDA]]))
        state = sub.solve(pin_ports(sub.pinned, ports))
        reference = _Subdomain(bkd, port_essential=True)
        bkd.assert_allclose(state, reference.solve(reference.pinned), atol=1e-12)


class TestRefusals:
    def test_port_on_an_essential_row(self, bkd: Backend[Array]) -> None:
        sub = _Subdomain(bkd)
        with pytest.raises(ValueError, match="1 port rows are also essential"):
            pin_ports(sub.pinned, sub.ports((0,)))
        # A partial overlap: one of two port rows is essential.
        with pytest.raises(ValueError, match="1 port rows are also essential"):
            pin_ports(sub.pinned, sub.ports((_PORT, 0)))

    def test_a_residual_that_is_not_pinned(self, bkd: Backend[Array]) -> None:
        sub = _Subdomain(bkd)
        with pytest.raises(TypeError, match="PinnedResidual"):
            PortParameterResidual(
                sub.pinned.unpinned(), sub.ports()  # type: ignore[arg-type]
            )

    def test_residual_with_parameters(self, bkd: Backend[Array]) -> None:
        sub = _Subdomain(bkd)
        parameters = ParameterState(
            ModalCoefficientParameterization(sub.operator, bkd)
        )
        pinned = galerkin_steady_residual(sub.discretization, parameters)
        with pytest.raises(ValueError, match="parameters of its own"):
            pin_ports(pinned, sub.ports())

    def test_mismatched_counts_and_types(self, bkd: Backend[Array]) -> None:
        sub = _Subdomain(bkd)
        states = sub.discretization.space()
        params = sub.discretization.parameter_space(2)
        descriptor = states.descriptor()
        with pytest.raises(ValueError, match="one row count"):
            PortPinnedData[Array, Array](
                BackendRowSet(descriptor, [_PORT], bkd),
                BackendPortRows(descriptor, [_PORT], bkd),
                BackendPortRows(params.descriptor(), [0, 1], bkd),
                states,
                params,
            )
        with pytest.raises(TypeError, match="PortRowsProtocol"):
            PortPinnedData[Array, Array](
                BackendRowSet(descriptor, [_PORT], bkd),
                object(),  # type: ignore[arg-type]
                BackendPortRows(params.descriptor(), [0], bkd),
                states,
                params,
            )

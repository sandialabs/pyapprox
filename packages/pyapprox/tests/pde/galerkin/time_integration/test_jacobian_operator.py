"""GalerkinJacobianOperator with injected assemblers, on several stacks.

The operator is compared with the residual adapter's matrix Jacobian
(essential rows zeroed) for a linear scalar physics, a nonlinear one and
a mixed one; for the nonlinear one its action is also checked against a
central difference of the residual, which shares no code with either
Jacobian. The skfem assembler runs on the SciPy and mock stacks and is
refused, statically and at runtime, by a stack without whole-matrix
adds; an assembler that adds element blocks only runs on that stack, so
the operator itself is shown not to depend on the capability.
"""

from typing import Any, Callable, Generic, TypeVar

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix

from pyapprox.ode.protocols import JacobianOperatorProtocol
from pyapprox.pde.assembly import (
    ElementDofMapProtocol,
    JacobianAssemblerProtocol,
    ScipyAssemblyStack,
)
from pyapprox.pde.assembly.protocols import (
    AssemblyStackProtocol,
    MatrixSinkProtocol,
)
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import skfem_ode_residual
from pyapprox.pde.galerkin.time_integration.jacobian_operator import (
    GalerkinJacobianOperator,
    essential_rows,
)
from pyapprox.pde.galerkin.time_integration.skfem_jacobian_assembler import (
    SkfemJacobianAssembler,
    require_sparse,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    LinearizationPoint,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import CSRFormat
from tests._helpers.galerkin_systems import galerkin_system
from tests._helpers.linalg.mock_stack import (
    MockAssemblyStack,
    MockFormat,
    MockVector,
    peek,
    wrap,
)
from tests._helpers.linalg.no_bulk_stack import NoBulkAssemblyStack

_Np = NDArray[Any]
_Dofs = NDArray[np.int64]
_Values = NDArray[np.float64]
_System = GalerkinSystem[_Np]
V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)


class _ElementBlockAssembler(Generic[V, F]):
    """Adds element blocks only, so it runs on any stack.

    A test stand-in for an element-wise assembler: its blocks are the
    nonzeros of skfem's matrix, one 1x1 block each.
    """

    def __init__(
        self,
        system: _System,
        stack: AssemblyStackProtocol[V, F, _Dofs, _Values],
        dof_map: ElementDofMapProtocol[_Dofs],
        descriptor: SpaceDescriptor,
    ) -> None:
        dofs = dof_map.element_dofs()
        self._sink: MatrixSinkProtocol[V, F, _Dofs, _Values] = stack.matrix_sink(
            descriptor, descriptor, dofs, dofs
        )
        self._spatial_operator = system.spatial_operator()

    def assemble_jacobian(
        self, state: _Np, time: float
    ) -> ActionOperatorProtocol[V, F]:
        coo = require_sparse(
            self._spatial_operator.spatial_jacobian(state, time)
        ).tocoo()
        rows: _Dofs = np.asarray(coo.row, dtype=np.int64)[:, None]
        cols: _Dofs = np.asarray(coo.col, dtype=np.int64)[:, None]
        values: _Values = np.asarray(coo.data, dtype=np.float64)[:, None, None]
        self._sink.zero()
        self._sink.add_element_matrices(rows, cols, values)
        return self._sink.assemble()


def _state(system: _System, seed: int) -> _Np:
    return np.random.default_rng(seed).normal(size=system.nstates())


def _reference(system: _System, state: _Np) -> _Np:
    """The adapter's Jacobian, essential rows zeroed, as a dense array."""
    jacobian = skfem_ode_residual(system).jacobian(state, 0.0)
    dense: _Np = require_sparse(jacobian).toarray()
    return dense


def _identity(block: _Np, descriptor: SpaceDescriptor) -> _Np:
    return block


def _same(vec: _Np) -> _Np:
    return vec


def _csr_dense(fmt: CSRFormat) -> _Np:
    dense: _Np = fmt.matrix().toarray()
    return dense


def _mock_dense(fmt: MockFormat) -> _Np:
    return fmt.dense()


def _check_against_reference(
    assembler: JacobianAssemblerProtocol[_Np, V, F],
    stack: AssemblyStackProtocol[V, F, _Dofs, _Values],
    system: _System,
    descriptor: SpaceDescriptor,
    to_vec: Callable[[_Np, SpaceDescriptor], V],
    from_vec: Callable[[V], _Np],
    format_dense: Callable[[F], _Np],
    format_type: type[F],
    bkd: NumpyBkd,
) -> None:
    """Action, transpose, stored format, label and spaces against the
    adapter's matrix Jacobian."""
    producer = GalerkinJacobianOperator(
        assembler, essential_rows(system, stack, descriptor)
    )
    state = _state(system, 0)
    point = LinearizationPoint(0, ("step", 0))
    op = producer.jacobian_operator(state, 0.0, point)
    reference = _reference(system, state)
    x = np.random.default_rng(1).normal(size=(system.nstates(), 2))
    bkd.assert_allclose(from_vec(op.apply(to_vec(x, descriptor))), reference @ x)
    bkd.assert_allclose(
        from_vec(op.apply_transpose(to_vec(x, descriptor))), reference.T @ x
    )
    essential = bkd.to_numpy(system.constraint_set().dofs())
    assert essential.size > 0
    bkd.assert_allclose(
        reference[essential], np.zeros((essential.size, system.nstates()))
    )
    assert op.linearization() == point
    assert (op.domain(), op.range()) == (descriptor, descriptor)
    fmt = op.format()
    assert isinstance(fmt, format_type)
    bkd.assert_allclose(format_dense(fmt), reference)


@pytest.mark.parametrize("physics", ["adr", "burgers", "stokes"])
class TestMatchesTheMatrixJacobian:
    def test_skfem_assembler_on_the_scipy_stack(
        self, numpy_bkd: NumpyBkd, physics: str
    ) -> None:
        system, basis, descriptor = galerkin_system(physics, numpy_bkd)
        stack = ScipyAssemblyStack(np.float64)
        _check_against_reference(
            SkfemJacobianAssembler(system, stack, basis, descriptor),
            stack,
            system,
            descriptor,
            _identity,
            _same,
            _csr_dense,
            CSRFormat,
            numpy_bkd,
        )

    def test_skfem_assembler_on_the_mock_stack(
        self, numpy_bkd: NumpyBkd, physics: str
    ) -> None:
        system, basis, descriptor = galerkin_system(physics, numpy_bkd)
        stack = MockAssemblyStack()
        _check_against_reference(
            SkfemJacobianAssembler(system, stack, basis, descriptor),
            stack,
            system,
            descriptor,
            wrap,
            peek,
            _mock_dense,
            MockFormat,
            numpy_bkd,
        )

    def test_element_block_assembler_on_the_no_bulk_stack(
        self, numpy_bkd: NumpyBkd, physics: str
    ) -> None:
        system, basis, descriptor = galerkin_system(physics, numpy_bkd)
        stack = NoBulkAssemblyStack(np.float64)
        _check_against_reference(
            _ElementBlockAssembler(system, stack, basis, descriptor),
            stack,
            system,
            descriptor,
            _identity,
            _same,
            _csr_dense,
            CSRFormat,
            numpy_bkd,
        )


def _burgers_on_the_mock(
    bkd: NumpyBkd,
) -> tuple[_System, GalerkinJacobianOperator[_Np, MockVector, MockFormat]]:
    system, basis, descriptor = galerkin_system("burgers", bkd)
    stack = MockAssemblyStack()
    return system, GalerkinJacobianOperator(
        SkfemJacobianAssembler(system, stack, basis, descriptor),
        essential_rows(system, stack, descriptor),
    )


class TestNonlinear:
    def test_satisfies_the_ode_protocol(self, numpy_bkd: NumpyBkd) -> None:
        """What a time stepper consumes, statically and at runtime."""
        _, producer = _burgers_on_the_mock(numpy_bkd)
        consumed: JacobianOperatorProtocol[_Np, MockVector, MockFormat] = producer
        assert isinstance(consumed, JacobianOperatorProtocol)

    def test_action_is_the_residual_derivative(self, numpy_bkd: NumpyBkd) -> None:
        """Central difference of the residual: essential rows of the
        residual carry no state, so they differentiate to zero too."""
        system, producer = _burgers_on_the_mock(numpy_bkd)
        residual = skfem_ode_residual(system)
        state, direction = _state(system, 2), _state(system, 3)
        step = 1e-6
        difference = (
            residual(state + step * direction, 0.0)
            - residual(state - step * direction, 0.0)
        ) / (2 * step)
        descriptor = SpaceDescriptor("state", system.nstates())
        op = producer.jacobian_operator(state, 0.0, LinearizationPoint(0))
        action = peek(op.apply(wrap(direction[:, None], descriptor)))[:, 0]
        numpy_bkd.assert_allclose(action, difference, rtol=1e-7, atol=1e-8)

    def test_results_are_independent_of_later_calls(self, numpy_bkd: NumpyBkd) -> None:
        system, producer = _burgers_on_the_mock(numpy_bkd)
        first_state, second_state = _state(system, 4), 3.0 * _state(system, 5)
        first = producer.jacobian_operator(first_state, 0.0, LinearizationPoint(0))
        before = first.format().dense().copy()
        producer.jacobian_operator(second_state, 0.0, LinearizationPoint(1))
        numpy_bkd.assert_allclose(first.format().dense(), before)
        numpy_bkd.assert_allclose(before, _reference(system, first_state))
        assert not np.allclose(before, _reference(system, second_state))


class TestPairing:
    def test_skfem_assembler_refuses_a_stack_without_bulk_adds(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A type error, and refused at construction, before assembly."""
        system, basis, descriptor = galerkin_system("adr", numpy_bkd)
        with pytest.raises(TypeError, match="whole assembled matrices"):
            SkfemJacobianAssembler(
                system,
                NoBulkAssemblyStack(np.float64),  # type: ignore[arg-type]
                basis,
                descriptor,
            )


class TestErrors:
    def test_refuses_a_descriptor_of_another_size(self, numpy_bkd: NumpyBkd) -> None:
        system, basis, descriptor = galerkin_system("adr", numpy_bkd)
        wrong = SpaceDescriptor("state", system.nstates() + 1)
        stack = MockAssemblyStack()
        with pytest.raises(ValueError, match="size"):
            SkfemJacobianAssembler(system, stack, basis, wrong)
        with pytest.raises(ValueError, match="size"):
            essential_rows(system, stack, wrong)

    def test_refuses_objects_without_their_protocols(self, numpy_bkd: NumpyBkd) -> None:
        system, basis, descriptor = galerkin_system("adr", numpy_bkd)
        stack = MockAssemblyStack()
        with pytest.raises(TypeError, match="system"):
            SkfemJacobianAssembler(basis, stack, basis, descriptor)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="stack"):
            SkfemJacobianAssembler(system, basis, basis, descriptor)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="dof_map"):
            SkfemJacobianAssembler(system, stack, system, descriptor)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="stack"):
            essential_rows(system, basis, descriptor)  # type: ignore[arg-type]
        rows = essential_rows(system, stack, descriptor)
        with pytest.raises(TypeError, match="assembler"):
            GalerkinJacobianOperator(system, rows)  # type: ignore[arg-type]
        assembler = SkfemJacobianAssembler(system, stack, basis, descriptor)
        with pytest.raises(TypeError, match="rows"):
            GalerkinJacobianOperator(assembler, basis)  # type: ignore[arg-type]

    def test_require_sparse_refuses_a_dense_jacobian(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(TypeError, match="sparse Jacobian"):
            require_sparse(numpy_bkd.eye(3))
        assert isinstance(require_sparse(csr_matrix(np.eye(3))), spmatrix)

"""SkfemMassAssembler, consistent and lumped, on the SciPy and mock stacks.

The assembled mass is compared with the residual adapter's mass before
its essential rows are replaced, and, with the rows replaced by
``RowReplacedOperator``, with the adapter's constrained mass, for a
scalar linear physics, a scalar nonlinear one and a mixed one (whose
mass is singular). The row-replaced mass keeps the CSR format, so it
pairs with sparse LU.
"""

from typing import Any, Callable, TypeVar

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.pde.assembly import MassAssemblerProtocol, ScipyAssemblyStack
from pyapprox.pde.assembly.protocols import AssemblyStackProtocol
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.galerkin.time_integration.constrained_mass import (
    ConstrainedMassMatrixProtocol,
)
from pyapprox.pde.galerkin.time_integration.jacobian_operator import essential_rows
from pyapprox.pde.galerkin.time_integration.skfem_jacobian_assembler import (
    require_sparse,
)
from pyapprox.pde.galerkin.time_integration.skfem_mass_assembler import (
    SkfemMassAssembler,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    RowReplacedOperator,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from tests._helpers.galerkin_systems import galerkin_system
from tests._helpers.linalg.mock_stack import MockAssemblyStack, MockFormat
from tests._helpers.linalg.no_bulk_stack import NoBulkAssemblyStack

_Np = NDArray[Any]
_Dofs = NDArray[np.int64]
_Values = NDArray[np.float64]
_System = GalerkinSystem[_Np]
V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)


def _dense(matrix: _Np) -> _Np:
    """A mass matrix the adapter holds, sparse or dense, as an array."""
    dense: _Np = csr_matrix(matrix).toarray()
    return dense


def _adapter_masses(system: _System, lumped: bool) -> tuple[_Np, _Np]:
    """The adapter's (unconstrained, constrained) mass, as arrays."""
    mass = GalerkinPhysicsToODEResidualAdapter(system, lumped_mass=lumped).mass_matrix()
    assert isinstance(mass, ConstrainedMassMatrixProtocol)
    return _dense(mass.unconstrained().as_matrix()), _dense(mass.as_matrix())


def _csr_dense(fmt: CSRFormat) -> _Np:
    dense: _Np = fmt.matrix().toarray()
    return dense


def _mock_dense(fmt: MockFormat) -> _Np:
    return fmt.dense()


def _check_against_adapter(
    assembler: MassAssemblerProtocol[V, F],
    stack: AssemblyStackProtocol[V, F, _Dofs, _Values],
    system: _System,
    descriptor: SpaceDescriptor,
    lumped: bool,
    format_dense: Callable[[F], _Np],
    format_type: type[F],
    bkd: NumpyBkd,
) -> None:
    """Stored mass, its declared properties, and the row-replaced mass."""
    unconstrained, constrained = _adapter_masses(system, lumped)
    mass = assembler.assemble_mass()
    assert (mass.domain(), mass.range()) == (descriptor, descriptor)
    fmt = mass.format()
    assert isinstance(fmt, format_type)
    bkd.assert_allclose(format_dense(fmt), unconstrained, atol=1e-15)
    assert mass.properties().symmetric
    assert mass.properties().diagonal == lumped
    assert not mass.properties().spd
    replaced = RowReplacedOperator(mass, essential_rows(system, stack, descriptor))
    replaced_fmt = replaced.format()
    assert isinstance(replaced_fmt, format_type)
    bkd.assert_allclose(format_dense(replaced_fmt), constrained, atol=1e-15)
    essential = bkd.to_numpy(system.constraint_set().dofs())
    assert essential.size > 0
    bkd.assert_allclose(
        format_dense(replaced_fmt)[essential], np.eye(system.nstates())[essential]
    )
    assert replaced.properties().diagonal == lumped


@pytest.mark.parametrize("lumped", [False, True])
@pytest.mark.parametrize("physics", ["adr", "burgers", "stokes"])
class TestMatchesTheAdaptersMass:
    def test_on_the_scipy_stack(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        system, basis, descriptor = galerkin_system(physics, numpy_bkd)
        stack = ScipyAssemblyStack(np.float64)
        _check_against_adapter(
            SkfemMassAssembler(system, stack, basis, descriptor, lumped=lumped),
            stack,
            system,
            descriptor,
            lumped,
            _csr_dense,
            CSRFormat,
            numpy_bkd,
        )

    def test_on_the_mock_stack(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        system, basis, descriptor = galerkin_system(physics, numpy_bkd)
        stack = MockAssemblyStack()
        _check_against_adapter(
            SkfemMassAssembler(system, stack, basis, descriptor, lumped=lumped),
            stack,
            system,
            descriptor,
            lumped,
            _mock_dense,
            MockFormat,
            numpy_bkd,
        )


@pytest.mark.parametrize("lumped", [False, True])
@pytest.mark.parametrize("physics", ["adr", "burgers"])
def test_row_replaced_mass_pairs_with_sparse_lu(
    numpy_bkd: NumpyBkd, physics: str, lumped: bool
) -> None:
    """Nonsingular once its essential rows are replaced (the mixed mass
    stays singular: its pressure block is zero), so it solves directly."""
    system, basis, descriptor = galerkin_system(physics, numpy_bkd)
    stack = ScipyAssemblyStack(np.float64)
    mass = SkfemMassAssembler(
        system, stack, basis, descriptor, lumped=lumped
    ).assemble_mass()
    replaced = RowReplacedOperator(mass, essential_rows(system, stack, descriptor))
    rhs = np.random.default_rng(7).normal(size=(system.nstates(), 2))
    solution, _ = ScipySparseLU().setup(replaced).solve(rhs)
    _, constrained = _adapter_masses(system, lumped)
    numpy_bkd.assert_allclose(solution, np.linalg.solve(constrained, rhs), rtol=1e-10)


def test_results_are_independent_of_later_calls(numpy_bkd: NumpyBkd) -> None:
    system, basis, descriptor = galerkin_system("burgers", numpy_bkd)
    stack = MockAssemblyStack()
    assembler = SkfemMassAssembler(system, stack, basis, descriptor)
    first = assembler.assemble_mass()
    before = first.format().dense().copy()
    assembler.assemble_mass()
    numpy_bkd.assert_allclose(first.format().dense(), before)


class TestErrors:
    def test_refuses_a_stack_without_bulk_adds(self, numpy_bkd: NumpyBkd) -> None:
        """A type error, and refused at construction, before assembly."""
        system, basis, descriptor = galerkin_system("adr", numpy_bkd)
        with pytest.raises(TypeError, match="whole assembled matrices"):
            SkfemMassAssembler(
                system,
                NoBulkAssemblyStack(np.float64),  # type: ignore[arg-type]
                basis,
                descriptor,
            )

    def test_refuses_a_descriptor_of_another_size(self, numpy_bkd: NumpyBkd) -> None:
        system, basis, descriptor = galerkin_system("adr", numpy_bkd)
        wrong = SpaceDescriptor("state", system.nstates() + 1)
        with pytest.raises(ValueError, match="size"):
            SkfemMassAssembler(system, MockAssemblyStack(), basis, wrong)

    def test_refuses_objects_without_their_protocols(self, numpy_bkd: NumpyBkd) -> None:
        system, basis, descriptor = galerkin_system("adr", numpy_bkd)
        stack = MockAssemblyStack()
        with pytest.raises(TypeError, match="system"):
            SkfemMassAssembler(basis, stack, basis, descriptor)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="stack"):
            SkfemMassAssembler(system, basis, basis, descriptor)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="dof_map"):
            SkfemMassAssembler(system, stack, system, descriptor)  # type: ignore[arg-type]

    def test_require_sparse_names_the_mass(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(TypeError, match="sparse mass"):
            require_sparse(numpy_bkd.eye(3), "mass")

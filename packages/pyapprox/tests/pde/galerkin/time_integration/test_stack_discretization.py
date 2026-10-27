"""GalerkinStackDiscretization and the adapter's stack system built on it.

The discretization applies no boundary condition: its Jacobian and mass
are the system's as assembled, compared with the system's own matrices,
and its row set selects exactly the constraint set's DOFs. The adapter
applies the essential rows: its implicit system's mass has identity rows
and its Jacobian zero rows there, matching the adapter's matrix forms,
and a shifted-Jacobian solve equals a dense solve with sigma M - J. For a
scalar linear physics, a scalar nonlinear one and a mixed one, with the
consistent and the lumped mass.
"""

from typing import Any

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.pde.assembly import ElementDofMapProtocol
from pyapprox.pde.galerkin.time_integration import skfem_ode_residual
from pyapprox.pde.galerkin.time_integration.constrained_mass import (
    ConstrainedMassMatrixProtocol,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.galerkin.time_integration.skfem_jacobian_assembler import (
    SkfemJacobianAssembler,
    require_sparse,
)
from pyapprox.pde.galerkin.time_integration.skfem_mass_assembler import (
    SkfemMassAssembler,
)
from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import LinearizationPoint, SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import CSRFormat
from tests._helpers.galerkin_systems import (
    SYSTEM_NAMES,
    MassScaledSystem,
    NpSystem,
    galerkin_system,
)
from tests._helpers.linalg.mock_stack import (
    MockAssemblyStack,
    MockFormat,
    MockVector,
    peek,
    wrap,
)

_Np = NDArray[Any]


def _dense(matrix: _Np) -> _Np:
    """A matrix the adapter or system holds, sparse or dense, as an array."""
    dense: _Np = csr_matrix(matrix).toarray()
    return dense


def _adapter_masses(system: NpSystem, lumped: bool) -> tuple[_Np, _Np]:
    """The adapter's matrix-form (unconstrained, constrained) mass."""
    mass = skfem_ode_residual(system, lumped_mass=lumped).mass_matrix()
    assert isinstance(mass, ConstrainedMassMatrixProtocol)
    return _dense(mass.unconstrained().as_matrix()), _dense(mass.as_matrix())


def _raw_jacobian(system: NpSystem, state: _Np, time: float) -> _Np:
    dense: _Np = require_sparse(
        system.spatial_operator().spatial_jacobian(state, time)
    ).toarray()
    return dense


def _state(system: NpSystem) -> _Np:
    return np.random.default_rng(0).normal(size=system.nstates())


def _csr(fmt: CSRFormat) -> _Np:
    dense: _Np = fmt.matrix().toarray()
    return dense


def _essential(system: NpSystem, bkd: NumpyBkd) -> _Np:
    dofs: _Np = np.sort(bkd.to_numpy(system.constraint_set().dofs()))
    return dofs


@pytest.mark.parametrize("lumped", [False, True])
@pytest.mark.parametrize("physics", SYSTEM_NAMES)
class TestSkfemDiscretization:
    def test_mass_is_the_systems_as_assembled(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        system, _, _ = galerkin_system(physics, numpy_bkd)
        disc = skfem_discretization(system, lumped_mass=lumped)
        unconstrained, _ = _adapter_masses(system, lumped)
        stored = disc.assemble_mass().format()
        assert isinstance(stored, CSRFormat)
        numpy_bkd.assert_allclose(_csr(stored), unconstrained, atol=1e-15)
        assert disc.assemble_mass().properties().diagonal == lumped

    def test_jacobian_is_the_systems_as_assembled(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        system, _, _ = galerkin_system(physics, numpy_bkd)
        disc = skfem_discretization(system, lumped_mass=lumped)
        state = _state(system)
        op = disc.assemble_jacobian(state, 0.3)
        numpy_bkd.assert_allclose(
            _csr(op.format()), _raw_jacobian(system, state, 0.3), atol=1e-14
        )

    def test_row_set_is_the_constraint_sets(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        system, _, _ = galerkin_system(physics, numpy_bkd)
        disc = skfem_discretization(system, lumped_mass=lumped)
        n = system.nstates()
        zeroed = disc.essential_rows().zero_rows(np.ones((n, 1)))[:, 0]
        expected = _essential(system, numpy_bkd)
        np.testing.assert_array_equal(np.flatnonzero(zeroed == 0.0), expected)
        assert disc.essential_rows().count() == expected.size > 0

    def test_spaces(self, numpy_bkd: NumpyBkd, physics: str, lumped: bool) -> None:
        system, _, _ = galerkin_system(physics, numpy_bkd)
        disc = skfem_discretization(system, lumped_mass=lumped)
        descriptor = SpaceDescriptor("state", system.nstates(), "float64")
        assert disc.descriptor() == descriptor
        assert disc.nstates() == system.nstates()
        assert disc.system() is system
        op = disc.assemble_mass()
        assert (op.domain(), op.range()) == (descriptor, descriptor)


@pytest.mark.parametrize("lumped", [False, True])
@pytest.mark.parametrize("physics", SYSTEM_NAMES)
class TestAdapterStackSystem:
    def test_essential_rows_are_applied_by_the_adapter(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        """Identity mass rows and zero Jacobian rows at the constrained
        DOFs, equal to the adapter's matrix forms everywhere."""
        system, _, _ = galerkin_system(physics, numpy_bkd)
        adapter = skfem_ode_residual(system, lumped_mass=lumped)
        assert adapter.lumped_mass() == lumped
        implicit = adapter.implicit_system()
        _, constrained = _adapter_masses(system, lumped)
        mass = implicit.mass()
        mass_format = mass.format()
        assert isinstance(mass_format, CSRFormat)
        numpy_bkd.assert_allclose(_csr(mass_format), constrained, atol=1e-15)
        assert mass.properties().diagonal == lumped
        state = _state(system)
        shift = 7.0
        op = implicit.shifted_jacobian(
            state, np.zeros_like(state), 0.3, shift, LinearizationPoint(0)
        )
        expected = shift * constrained - _dense(adapter.jacobian(state, 0.3))
        op_format = op.format()
        assert isinstance(op_format, CSRFormat)
        numpy_bkd.assert_allclose(_csr(op_format), expected, atol=1e-13)
        n = system.nstates()
        essential = _essential(system, numpy_bkd)
        numpy_bkd.assert_allclose(
            expected[essential], shift * np.eye(n)[essential], atol=1e-15
        )

    def test_shifted_solves_match_the_matrix_form(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        system, _, _ = galerkin_system(physics, numpy_bkd)
        adapter = skfem_ode_residual(system, lumped_mass=lumped)
        implicit = adapter.implicit_system()
        state, shift = _state(system), 7.0
        op = implicit.shifted_jacobian(
            state, np.zeros_like(state), 0.3, shift, LinearizationPoint(0)
        )
        _, constrained = _adapter_masses(system, lumped)
        matrix = shift * constrained - _dense(adapter.jacobian(state, 0.3))
        rhs = np.random.default_rng(2).normal(size=(system.nstates(), 2))
        handle = implicit.solver().setup(op)
        solution, info = handle.solve(rhs)
        assert info.converged
        numpy_bkd.assert_allclose(solution, np.linalg.solve(matrix, rhs), rtol=1e-10)
        transposed = handle.transpose()
        assert transposed is not None
        solution, _ = transposed.solve(rhs)
        numpy_bkd.assert_allclose(solution, np.linalg.solve(matrix.T, rhs), rtol=1e-10)


@pytest.mark.parametrize("lumped", [False, True])
class TestMassCaching:
    def test_repeated_calls_share_one_mass(
        self, numpy_bkd: NumpyBkd, lumped: bool
    ) -> None:
        system, _, _ = galerkin_system("burgers", numpy_bkd)
        adapter = skfem_ode_residual(system, lumped_mass=lumped)
        assert adapter.implicit_system().mass() is adapter.implicit_system().mass()
        disc = skfem_discretization(system, lumped_mass=lumped)
        assert disc.assemble_mass() is disc.assemble_mass()

    def test_a_new_mass_is_assembled_again(
        self, numpy_bkd: NumpyBkd, lumped: bool
    ) -> None:
        """Rescaling the system's mass yields new operators with the new
        values; the earlier ones keep theirs."""
        inner, _, _ = galerkin_system("burgers", numpy_bkd)
        system = MassScaledSystem(inner)
        adapter = skfem_ode_residual(system, lumped_mass=lumped)
        disc = skfem_discretization(system, lumped_mass=lumped)
        first, first_assembled = (
            adapter.implicit_system().mass(),
            disc.assemble_mass(),
        )
        before = _csr(first_assembled.format()).copy()
        system.scale(2.0)
        second, second_assembled = (
            adapter.implicit_system().mass(),
            disc.assemble_mass(),
        )
        assert second is not first
        assert second_assembled is not first_assembled
        numpy_bkd.assert_allclose(_csr(second_assembled.format()), 2.0 * before)
        numpy_bkd.assert_allclose(_csr(first_assembled.format()), before)
        second_format = second.format()
        assert isinstance(second_format, CSRFormat)
        essential = _essential(inner, numpy_bkd)
        numpy_bkd.assert_allclose(
            _csr(second_format)[essential], np.eye(inner.nstates())[essential]
        )


def test_the_builder_takes_the_basis_from_the_system(numpy_bkd: NumpyBkd) -> None:
    """Systems carry their basis, whose element DOF maps give the pattern."""
    system, basis, _ = galerkin_system("adr", numpy_bkd)
    assert system.basis() is basis
    assert isinstance(system.basis(), ElementDofMapProtocol)
    assert skfem_discretization(system).nstates() == system.nstates()


def _mock_discretization(
    bkd: NumpyBkd,
) -> tuple[NpSystem, GalerkinStackDiscretization[_Np, MockVector, MockFormat]]:
    system, basis, descriptor = galerkin_system("burgers", bkd)
    stack = MockAssemblyStack()
    return system, GalerkinStackDiscretization(
        system,
        stack,
        SkfemJacobianAssembler(system, stack, basis, descriptor),
        SkfemMassAssembler(system, stack, basis, descriptor),
        descriptor,
    )


class TestOnAnotherStack:
    def test_operators_match_the_systems(self, numpy_bkd: NumpyBkd) -> None:
        """Nothing in the discretization depends on the SciPy stack."""
        system, disc = _mock_discretization(numpy_bkd)
        unconstrained, _ = _adapter_masses(system, False)
        numpy_bkd.assert_allclose(disc.assemble_mass().format().dense(), unconstrained)
        state = _state(system)
        op = disc.assemble_jacobian(state, 0.0)
        x = np.random.default_rng(1).normal(size=(system.nstates(), 2))
        reference = _raw_jacobian(system, state, 0.0)
        numpy_bkd.assert_allclose(
            peek(op.apply(wrap(x, disc.descriptor()))), reference @ x
        )
        numpy_bkd.assert_allclose(
            peek(op.apply_transpose(wrap(x, disc.descriptor()))), reference.T @ x
        )

    def test_masses_are_independent_of_later_calls(self, numpy_bkd: NumpyBkd) -> None:
        _, disc = _mock_discretization(numpy_bkd)
        first = disc.assemble_mass()
        before = first.format().dense().copy()
        disc.assemble_mass()
        numpy_bkd.assert_allclose(first.format().dense(), before)


class TestErrors:
    def test_refuses_a_descriptor_of_another_size(self, numpy_bkd: NumpyBkd) -> None:
        system, basis, descriptor = galerkin_system("adr", numpy_bkd)
        stack = MockAssemblyStack()
        wrong = SpaceDescriptor("state", system.nstates() + 1)
        with pytest.raises(ValueError, match="size"):
            GalerkinStackDiscretization(
                system,
                stack,
                SkfemJacobianAssembler(system, stack, basis, descriptor),
                SkfemMassAssembler(system, stack, basis, descriptor),
                wrong,
            )

    def test_refuses_objects_without_their_protocols(self, numpy_bkd: NumpyBkd) -> None:
        system, basis, descriptor = galerkin_system("adr", numpy_bkd)
        stack = MockAssemblyStack()
        jacobian = SkfemJacobianAssembler(system, stack, basis, descriptor)
        mass = SkfemMassAssembler(system, stack, basis, descriptor)
        with pytest.raises(TypeError, match="mass_assembler"):
            GalerkinStackDiscretization(
                system,
                stack,
                jacobian,
                jacobian,  # type: ignore[arg-type]
                descriptor,
            )
        with pytest.raises(TypeError, match="jacobian_assembler"):
            GalerkinStackDiscretization(
                system,
                stack,
                mass,  # type: ignore[arg-type]
                mass,
                descriptor,
            )
        with pytest.raises(TypeError, match="stack"):
            GalerkinStackDiscretization(
                system,
                basis,  # type: ignore[arg-type]
                jacobian,
                mass,
                descriptor,
            )

"""GalerkinStackDiscretization, from the skfem builder and on the mock stack.

Its state Jacobian and its masses (as assembled, and with the essential
rows replaced) are compared with the residual adapter's matrices, for a
scalar linear physics, a scalar nonlinear one and a mixed one, with the
consistent and the lumped mass. Its row set selects exactly the
constraint set's DOFs, and the same row set zeroes the Jacobian's rows
and replaces the mass's.
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
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.time_integration import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.galerkin.time_integration.constrained_mass import (
    ConstrainedMassMatrixProtocol,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.galerkin.time_integration.skfem_jacobian_assembler import (
    SkfemJacobianAssembler,
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
    adr_system,
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
    """A matrix the adapter holds, sparse or dense, as an array."""
    dense: _Np = csr_matrix(matrix).toarray()
    return dense


def _adapter(
    system: NpSystem, lumped: bool
) -> GalerkinPhysicsToODEResidualAdapter[_Np]:
    return GalerkinPhysicsToODEResidualAdapter(system, lumped_mass=lumped)


def _adapter_masses(system: NpSystem, lumped: bool) -> tuple[_Np, _Np]:
    """The adapter's (unconstrained, constrained) mass, as arrays."""
    mass = _adapter(system, lumped).mass_matrix()
    assert isinstance(mass, ConstrainedMassMatrixProtocol)
    return _dense(mass.unconstrained().as_matrix()), _dense(mass.as_matrix())


def _state(system: NpSystem) -> _Np:
    return np.random.default_rng(0).normal(size=system.nstates())


def _csr(fmt: CSRFormat) -> _Np:
    dense: _Np = fmt.matrix().toarray()
    return dense


@pytest.mark.parametrize("lumped", [False, True])
@pytest.mark.parametrize("physics", SYSTEM_NAMES)
class TestSkfemDiscretization:
    def test_masses_match_the_adapters(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        system, basis, _ = galerkin_system(physics, numpy_bkd)
        disc = skfem_discretization(system, basis, lumped_mass=lumped)
        unconstrained, constrained = _adapter_masses(system, lumped)
        stored = disc.unconstrained_mass().format()
        assert isinstance(stored, CSRFormat)
        numpy_bkd.assert_allclose(_csr(stored), unconstrained, atol=1e-15)
        numpy_bkd.assert_allclose(_csr(disc.mass().format()), constrained, atol=1e-15)
        assert disc.mass().properties().diagonal == lumped

    def test_jacobian_matches_the_adapters(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        system, basis, _ = galerkin_system(physics, numpy_bkd)
        disc = skfem_discretization(system, basis, lumped_mass=lumped)
        state = _state(system)
        point = LinearizationPoint(0, ("step", 0))
        op = disc.jacobian_operator(state, 0.3, point)
        reference = _dense(_adapter(system, lumped).jacobian(state, 0.3))
        numpy_bkd.assert_allclose(_csr(op.format()), reference, atol=1e-14)
        assert op.linearization() == point

    def test_one_row_set_is_the_constraint_sets(
        self, numpy_bkd: NumpyBkd, physics: str, lumped: bool
    ) -> None:
        """The row set selects the constrained DOFs; the Jacobian's zero
        rows and the mass's identity rows are those rows."""
        system, basis, _ = galerkin_system(physics, numpy_bkd)
        disc = skfem_discretization(system, basis, lumped_mass=lumped)
        n = system.nstates()
        zeroed = disc.essential_rows().zero_rows(np.ones((n, 1)))[:, 0]
        selected = np.flatnonzero(zeroed == 0.0)
        expected = np.sort(numpy_bkd.to_numpy(system.constraint_set().dofs()))
        np.testing.assert_array_equal(selected, expected)
        assert disc.essential_rows().count() == expected.size > 0
        jacobian = _csr(
            disc.jacobian_operator(_state(system), 0.0, LinearizationPoint(0)).format()
        )
        numpy_bkd.assert_allclose(jacobian[selected], np.zeros((selected.size, n)))
        numpy_bkd.assert_allclose(
            _csr(disc.mass().format())[selected], np.eye(n)[selected]
        )

    def test_spaces(self, numpy_bkd: NumpyBkd, physics: str, lumped: bool) -> None:
        system, basis, _ = galerkin_system(physics, numpy_bkd)
        disc = skfem_discretization(system, basis, lumped_mass=lumped)
        descriptor = SpaceDescriptor("state", system.nstates(), "float64")
        assert disc.descriptor() == descriptor
        assert disc.nstates() == system.nstates()
        assert disc.system() is system
        for op in (disc.mass(), disc.unconstrained_mass()):
            assert (op.domain(), op.range()) == (descriptor, descriptor)


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


@pytest.mark.parametrize("lumped", [False, True])
class TestMassCaching:
    def test_repeated_calls_share_one_mass(
        self, numpy_bkd: NumpyBkd, lumped: bool
    ) -> None:
        system, basis, _ = galerkin_system("burgers", numpy_bkd)
        disc = skfem_discretization(system, basis, lumped_mass=lumped)
        assert disc.mass() is disc.mass()
        assert disc.unconstrained_mass() is disc.unconstrained_mass()

    def test_a_new_mass_is_assembled_again(
        self, numpy_bkd: NumpyBkd, lumped: bool
    ) -> None:
        """Rescaling the system's mass yields new operators with the new
        values; the earlier ones keep theirs."""
        inner, basis, _ = galerkin_system("burgers", numpy_bkd)
        system = MassScaledSystem(inner)
        disc = skfem_discretization(system, basis, lumped_mass=lumped)
        first, first_unconstrained = disc.mass(), disc.unconstrained_mass()
        before = _csr(first_unconstrained.format()).copy()
        system.scale(2.0)
        second, second_unconstrained = disc.mass(), disc.unconstrained_mass()
        assert second is not first
        assert second_unconstrained is not first_unconstrained
        numpy_bkd.assert_allclose(_csr(second_unconstrained.format()), 2.0 * before)
        numpy_bkd.assert_allclose(_csr(first_unconstrained.format()), before)
        essential = numpy_bkd.to_numpy(inner.constraint_set().dofs())
        numpy_bkd.assert_allclose(
            _csr(second.format())[essential],
            np.eye(inner.nstates())[essential],
        )


def test_a_basis_typed_by_its_protocol_feeds_the_builder(
    numpy_bkd: NumpyBkd,
) -> None:
    """Physics hand out their basis as a ``GalerkinBasisProtocol``, which
    carries the element DOF maps the builder needs."""
    system, lagrange = adr_system(numpy_bkd)
    basis: GalerkinBasisProtocol[_Np] = lagrange
    assert isinstance(basis, ElementDofMapProtocol)
    disc = skfem_discretization(system, basis)
    assert disc.nstates() == system.nstates()


class TestOnAnotherStack:
    def test_operators_match_the_adapters(self, numpy_bkd: NumpyBkd) -> None:
        """Nothing in the discretization depends on the SciPy stack."""
        system, disc = _mock_discretization(numpy_bkd)
        _, constrained = _adapter_masses(system, False)
        numpy_bkd.assert_allclose(disc.mass().format().dense(), constrained)
        state = _state(system)
        op = disc.jacobian_operator(state, 0.0, LinearizationPoint(0))
        x = np.random.default_rng(1).normal(size=(system.nstates(), 2))
        reference = _dense(_adapter(system, False).jacobian(state, 0.0))
        numpy_bkd.assert_allclose(
            peek(op.apply(wrap(x, disc.descriptor()))), reference @ x
        )
        numpy_bkd.assert_allclose(
            peek(op.apply_transpose(wrap(x, disc.descriptor()))), reference.T @ x
        )

    def test_masses_are_independent_of_later_calls(self, numpy_bkd: NumpyBkd) -> None:
        _, disc = _mock_discretization(numpy_bkd)
        first = disc.mass()
        before = first.format().dense().copy()
        disc.mass()
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
        with pytest.raises(TypeError, match="assembler"):
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

"""GalerkinJacobianOperator on the SciPy, mock and no-bulk stacks.

The stack operator is compared with the residual adapter's matrix
Jacobian (essential rows zeroed) for a linear scalar physics, a
nonlinear one and a mixed one; for the nonlinear one its action is also
checked against a central difference of the residual, which shares no
code with either Jacobian. The no-bulk stack exercises the fallback that
adds nonzeros as element blocks.
"""

from typing import Any, Callable, Sequence, TypeVar

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix

from pyapprox.pde.assembly import ElementDofMapProtocol, ScipyAssemblyStack
from pyapprox.pde.assembly.protocols import AssemblyStackProtocol
from pyapprox.pde.boundary import BoundaryConditionRole
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import LagrangeBasis, MixedBasis
from pyapprox.pde.galerkin.basis.mixed import MixedFieldBasisProtocol
from pyapprox.pde.galerkin.basis.vector_lagrange import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.physics.burgers import BurgersPhysics
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.galerkin.time_integration.jacobian_operator import (
    GalerkinJacobianOperator,
    require_sparse,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    LinearizationPoint,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import CSRFormat
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


def _zero(x: _Np) -> _Np:
    return np.zeros(x.shape[1])


def _ones(x: _Np) -> _Np:
    return np.ones(x.shape[-1])


def _zero_velocity(x: _Np) -> _Np:
    return np.zeros((2, x.shape[1]))


def _adr(bkd: NumpyBkd) -> tuple[_System, LagrangeBasis[_Np]]:
    """Linear diffusion-reaction, P1 on a 3x2 mesh, Dirichlet on the left."""
    mesh = StructuredMesh2D(3, 2, [(0.0, 1.0), (0.0, 1.0)], bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=0.5,
        bkd=bkd,
        reaction=2.0,
        forcing=TimeIndependent(_zero),
    )
    bc = DirichletBC(basis, "left", TimeIndependent(_zero), bkd)
    return compose_galerkin_system(physics, [bc]), basis


def _burgers(bkd: NumpyBkd) -> tuple[_System, LagrangeBasis[_Np]]:
    """Viscous Burgers, P2 on 6 elements, Dirichlet at both ends."""
    mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)
    physics = BurgersPhysics(
        basis=basis, viscosity=0.1, bkd=bkd, forcing=TimeIndependent(_ones)
    )
    bcs: Sequence[BoundaryConditionRole[_Np]] = [
        DirichletBC(basis, name, TimeIndependent(_zero), bkd)
        for name in ("left", "right")
    ]
    return compose_galerkin_system(physics, bcs), basis


def _stokes(bkd: NumpyBkd) -> tuple[_System, MixedBasis[_Np]]:
    """Taylor-Hood Stokes on a 3x3 mesh, velocity fixed on the left."""
    mesh = StructuredMesh2D(nx=3, ny=3, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
    fields: list[tuple[str, MixedFieldBasisProtocol[_Np]]] = [
        ("velocity", VectorLagrangeBasis(mesh, degree=2)),
        ("pressure", LagrangeBasis(mesh, degree=1)),
    ]
    basis = MixedBasis(fields)
    bc = DirichletBC(
        basis.field("velocity"), "left", TimeIndependent(_zero_velocity), bkd
    )
    return compose_galerkin_system(StokesPhysics(basis, bkd), [bc]), basis


_Builder = Callable[[NumpyBkd], tuple[_System, ElementDofMapProtocol[_Dofs]]]
_PHYSICS: dict[str, _Builder] = {"adr": _adr, "burgers": _burgers, "stokes": _stokes}


def _state(system: _System, seed: int) -> _Np:
    return np.random.default_rng(seed).normal(size=system.nstates())


def _reference(system: _System, state: _Np) -> _Np:
    """The adapter's Jacobian, essential rows zeroed, as a dense array."""
    jacobian = GalerkinPhysicsToODEResidualAdapter(system).jacobian(state, 0.0)
    dense: _Np = require_sparse(jacobian).toarray()
    return dense


def _check_against_reference(
    stack: AssemblyStackProtocol[V, F, _Dofs, _Values],
    to_vec: Callable[[_Np, SpaceDescriptor], V],
    from_vec: Callable[[V], _Np],
    format_dense: Callable[[F], _Np],
    format_type: type[F],
    physics: str,
    bkd: NumpyBkd,
) -> None:
    system, basis = _PHYSICS[physics](bkd)
    descriptor = SpaceDescriptor("state", system.nstates())
    producer = GalerkinJacobianOperator(system, stack, basis, descriptor)
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
    def test_scipy_stack(self, numpy_bkd: NumpyBkd, physics: str) -> None:
        _check_against_reference(
            ScipyAssemblyStack(np.float64),
            lambda x, _: x,
            lambda v: v,
            lambda fmt: fmt.matrix().toarray(),
            CSRFormat,
            physics,
            numpy_bkd,
        )

    def test_mock_stack(self, numpy_bkd: NumpyBkd, physics: str) -> None:
        _check_against_reference(
            MockAssemblyStack(),
            wrap,
            peek,
            lambda fmt: fmt.dense(),
            MockFormat,
            physics,
            numpy_bkd,
        )

    def test_no_bulk_stack(self, numpy_bkd: NumpyBkd, physics: str) -> None:
        _check_against_reference(
            NoBulkAssemblyStack(np.float64),
            lambda x, _: x,
            lambda v: v,
            lambda fmt: fmt.matrix().toarray(),
            CSRFormat,
            physics,
            numpy_bkd,
        )


class TestNonlinear:
    def _producer(
        self, bkd: NumpyBkd
    ) -> tuple[_System, GalerkinJacobianOperator[_Np, MockVector, MockFormat]]:
        system, basis = _burgers(bkd)
        descriptor = SpaceDescriptor("state", system.nstates())
        return system, GalerkinJacobianOperator(
            system, MockAssemblyStack(), basis, descriptor
        )

    def test_action_is_the_residual_derivative(self, numpy_bkd: NumpyBkd) -> None:
        """Central difference of the residual: essential rows of the
        residual carry no state, so they differentiate to zero too."""
        system, producer = self._producer(numpy_bkd)
        residual = GalerkinPhysicsToODEResidualAdapter(system)
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
        system, producer = self._producer(numpy_bkd)
        first_state, second_state = _state(system, 4), 3.0 * _state(system, 5)
        first = producer.jacobian_operator(first_state, 0.0, LinearizationPoint(0))
        before = first.format().dense().copy()
        producer.jacobian_operator(second_state, 0.0, LinearizationPoint(1))
        numpy_bkd.assert_allclose(first.format().dense(), before)
        numpy_bkd.assert_allclose(before, _reference(system, first_state))
        assert not np.allclose(before, _reference(system, second_state))


class TestErrors:
    def test_refuses_a_descriptor_of_another_size(self, numpy_bkd: NumpyBkd) -> None:
        system, basis = _adr(numpy_bkd)
        with pytest.raises(ValueError, match="size"):
            GalerkinJacobianOperator(
                system,
                MockAssemblyStack(),
                basis,
                SpaceDescriptor("state", system.nstates() + 1),
            )

    def test_refuses_objects_without_their_protocols(self, numpy_bkd: NumpyBkd) -> None:
        system, basis = _adr(numpy_bkd)
        descriptor = SpaceDescriptor("state", system.nstates())
        stack = MockAssemblyStack()
        with pytest.raises(TypeError, match="system"):
            GalerkinJacobianOperator(basis, stack, basis, descriptor)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="stack"):
            GalerkinJacobianOperator(system, basis, basis, descriptor)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="dof_map"):
            GalerkinJacobianOperator(system, stack, system, descriptor)  # type: ignore[arg-type]

    def test_require_sparse_refuses_a_dense_jacobian(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(TypeError, match="sparse Jacobian"):
            require_sparse(numpy_bkd.eye(3))
        assert isinstance(require_sparse(csr_matrix(np.eye(3))), spmatrix)

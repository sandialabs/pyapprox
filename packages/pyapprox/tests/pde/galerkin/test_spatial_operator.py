"""Tests for ComposedSpatialOperator, F = F_Omega + F_Gamma."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Generic, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pyapprox.ode.protocols import SpatialOperatorProtocol
from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.boundary import NaturalBCOperator
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.boundary.implementations import (
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.mesh.structured import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.spatial_operator import ComposedSpatialOperator
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.util.backends.protocols import Array, Backend


class _StubInterior(Generic[Array]):
    """A linear interior ``F_Omega(u) = -A u + b`` with no base class.

    Its declared curvature is configurable, so composition can be
    checked with curvature present, zero, or absent.
    """

    def __init__(
        self,
        matrix: Array,
        load: Array,
        bkd: Backend[Array],
        derivatives: Optional[StateDerivatives[Array]] = None,
    ) -> None:
        self._matrix = matrix
        self._load = load
        self._bkd = bkd
        self._derivatives = (
            StateDerivatives.linear(bkd) if derivatives is None else derivatives
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return int(self._matrix.shape[0])

    def interior_residual(self, state: Array, time: float) -> Array:
        return self._load - self._matrix @ state

    def interior_jacobian(self, state: Array, time: float) -> Array:
        return -self._matrix

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        return self._derivatives

    def interior_is_time_invariant(self) -> bool:
        return True


def _elementwise_hvp(
    state: Array, adj_state: Array, wvec: Array, time: float
) -> Array:
    """A stand-in curvature: ``adj * w * u`` elementwise."""
    return adj_state * wvec * state


class _CurvedNeumann(NeumannBC[Any]):
    """A Neumann term that declares a stand-in curvature."""

    def state_derivatives(self) -> StateDerivatives[Any]:
        return StateDerivatives.second_order(_elementwise_hvp)


class _UncurvedNeumann(NeumannBC[Any]):
    """A term that supplies no curvature."""

    def state_derivatives(self) -> StateDerivatives[Any]:
        return StateDerivatives.none()


class TestComposedSpatialOperator:
    def _setup(
        self, bkd: Backend[Array]
    ) -> Tuple[LagrangeBasis[Array], _StubInterior[Array], Array]:
        mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0, 1], [0, 1]], bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        n = basis.ndofs()
        matrix = bkd.eye(n) * 2.0 + bkd.ones((n, n)) * 0.1
        load = bkd.linspace(0.0, 1.0, n)
        interior = _StubInterior(matrix, load, bkd)
        state = bkd.linspace(-1.0, 2.0, n)
        return basis, interior, state

    def test_satisfies_protocol(self, bkd: Backend[Array]) -> None:
        _, interior, _ = self._setup(bkd)
        op = ComposedSpatialOperator(interior, NaturalBCOperator([]))
        assert isinstance(op, SpatialOperatorProtocol)
        assert op.nstates() == interior.nstates()

    def test_empty_bcs_is_interior(self, bkd: Backend[Array]) -> None:
        _, interior, state = self._setup(bkd)
        op = ComposedSpatialOperator(interior, NaturalBCOperator([]))
        bkd.assert_allclose(
            op.spatial_residual(state, 0.0),
            interior.interior_residual(state, 0.0),
        )
        bkd.assert_allclose(
            op.spatial_jacobian(state, 0.0),
            interior.interior_jacobian(state, 0.0),
        )

    def test_adds_each_term(self, bkd: Backend[Array]) -> None:
        basis, interior, state = self._setup(bkd)
        terms = [
            RobinBC(basis, "left", alpha=2.0, value_func=1.0, bkd=bkd),
            NeumannBC(basis, "top", flux_func=3.0, bkd=bkd),
        ]
        op = ComposedSpatialOperator(interior, NaturalBCOperator(terms))
        n = interior.nstates()
        expected_res = interior.interior_residual(state, 0.0)
        expected_jac = interior.interior_jacobian(state, 0.0)
        for term in terms:
            expected_res = expected_res + term.apply_to_residual(
                bkd.zeros((n,)), state, 0.0
            )
            expected_jac = expected_jac + term.apply_to_jacobian(
                bkd.zeros((n, n)), state, 0.0
            )
        bkd.assert_allclose(op.spatial_residual(state, 0.0), expected_res)
        bkd.assert_allclose(op.spatial_jacobian(state, 0.0), expected_jac)

    def test_rejects_non_interior(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="GalerkinInteriorOperatorProtocol"):
            ComposedSpatialOperator(object(), NaturalBCOperator([]))

    def test_rejects_non_operator_bcs(self, bkd: Backend[Array]) -> None:
        _, interior, _ = self._setup(bkd)
        with pytest.raises(TypeError, match="NaturalBCOperator"):
            ComposedSpatialOperator(interior, [])


class TestComposedStateDerivatives:
    def _setup(
        self,
        bkd: Backend[Array],
        derivatives: Optional[StateDerivatives[Array]] = None,
    ) -> Tuple[LagrangeBasis[Array], _StubInterior[Array], Array]:
        mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0, 1], [0, 1]], bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        n = basis.ndofs()
        interior = _StubInterior(
            bkd.eye(n), bkd.zeros((n,)), bkd, derivatives
        )
        return basis, interior, bkd.linspace(-1.0, 2.0, n)

    def _hvp(self, op: Any, state: Array, bkd: Backend[Array]) -> Array:
        hvp = op.state_derivatives().state_state_hvp
        assert hvp is not None
        n = state.shape[0]
        return hvp(state, bkd.linspace(0.5, 1.5, n), bkd.ones((n,)), 0.0)

    def test_linear_parts_compose_to_exact_zero(
        self, bkd: Backend[Array]
    ) -> None:
        basis, interior, state = self._setup(bkd)
        terms = [
            RobinBC(basis, "left", alpha=2.0, value_func=1.0, bkd=bkd),
            NeumannBC(basis, "top", flux_func=3.0, bkd=bkd),
        ]
        op = ComposedSpatialOperator(interior, NaturalBCOperator(terms))
        bkd.assert_allclose(
            self._hvp(op, state, bkd), bkd.zeros((state.shape[0],))
        )

    def test_curvature_is_summed_over_interior_and_terms(
        self, bkd: Backend[Array]
    ) -> None:
        basis, interior, state = self._setup(
            bkd, StateDerivatives.second_order(_elementwise_hvp)
        )
        terms = [_CurvedNeumann(basis, "top", flux_func=0.0, bkd=bkd)]
        op = ComposedSpatialOperator(interior, NaturalBCOperator(terms))
        n = state.shape[0]
        single = _elementwise_hvp(
            state, bkd.linspace(0.5, 1.5, n), bkd.ones((n,)), 0.0
        )
        bkd.assert_allclose(self._hvp(op, state, bkd), 2.0 * single)

    def test_absent_interior_curvature_is_absent(
        self, bkd: Backend[Array]
    ) -> None:
        _, interior, _ = self._setup(bkd, StateDerivatives.none())
        op = ComposedSpatialOperator(interior, NaturalBCOperator([]))
        assert op.state_derivatives().state_state_hvp is None

    def test_one_term_without_curvature_makes_it_absent(
        self, bkd: Backend[Array]
    ) -> None:
        """A part without curvature is never silently dropped."""
        basis, interior, _ = self._setup(bkd)
        terms = [
            RobinBC(basis, "left", alpha=2.0, value_func=1.0, bkd=bkd),
            _UncurvedNeumann(basis, "top", flux_func=0.0, bkd=bkd),
        ]
        op = ComposedSpatialOperator(interior, NaturalBCOperator(terms))
        assert op.state_derivatives().state_state_hvp is None


def _square(x: NDArray[Any], u: NDArray[Any]) -> NDArray[Any]:
    return u**2


def _square_derivative(x: NDArray[Any], u: NDArray[Any]) -> NDArray[Any]:
    return 2.0 * u


def _square_second_derivative(
    x: NDArray[Any], u: NDArray[Any]
) -> NDArray[Any]:
    return np.full_like(u, 2.0)


def _one(x: NDArray[Any]) -> NDArray[Any]:
    return np.ones(x.shape[1])


class TestPhysicsStateDerivatives:
    def _adr(
        self, numpy_bkd: Backend[Array], second_derivative: bool
    ) -> GalerkinSystem[Array]:
        basis = LagrangeBasis(
            StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=numpy_bkd), 1
        )
        reaction = CallableReaction(
            _square,
            _square_derivative,
            _square_second_derivative if second_derivative else None,
        )
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            bkd=numpy_bkd,
            reaction=reaction,
            forcing=TimeIndependent(_one),
        )
        return compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, numpy_bkd),
                RobinBC(
                    basis, "right", alpha=2.0, value_func=1.0, bkd=numpy_bkd
                ),
            ],
        )

    def test_composed_curvature_matches_fd_of_jacobian(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """lambda^T (d^2F/du^2) w of the composed F (Robin included)
        against a central difference of J(u)^T lambda along w."""
        bkd = numpy_bkd
        system = self._adr(bkd, second_derivative=True)
        operator = system.spatial_operator()
        hvp = operator.state_derivatives().state_state_hvp
        assert hvp is not None
        n = system.nstates()
        state = bkd.linspace(0.2, 0.9, n)
        adj = bkd.linspace(1.0, -1.0, n)
        wvec = bkd.linspace(0.3, 0.6, n)
        eps = 1e-6

        def jac_t_adj(u: Array) -> Array:
            jac = operator.spatial_jacobian(u, 0.0)
            return bkd.asarray(jac.T @ bkd.to_numpy(adj))

        fd = (jac_t_adj(state + eps * wvec) - jac_t_adj(state - eps * wvec)) / (
            2 * eps
        )
        bkd.assert_allclose(hvp(state, adj, wvec, 0.0), fd, rtol=1e-6, atol=1e-10)

    def test_nonlinear_reaction_without_second_derivative_is_absent(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """Decided up front, not by state_state_hvp raising mid-solve."""
        system = self._adr(numpy_bkd, second_derivative=False)
        assert (
            system.spatial_operator().state_derivatives().state_state_hvp
            is None
        )

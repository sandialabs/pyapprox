"""Tests for ComposedSpatialOperator, F = F_Omega + F_Gamma."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Generic, Tuple

from pyapprox.ode.protocols import SpatialOperatorProtocol
from pyapprox.pde.boundary import NaturalBCOperator
from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh.structured import StructuredMesh2D
from pyapprox.pde.galerkin.spatial_operator import ComposedSpatialOperator
from pyapprox.util.backends.protocols import Array, Backend


class _StubInterior(Generic[Array]):
    """A linear interior ``F_Omega(u) = -A u + b`` with no base class."""

    def __init__(self, matrix: Array, load: Array, bkd: Backend[Array]) -> None:
        self._matrix = matrix
        self._load = load
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return int(self._matrix.shape[0])

    def interior_residual(self, state: Array, time: float) -> Array:
        return self._load - self._matrix @ state

    def interior_jacobian(self, state: Array, time: float) -> Array:
        return -self._matrix


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

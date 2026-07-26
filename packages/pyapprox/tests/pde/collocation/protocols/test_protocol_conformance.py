"""Protocol-conformance tests for collocation physics and boundary protocols.

Verifies that the concrete implementations structurally satisfy the
runtime-checkable protocols they are documented against, and that the
typed BCPhysicalSensitivities carrier flows through Robin parameter
Jacobians.
"""

import dataclasses

import pytest
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import (
    constant_dirichlet_bc,
    gradient_robin_bc,
)
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    HyperelasticTractionNormalOperator,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
    TractionNormalOperator,
    _LegacyNormalOperator,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.protocols import (
    BCPhysicalSensitivities,
    BoundaryConditionProtocol,
    BoundaryConditionWithNormalOperatorProtocol,
    BoundaryConditionWithParamJacobianProtocol,
    NormalOperatorProtocol,
    PhysicsProtocol,
)


def _make_1d_basis(npts, bkd):
    mesh = TransformedMesh1D(npts, bkd)
    return ChebyshevBasis1D(mesh, bkd)


def _make_gradient_robin_bc(basis, bkd, alpha=1.0, beta=2.0):
    left_idx = bkd.array([0], dtype=int)
    normals = bkd.array([[-1.0]])
    return gradient_robin_bc(
        bkd,
        left_idx,
        normals,
        [basis.derivative_matrix()],
        alpha,
        beta,
        0.0,
    )


class TestNormalOperatorConformance:
    """All concrete normal operators satisfy NormalOperatorProtocol."""

    @pytest.mark.parametrize(
        "operator_class",
        [
            GradientNormalOperator,
            FluxNormalOperator,
            TractionNormalOperator,
            _LegacyNormalOperator,
            HyperelasticTractionNormalOperator,
        ],
    )
    def test_operator_class_conforms(self, numpy_bkd, operator_class):
        # NormalOperatorProtocol is method-only, so issubclass checks
        # structural conformance without constructing the operator.
        assert issubclass(operator_class, NormalOperatorProtocol)

    def test_gradient_operator_instance_conforms(self, bkd):
        basis = _make_1d_basis(8, bkd)
        normal_op = GradientNormalOperator(
            bkd,
            bkd.array([0], dtype=int),
            bkd.array([[-1.0]]),
            [basis.derivative_matrix()],
        )
        assert isinstance(normal_op, NormalOperatorProtocol)
        assert not normal_op.has_coefficient_dependence()
        bkd.assert_allclose(normal_op.normals(), bkd.array([[-1.0]]))

    def test_legacy_operator_normals(self, bkd):
        basis = _make_1d_basis(8, bkd)
        deriv_rows = basis.derivative_matrix()[:2, :]
        normal_op = _LegacyNormalOperator(bkd, deriv_rows, -1.0)
        assert isinstance(normal_op, NormalOperatorProtocol)
        # Legacy operator carries only a scalar sign: normals are the
        # sign replicated per boundary point with ndim = 1.
        bkd.assert_allclose(normal_op.normals(), bkd.full((2, 1), -1.0))

    def test_traction_operator_normals(self, bkd):
        basis = _make_1d_basis(4, bkd)
        D = basis.derivative_matrix()
        normals = bkd.array([[0.0, 1.0]])
        normal_op = TractionNormalOperator(
            bkd,
            bkd.array([0], dtype=int),
            normals,
            [D, D],
            1.0,
            1.0,
            0,
            4,
        )
        assert isinstance(normal_op, NormalOperatorProtocol)
        bkd.assert_allclose(normal_op.normals(), normals)


class TestBoundaryConditionConformance:
    """BC implementations satisfy the boundary protocols they document."""

    def test_robin_bc_has_normal_operator(self, bkd):
        basis = _make_1d_basis(8, bkd)
        bc = _make_gradient_robin_bc(basis, bkd)
        assert isinstance(bc, BoundaryConditionProtocol)
        assert isinstance(bc, BoundaryConditionWithParamJacobianProtocol)
        assert isinstance(bc, BoundaryConditionWithNormalOperatorProtocol)
        assert isinstance(bc.normal_operator(), NormalOperatorProtocol)

    def test_dirichlet_bc_lacks_normal_operator(self, bkd):
        bc = constant_dirichlet_bc(bkd, bkd.array([0], dtype=int), 1.0)
        assert isinstance(bc, BoundaryConditionProtocol)
        assert isinstance(bc, BoundaryConditionWithParamJacobianProtocol)
        assert not isinstance(
            bc, BoundaryConditionWithNormalOperatorProtocol
        )


class TestPhysicsProtocolConformance:
    """Physics satisfies PhysicsProtocol including the repaired members."""

    def test_adr_physics_conforms(self, bkd):
        basis = _make_1d_basis(8, bkd)
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        physics.set_boundary_conditions(
            [
                constant_dirichlet_bc(bkd, bkd.array([0], dtype=int), 0.0),
                constant_dirichlet_bc(bkd, bkd.array([7], dtype=int), 0.0),
            ]
        )
        assert isinstance(physics, PhysicsProtocol)
        assert len(physics.boundary_conditions()) == 2

    def test_bc_dof_classification_invariant(self, bkd):
        basis = _make_1d_basis(8, bkd)
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        physics.set_boundary_conditions(
            [
                constant_dirichlet_bc(bkd, bkd.array([0], dtype=int), 0.0),
                _make_gradient_robin_bc(basis, bkd),
            ]
        )
        bc_class = physics.bc_dof_classification()
        assert set(bc_class.essential) <= set(bc_class.row_replaced)


class TestBCPhysicalSensitivities:
    """Typed sensitivities carrier semantics."""

    def test_frozen(self, bkd):
        sens = BCPhysicalSensitivities(dflux_n_dp=bkd.ones((1, 2)))
        with pytest.raises(dataclasses.FrozenInstanceError):
            sens.dflux_n_dp = bkd.zeros((1, 2))

    def test_robin_param_jacobian_uses_sensitivities(self, bkd):
        npts = 8
        basis = _make_1d_basis(npts, bkd)
        beta = 2.0
        bc = _make_gradient_robin_bc(basis, bkd, alpha=1.0, beta=beta)
        nparams = 3
        pjac = bkd.ones((npts, nparams))
        dflux_n_dp = bkd.array([[1.0, 2.0, 3.0]])

        result = bc.apply_to_param_jacobian(
            pjac,
            bkd.zeros((npts,)),
            0.0,
            physical_sensitivities=BCPhysicalSensitivities(
                dflux_n_dp=dflux_n_dp
            ),
        )
        bkd.assert_allclose(result[0, :], beta * dflux_n_dp[0, :])
        bkd.assert_allclose(result[1:, :], pjac[1:, :])

        # Without sensitivities the BC rows are zeroed.
        result = bc.apply_to_param_jacobian(pjac, bkd.zeros((npts,)), 0.0)
        bkd.assert_allclose(result[0, :], bkd.zeros((nparams,)))

"""The collocation adapter tiers' derivatives bundles against their methods.

Each tier's bundle matches its capability: the base tier states only the
Jacobian, the first-order tier adds the parameter and initial-state
Jacobians, the HVP tier adds the second derivatives. On a Chebyshev
advection-diffusion problem with a diffusion parameterization, every
populated operator reproduces the adapter's matrix or vector method,
applied and transposed.
"""

import math

from pyapprox.ode.residual_derivatives import DerivativeOperator
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.models.collocation import (
    CollocationPhysicsToODEResidualWithHVPAdapter,
    CollocationPhysicsToODEResidualWithParamJacobianAdapter,
    create_collocation_physics_ode_residual,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import LinearizationPoint
from tests._helpers.adjoint_checks import NoHVPQuadraticFieldMap

_NPTS = 8
_POINT = LinearizationPoint(0)


def _problem(
    bkd: Backend[Array],
) -> tuple[AdvectionDiffusionReaction[Array], Array]:
    """The physics and its collocation nodes."""
    basis = ChebyshevBasis1D(TransformedMesh1D(_NPTS, bkd), bkd)
    return AdvectionDiffusionReaction(basis, bkd, diffusion=1.0), basis.nodes()


def _state(bkd: Backend[Array], nodes: Array) -> Array:
    return bkd.sin(math.pi * nodes) + 0.3 * nodes


def _column(bkd: Backend[Array], vec: Array) -> Array:
    return bkd.reshape(vec, (vec.shape[0], 1))


def _applied(bkd: Backend[Array], op: DerivativeOperator[Array], vec: Array) -> Array:
    return op.apply(_column(bkd, vec))[:, 0]


def _transposed(
    bkd: Backend[Array], op: DerivativeOperator[Array], vec: Array
) -> Array:
    return op.apply_transpose(_column(bkd, vec))[:, 0]


class TestCollocationBundles:
    def test_base_tier_states_the_jacobian_only(self, bkd: Backend[Array]) -> None:
        physics, nodes = _problem(bkd)
        adapter = create_collocation_physics_ode_residual(physics, bkd)
        bundle = adapter.derivatives()
        assert bundle.param is None and bundle.state_state is None
        assert bundle.jacobian is not None
        state = _state(bkd, nodes)
        w = bkd.linspace(-1.0, 1.0, _NPTS)
        bkd.assert_allclose(
            _applied(bkd, bundle.jacobian(state, 0.2, _POINT), w),
            bkd.dot(adapter.jacobian(state, 0.2), w),
            rtol=1e-12,
        )
        assert adapter.derivatives() is bundle  # built once

    def test_first_order_tier(self, bkd: Backend[Array]) -> None:
        physics, nodes = _problem(bkd)
        basis = bkd.ones((_NPTS,))[:, None]
        field_map = NoHVPQuadraticFieldMap(bkd, bkd.full((_NPTS,), 1.0), basis)
        adapter = create_collocation_physics_ode_residual(
            physics, bkd, create_diffusion_parameterization(physics, bkd, field_map)
        )
        assert isinstance(
            adapter, CollocationPhysicsToODEResidualWithParamJacobianAdapter
        )
        assert not isinstance(adapter, CollocationPhysicsToODEResidualWithHVPAdapter)
        adapter.set_param(bkd.asarray([0.5]))
        bundle = adapter.derivatives()
        assert bundle.state_state is None
        assert bundle.param is not None and bundle.initial_state is not None
        state = _state(bkd, nodes)
        v, w = bkd.asarray([0.7]), bkd.linspace(-1.0, 1.0, _NPTS)
        matrix = adapter.param_jacobian(state, 0.2)
        initial = adapter.initial_param_jacobian()
        pairs = [
            (_applied(bkd, bundle.param(state, 0.2), v), bkd.dot(matrix, v)),
            (_transposed(bkd, bundle.param(state, 0.2), w), bkd.dot(matrix.T, w)),
            (_applied(bkd, bundle.initial_state(), v), bkd.dot(initial, v)),
            (_transposed(bkd, bundle.initial_state(), w), bkd.dot(initial.T, w)),
        ]
        for actual, expected in pairs:
            bkd.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_hvp_tier(self, bkd: Backend[Array]) -> None:
        physics, nodes = _problem(bkd)
        field_map = BasisExpansion(bkd, 1.0, [bkd.ones((_NPTS,)), nodes])
        adapter = create_collocation_physics_ode_residual(
            physics, bkd, create_diffusion_parameterization(physics, bkd, field_map)
        )
        assert isinstance(adapter, CollocationPhysicsToODEResidualWithHVPAdapter)
        adapter.set_param(bkd.asarray([0.3, -0.2]))
        bundle = adapter.derivatives()
        assert bundle.state_state is not None and bundle.state_param is not None
        assert bundle.param_param is not None
        assert bundle.initial_state_curvature is not None
        state = _state(bkd, nodes)
        adj = bkd.cos(nodes)
        v, w = bkd.asarray([0.7, -0.4]), bkd.linspace(-1.0, 1.0, _NPTS)
        time = 0.2
        mixed = bundle.state_param(state, time, adj)
        pairs = [
            (
                _applied(bkd, bundle.state_state(state, time, adj), w),
                adapter.state_state_hvp(state, time, adj, w),
            ),
            (_applied(bkd, mixed, v), adapter.state_param_hvp(state, time, adj, v)),
            (_transposed(bkd, mixed, w), adapter.param_state_hvp(state, time, adj, w)),
            (
                _applied(bkd, bundle.param_param(state, time, adj), v),
                adapter.param_param_hvp(state, time, adj, v),
            ),
            (
                _applied(bkd, bundle.initial_state_curvature(w), v),
                adapter.initial_param_hvp(w, v),
            ),
        ]
        for actual, expected in pairs:
            bkd.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

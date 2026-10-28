"""The collocation adapter tiers' derivatives bundles, by finite differences.

Each tier's bundle matches its capability: the base tier states only the
Jacobian, the first-order tier adds the parameter and initial-state
Jacobians, the HVP tier adds the second derivatives. On a Chebyshev
advection-diffusion problem (8 nodes, unit diffusion) with a diffusion
parameterization, every declared term passes the bundle checker at
t = 0.2: forward-difference errors below 1e-6, dot tests below 1e-12.
"""

import math

from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
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
from tests._helpers.adjoint_checks import NoHVPQuadraticFieldMap

_NPTS = 8


def _problem(
    bkd: Backend[Array],
) -> tuple[AdvectionDiffusionReaction[Array], Array]:
    """The physics and its collocation nodes."""
    basis = ChebyshevBasis1D(TransformedMesh1D(_NPTS, bkd), bkd)
    return AdvectionDiffusionReaction(basis, bkd, diffusion=1.0), basis.nodes()


def _state(bkd: Backend[Array], nodes: Array) -> Array:
    return bkd.sin(math.pi * nodes) + 0.3 * nodes


class TestCollocationBundles:
    def test_base_tier_states_the_jacobian_only(self, bkd: Backend[Array]) -> None:
        physics, nodes = _problem(bkd)
        adapter = create_collocation_physics_ode_residual(physics, bkd)
        bundle = adapter.derivatives()
        assert bundle.param is None and bundle.state_state is None
        assert bundle.jacobian is not None
        report = ODEResidualBundleChecker(adapter).check(_state(bkd, nodes), 0.2)
        assert set(report.fd_errors) == {"jacobian"}
        assert report.failures() == []
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
        param = bkd.asarray([0.5])
        adapter.set_param(param)
        bundle = adapter.derivatives()
        assert bundle.state_state is None
        assert bundle.param is not None and bundle.initial_state is not None
        report = ODEResidualBundleChecker(adapter).check(
            _state(bkd, nodes), 0.2, param=param
        )
        assert set(report.fd_errors) == {"jacobian", "param"}
        assert report.failures() == []

    def test_hvp_tier(self, bkd: Backend[Array]) -> None:
        physics, nodes = _problem(bkd)
        field_map = BasisExpansion(bkd, 1.0, [bkd.ones((_NPTS,)), nodes])
        adapter = create_collocation_physics_ode_residual(
            physics, bkd, create_diffusion_parameterization(physics, bkd, field_map)
        )
        assert isinstance(adapter, CollocationPhysicsToODEResidualWithHVPAdapter)
        param = bkd.asarray([0.3, -0.2])
        adapter.set_param(param)
        bundle = adapter.derivatives()
        assert bundle.state_state is not None and bundle.state_param is not None
        assert bundle.param_param is not None
        assert bundle.initial_state_curvature is not None
        report = ODEResidualBundleChecker(adapter).check(
            _state(bkd, nodes), 0.2, bkd.cos(nodes), param
        )
        assert set(report.fd_errors) == {
            "jacobian",
            "param",
            "state_state",
            "state_param",
            "param_state",
            "param_param",
        }
        assert report.failures() == []

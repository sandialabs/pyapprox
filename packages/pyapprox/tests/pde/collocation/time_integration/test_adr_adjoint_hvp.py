"""Transient adjoint gradient + HVP for collocation ADR with log-KLE.

DerivativeChecker-validated dQ/dp and d^2Q/dp^2 v through the full
collocation stack (DiffusionParameterization -> HVP-tier ODE adapter ->
stepper -> BCEnforcingHVPResidual -> TimeAdjointOperatorWithHVP) with
parameters the coefficients of a lognormal KLE of the diffusion field.
The exp map makes every parameter-facing HVP nonzero, and Dirichlet row
replacement makes the wrapped tensors non-symmetric — the configuration
that requires the wrapper's true-tensor convention (adjoint zeroed into
every contraction, outputs unmasked).
"""

from typing import Any, Tuple

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    create_uniform_mesh_1d,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.time_integration.bc_time_residual_adapter import (
    BCEnforcingHVPResidual,
    create_bc_enforcing_residual,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.models.collocation import (
    CollocationPhysicsToODEResidualWithHVPAdapter,
    create_collocation_physics_ode_residual,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.rootfinding.newton import NewtonSolver

from tests._helpers.adjoint_checks import HVPOperatorFunction, NumpyArray

_NPARAMS = 3
_FINAL_TIME, _DELTAT = 0.4, 0.1


def _lognormal_kle_map(
    bkd: NumpyBkd, nodes: NumpyArray
) -> TransformedFieldMap[NumpyArray]:
    """D = exp(W theta): hand-built smooth modes on the nodes."""
    coords = bkd.to_numpy(nodes)
    modes = np.stack(
        [
            0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1)
            for k in range(_NPARAMS)
        ],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _build_pipeline(
    bkd: NumpyBkd, method: str, final_time: float = _FINAL_TIME
) -> Tuple[
    TimeIntegrator[NumpyArray],
    Any,
    AdvectionDiffusionReaction[NumpyArray],
]:
    npts = 20
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
    nodes = basis.nodes()

    def forcing(t: float) -> NumpyArray:
        return (np.pi**2) * bkd.sin(np.pi * nodes)

    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=1.0, forcing=forcing
    )
    physics.set_boundary_conditions(
        [
            zero_dirichlet_bc(bkd, mesh_obj.boundary_indices(0)),
            zero_dirichlet_bc(bkd, mesh_obj.boundary_indices(1)),
        ]
    )
    param_obj = create_diffusion_parameterization(
        physics, bkd, basis, _lognormal_kle_map(bkd, nodes)
    )
    adapter = create_collocation_physics_ode_residual(
        physics, bkd, param_obj
    )
    assert isinstance(
        adapter, CollocationPhysicsToODEResidualWithHVPAdapter
    )
    stepper = create_stepper(method, adapter)
    wrapper = create_bc_enforcing_residual(stepper, physics, bkd)
    assert isinstance(wrapper, BCEnforcingHVPResidual)
    newton = NewtonSolver(wrapper)
    newton.set_options(maxiters=20, atol=1e-12, rtol=0.0)
    integrator = TimeIntegrator(0.0, final_time, _DELTAT, newton)
    return integrator, adapter, physics


class TestCollocationADRLogKLEAdjointHVP:
    # final_time=0.35 with dt=0.1 forces a NON-UNIFORM last step: any
    # backward-sweep method reading stale bound step state (last
    # forward step's deltat/t) instead of its ctx argument fails at
    # O(1) — uniform-dt autonomous problems cannot see this.
    @pytest.mark.parametrize("final_time", [_FINAL_TIME, 0.35])
    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_endpoint_gradient_and_hvp_match_fd(
        self, numpy_bkd: NumpyBkd, method: str, final_time: float
    ) -> None:
        bkd = numpy_bkd
        integrator, adapter, physics = _build_pipeline(
            bkd, method, final_time
        )
        npts = physics.npts()
        functional = EndpointFunctional(npts // 2, npts, _NPARAMS, bkd)
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        y0 = bkd.asarray(np.zeros(npts))
        fn = HVPOperatorFunction(operator, adapter, y0, bkd)
        checker = DerivativeChecker(fn)

        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        direction = bkd.asarray(np.array([[0.5], [0.7], [-0.6]]))
        errors = checker.check_derivatives(
            param, direction=direction, relative=True
        )

        # One-sided FD of iteratively solved quantities floors the
        # error ratio near 1e-6 (the min/max denominator is the
        # eps=1e-13 roundoff blowup, which varies by stepper); assert
        # the V-bottom (a genuine bug plateaus orders of magnitude
        # higher) and a looser ratio.
        jac_min = float(bkd.to_numpy(bkd.min(errors[0])))
        assert jac_min <= 1e-7
        jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert jac_ratio <= 1e-5
        hvp_min = float(bkd.to_numpy(bkd.min(errors[1])))
        assert hvp_min <= 1e-6
        hvp_ratio = float(bkd.to_numpy(checker.error_ratio(errors[1])))
        assert hvp_ratio <= 1e-5

        # FD-noise-immune symmetry identity <Hu, v> = <Hv, u>:
        # breaks for step-pairing/contraction bugs.
        hvp_fn = fn.derivatives().hvp
        assert hvp_fn is not None
        other = bkd.asarray(np.array([[-0.2], [0.9], [0.3]]))
        h_dir = bkd.flatten(hvp_fn(param, direction))
        h_other = bkd.flatten(hvp_fn(param, other))
        bkd.assert_allclose(
            bkd.sum(h_dir * bkd.flatten(other)),
            bkd.sum(h_other * bkd.flatten(direction)),
            rtol=1e-12,
        )

"""Transient adjoint gradient + HVP for ADR with log-KLE diffusivity.

DerivativeChecker-validated dQ/dp and d^2Q/dp^2 v with parameters the
coefficients of a lognormal KLE of the diffusivity field
(kappa = exp(W theta + mean), hand-built modes — type-identical to the
SPDE/bilaplacian factory products). The exp map makes param_param_hvp,
state_param_hvp, and param_state_hvp all nonzero; the cubic-reaction
variant additionally makes state_state_hvp nonzero, so every
second-derivative pathway of the HVP recursion is exercised.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Optional, Tuple

import numpy as np
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration.bc_time_residual_adapter import (
    GalerkinBCEnforcingHVPResidual,
    create_galerkin_bc_enforcing_residual,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithHVPAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.galerkin_diffusivity import (
    AffineDiffusivityFieldParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.rootfinding.newton import NewtonSolver

from tests._helpers.adjoint_checks import HVPOperatorFunction, NumpyArray

_NPARAMS = 3
# Explicit steppers need dt below the diffusion stability limit
# (~2/lambda_max(M^-1 K) ~ 1.6e-3 for nx=10 P1 with kappa ~ 1).
_METHOD_TIMES = {
    "backward_euler": (0.4, 0.1),
    "crank_nicolson": (0.4, 0.1),
    "forward_euler": (2.5e-3, 5e-4),
    "heun": (2.5e-3, 5e-4),
}


def _lognormal_kle_map(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]
) -> TransformedFieldMap[NumpyArray]:
    """kappa = exp(W theta): hand-built smooth modes on the DOF nodes."""
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
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
    bkd: NumpyBkd,
    method: str,
    nonlinear_reaction: bool,
    final_time: Optional[float] = None,
) -> Tuple[
    TimeIntegrator[NumpyArray],
    Any,
    AdvectionDiffusionReaction[NumpyArray],
]:
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    reaction = (
        CallableReaction(
            lambda x, u: u**2,
            lambda x, u: 2.0 * u,
            lambda x, u: np.full_like(u, 2.0),
        )
        if nonlinear_reaction
        else None
    )
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        reaction=reaction,
        forcing=lambda x: np.ones(x.shape[1]),
        boundary_conditions=[
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
        ],
    )
    param = AffineDiffusivityFieldParameterization(
        physics, _lognormal_kle_map(bkd, basis), bkd
    )
    adapter = create_galerkin_physics_ode_residual(physics, param)
    assert isinstance(adapter, GalerkinPhysicsToODEResidualWithHVPAdapter)
    stepper = create_stepper(method, adapter)
    wrapper = create_galerkin_bc_enforcing_residual(stepper, physics, bkd)
    assert isinstance(wrapper, GalerkinBCEnforcingHVPResidual)
    newton = NewtonSolver(wrapper)
    newton.set_options(maxiters=20, atol=1e-12, rtol=0.0)
    default_final_time, deltat = _METHOD_TIMES[method]
    if final_time is None:
        final_time = default_final_time
    integrator = TimeIntegrator(0.0, final_time, deltat, newton)
    return integrator, adapter, physics


class TestADRLogKLEAdjointHVP:
    @pytest.mark.parametrize("nonlinear_reaction", [False, True])
    @pytest.mark.parametrize(
        "method",
        ["backward_euler", "crank_nicolson", "forward_euler", "heun"],
    )
    def test_endpoint_gradient_and_hvp_match_fd(
        self, numpy_bkd: NumpyBkd, method: str, nonlinear_reaction: bool
    ) -> None:
        bkd = numpy_bkd
        integrator, adapter, physics = _build_pipeline(
            bkd, method, nonlinear_reaction
        )
        nstates = physics.nstates()
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        functional = EndpointFunctional(state_idx, nstates, _NPARAMS, bkd)
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        y0 = bkd.asarray(np.zeros(nstates))
        fn = HVPOperatorFunction(operator, adapter, y0, bkd)
        checker = DerivativeChecker(fn)

        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        direction = bkd.asarray(np.array([[0.5], [0.7], [-0.6]]))
        errors = checker.check_derivatives(
            param, direction=direction, relative=True
        )

        jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert jac_ratio <= 1e-6
        # The HVP FD check one-sided-differences an iteratively solved
        # gradient, so its error floor (~1e-7) sits above the usual
        # 1e-6 ratio; assert the V-bottom (a genuine bug plateaus
        # orders of magnitude higher) and a looser ratio.
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

    @pytest.mark.parametrize("method", ["backward_euler", "crank_nicolson"])
    def test_gradient_and_hvp_nonuniform_dt(
        self, numpy_bkd: NumpyBkd, method: str
    ) -> None:
        """Non-uniform last step (T=0.35, dt=0.1): backward-sweep
        methods must use their ctx argument, not stale bound step
        state — invisible under uniform dt."""
        bkd = numpy_bkd
        integrator, adapter, physics = _build_pipeline(
            bkd, method, nonlinear_reaction=True, final_time=0.35
        )
        nstates = physics.nstates()
        constrained = set(
            int(d) for d in bkd.to_numpy(physics.constraint_set().dofs())
        )
        state_idx = next(
            ii for ii in range(nstates) if ii not in constrained
        )
        functional = EndpointFunctional(state_idx, nstates, _NPARAMS, bkd)
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        y0 = bkd.asarray(np.zeros(nstates))
        fn = HVPOperatorFunction(operator, adapter, y0, bkd)
        checker = DerivativeChecker(fn)

        param = bkd.asarray(np.array([[0.4], [-0.3], [0.2]]))
        direction = bkd.asarray(np.array([[0.5], [0.7], [-0.6]]))
        errors = checker.check_derivatives(
            param, direction=direction, relative=True
        )
        # V-bottom + looser ratio: the min/max denominator is the
        # eps=1e-13 roundoff blowup, which varies by configuration.
        jac_min = float(bkd.to_numpy(bkd.min(errors[0])))
        assert jac_min <= 1e-7
        jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        assert jac_ratio <= 1e-5
        hvp_min = float(bkd.to_numpy(bkd.min(errors[1])))
        assert hvp_min <= 1e-6
        hvp_ratio = float(bkd.to_numpy(checker.error_ratio(errors[1])))
        assert hvp_ratio <= 1e-5

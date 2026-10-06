"""Adjoint gradients with collocation periodic boundary conditions.

A periodic condition prescribes no value: it relates the primary and partner
boundary nodes (value match on the primary row, derivative match on the
partner row). Both nodes' values depend on the parameters, so neither belongs
to the essential set E (``dy_E/dp = 0``); both rows are replaced, so both
belong to R.

The check needs no derivation: the adjoint gradient is compared with finite
differences of the forward model.
"""

import math
from typing import Any

import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import PeriodicBC
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.models.collocation.transient import TransientForwardModel
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.util.backends.protocols import Array, Backend

_NPTS = 15


def _model(bkd: Backend[Array], method: str, qoi_node: str) -> Any:
    mesh = TransformedMesh1D(_NPTS, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()
    physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
    left = mesh.boundary_indices(0)
    right = mesh.boundary_indices(1)
    physics.set_boundary_conditions(
        [PeriodicBC(bkd, left, right, basis.derivative_matrix())]
    )
    # D(x) = 1 + p_0 + p_1 cos(pi x): periodic, so the problem is too.
    field_map = BasisExpansion(
        bkd, 1.0, [bkd.ones((_NPTS,)), bkd.cos(math.pi * nodes)]
    )
    param = create_diffusion_parameterization(physics, bkd, field_map)
    nstates = physics.nstates()
    state_idx = (
        bkd.to_int(left[0]) if qoi_node == "primary" else nstates // 3
    )
    functional = EndpointFunctional(state_idx, nstates, param.nparams(), bkd)
    time_config = TimeIntegrationConfig(
        method=method,
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-12,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )
    # A periodic initial state that is not an equilibrium.
    init_state = 1.0 + bkd.cos(math.pi * nodes) + 0.5 * bkd.sin(math.pi * nodes)
    return TransientForwardModel(
        physics, bkd, init_state, time_config,
        functional=functional, parameterization=param,
    )


def _steady_model(bkd: Backend[Array]) -> Any:
    """Steady periodic diffusion-reaction; Q is the primary-node value.

    The reaction makes the periodic problem nonsingular.
    """
    from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
        WeightedSumFunctional,
    )
    from pyapprox.pde.models.collocation.steady import SteadyForwardModel

    mesh = TransformedMesh1D(_NPTS, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()
    forcing = 1.0 + bkd.cos(math.pi * nodes) + 0.5 * bkd.sin(math.pi * nodes)
    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=1.0, reaction=1.0, forcing=lambda t: forcing
    )
    left = mesh.boundary_indices(0)
    right = mesh.boundary_indices(1)
    physics.set_boundary_conditions(
        [PeriodicBC(bkd, left, right, basis.derivative_matrix())]
    )
    field_map = BasisExpansion(
        bkd, 1.0, [bkd.ones((_NPTS,)), bkd.cos(math.pi * nodes)]
    )
    param = create_diffusion_parameterization(physics, bkd, field_map)
    weights = bkd.copy(bkd.zeros((physics.nstates(), 1)))
    weights[bkd.to_int(left[0])] = 1.0
    functional = WeightedSumFunctional(weights, param.nparams(), bkd)
    return SteadyForwardModel(
        physics, bkd, bkd.zeros((physics.nstates(),)),
        functional=functional, parameterization=param,
    )


class TestPeriodicSteadyDerivatives:
    def test_gradient_and_hvp_match_finite_differences(
        self, bkd: Backend[Array]
    ) -> None:
        """The steady HVP zeroes the adjoint on every replaced row; the
        periodic partner rows must be among them. (Without them the HVP
        error plateaus near 1.1; the gradient is unaffected.)"""
        model = _steady_model(bkd)
        assert model.derivatives().hvp is not None
        checker = DerivativeChecker(model)
        grad_errors, hvp_errors = checker.check_derivatives(
            bkd.array([0.3, 0.2])[:, None], relative=True
        )
        # The steady solver's default tolerance puts the floor of the
        # one-sided sweep near 1e-6, and both bounds sit just above the
        # worst value measured rather than at a round number, so the
        # sweep still has to reach its floor instead of passing a
        # gradient that is actually wrong. Measurements: the floor
        # reaches 2.4e-6 on CI's macOS runners against 1.2e-6 on this
        # machine, and the ratio 1.3e-5 under torch on x86_64 Linux
        # against under 1e-5 elsewhere.
        for errors in (grad_errors, hvp_errors):
            assert float(bkd.to_numpy(bkd.min(errors))) <= 3e-6
            assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 1.5e-5


class TestPeriodicAdjointGradient:
    @pytest.mark.parametrize("qoi_node", ["primary", "interior"])
    @pytest.mark.parametrize(
        "method", ["crank_nicolson", "backward_euler"]
    )
    def test_gradient_matches_finite_differences(
        self, bkd: Backend[Array], method: str, qoi_node: str
    ) -> None:
        model = _model(bkd, method, qoi_node)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=model.nvars(),
            fun=model,
            jacobian=model.derivatives().jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            bkd.array([0.3, 0.2])[:, None], relative=True
        )[0]
        # A misclassified row plateaus at O(1); a correct gradient's
        # one-sided finite-difference sweep bottoms near 1e-8.
        assert float(bkd.to_numpy(bkd.min(errors))) <= 1e-6
        assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 1e-5

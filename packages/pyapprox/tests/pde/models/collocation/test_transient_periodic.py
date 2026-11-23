r"""Adjoint gradients with collocation periodic rows.

A periodic relation prescribes no value: it matches the primary and partner
boundary nodes' values on the primary row and their derivatives on the
partner row. Both values depend on the parameters, so neither node is
essential; both rows are closure rows.

Diffusion :math:`1 + p_0 + p_1 \cos(\pi x)` (periodic, so the problem is
too) on 15 Chebyshev nodes, initial state
:math:`1 + \cos(\pi x) + 0.5 \sin(\pi x)` (periodic, not an equilibrium),
five steps of 0.02. The endpoint value at the primary node and at an
interior node; BE and CN, with the step-end pin and the pin at every
block. The adjoint gradient is compared with central differences: a
misclassified row plateaus at O(1).
"""

import math
from typing import Callable

import pytest

from pyapprox.implicitfunction.implicit_map import (
    AdjointRows,
    ArrayParameters,
    ImplicitMapFunction,
)
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import AtEveryBlock, AtStepEnd, PinPlacementProtocol
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import ButcherTableau, backward_euler, crank_nicolson
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TransientImplicitMap,
    TransientPoint,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary.periodic_rows import (
    PeriodicPart,
    PeriodicRows,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.transient_system import (
    collocation_transient_system,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton

_NPTS = 15

_TABLEAUX: dict[str, Callable[[], ButcherTableau]] = {
    "backward_euler": backward_euler,
    "crank_nicolson": crank_nicolson,
}
_PLACEMENTS: dict[str, Callable[[], PinPlacementProtocol]] = {
    "step_end": AtStepEnd,
    "every_block": AtEveryBlock,
}


def _function(
    bkd: Backend[Array], method: str, placement: str, qoi_node: str
) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
    mesh = TransformedMesh1D(_NPTS, bkd)
    nodes = mesh.points()[0]
    primary, partner = mesh.boundary_indices(0), mesh.boundary_indices(1)
    physics = AdvectionDiffusionReaction(
        ChebyshevBasis1D(mesh, bkd), bkd, diffusion=1.0
    )
    periodic = PeriodicRows(
        bkd,
        StateLayout.uniform(_NPTS, 1),
        mesh.reference_points(),
        [
            PeriodicPart(
                primary,
                partner,
                ChebyshevBasis1D(mesh, bkd).derivative_matrix(1, 0),
            )
        ],
    )
    system: CollocationSystem[Array] = CollocationSystem(physics, None, [periodic])
    parameters = ParameterState(
        create_diffusion_parameterization(
            physics,
            bkd,
            BasisExpansion(bkd, 1.0, [bkd.ones((_NPTS,)), bkd.cos(math.pi * nodes)]),
        )
    )
    transient = collocation_transient_system(
        system, _PLACEMENTS[placement](), None, parameters
    )
    entry = bkd.to_int(primary[0]) if qoi_node == "primary" else _NPTS // 3
    init = bkd.reshape(
        1.0 + bkd.cos(math.pi * nodes) + 0.5 * bkd.sin(math.pi * nodes), (-1, 1)
    )
    newton: Newton[Array, Array] = Newton(atol=1e-12, rtol=0.0, maxiters=20)
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd,
            1,
            state_terms=(EndpointTerm(StateEntries(bkd, _NPTS, [entry])),),
        )
    )
    implicit = TransientImplicitMap(
        transient,
        transient,
        RungeKutta(_TABLEAUX[method](), newton),
        RecomputeAlways(transient.solved.solver),
        FixedStep(0.02, (0.0, 0.1)),
        init,
        functional,
    )
    return ImplicitMapFunction(implicit, ArrayParameters(bkd), AdjointRows())


@pytest.mark.parametrize("qoi_node", ["primary", "interior"])
@pytest.mark.parametrize("placement", list(_PLACEMENTS))
@pytest.mark.parametrize("method", list(_TABLEAUX))
def test_gradient_matches_central_differences(
    bkd: Backend[Array], method: str, placement: str, qoi_node: str
) -> None:
    function = _function(bkd, method, placement, qoi_node)
    checker = DerivativeChecker(function)
    errors = checker.check_derivatives(
        bkd.asarray([[0.3], [0.2]]),
        direction=bkd.asarray([[0.6], [-0.8]]),
        relative=True,
        central=True,
    )[0]
    assert bkd.to_float(bkd.min(errors)) <= 1e-6
    assert bkd.to_float(checker.error_ratio(errors)) <= 1e-6

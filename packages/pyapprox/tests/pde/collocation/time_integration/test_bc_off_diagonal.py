"""The collocation BC wrapper's off-diagonal block as an operator.

Crank-Nicolson on a Chebyshev diffusion problem with Dirichlet ends: the
wrapper's operator applies the sensitivity block with its row-replaced
rows zeroed, and its transpose at the next step's context the adjoint
block with those columns zeroed, as the matrix methods do today.
"""

import math

import numpy as np

from pyapprox.ode.implicit_steppers.crank_nicolson import CrankNicolsonHVP
from pyapprox.ode.step_context import StepContext
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.mesh import TransformedMesh1D, create_uniform_mesh_1d
from pyapprox.pde.collocation.physics import AdvectionDiffusionReaction
from pyapprox.pde.collocation.time_integration import (
    BCEnforcingAdjointResidual,
    CollocationPhysicsToODEResidualAdapter,
    create_bc_enforcing_residual,
)
from pyapprox.util.backends.protocols import Array, Backend

_NPTS = 9


def test_rows_zeroed_and_its_transpose_match_todays_blocks(
    bkd: Backend[Array],
) -> None:
    basis = ChebyshevBasis1D(TransformedMesh1D(_NPTS, bkd), bkd)
    umesh = create_uniform_mesh_1d(_NPTS, (-1.0, 1.0), bkd)
    physics = AdvectionDiffusionReaction(basis, bkd, diffusion=0.5)
    physics.set_boundary_conditions(
        [
            zero_dirichlet_bc(bkd, umesh.boundary_indices(0)),
            zero_dirichlet_bc(bkd, umesh.boundary_indices(1)),
        ]
    )
    stepper = CrankNicolsonHVP(CollocationPhysicsToODEResidualAdapter(physics, bkd))
    wrapper = create_bc_enforcing_residual(stepper, physics, bkd)
    assert isinstance(wrapper, BCEnforcingAdjointResidual)
    nodes = basis.nodes()
    ctx = StepContext(t_prev=0.1, deltat=0.05, y_prev=bkd.sin(math.pi * nodes))
    y_curr = bkd.cos(0.5 * math.pi * nodes)
    x = bkd.asarray(np.random.default_rng(6).normal(size=(_NPTS, 2)))
    op = wrapper.off_diag_operator(ctx, y_curr)
    block = wrapper.sensitivity_off_diag_jacobian(ctx, y_curr)
    bkd.assert_allclose(op.apply(x), bkd.dot(block, x), rtol=1e-12, atol=1e-12)
    next_ctx = StepContext(t_prev=0.15, deltat=0.05, y_prev=y_curr)
    y_next = bkd.sin(0.3 * math.pi * nodes)
    adjoint = wrapper.adjoint_off_diag_jacobian(next_ctx, y_next)
    bkd.assert_allclose(
        wrapper.off_diag_operator(next_ctx, y_next).apply_transpose(x),
        bkd.dot(adjoint, x),
        rtol=1e-12,
        atol=1e-12,
    )

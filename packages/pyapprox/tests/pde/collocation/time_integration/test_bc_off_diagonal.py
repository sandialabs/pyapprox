"""The collocation BC wrapper's off-diagonal block as an operator.

Crank-Nicolson on a Chebyshev diffusion problem with Dirichlet ends: the
wrapper's operator applies the inner stepper's block with the Dirichlet
rows zeroed, and its transpose applies the inner transpose to a block
whose Dirichlet entries are zeroed (the block's columns zeroed).
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


def test_rows_zeroed_and_its_transpose_columns_zeroed(
    bkd: Backend[Array],
) -> None:
    basis = ChebyshevBasis1D(TransformedMesh1D(_NPTS, bkd), bkd)
    umesh = create_uniform_mesh_1d(_NPTS, (-1.0, 1.0), bkd)
    left, right = umesh.boundary_indices(0), umesh.boundary_indices(1)
    physics = AdvectionDiffusionReaction(basis, bkd, diffusion=0.5)
    physics.set_boundary_conditions(
        [zero_dirichlet_bc(bkd, left), zero_dirichlet_bc(bkd, right)]
    )
    stepper = CrankNicolsonHVP(CollocationPhysicsToODEResidualAdapter(physics, bkd))
    wrapper = create_bc_enforcing_residual(stepper, physics, bkd)
    assert isinstance(wrapper, BCEnforcingAdjointResidual)
    keep = np.ones((_NPTS, 1))
    keep[np.concatenate([bkd.to_numpy(left), bkd.to_numpy(right)])] = 0.0
    mask = bkd.asarray(keep)
    nodes = basis.nodes()
    ctx = StepContext(t_prev=0.1, deltat=0.05, y_prev=bkd.sin(math.pi * nodes))
    y_curr = bkd.cos(0.5 * math.pi * nodes)
    x = bkd.asarray(np.random.default_rng(6).normal(size=(_NPTS, 2)))
    op = wrapper.off_diag_operator(ctx, y_curr)
    inner = stepper.off_diag_operator(ctx, y_curr)
    assert float(bkd.to_numpy(bkd.sum(bkd.abs(inner.apply(x) * (1.0 - mask))))) > 0
    bkd.assert_allclose(op.apply(x), mask * inner.apply(x), rtol=1e-12, atol=1e-12)
    bkd.assert_allclose(
        op.apply_transpose(x),
        inner.apply_transpose(mask * x),
        rtol=1e-12,
        atol=1e-12,
    )

"""The steppers' off-diagonal blocks as operators.

For each stepper, :math:`B_n = \\partial R_n/\\partial y_{n-1}` as an
operator must agree with finite differences of the step residual in the
previous state (so the formulas are checked independently), with the
matrix the stepper returns today in both directions (forward
sensitivity, and transposed at the next step's context for the adjoint),
and pass a dot test. Dense on both backends with a nonsymmetric mass,
and SciPy-sparse on a Galerkin problem.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerHVP
from pyapprox.ode.explicit_steppers.heun import HeunHVP
from pyapprox.ode.implicit_steppers import (
    BackwardEulerHVP,
    CrankNicolsonHVP,
    ImplicitMidpointHVP,
)
from pyapprox.ode.protocols.ode_residual import ODEResidualProtocol
from pyapprox.ode.protocols.time_stepping import (
    HVPEnabledTimeSteppingResidualProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available
from tests._helpers.ode_models import LotkaVolterra

_MASS = [[2.0, 0.5], [0.3, 1.0]]
_NAMES = ["backward-euler", "crank-nicolson", "midpoint", "forward-euler", "heun"]


def _stepper(
    name: str, model: ODEResidualProtocol[Array]
) -> HVPEnabledTimeSteppingResidualProtocol[Array]:
    """The method's HVP-tier stepper."""
    if name == "backward-euler":
        return BackwardEulerHVP(model)
    if name == "crank-nicolson":
        return CrankNicolsonHVP(model)
    if name == "midpoint":
        return ImplicitMidpointHVP(model)
    if name == "forward-euler":
        return ForwardEulerHVP(model)
    return HeunHVP(model)


def _ctx(bkd: Backend[Array], y_prev: list[float], t_prev: float) -> StepContext[Array]:
    return StepContext(t_prev=t_prev, deltat=0.1, y_prev=bkd.asarray(y_prev))


@pytest.mark.parametrize("name", _NAMES)
class TestDenseOffDiagonal:
    def test_matches_finite_differences(self, bkd: Backend[Array], name: str) -> None:
        r""":math:`B_n v` against central differences of :math:`R_n` in
        :math:`y_{n-1}`, with :math:`y_n` held fixed."""
        stepper = _stepper(name, LotkaVolterra(bkd, mass=bkd.asarray(_MASS)))
        y_prev, y_curr = [9.0, 4.5], bkd.asarray([10.0, 5.0])
        direction, step = np.array([0.3, -0.8]), 1e-6

        def residual(shift: float) -> Array:
            stepper.bind(_ctx(bkd, list(np.array(y_prev) + shift * direction), 0.2))
            value: Array = stepper(y_curr)
            return value

        difference = (residual(step) - residual(-step)) / (2 * step)
        op = stepper.off_diag_operator(_ctx(bkd, y_prev, 0.2), y_curr)
        bkd.assert_allclose(
            op.apply(bkd.asarray(direction[:, None]))[:, 0],
            difference,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_matches_todays_blocks_and_transposes(
        self, bkd: Backend[Array], name: str
    ) -> None:
        stepper = _stepper(name, LotkaVolterra(bkd, mass=bkd.asarray(_MASS)))
        ctx, y_curr = _ctx(bkd, [9.0, 4.5], 0.2), bkd.asarray([10.0, 5.0])
        next_ctx, y_next = _ctx(bkd, [10.0, 5.0], 0.3), bkd.asarray([10.4, 5.3])
        rng = np.random.default_rng(3)
        x, y = (
            bkd.asarray(rng.normal(size=(2, 3))),
            bkd.asarray(rng.normal(size=(2, 3))),
        )
        op = stepper.off_diag_operator(ctx, y_curr)
        block = stepper.sensitivity_off_diag_jacobian(ctx, y_curr)
        bkd.assert_allclose(op.apply(x), bkd.dot(block, x), rtol=1e-12)
        bkd.assert_allclose(op.apply_transpose(y), bkd.dot(block.T, y), rtol=1e-12)
        adjoint = stepper.adjoint_off_diag_jacobian(next_ctx, y_next)
        bkd.assert_allclose(
            stepper.off_diag_operator(next_ctx, y_next).apply_transpose(y),
            bkd.dot(adjoint, y),
            rtol=1e-12,
        )
        # Dot test: <B x, y> = <x, B^T y>.
        bkd.assert_allclose(
            bkd.sum(op.apply(x) * y), bkd.sum(x * op.apply_transpose(y)), rtol=1e-12
        )


if package_available("skfem"):
    from pyapprox.pde.galerkin.basis import LagrangeBasis
    from pyapprox.pde.galerkin.compose import compose_galerkin_system
    from pyapprox.pde.galerkin.mesh import StructuredMesh1D
    from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
    from pyapprox.pde.galerkin.time_integration import (
        GalerkinPhysicsToODEResidualAdapter,
    )
    from pyapprox.util.backends.numpy import NumpyBkd


@pytest.mark.skipif(not package_available("skfem"), reason="skfem not installed")
@pytest.mark.parametrize("name", ["crank-nicolson", "forward-euler", "heun"])
def test_sparse_galerkin_blocks(numpy_bkd: "NumpyBkd", name: str) -> None:
    """With SciPy-sparse J and M (a Galerkin advection-diffusion-reaction
    adapter), the operator applies the same block as today's matrix,
    which is sparse for Crank-Nicolson and forward Euler and dense for
    Heun."""
    bkd = numpy_bkd
    mesh = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
    physics = AdvectionDiffusionReaction(
        basis=LagrangeBasis(mesh, degree=2), diffusivity=0.5, bkd=bkd, reaction=2.0
    )
    adapter = GalerkinPhysicsToODEResidualAdapter(compose_galerkin_system(physics))
    stepper = _stepper(name, adapter)
    n = adapter.mass_matrix().as_matrix().shape[0]
    y_prev = bkd.asarray(np.linspace(0.0, 1.0, n))
    y_curr = bkd.asarray(np.linspace(0.2, 0.9, n))
    ctx = StepContext(t_prev=0.0, deltat=0.05, y_prev=y_prev)
    x = np.random.default_rng(4).normal(size=(n, 2))
    block: NDArray[Any] = np.asarray(
        stepper.sensitivity_off_diag_jacobian(ctx, y_curr) @ x
    )
    op = stepper.off_diag_operator(ctx, y_curr)
    bkd.assert_allclose(op.apply(x), block, rtol=1e-12, atol=1e-12)
    transposed = np.asarray(stepper.sensitivity_off_diag_jacobian(ctx, y_curr).T @ x)
    bkd.assert_allclose(op.apply_transpose(x), transposed, rtol=1e-12, atol=1e-12)

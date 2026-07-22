"""Tests for explicit steppers with non-identity mass matrices.

The Heun stage is y_{n-1} + dt*M^{-1}*k1 (the ODE is M*dy/dt = f), and
its sensitivity block must differentiate the same scheme. Identity-mass
behavior is covered implicitly: mass.solve degenerates to a no-op.
"""

from typing import Any

import numpy as np
from pyapprox.ode.explicit_steppers.heun import HeunHVP
from pyapprox.ode.mass_matrix import MassMatrixProtocol, create_mass_matrix
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Backend


class _LinearResidual:
    """f(y, t) = A y + t b with mass M (dense), ODEResidualProtocol."""

    def __init__(self, bkd: Backend[Any]) -> None:
        self._bkd = bkd
        self._amat = bkd.asarray(
            np.array([[-2.0, 1.0, 0.0], [1.0, -3.0, 1.0], [0.0, 1.0, -1.5]])
        )
        self._bvec = bkd.asarray(np.array([0.5, -1.0, 2.0]))
        self._mass = create_mass_matrix(
            bkd.asarray(np.diag([2.0, 3.0, 4.0])), bkd
        )
        self._time = 0.0

    def bkd(self) -> Backend[Any]:
        return self._bkd

    def set_time(self, time: float) -> None:
        self._time = time

    def __call__(self, state: Any) -> Any:
        return self._bkd.dot(self._amat, state) + self._time * self._bvec

    def jacobian(self, state: Any) -> Any:
        return self._amat

    def mass_matrix(self) -> MassMatrixProtocol[Any]:
        return self._mass


class TestHeunWithMassMatrix:
    def _bind(
        self, stepper: HeunHVP[Any], y_prev: Any
    ) -> StepContext[Any]:
        ctx = StepContext(t_prev=0.3, deltat=0.1, y_prev=y_prev)
        stepper.bind(ctx)
        return ctx

    def test_stage_uses_mass_solve(self, bkd: Backend[Any]) -> None:
        """One Heun step matches the hand-computed M^{-1}-staged update."""
        residual = _LinearResidual(bkd)
        stepper = HeunHVP(residual)
        y_prev = bkd.asarray(np.array([1.0, -0.5, 2.0]))
        ctx = self._bind(stepper, y_prev)

        # solve R(y) = 0 via the stepper's own one-step machinery
        y_new = y_prev - stepper.linsolve(y_prev, stepper(y_prev))

        # hand-computed reference
        minv = np.diag(1.0 / np.array([2.0, 3.0, 4.0]))
        amat = bkd.to_numpy(residual.jacobian(y_prev))
        bvec = np.array([0.5, -1.0, 2.0])
        y0 = bkd.to_numpy(y_prev)
        k1 = amat @ y0 + ctx.t_prev * bvec
        stage = y0 + ctx.deltat * minv @ k1
        k2 = amat @ stage + ctx.t_curr * bvec
        expected = y0 + 0.5 * ctx.deltat * minv @ (k1 + k2)

        bkd.assert_allclose(y_new, bkd.asarray(expected), rtol=1e-13)

    def test_sensitivity_off_diag_matches_finite_difference(
        self, bkd: Backend[Any]
    ) -> None:
        """dR/dy_prev from the formula matches central differences."""
        residual = _LinearResidual(bkd)
        stepper = HeunHVP(residual)
        y_prev = bkd.asarray(np.array([1.0, -0.5, 2.0]))
        y_curr = bkd.asarray(np.array([0.8, -0.2, 1.9]))
        ctx = self._bind(stepper, y_prev)

        analytic = stepper.sensitivity_off_diag_jacobian(ctx, y_curr)

        eps = 1e-7
        n = 3
        fd = np.zeros((n, n))
        y0 = bkd.to_numpy(y_prev)
        for jj in range(n):
            for sign in (1.0, -1.0):
                y_pert = y0.copy()
                y_pert[jj] += sign * eps
                ctx_pert = StepContext(
                    t_prev=ctx.t_prev,
                    deltat=ctx.deltat,
                    y_prev=bkd.asarray(y_pert),
                )
                stepper.bind(ctx_pert)
                fd[:, jj] += sign * bkd.to_numpy(stepper(y_curr)) / (2 * eps)

        bkd.assert_allclose(
            bkd.asarray(analytic), bkd.asarray(fd), rtol=1e-6, atol=1e-8
        )

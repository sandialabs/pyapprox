"""Adjoint gradient and HVP when parameters set the initial state.

ODE: ``M dy/dt = -p * y + s`` (elementwise), with a dense non-symmetric mass
``M``. Q is an endpoint value. Three initial states:

- ``fixed``:  ``y_0 = y_base``, independent of p (a control: this path was
  already exercised, so it checks the harness);
- ``affine``: ``y_0 = y_base + J p``;
- ``curved``: ``y_0 = y_base + J p + c * p**2 / 2`` (elementwise), which
  needs the residual's stated initial-state curvature.

The references need no derivation: the gradient is compared with finite
differences of the forward solve, and the HVP with finite differences of the
gradient, rebuilding ``y_0(p)`` for every perturbation.
"""

from typing import Any

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.mass_matrix import ConstantDenseMassMatrix
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver

_NSTATES = 3
_BASE_PARAM = np.array([0.7, 1.2, 0.4])
_Y_BASE = np.array([1.0, -0.5, 0.8])
_SOURCE = np.array([0.5, -0.3, 0.2])
_IC_JACOBIAN = np.array(
    [[1.0, 0.3, 0.0], [-0.4, 0.8, 0.2], [0.1, 0.0, -0.6]]
)
_IC_CURVATURE = np.array([0.5, -1.2, 0.8])
# Non-symmetric, so a transposed or missing mass cannot pass.
_MASS = (
    np.eye(_NSTATES)
    + 0.3 * np.diag(np.ones(_NSTATES - 1), 1)
    + 0.1 * np.diag(np.ones(_NSTATES - 1), -1)
)

_IC_KINDS = {
    "fixed": (np.zeros((_NSTATES, _NSTATES)), np.zeros(_NSTATES)),
    "affine": (_IC_JACOBIAN, np.zeros(_NSTATES)),
    "curved": (_IC_JACOBIAN, _IC_CURVATURE),
}


class _DecayResidualWithoutCurvature(DefaultNewtonJacobianMixin):
    """``f(y; p) = -p * y + s`` with the initial state
    ``y_0 = y_base + J p + c * p**2 / 2``; states no initial curvature."""

    def __init__(
        self, bkd: Backend[Array], ic_jacobian: Array, ic_curvature: Array
    ) -> None:
        self._bkd = bkd
        self._mass = ConstantDenseMassMatrix(bkd.asarray(_MASS), bkd)
        self._source = bkd.asarray(_SOURCE)
        self._ic_jacobian = ic_jacobian
        self._ic_curvature = ic_curvature
        self._param = bkd.zeros((_NSTATES,))

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_time(self, time: float) -> None:
        pass

    def __call__(self, state: Array) -> Array:
        return -self._param * state + self._source

    def jacobian(self, state: Array) -> Array:
        return -self._bkd.diag(self._param)

    def mass_matrix(self) -> Any:
        return self._mass

    def nparams(self) -> int:
        return _NSTATES

    def set_param(self, param: Array) -> None:
        self._param = self._bkd.flatten(param)

    def param_jacobian(self, state: Array) -> Array:
        return -self._bkd.diag(state)

    def initial_param_jacobian(self) -> Array:
        return self._ic_jacobian + self._bkd.diag(self._ic_curvature * self._param)

    def state_state_hvp(self, state: Array, adj_state: Array, wvec: Array) -> Array:
        return self._bkd.zeros((_NSTATES,))

    def state_param_hvp(self, state: Array, adj_state: Array, vvec: Array) -> Array:
        return -adj_state * self._bkd.flatten(vvec)

    def param_state_hvp(self, state: Array, adj_state: Array, wvec: Array) -> Array:
        return -adj_state * self._bkd.flatten(wvec)

    def param_param_hvp(self, state: Array, adj_state: Array, vvec: Array) -> Array:
        return self._bkd.zeros((_NSTATES,))


class _DecayResidual(_DecayResidualWithoutCurvature):
    """Adds the stated initial-state curvature, ``(w * c * v)_i``."""

    def initial_param_hvp(self, weight: Array, vvec: Array) -> Array:
        return weight * self._ic_curvature * self._bkd.flatten(vvec)


def _initial_state(bkd: Backend[Array], kind: str, param: Array) -> Array:
    jac, curv = _IC_KINDS[kind]
    p = bkd.flatten(param)
    return (
        bkd.asarray(_Y_BASE)
        + bkd.asarray(jac) @ p
        + 0.5 * bkd.asarray(curv) * p * p
    )


def _operator(
    bkd: Backend[Array], method: str, kind: str, residual_cls: Any = _DecayResidual
) -> TimeAdjointOperatorWithHVP[Array]:
    jac, curv = _IC_KINDS[kind]
    residual = residual_cls(bkd, bkd.asarray(jac), bkd.asarray(curv))
    stepper = create_stepper(method, residual)
    newton = NewtonSolver(stepper)
    newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
    integrator = TimeIntegrator(0.0, 0.5, 0.05, newton)
    functional = EndpointFunctional(1, _NSTATES, _NSTATES, bkd)
    return TimeAdjointOperatorWithHVP(integrator, functional)


_METHODS = ["backward_euler", "crank_nicolson", "implicit_midpoint"]


class TestInitialStateParameters:
    @pytest.mark.parametrize("kind", list(_IC_KINDS))
    @pytest.mark.parametrize("method", _METHODS)
    def test_gradient_and_hvp(
        self, bkd: Backend[Array], method: str, kind: str
    ) -> None:
        """The checker perturbs p along one direction v and compares finite
        differences of Q with the adjoint gradient, and finite differences
        of the gradient with H v."""
        base = bkd.asarray(_BASE_PARAM.reshape(-1, 1))

        def fun(samples: Array) -> Array:
            # One column per perturbed parameter; y_0 moves with p.
            vals = []
            for ii in range(samples.shape[1]):
                p = samples[:, ii : ii + 1]
                op = _operator(bkd, method, kind)
                vals.append(op(_initial_state(bkd, kind, p), p))
            return bkd.hstack(vals)

        def jacobian(sample: Array) -> Array:
            op = _operator(bkd, method, kind)
            return op.jacobian(_initial_state(bkd, kind, sample), sample)

        def hvp(sample: Array, vec: Array) -> Array:
            op = _operator(bkd, method, kind)
            return op.hvp(_initial_state(bkd, kind, sample), sample, vec)

        wrapper = FunctionWithJacobianAndHVPFromCallable(
            nvars=_NSTATES, fun=fun, jacobian=jacobian, hvp=hvp, bkd=bkd
        )
        checker = DerivativeChecker(wrapper)
        grad_errors, hvp_errors = checker.check_derivatives(base, relative=True)
        # The finite-difference sweep of an iterative solve bottoms near
        # 1e-8, which puts the ratio astride 1e-6; a missing term plateaus
        # at O(1) instead.
        for errors in (grad_errors, hvp_errors):
            assert float(bkd.to_numpy(bkd.min(errors))) <= 1e-7
            assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 5e-6

    def test_hvp_refused_without_stated_curvature(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """A residual that does not state its initial-state curvature gets
        no HVP: it raises instead of silently assuming zero."""
        bkd = numpy_bkd
        base = bkd.asarray(_BASE_PARAM.reshape(-1, 1))
        op = _operator(
            bkd, "crank_nicolson", "curved", _DecayResidualWithoutCurvature
        )
        y0 = _initial_state(bkd, "curved", base)
        op.jacobian(y0, base)  # the gradient needs no curvature
        with pytest.raises(TypeError):
            op.hvp(y0, base, bkd.ones((_NSTATES, 1)))

"""Adjoint gradient + HVP stepper validation with non-identity masses.

The identity-mass HVP suite (test_adjoint_hvp.py) cannot catch
second-derivative terms illegitimately entangled with the mass matrix
(M is linear in the state, so no second-derivative contraction carries
it) or sweep solves that silently assume M = I. Parametrize the
quadratic ODE (nonzero d^2f/dy^2) over consistent (SPD dense) and
lumped (diagonal) masses for every stepper, validating the assembled
gradient and HVP against FD plus the FD-noise-immune HVP symmetry
identity.
"""

import numpy as np
import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerHVP
from pyapprox.ode.explicit_steppers.heun import HeunHVP
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerHVP
from pyapprox.ode.implicit_steppers.crank_nicolson import CrankNicolsonHVP
from pyapprox.ode.implicit_steppers.implicit_midpoint import (
    ImplicitMidpointHVP,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.mass_matrix import (
    ConstantDenseMassMatrix,
    DiagonalMassMatrix,
    MassMatrixProtocol,
)
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver
from pyapprox_benchmarks.functions.ode.linear_ode import QuadraticODEResidual


class _OperatorFunction:
    """Adapt TimeAdjointOperatorWithHVP to FunctionProtocol
    (backend-generic; the operator sets parameters internally)."""

    def __init__(self, operator, init_state, bkd):
        self._operator = operator
        self._init_state = init_state
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._operator.nparams()

    def __call__(self, samples: Array) -> Array:
        self._operator.storage()._clear()
        return self._operator(self._init_state, samples)

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.second_order(jacobian=self._jacobian, hvp=self._hvp)

    def _jacobian(self, param: Array) -> Array:
        self._operator.storage()._clear()
        return self._operator.jacobian(self._init_state, param)

    def _hvp(self, param: Array, vvec: Array) -> Array:
        self._operator.storage()._clear()
        return self._operator.hvp(self._init_state, param, vvec).T


def _make_mass(bkd, kind: str) -> MassMatrixProtocol:
    if kind == "consistent":
        return ConstantDenseMassMatrix(
            bkd.asarray(np.array([[2.0, 0.5], [0.5, 1.5]])), bkd
        )
    return DiagonalMassMatrix(bkd.asarray(np.array([2.0, 0.5])), bkd)


class TestAdjointHVPMassMatrix:
    @pytest.mark.parametrize("mass_kind", ["consistent", "lumped"])
    @pytest.mark.parametrize(
        "stepper_class",
        [
            BackwardEulerHVP,
            CrankNicolsonHVP,
            ImplicitMidpointHVP,
            ForwardEulerHVP,
            HeunHVP,
        ],
    )
    def test_gradient_and_hvp_match_fd(
        self, bkd, stepper_class, mass_kind
    ) -> None:
        np.random.seed(42)
        Amat = bkd.asarray(np.array([[-1.0, 0.1], [0.1, -2.0]]))
        ode_residual = QuadraticODEResidual(
            Amat, bkd, mass_matrix=_make_mass(bkd, mass_kind)
        )
        nstates, nparams = 2, ode_residual.nparams()

        time_residual = stepper_class(ode_residual)
        newton_solver = NewtonSolver(time_residual)
        newton_solver.set_options(atol=1e-12, rtol=1e-12)
        integrator = TimeIntegrator(0.0, 0.3, 0.1, newton_solver)

        functional = EndpointFunctional(
            state_idx=0, nstates=nstates, nparams=nparams, bkd=bkd
        )
        operator = TimeAdjointOperatorWithHVP(integrator, functional)

        param = bkd.asarray(np.array([[0.1], [0.5]]))
        init_state = bkd.asarray(np.array([0.5, 0.3]))

        wrapper = _OperatorFunction(operator, init_state, bkd)
        checker = DerivativeChecker(wrapper)
        direction = bkd.asarray(np.random.randn(nparams, 1))
        direction = direction / bkd.norm(direction)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            param, direction=direction, fd_eps=fd_eps, relative=True
        )
        jac_ratio = float(checker.error_ratio(errors[0]))
        assert jac_ratio < 1e-6
        hvp_ratio = float(checker.error_ratio(errors[1]))
        assert hvp_ratio < 1e-6

        # FD-noise-immune symmetry identity <Hu, v> = <Hv, u>.
        other = bkd.asarray(np.array([[-0.7], [0.4]]))
        operator.storage()._clear()
        h_dir = operator.hvp(init_state, param, direction)
        operator.storage()._clear()
        h_other = operator.hvp(init_state, param, other)
        bkd.assert_allclose(
            bkd.sum(bkd.flatten(h_dir) * bkd.flatten(other)),
            bkd.sum(bkd.flatten(h_other) * bkd.flatten(direction)),
            rtol=1e-10,
        )

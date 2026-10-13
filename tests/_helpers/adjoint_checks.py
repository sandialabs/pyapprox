"""Shared harness pieces for adjoint/HVP DerivativeChecker tests."""

from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.util.backends.numpy import NumpyBkd

NumpyArray = NDArray[Any]


class NoHVPQuadraticFieldMap:
    """Field map with genuine curvature but no declared hvp.

    The one honestly first-order case: field = mean + (W p)^2 has
    nonzero parameter curvature that the map does not provide, so
    consumers must degrade instead of assuming zeros. Used by the
    HVP-tier negative tests.
    """

    def __init__(
        self, bkd: NumpyBkd, mean: NumpyArray, modes: NumpyArray
    ) -> None:
        self._bkd = bkd
        self._mean = mean
        self._W = modes

    def nvars(self) -> int:
        return int(self._W.shape[1])

    def __call__(self, params_1d: NumpyArray) -> NumpyArray:
        z = self._bkd.dot(self._W, params_1d)
        return self._mean + z * z

    def jacobian(self, params_1d: NumpyArray) -> NumpyArray:
        z = self._bkd.dot(self._W, params_1d)
        return 2.0 * z[:, None] * self._W


def solve_based_derivatives_pass_fd(model: Any, sample: NumpyArray) -> List[bool]:
    """Finite-difference check of each declared derivative of a model
    whose every evaluation goes through a solve.

    The solve's results are at roundoff, and its floor sits near an error
    ratio of 1e-6, so whether the usual ``ratio <= 1e-6`` or a V-shape fit
    passes depends on the random direction. Each derivative must instead
    satisfy both:

    - error ratio at most 5e-6 over steps 1e-13 to 1e-1, the bound used
      for noise-limited checks elsewhere;
    - first-order convergence over steps 1e-1 to 1e-5, the truncation
      regime, which is clean for every direction; a wrong derivative
      does not converge there (``convergence_order`` then raises, which
      fails the test).
    """
    bkd = model.bkd()
    checker = DerivativeChecker(model)
    fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
    passed = []
    for errors in checker.check_derivatives(
        sample, fd_eps=fd_eps, relative=True
    ):
        ratio = bkd.to_float(checker.error_ratio(errors))
        # fd_eps descends, so the first five steps are 1e-1 to 1e-5.
        order = bkd.to_float(
            checker.convergence_order(errors[:5], fd_eps[:5])
        )
        passed.append(ratio <= 5e-6 and abs(order - 1.0) < 0.1)
    return passed


class HVPOperatorFunction:
    """Adapt TimeAdjointOperatorWithHVP to FunctionProtocol for
    DerivativeChecker: fixes the initial state, re-applies parameters
    through the ODE adapter before every evaluation, and declares a
    second-order Derivatives bundle (jacobian + hvp).
    """

    def __init__(
        self,
        operator: "TimeAdjointOperatorWithHVP[NumpyArray]",
        adapter: Any,
        init_state: NumpyArray,
        bkd: NumpyBkd,
    ) -> None:
        self._operator = operator
        self._adapter = adapter
        self._init_state = init_state
        self._bkd = bkd

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._operator.nparams()

    def _prepare(self, params: NumpyArray) -> None:
        self._adapter.set_param(params[:, 0])
        self._operator.storage()._clear()

    def __call__(self, samples: NumpyArray) -> NumpyArray:
        self._prepare(samples)
        return self._operator(self._init_state, samples)

    def derivatives(self) -> Derivatives[NumpyArray]:
        return Derivatives.second_order(
            jacobian=self._jacobian, hvp=self._hvp
        )

    def _jacobian(self, params: NumpyArray) -> NumpyArray:
        self._prepare(params)
        return self._operator.jacobian(self._init_state, params)

    def _hvp(self, params: NumpyArray, vvec: NumpyArray) -> NumpyArray:
        self._prepare(params)
        return np.asarray(
            self._operator.hvp(self._init_state, params, vvec)
        ).T

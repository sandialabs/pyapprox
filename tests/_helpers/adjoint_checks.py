"""Shared harness pieces for adjoint/HVP DerivativeChecker tests."""

from typing import Any, Generic, List, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
    VShapeReport,
    predicted_rounding,
)
from pyapprox.interface.functions.derivative_checks.resolve import resolve_bundle
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

NumpyArray = NDArray[Any]


class NoHVPQuadraticFieldMap(Generic[Array]):
    """Field map with genuine curvature but no declared hvp.

    The one honestly first-order case: field = mean + (W p)^2 has
    nonzero parameter curvature that the map does not provide, so
    consumers must degrade instead of assuming zeros. Used by the
    HVP-tier negative tests.
    """

    def __init__(self, bkd: Backend[Array], mean: Array, modes: Array) -> None:
        self._bkd = bkd
        self._mean = mean
        self._W = modes

    def nvars(self) -> int:
        return int(self._W.shape[1])

    def __call__(self, params_1d: Array) -> Array:
        z = self._bkd.dot(self._W, params_1d)
        return self._mean + z * z

    def jacobian(self, params_1d: Array) -> Array:
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
    for errors in checker.check_derivatives(sample, fd_eps=fd_eps, relative=True):
        ratio = bkd.to_float(checker.error_ratio(errors))
        # fd_eps descends, so the first five steps are 1e-1 to 1e-5.
        order = bkd.to_float(checker.convergence_order(errors[:5], fd_eps[:5]))
        passed.append(ratio <= 5e-6 and abs(order - 1.0) < 0.1)
    return passed


def central_derivatives_pass_fd(
    function: FunctionProtocol[NumpyArray],
    sample: NumpyArray,
    direction: NumpyArray,
    weights: Optional[NumpyArray] = None,
) -> List[VShapeReport]:
    """Central-difference check of a function's Jacobian and, when it has
    one, its (weighted) Hessian-vector product, along ``direction``, over
    steps 1e-1 to 1e-13; one report per derivative, each to be asserted
    ``passed``.

    Each curve is judged against its predicted rounding
    (``predicted_rounding``), so the second-order fit covers the two
    decades above where the error nears it and stays off a floor that
    moves with the solve and the platform. A curve that is rounding
    throughout (a quantity affine or quadratic along the direction)
    passes as truncation free. The sizes are the value and the Jacobian's
    action for the Jacobian, and the weighted gradient and the product for
    the Hessian; ``weights`` defaults to ones, as the checker's.
    """
    bkd = function.bkd()
    steps = [10.0**-k for k in range(1, 14)]
    checker = DerivativeChecker(function)
    derivatives = resolve_bundle(function)
    jacobian = derivatives.jacobian
    if jacobian is None:
        raise ValueError("the function declares no Jacobian")
    if weights is None:
        weights = bkd.ones((function.nqoi(), 1))
    errors = checker.check_derivatives(
        sample,
        fd_eps=bkd.asarray(steps),
        direction=direction,
        relative=True,
        central=True,
        weights=weights,
    )
    gradient = jacobian(sample)
    sizes = [(function(sample), bkd.dot(gradient, direction))]
    whvp = derivatives.resolved_whvp(function.nqoi())
    if whvp is not None:
        sizes.append((bkd.dot(gradient.T, weights), whvp(sample, direction, weights)))
    eps = float(np.finfo(bkd.to_numpy(sample).dtype).eps)
    point_size = bkd.to_float(bkd.norm(sample))
    reports = []
    for error, (value, action) in zip(errors, sizes):
        rounding = predicted_rounding(
            eps,
            bkd.to_float(bkd.norm(value)),
            bkd.to_float(bkd.norm(action)),
            point_size,
            steps,
        )
        curve = checker.curve(
            error, bkd.asarray(steps), central=True, rounding=rounding
        )
        reports.append(curve.v_shape())
    return reports


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
        return Derivatives.second_order(jacobian=self._jacobian, hvp=self._hvp)

    def _jacobian(self, params: NumpyArray) -> NumpyArray:
        self._prepare(params)
        return self._operator.jacobian(self._init_state, params)

    def _hvp(self, params: NumpyArray, vvec: NumpyArray) -> NumpyArray:
        self._prepare(params)
        return np.asarray(self._operator.hvp(self._init_state, params, vvec)).T

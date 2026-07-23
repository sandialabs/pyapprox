"""Shared harness pieces for adjoint/HVP DerivativeChecker tests."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
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

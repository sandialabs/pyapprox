r"""The weighted squared norm :math:`q = y^\top W y` as an integrand.

On four states with the symmetric positive-definite tridiagonal weight
:math:`W = I + 0.3 (E_{+1} + E_{-1})`, at a random state:

- the value is :math:`y^\top W y` to 1e-14;
- the state derivative passes central finite differences of the value,
  and the weighted second derivative (weight 1) passes central finite
  differences of the gradient; the weight :math:`\omega` scales the
  second derivative linearly;
- the integrand reads no parameters, and refuses a weight that is not a
  ``SymmetricWeightProtocol``.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.integrands import WeightedSquaredNorm
from pyapprox.ode.time_integral import IntegrandProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.symmetric_weight import DenseMatrixWeight

_N = 4


def _matrix(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.eye(_N) + 0.3 * (np.eye(_N, k=1) + np.eye(_N, k=-1)))


def _integrand(bkd: Backend[Array]) -> WeightedSquaredNorm[Array]:
    return WeightedSquaredNorm(DenseMatrixWeight(_matrix(bkd), bkd), bkd)


class TestWeightedSquaredNorm:
    def test_value(self, bkd: Backend[Array]) -> None:
        integrand = _integrand(bkd)
        assert isinstance(integrand, IntegrandProtocol)
        assert integrand.nqoi() == 1
        state = bkd.asarray(np.random.default_rng(5).normal(size=(_N, 1)))
        bkd.assert_allclose(
            integrand.value(state, 0.3),
            bkd.dot(state.T, bkd.dot(_matrix(bkd), state)),
            rtol=1e-14,
        )

    def test_derivatives_pass_finite_differences(self, bkd: Backend[Array]) -> None:
        integrand = _integrand(bkd)
        derivatives = integrand.derivatives()
        assert derivatives.param is None
        state_fn, curvature = derivatives.state, derivatives.state_state
        assert state_fn is not None and curvature is not None
        one = bkd.ones((1, 1))

        def values(samples: Array) -> Array:
            return bkd.hstack(
                [
                    integrand.value(samples[:, k : k + 1], 0.0)
                    for k in range(samples.shape[1])
                ]
            )

        def gradient(sample: Array) -> Array:
            return state_fn(sample, 0.0).apply_transpose(one)

        def jacobian(sample: Array) -> Array:
            return gradient(sample).T

        def gradients(samples: Array) -> Array:
            return bkd.hstack(
                [gradient(samples[:, k : k + 1]) for k in range(samples.shape[1])]
            )

        def hvp(sample: Array, direction: Array) -> Array:
            return curvature(sample, 0.0, one, direction)

        state = bkd.asarray(np.random.default_rng(5).normal(size=(_N, 1)))
        first = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=1, nvars=_N, fun=values, jacobian=jacobian, bkd=bkd
            )
        ).check_derivatives(state, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(first)) < 1e-7
        second = DerivativeChecker(
            FunctionWithJVPFromCallable(
                nqoi=_N, nvars=_N, fun=gradients, jvp=hvp, bkd=bkd
            )
        ).check_derivatives(state, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(second)) < 1e-7
        direction = bkd.asarray(np.random.default_rng(6).normal(size=(_N, 1)))
        bkd.assert_allclose(
            curvature(state, 0.0, 2.5 * one, direction),
            2.5 * hvp(state, direction),
            rtol=1e-14,
        )


class TestWeightedSquaredNormRefusals:
    def test_refuses_a_bare_matrix(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(TypeError, match="SymmetricWeightProtocol"):
            WeightedSquaredNorm(
                _matrix(numpy_bkd),  # type: ignore[arg-type]
                numpy_bkd,
            )

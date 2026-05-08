"""Quadratic test functions for benchmarking.

Simple sum-of-squares functions with analytical derivatives,
useful for verifying parallel execution, optimization, and
derivative-checking infrastructure.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


def square(x: int) -> int:
    """Square a number. Picklable helper for parallel map tests."""
    return x * x


def add(x: int, y: int) -> int:
    """Add two numbers. Picklable helper for parallel starmap tests."""
    return x + y


class QuadraticFunction(Generic[Array]):
    """Sum-of-squares function: f(x) = sum(x_i^2).

    Single-output (nqoi=1) with analytical jacobian, hessian, hvp, and whvp.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    nvars : int
        Number of input variables.
    """

    def __init__(self, bkd: Backend[Array], nvars: int = 3) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._eye = bkd.eye(nvars)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return self._bkd.sum(samples**2, axis=0, keepdims=True)

    def jacobian(self, sample: Array) -> Array:
        return 2 * sample.T

    def hessian(self, sample: Array) -> Array:
        return 2 * self._eye

    def hvp(self, sample: Array, vec: Array) -> Array:
        return 2 * vec

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        return weights[0, 0] * 2 * vec


class DiagonalQuadraticFunction(Generic[Array]):
    """Multi-output diagonal quadratic: f_i(x) = x_i^2.

    Each output depends on exactly one input variable.
    nqoi = nvars = 2.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 2

    def __call__(self, samples: Array) -> Array:
        x = samples[0:1, :]
        y = samples[1:2, :]
        return self._bkd.vstack([x**2, y**2])

    def jacobian(self, sample: Array) -> Array:
        x = sample[0, 0]
        y = sample[1, 0]
        return self._bkd.asarray([[2 * x, 0.0], [0.0, 2 * y]])

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        w0 = weights[0, 0]
        w1 = weights[1, 0]
        result = self._bkd.asarray([[2 * w0 * vec[0, 0]], [2 * w1 * vec[1, 0]]])
        return result

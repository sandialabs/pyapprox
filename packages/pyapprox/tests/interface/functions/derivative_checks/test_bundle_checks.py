r"""The array bundle-check tools catch what they exist to catch.

On :math:`q(x) = x^2` (entrywise, three entries, at
:math:`x = (0.3, -1.2, 0.8)`) and the 3x3 matrix
:math:`A_{ij} = i - 2j + \delta_{ij}`:

- ``fd_check``: the true action :math:`2 x \odot v` passes the report's
  default tolerances; :math:`3 x \odot v` and a stated zero fail.
- ``dot_test_error``: :math:`A` with its transpose is round-off; with
  :math:`A` as its own "transpose" it is of order one.
- ``transpose_errors``: an operator declaring itself symmetric gets a
  ``"(symmetric)"`` entry, round-off when true and of order one when false.
- ``refetched`` returns a present field and refuses an absent one.
"""

from typing import Callable, Generic, Optional

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.bundle_checks import (
    BundleCheckReport,
    dot_test_error,
    fd_check,
    refetched,
    transpose_errors,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import backend_dtype_name
from pyapprox.util.linalg.operators import OperatorProperties, SpaceDescriptor


class _Matrices(Generic[Array]):
    """A linear map given by one matrix for its action and another for its
    transpose's, so a wrong transpose or a false symmetry can be built."""

    def __init__(
        self,
        bkd: Backend[Array],
        forward: Array,
        transpose: Array,
        symmetric: bool = False,
    ) -> None:
        self._forward, self._transpose = forward, transpose
        self._space = SpaceDescriptor("x", forward.shape[1], backend_dtype_name(bkd))
        self._properties = OperatorProperties(symmetric=symmetric)

    def domain(self) -> SpaceDescriptor:
        return self._space

    def range(self) -> SpaceDescriptor:
        return self._space

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        return self._forward @ x

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        return self._transpose @ y

    def properties(self) -> OperatorProperties:
        return self._properties


def _matrix(bkd: Backend[Array]) -> Array:
    return bkd.asarray(
        [[float(i - 2 * j + (i == j)) for j in range(3)] for i in range(3)]
    )


def _square(x: Array) -> Array:
    return x**2


def _scaled_action(factor: float) -> Callable[[Array, Array], Array]:
    def action(x: Array, v: Array) -> Array:
        return factor * x[:, None] * v

    return action


def _fd_failures(bkd: Backend[Array], action: Callable[[Array, Array], Array]) -> int:
    point = bkd.asarray([0.3, -1.2, 0.8])
    curve = fd_check(bkd, point, _square, action, np.random.default_rng(0))
    return len(BundleCheckReport({"q": curve}, {}).failures())


class TestFDCheck:
    def test_the_true_action_passes(self, bkd: Backend[Array]) -> None:
        assert _fd_failures(bkd, _scaled_action(2.0)) == 0

    def test_a_wrong_action_fails(self, bkd: Backend[Array]) -> None:
        assert _fd_failures(bkd, _scaled_action(3.0)) == 1

    def test_a_false_stated_zero_fails(self, bkd: Backend[Array]) -> None:
        assert _fd_failures(bkd, _scaled_action(0.0)) == 1


class TestDotTest:
    def test_a_true_transpose(self, bkd: Backend[Array]) -> None:
        a = _matrix(bkd)
        error = dot_test_error(bkd, _Matrices(bkd, a, a.T), np.random.default_rng(0))
        assert error < 1e-14

    def test_a_wrong_transpose(self, bkd: Backend[Array]) -> None:
        a = _matrix(bkd)
        error = dot_test_error(bkd, _Matrices(bkd, a, a), np.random.default_rng(0))
        assert error > 1e-2


class TestSymmetry:
    def test_only_a_declared_symmetry_is_checked(self, bkd: Backend[Array]) -> None:
        a = _matrix(bkd)
        s = a + a.T
        errors = transpose_errors(
            bkd,
            [
                ("plain", _Matrices(bkd, a, a.T)),
                ("true", _Matrices(bkd, s, s, symmetric=True)),
                ("false", _Matrices(bkd, a, a.T, symmetric=True)),
            ],
            np.random.default_rng(0),
        )
        assert set(errors) == {
            "plain",
            "true",
            "true (symmetric)",
            "false",
            "false (symmetric)",
        }
        assert errors["true (symmetric)"] < 1e-14
        assert errors["false (symmetric)"] > 1e-2
        report = BundleCheckReport({}, errors)
        assert report.failures() == ["false (symmetric)"]


class TestRefetched:
    def test_present_and_absent(self) -> None:
        assert refetched(3, "jacobian") == 3
        with pytest.raises(ValueError, match="lacks 'jacobian'"):
            refetched(None, "jacobian")

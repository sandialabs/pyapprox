"""Tests for ScipyOptimizerResultWrapper."""

import numpy as np
from scipy.optimize import OptimizeResult

from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestMessage:
    def test_status_and_iterations_are_appended(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """Whether the solver stepped at all is what tells a bad first
        step from a stall, and the text alone does not say."""
        result = OptimizeResult(
            x=np.zeros(2),
            fun=0.0,
            success=False,
            status=8,
            nit=0,
            message="Positive directional derivative for linesearch",
        )
        message = ScipyOptimizerResultWrapper(result, numpy_bkd).message()
        assert message == (
            "Positive directional derivative for linesearch (status 8, nit 0)"
        )

    def test_a_result_without_them_keeps_its_text(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        result = OptimizeResult(
            x=np.zeros(2), fun=0.0, success=True, message="done"
        )
        assert ScipyOptimizerResultWrapper(result, numpy_bkd).message() == "done"

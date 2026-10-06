from typing import Generic

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestDerivativeChecker:
    def test_derivative_checker(self, bkd) -> None:
        """
        Test the derivative checker for a simple quadratic function.
        """

        # Define the value function
        def value_function(x):
            return bkd.reshape(x[0] ** 3 + x[1] ** 2, (1, x.shape[1]))

        def jacobian_function(x):
            return bkd.stack([3 * x[0] ** 2, 2 * x[1]], axis=1)

        def hvp_function(x, v):
            return bkd.stack([6 * x[0] * v[0], 2 * v[1]], axis=0)

        # Wrap the function using FunctionWithJacobianAndHVPFromCallable
        function_object = FunctionWithJacobianAndHVPFromCallable(
            nvars=2,
            fun=value_function,
            jacobian=jacobian_function,
            hvp=hvp_function,
            bkd=bkd,
        )

        # Initialize DerivativeChecker
        checker = DerivativeChecker(function_object)

        # Define a sample point
        sample = bkd.asarray([[2.0, 1.0]]).T

        # Check derivatives
        errors = checker.check_derivatives(sample)

        # Assert that the gradient errors are below a tolerance
        assert checker.error_ratio(errors[0]) <= 1e-5

        # Assert that the Hessian errors are below a tolerance
        assert checker.error_ratio(errors[1]) <= 1e-5

    def test_hvp_checked_at_a_weight(self, bkd) -> None:
        """A single QoI's hvp, weighted as a constraint's multiplier."""
        checker = DerivativeChecker(_SingleQoI(bkd, declares_hvp=True))
        errors = checker.check_derivatives(
            bkd.asarray([[2.0], [1.0]]), weights=bkd.asarray([[0.5]])
        )
        assert checker.error_ratio(errors[1]) <= 1e-5


class _SingleQoI(Generic[Array]):
    """``x0^3 + x1^2``, one QoI, with its curvature declared as asked.

    A constraint with one row still has a multiplier, so its curvature
    reaches an optimizer weighted; this is that case.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        declares_hvp: bool = False,
        ignores_weight: bool = False,
    ) -> None:
        self._bkd = bkd
        self._ignores_weight = ignores_weight
        self._derivs = (
            Derivatives.second_order(jacobian=self.jacobian, hvp=self.hvp)
            if declares_hvp
            else Derivatives.second_order_weighted(
                jacobian=self.jacobian, whvp=self.whvp
            )
        )

    def derivatives(self) -> Derivatives[Array]:
        return self._derivs

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        return (samples[0] ** 3 + samples[1] ** 2)[None, :]

    def jacobian(self, sample: Array) -> Array:
        x = sample[:, 0]
        return self._bkd.stack([3 * x[0] ** 2, 2 * x[1]])[None, :]

    def hvp(self, sample: Array, vec: Array) -> Array:
        x = sample[:, 0]
        return self._bkd.stack([6 * x[0] * vec[0], 2 * vec[1]])

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        weight = 1.0 if self._ignores_weight else weights[0, 0]
        return weight * self.hvp(sample, vec)


class TestWeightedSingleQoI:
    """The checker weights the gradient and its derivative alike.

    It used to difference the unweighted gradient of a single QoI while
    comparing against the weighted whvp, so they differed by exactly the
    weight: a correct whvp failed at any weight but one, and a whvp that
    ignored its weight passed.
    """

    def test_a_correct_whvp_passes_at_any_weight(
        self, bkd: Backend[Array]
    ) -> None:
        checker = DerivativeChecker(_SingleQoI(bkd))
        errors = checker.check_derivatives(
            bkd.asarray([[2.0], [1.0]]), weights=bkd.asarray([[0.5]])
        )
        assert checker.error_ratio(errors[1]) <= 1e-5

    def test_a_whvp_ignoring_its_weight_fails(
        self, bkd: Backend[Array]
    ) -> None:
        checker = DerivativeChecker(_SingleQoI(bkd, ignores_weight=True))
        errors = checker.check_derivatives(
            bkd.asarray([[2.0], [1.0]]), weights=bkd.asarray([[0.5]])
        )
        assert checker.error_ratio(errors[1]) > 1e-1

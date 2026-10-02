"""Assert that declared boundary time derivatives are derivatives.

The framework trusts analytic time derivatives (it never finite
differences them), so a wrong ``s_dot`` or ``s_ddot`` would surface only
indirectly, as lost convergence. This checks each declared order
against a finite difference of the order below, through the public
``time_derivative_functions`` view and ``DerivativeChecker``.

Pass criterion: the SMALLEST finite-difference error over the step
sizes is below ``tol``. The usual error ratio is unusable here: for a
signal polynomial in time, finite differences are exact at every step,
so the ratio is about one even when the derivative is right. A wrong
derivative never becomes small at any step.
"""

from typing import Callable, Optional, Sequence

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.boundary import time_derivative_functions
from pyapprox.util.backends.protocols import Array, Backend


def assert_time_derivatives_match(
    values: Callable[[float], Array],
    derivative_of_order: Callable[[int], Optional[Callable[[float], Array]]],
    norders: int,
    ndofs: int,
    bkd: Backend[Array],
    times: Sequence[float] = (0.3, 0.71),
    tol: float = 1e-6,
) -> None:
    """Raise AssertionError unless orders ``1..norders`` match FD.

    Parameters
    ----------
    values, derivative_of_order
        ``bc.constrained_values`` and ``bc.constrained_values_derivative``,
        or ``constraint_set.values`` and ``constraint_set.values_derivative``.
    norders : int
        Highest order to check. An absent order raises ValueError.
    ndofs : int
        Number of constrained DOFs.
    times : sequence of float
        Times at which to check.
    tol : float
        Bound on the smallest finite-difference error.
    """
    functions = time_derivative_functions(
        values, derivative_of_order, norders, ndofs, bkd
    )
    for order, function in enumerate(functions, start=1):
        checker = DerivativeChecker(function)
        for time in times:
            errors = checker.check_derivatives(bkd.asarray([[time]]))[0]
            min_error = bkd.to_float(bkd.min(errors))
            if not min_error <= tol:
                raise AssertionError(
                    f"time derivative of order {order} at t={time} does "
                    f"not match finite differences of order {order - 1}: "
                    f"smallest FD error {min_error:.3e} > {tol:.1e}"
                )

"""Boundary time derivatives as functions, for derivative checking.

A signal's time derivatives are callables ``t -> Array`` of shape
``(ndofs,)``; ``DerivativeChecker`` checks a ``FunctionProtocol`` of a
sample ``(nvars, 1)``. ``TimeDerivativeFunction`` bridges the two: it
presents the ``(k - 1)``-th time derivative as a function of time (one
variable) whose Jacobian is the ``k``-th, so the declared analytic
derivatives of any essential BC or constraint set can be checked
against finite differences with the standard checker:

>>> functions = time_derivative_functions(
...     bc.constrained_values, bc.constrained_values_derivative,
...     norders=2, ndofs=bc.constrained_dofs().shape[0], bkd=bkd,
... )
>>> for function in functions:
...     errors = DerivativeChecker(function).check_derivatives(
...         bkd.asarray([[0.3]])
...     )[0]

Time enters the BC API as a float, so the sample is converted to one;
this view does not carry autograd through time, which the
finite-difference checker does not need.
"""

from typing import Callable, Generic, List, Optional

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.validation import (
    validate_sample,
    validate_samples,
)
from pyapprox.util.backends.protocols import Array, Backend


class TimeDerivativeFunction(Generic[Array]):
    """``t -> s^(k-1)(t)`` with Jacobian ``s^(k)(t)``.

    Satisfies ``ObjectiveProtocol`` with ``nvars() == 1`` (the time) and
    ``nqoi() == ndofs``.

    Parameters
    ----------
    values : Callable[[float], Array]
        ``s^(k-1)``: maps a time to values of shape (ndofs,).
    derivative : Callable[[float], Array]
        ``s^(k)``: its claimed analytic time derivative, same shape.
    ndofs : int
        Number of constrained DOFs.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        values: Callable[[float], Array],
        derivative: Callable[[float], Array],
        ndofs: int,
        bkd: Backend[Array],
    ) -> None:
        if ndofs <= 0:
            raise ValueError(f"ndofs must be positive, got {ndofs}")
        self._values = values
        self._derivative = derivative
        self._ndofs = ndofs
        self._bkd = bkd
        self._derivs: Derivatives[Array] = Derivatives.first_order(
            jacobian=self.jacobian
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return 1: the only variable is the time."""
        return 1

    def nqoi(self) -> int:
        """Return the number of constrained DOFs."""
        return self._ndofs

    def derivatives(self) -> Derivatives[Array]:
        """Return the bundle declaring the Jacobian ``s^(k)``."""
        return self._derivs

    def _column(self, func: Callable[[float], Array], sample: Array) -> Array:
        """Evaluate ``func`` at the sample's time as an (ndofs, 1) column."""
        vals = func(self._bkd.to_float(sample[0, 0]))
        if vals.shape != (self._ndofs,):
            raise ValueError(
                f"expected values of shape ({self._ndofs},), got {vals.shape}"
            )
        return self._bkd.reshape(vals, (self._ndofs, 1))

    def __call__(self, samples: Array) -> Array:
        """Evaluate ``s^(k-1)`` at each time. Shape: (ndofs, nsamples)."""
        validate_samples(1, samples)
        return self._bkd.hstack(
            [
                self._column(self._values, samples[:, ii : ii + 1])
                for ii in range(samples.shape[1])
            ]
        )

    def jacobian(self, sample: Array) -> Array:
        """Return ``s^(k)`` at the sample's time. Shape: (ndofs, 1)."""
        validate_sample(1, sample)
        return self._column(self._derivative, sample)


def time_derivative_functions(
    values: Callable[[float], Array],
    derivative_of_order: Callable[[int], Optional[Callable[[float], Array]]],
    norders: int,
    ndofs: int,
    bkd: Backend[Array],
) -> List[TimeDerivativeFunction[Array]]:
    """Return one checkable function per derivative order ``1..norders``.

    Entry ``k - 1`` is ``s^(k-1)`` with Jacobian ``s^(k)``, so checking
    every entry checks each declared order against the one below it.

    Parameters
    ----------
    values : Callable[[float], Array]
        ``s``, e.g. ``bc.constrained_values`` or ``constraint_set.values``.
    derivative_of_order : Callable[[int], Optional[Callable]]
        The derivative accessor, e.g. ``bc.constrained_values_derivative``
        or ``constraint_set.values_derivative``.
    norders : int
        Highest order to check, at least 1.
    ndofs : int
        Number of constrained DOFs.
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If an order up to ``norders`` is absent: there is nothing to
        check, and silently skipping it would pass vacuously.
    """
    if norders < 1:
        raise ValueError(f"norders must be at least 1, got {norders}")
    functions: List[TimeDerivativeFunction[Array]] = []
    lower = values
    for order in range(1, norders + 1):
        derivative = derivative_of_order(order)
        if derivative is None:
            raise ValueError(
                f"time derivative of order {order} is absent, so it "
                "cannot be checked"
            )
        functions.append(TimeDerivativeFunction(lower, derivative, ndofs, bkd))
        lower = derivative
    return functions

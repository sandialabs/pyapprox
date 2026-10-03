"""The time signal of an essential BC: values and time derivatives.

An essential BC is a selection of rows times a signal on them. The
selection (a boundary's DOFs, component masks, explicit indices) is
discretization-specific; the signal is shared. ``BoundarySignal`` holds
``s(x, t)`` together with its time derivatives indexed by order, each
declared present or absent once, at construction:

- whether ``s`` depends on time is DECLARED (``TimeIndependent`` /
  ``TimeDependent``, or a constant), never inferred. A time-independent
  signal has exactly zero time derivatives of every order, without any
  being supplied;
- a time-dependent signal has the derivatives it was given, and the
  higher orders are absent (``None``), never approximated.

``DofSignal`` is the coordinate-free counterpart for selections given
as explicit DOF indices: its suppliers take only the time.

Supplied derivatives are TRUSTED, like any analytic derivative in a
``Derivatives`` bundle: a wrong one gives wrong numbers, not an error.
To verify them, check each order against finite differences of the one
below with ``time_derivative_functions`` and ``DerivativeChecker``
(``pde.boundary.time_derivative_function``).

The parameter Jacobian ``s_p`` of BC data is reserved for a BC-data
parameterization and not built: it would be one more optional entry on
a signal, so adding it changes no BC class.
"""

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.constitutive.coefficient_functions import (
    TimeAwareCallableProtocol,
    TimeVaryingProtocol,
    as_time_aware,
)

_Values = NDArray[np.floating[Any]]


class _ConstantSignalValue:
    """A constant boundary value as a picklable time-independent supplier."""

    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(self, coords: _Values, time: float) -> _Values:
        return np.full(coords.shape[1], self._value)

    def is_time_dependent(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{self._value!r}"


class _ZeroTimeDerivative:
    """The exact zero time derivative of a time-independent signal.

    Evaluates the values only for their shape, so per-component
    (vector) values get a zero of the matching shape.
    """

    def __init__(self, values: TimeAwareCallableProtocol) -> None:
        self._values = values

    def __call__(self, coords: _Values, time: float) -> _Values:
        return np.zeros_like(
            np.asarray(self._values(coords, time), dtype=np.float64)
        )

    def is_time_dependent(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"zero time derivative of {self._values!r}"


class BoundarySignal:
    """Prescribed values ``s(x, t)`` and their time derivatives by order.

    Parameters
    ----------
    values : float, Callable or TimeAwareCallableProtocol
        ``s``: a constant; a bare ``s(coords)`` (time-independent); or a
        declared ``TimeIndependent(s)`` / ``TimeDependent(s)`` for
        ``s(coords, time)``. A bare callable that could take a time is
        rejected, so time dependence is never guessed.
    time_derivatives : sequence of callables, optional
        ANALYTIC time derivatives, ``time_derivatives[k - 1]`` being the
        ``k``-th, each with the same signature and return shape as
        ``values`` and declared the same way. Only meaningful for a
        time-dependent ``values``; a time-independent one has exact
        zero derivatives automatically.

    Raises
    ------
    ValueError
        If derivatives are supplied for time-independent values.
    TypeError
        If any supplier is a bare callable that could take a time.
    """

    def __init__(
        self,
        values: Union[float, Callable[..., Any]],
        time_derivatives: Sequence[Callable[..., Any]] = (),
    ) -> None:
        if callable(values):
            self._values: TimeAwareCallableProtocol = as_time_aware(values)
        else:
            self._values = _ConstantSignalValue(float(values))
        self._time_derivatives: Tuple[TimeAwareCallableProtocol, ...] = (
            tuple(as_time_aware(deriv) for deriv in time_derivatives)
        )
        if self._time_derivatives and not self._values.is_time_dependent():
            raise ValueError(
                "time derivatives are only meaningful for time-dependent "
                f"values; {self._values!r} is declared time-independent, "
                "so its derivatives are exactly zero automatically"
            )

    def values(self) -> TimeAwareCallableProtocol:
        """Return ``s`` as an ``s(coords, time)`` supplier."""
        return self._values

    def is_time_dependent(self) -> bool:
        """Whether ``s`` is declared to depend on time."""
        return self._values.is_time_dependent()

    def time_derivative(
        self, order: int
    ) -> Optional[TimeAwareCallableProtocol]:
        """Return the ``order``-th time derivative of ``s``, or ``None``.

        Exact zeros for a time-independent signal; ``None`` when a
        time-dependent signal was not given that order.

        Parameters
        ----------
        order : int
            Derivative order, at least 1.
        """
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        if not self.is_time_dependent():
            return _ZeroTimeDerivative(self._values)
        if order > len(self._time_derivatives):
            return None
        return self._time_derivatives[order - 1]

    def norders(self) -> int:
        """Return the number of supplied time derivatives."""
        return len(self._time_derivatives)

    def __repr__(self) -> str:
        return (
            f"BoundarySignal({self._values!r}, "
            f"norders={len(self._time_derivatives)})"
        )


_DofSupplier = Callable[[float], _Values]


class _ZeroDofTimeDerivative:
    """The exact zero time derivative of a time-independent DOF signal."""

    def __init__(self, values: _DofSupplier) -> None:
        self._values = values

    def __call__(self, time: float) -> _Values:
        return np.zeros_like(np.asarray(self._values(time), dtype=np.float64))


class DofSignal:
    """Prescribed values ``s(t)`` at fixed DOFs, with time derivatives.

    The coordinate-free counterpart of ``BoundarySignal``, for
    selections given as explicit DOF indices. Each supplier takes only
    the time, so a plain function is time-dependent; a supplier that
    declares itself time-independent (``TimeVaryingProtocol`` with
    ``is_time_dependent() == False``) is honored, and then has exact
    zero time derivatives.

    Parameters
    ----------
    values : Callable[[float], ndarray]
        ``s(t)``, returning one value per DOF.
    time_derivatives : sequence of Callable[[float], ndarray], optional
        ANALYTIC time derivatives, ``time_derivatives[k - 1]`` being the
        ``k``-th, each with the same signature and return shape as
        ``values``. Orders beyond the sequence are absent. Only
        meaningful for a time-dependent ``values``.

    Raises
    ------
    ValueError
        If derivatives are supplied for declared time-independent values.
    """

    def __init__(
        self,
        values: _DofSupplier,
        time_derivatives: Sequence[_DofSupplier] = (),
    ) -> None:
        self._values = values
        self._time_dependent = not (
            isinstance(values, TimeVaryingProtocol)
            and not values.is_time_dependent()
        )
        self._time_derivatives: Tuple[_DofSupplier, ...] = tuple(
            time_derivatives
        )
        if self._time_derivatives and not self._time_dependent:
            raise ValueError(
                "time derivatives are only meaningful for time-dependent "
                f"values; {values!r} is declared time-independent, so its "
                "derivatives are exactly zero automatically"
            )

    def values(self) -> _DofSupplier:
        """Return ``s`` as an ``s(time)`` supplier."""
        return self._values

    def is_time_dependent(self) -> bool:
        """Whether ``s`` depends on time (plain functions of time do)."""
        return self._time_dependent

    def time_derivative(self, order: int) -> Optional[_DofSupplier]:
        """Return the ``order``-th time derivative of ``s``, or ``None``.

        Exact zeros for a declared time-independent signal.

        Parameters
        ----------
        order : int
            Derivative order, at least 1.
        """
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        if not self._time_dependent:
            return _ZeroDofTimeDerivative(self._values)
        if order > len(self._time_derivatives):
            return None
        return self._time_derivatives[order - 1]

    def norders(self) -> int:
        """Return the number of supplied time derivatives."""
        return len(self._time_derivatives)

    def __repr__(self) -> str:
        return (
            f"DofSignal({self._values!r}, "
            f"norders={len(self._time_derivatives)})"
        )

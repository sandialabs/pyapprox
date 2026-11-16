r"""Coefficients of a collocation physics, given at the collocation nodes.

A collocation physics reads each coefficient and forcing as its values at
the nodes, at a time, and needs to know whether those values vary with
time (to declare its own time dependence) and when they were last
changed. That is :class:`NodalCoefficientProtocol`:

- :class:`NodalField` holds fixed nodal values that a parameterization
  rewrites through ``set_values``; it is the differentiable
  representation, the one a parameterized coefficient must be.
- :class:`NodalSupplier` evaluates a declared ``f(coords, time)``
  supplier (``TimeIndependent`` / ``TimeDependent``) at the nodes.
- :class:`NodalTimeFunction` holds a bare ``f(time)`` returning nodal
  values. It declares nothing, so it is taken as time-dependent, which
  can cost a reassembly or a refused steady solve but never a wrong
  value, and warns with :class:`UndeclaredTimeDependenceWarning` so that
  each such supplier can be found and declared.

:func:`as_nodal_coefficient` turns whatever a physics' constructor was
given into one of these. A physics knows its nodes only by number, so a
declared supplier reaches it already wrapped with the node coordinates.
"""

import warnings
from typing import Callable, Generic, Protocol, Union, runtime_checkable

from pyapprox.pde.constitutive.coefficient_functions import (
    TimeAwareCallableProtocol,
)
from pyapprox.util.backends.protocols import Array, Array_co, Backend


class UndeclaredTimeDependenceWarning(UserWarning):
    """A bare ``f(time)`` coefficient was taken as time-dependent."""


@runtime_checkable
class NodalCoefficientProtocol(Protocol, Generic[Array_co]):
    """A coefficient's values at the collocation nodes."""

    def values(self, time: float) -> Array_co:
        """The values at every node at ``time``. Shape: (npts,)"""
        ...

    def is_time_dependent(self) -> bool:
        """Whether the values are declared to vary with time."""
        ...

    def version(self) -> int:
        """A counter incremented on every change of the values."""
        ...


class NodalField(Generic[Array]):
    """Nodal values fixed in time, rewritten by a parameterization.

    Parameters
    ----------
    values : Array
        The values at the nodes. Shape: (npts,)

    Raises
    ------
    ValueError
        If ``values`` is not 1-D.
    """

    def __init__(self, values: Array) -> None:
        if values.ndim != 1:
            raise ValueError(
                "values must be 1-D with shape (npts,), got shape "
                f"{tuple(values.shape)}"
            )
        self._values = values
        self._npts = values.shape[0]
        self._version = 0

    def npts(self) -> int:
        """The number of nodes."""
        return self._npts

    def set_values(self, values: Array) -> None:
        """Replace the values.

        Raises
        ------
        ValueError
            If ``values`` does not have shape (npts,).
        """
        if values.ndim != 1 or values.shape[0] != self._npts:
            raise ValueError(
                f"values must have shape ({self._npts},), got shape "
                f"{tuple(values.shape)}"
            )
        self._values = values
        self._version += 1

    def values(self, time: float) -> Array:
        """The values; the time is accepted and ignored. Shape: (npts,)"""
        return self._values

    def is_time_dependent(self) -> bool:
        return False

    def version(self) -> int:
        return self._version

    def __repr__(self) -> str:
        return f"{type(self).__name__}(npts={self._npts}, version={self._version})"


class NodalSupplier(Generic[Array]):
    """A declared ``f(coords, time)`` supplier evaluated at the nodes.

    Parameters
    ----------
    supplier : TimeAwareCallableProtocol
        ``TimeIndependent(f)`` or ``TimeDependent(f)``; its declaration
        is the coefficient's.
    nodes : Array
        The physical node coordinates. Shape: (ndim, npts)

    Raises
    ------
    TypeError
        If ``supplier`` does not declare its time dependence.
    """

    def __init__(
        self, supplier: TimeAwareCallableProtocol[Array], nodes: Array
    ) -> None:
        if not isinstance(supplier, TimeAwareCallableProtocol):
            raise TypeError(
                "supplier must declare its time dependence (TimeIndependent "
                f"or TimeDependent), got {type(supplier).__name__}"
            )
        self._supplier = supplier
        self._nodes = nodes

    def values(self, time: float) -> Array:
        """``f(nodes, time)``. Shape: (npts,)"""
        return self._supplier(self._nodes, time)

    def is_time_dependent(self) -> bool:
        return self._supplier.is_time_dependent()

    def version(self) -> int:
        """Zero: the supplier is not rewritten."""
        return 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._supplier!r})"


class NodalTimeFunction(Generic[Array]):
    """A bare ``f(time)`` returning nodal values, taken as time-dependent.

    Warns with :class:`UndeclaredTimeDependenceWarning` on construction:
    replace it with a :class:`NodalField` or a declared
    :class:`NodalSupplier`.

    Parameters
    ----------
    func : callable
        ``time -> values``. Shape of the values: (npts,)
    """

    def __init__(self, func: Callable[[float], Array]) -> None:
        warnings.warn(
            f"the bare coefficient {func!r} declares no time dependence and "
            "is taken as time-dependent; give a constant, nodal values, or "
            "a TimeIndependent/TimeDependent f(coords, time) supplier",
            UndeclaredTimeDependenceWarning,
            stacklevel=3,
        )
        self._func = func

    def values(self, time: float) -> Array:
        """``f(time)``. Shape: (npts,)"""
        return self._func(time)

    def is_time_dependent(self) -> bool:
        return True

    def version(self) -> int:
        """Zero: the function is not rewritten."""
        return 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._func!r})"


CoefficientInput = Union[
    float,
    NodalCoefficientProtocol[Array],
    Callable[[float], Array],
]


def as_nodal_coefficient(
    value: CoefficientInput[Array], npts: int, bkd: Backend[Array]
) -> NodalCoefficientProtocol[Array]:
    """A physics' coefficient argument as a nodal coefficient.

    A constant becomes a :class:`NodalField`; a nodal coefficient (a
    :class:`NodalField` or :class:`NodalSupplier` among them) is kept; a
    bare ``f(time)`` becomes a warning :class:`NodalTimeFunction`.

    Parameters
    ----------
    value : float, NodalCoefficientProtocol or callable
        The coefficient as given.
    npts : int
        The number of nodes.
    bkd : Backend
        The backend of the values.

    Raises
    ------
    TypeError
        If ``value`` is an array of nodal values (it must arrive as
        ``NodalField(values)``), or a declared ``f(coords, time)``
        supplier (the physics has no node coordinates to evaluate it at,
        so it must arrive as ``NodalSupplier(supplier, nodes)``).
    """
    if isinstance(value, (int, float)):
        return NodalField(bkd.full((npts,), float(value)))
    if isinstance(value, NodalCoefficientProtocol):
        return value
    if isinstance(value, bkd.array_type()):
        raise TypeError(
            "nodal values must arrive as NodalField(values), which a "
            "parameterization can rewrite"
        )
    if isinstance(value, TimeAwareCallableProtocol):
        raise TypeError(
            f"{value!r} is an f(coords, time) supplier; pass it as "
            "NodalSupplier(supplier, nodes) with the physical node "
            "coordinates"
        )
    return NodalTimeFunction(value)

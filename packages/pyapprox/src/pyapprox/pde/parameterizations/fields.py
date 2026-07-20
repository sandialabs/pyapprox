"""Picklable field callables set on physics coefficient slots.

Physics coefficient setters take ``Callable[[float], Array]``. This
module provides picklable implementations (stdlib pickle cannot
serialize lambdas; module-level classes pickle by reference).

Coefficients are currently constant in time. If a time-dependent
coefficient is ever needed, the general mechanism is a time-dependent
field map ``G(p, t)`` decided at the facade/engine migration — not a
per-parameterization modulation hook.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array


class ConstantInTimeField(Generic[Array]):
    """Picklable callable returning the same field at every time."""

    def __init__(self, field: Array) -> None:
        self._field = field

    def __call__(self, time: float) -> Array:
        return self._field

r"""Shared cases for testing the linear multistep stepper.

- ``METHODS``: the library's multistep methods with their order and
  whether they are explicit.
- :func:`lm_stepper`: a backend stepper with a tight Newton when the
  method is implicit.
"""

from typing import Callable, List

from pyapprox.ode.linear_multistep import LinearMultistepStepper
from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.multistep_methods import (
    MultistepMethod,
    adams_bashforth2,
    adams_moulton3,
    bdf2,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton

METHODS: List[tuple[Callable[[], MultistepMethod], int, bool]] = [
    (bdf2, 2, False),
    (adams_bashforth2, 2, True),
    (adams_moulton3, 3, False),
]
"""``(method, order, explicit)``."""

METHOD_IDS = [make().name for make, *_ in METHODS]


def lm_stepper(
    method: MultistepMethod,
    system: MassFormSystem[Array, Array, DenseFormat[Array]],
) -> LinearMultistepStepper[Array, Array, DenseFormat[Array]]:
    """A backend stepper, with Newton (atol 1e-13) for implicit methods."""
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
    return LinearMultistepStepper(
        method, system, None if method.is_explicit() else newton
    )

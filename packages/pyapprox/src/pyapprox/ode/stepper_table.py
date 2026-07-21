"""Closed table of built-in time steppers and the typed factory handle.

One frozen name -> stepper-class table shared by every model that
constructs time steppers. There is deliberately NO registration API:
mutable registries carry import-order and drift diseases (the previous
per-model maps had already diverged). Extension uses the typed
``StepperFactory`` handle — a user stepper needs zero registration and
is protocol-checked at construction. Strings exist only because
config-as-data is genuine here (serialized benchmark configs); they are
resolved eagerly against this closed table, never open registration.

All built-ins are the HVP-tier classes: capability narrowing is lazy,
so the richer tier costs nothing for forward-only or gradient-only use.
"""

from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Union

from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerHVP
from pyapprox.ode.explicit_steppers.heun import HeunHVP
from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerHVP
from pyapprox.ode.implicit_steppers.crank_nicolson import CrankNicolsonHVP
from pyapprox.ode.implicit_steppers.implicit_midpoint import (
    ImplicitMidpointHVP,
)
from pyapprox.ode.protocols.ode_residual import ODEResidualProtocol
from pyapprox.ode.protocols.time_stepping import TimeSteppingResidualProtocol
from pyapprox.util.backends.protocols import Array

StepperFactory = Callable[
    [ODEResidualProtocol[Array]], TimeSteppingResidualProtocol[Array]
]
"""Typed extension handle: residual in, protocol-conforming stepper out.

The built-in stepper classes themselves satisfy this signature, so the
table below is a mapping of name -> StepperFactory.
"""

class _BackendGenericStepperFactory(Protocol):
    """Factory usable with EVERY backend array type.

    Module-level variables cannot carry a free type variable, so the
    table's value type is this protocol whose ``__call__`` is itself
    generic — which is precisely what the built-in stepper classes are
    (constructing one binds Array from the residual argument).
    """

    def __call__(
        self, residual: ODEResidualProtocol[Array]
    ) -> TimeSteppingResidualProtocol[Array]: ...


STEPPER_TABLE: Mapping[str, _BackendGenericStepperFactory] = MappingProxyType(
    {
        "backward_euler": BackwardEulerHVP,
        "crank_nicolson": CrankNicolsonHVP,
        "forward_euler": ForwardEulerHVP,
        "heun": HeunHVP,
        "implicit_midpoint": ImplicitMidpointHVP,
    }
)

EXPLICIT_METHOD_NAMES = frozenset({"forward_euler", "heun"})
"""Built-in names whose steppers need no Newton solve."""


def resolve_stepper_factory(
    method: Union[str, StepperFactory[Array]],
) -> StepperFactory[Array]:
    """Resolve a config ``method`` to a stepper factory.

    A callable passes through unchanged (the typed extension path); a
    string is looked up in the closed built-in table.

    Raises
    ------
    ValueError
        If ``method`` is a string not in the table (message lists the
        valid names).
    TypeError
        If ``method`` is neither a string nor a callable.
    """
    if isinstance(method, str):
        if method not in STEPPER_TABLE:
            raise ValueError(
                f"Unknown time integration method: '{method}'. "
                f"Valid names: {sorted(STEPPER_TABLE)}. Custom steppers "
                "are passed as a StepperFactory callable, not by name."
            )
        return STEPPER_TABLE[method]
    if not callable(method):
        raise TypeError(
            "method must be a built-in stepper name or a StepperFactory "
            f"callable, got {type(method).__name__}"
        )
    return method


def create_stepper(
    method: Union[str, StepperFactory[Array]],
    residual: ODEResidualProtocol[Array],
) -> TimeSteppingResidualProtocol[Array]:
    """Construct a stepper from a method name or factory handle.

    The constructed instance is protocol-checked so a misbehaving
    custom factory fails here with an actionable error rather than
    deep inside time integration.
    """
    stepper = resolve_stepper_factory(method)(residual)
    if not isinstance(stepper, TimeSteppingResidualProtocol):
        raise TypeError(
            f"stepper factory {method!r} returned "
            f"{type(stepper).__name__}, which does not satisfy "
            "TimeSteppingResidualProtocol"
        )
    return stepper

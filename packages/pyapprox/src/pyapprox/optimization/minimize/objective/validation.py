from typing import Any, List

from pyapprox.interface.functions.legacy_adapter import (
    as_derivatives,
)
from pyapprox.interface.functions.protocols.function import FunctionProtocol


def _missing_protocol_methods(obj: object, protocol: type) -> List[str]:
    """Return list of protocol methods not implemented by obj."""
    missing = []
    # Get protocol's required methods (excluding private/dunder methods except __call__)
    for name in dir(protocol):
        if name.startswith("_") and name != "__call__":
            continue
        if not hasattr(obj, name):
            missing.append(name)
        elif callable(getattr(protocol, name, None)) and not callable(
            getattr(obj, name, None)
        ):
            missing.append(name)
    return missing


def validate_objective(objective: Any) -> None:
    """
    Validate that the given objective can be consumed by an optimizer.

    INTERIM (derivatives-refactor migration) form: requires the base
    function shape (bkd/nvars/nqoi/__call__) and a well-formed Derivatives
    bundle via ``as_derivatives`` — which accepts both migrated producers
    (``derivatives()``) and legacy producers (capability attributes).
    Once every producer is migrated this tightens to
    ``isinstance(objective, ObjectiveProtocol)``.
    Additionally checks that the objective has exactly one quantity of
    interest (nqoi == 1).

    Parameters
    ----------
    objective : Any
        The objective function to validate.

    Raises
    ------
    TypeError
        If the objective does not have the base function shape, or its
        ``derivatives()`` does not return a Derivatives bundle.
    ValueError
        If the objective does not have exactly one quantity of interest (nqoi != 1).
    """
    if not isinstance(objective, FunctionProtocol):
        missing = _missing_protocol_methods(objective, FunctionProtocol)
        raise TypeError(
            f"Invalid objective type: expected an object implementing the "
            f"ObjectiveProtocol base shape (bkd(), nvars(), nqoi(), "
            f"__call__(samples)), got {type(objective).__name__}. "
            f"Missing or invalid methods: {missing}."
        )

    # raises TypeError on malformed derivatives(); harvests legacy attrs
    as_derivatives(objective)

    # Check that the objective has exactly one quantity of interest
    if objective.nqoi() != 1:
        raise ValueError(
            f"Invalid objective: expected nqoi=1, got nqoi={objective.nqoi()}."
        )

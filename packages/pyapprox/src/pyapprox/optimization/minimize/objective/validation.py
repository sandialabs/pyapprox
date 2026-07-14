from typing import Any, List

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)


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

    Requires ``ObjectiveProtocol`` conformance (bkd/nvars/nqoi/__call__
    plus a ``derivatives()`` accessor returning a well-formed
    ``Derivatives`` bundle) and exactly one quantity of interest
    (nqoi == 1).

    Parameters
    ----------
    objective : Any
        The objective function to validate.

    Raises
    ------
    TypeError
        If the objective does not satisfy ObjectiveProtocol, or its
        ``derivatives()`` does not return a Derivatives bundle.
    ValueError
        If the objective does not have exactly one quantity of interest (nqoi != 1).
    """
    if not isinstance(objective, ObjectiveProtocol):
        missing = _missing_protocol_methods(objective, ObjectiveProtocol)
        raise TypeError(
            f"Invalid objective type: expected an object implementing "
            f"ObjectiveProtocol (bkd(), nvars(), nqoi(), "
            f"__call__(samples), derivatives()), got "
            f"{type(objective).__name__}. "
            f"Missing or invalid methods: {missing}."
        )

    bundle = objective.derivatives()
    if not isinstance(bundle, Derivatives):
        raise TypeError(
            f"{type(objective).__name__}.derivatives() must return a "
            f"Derivatives bundle; got {type(bundle).__name__}"
        )

    # Check that the objective has exactly one quantity of interest
    if objective.nqoi() != 1:
        raise ValueError(
            f"Invalid objective: expected nqoi=1, got nqoi={objective.nqoi()}."
        )

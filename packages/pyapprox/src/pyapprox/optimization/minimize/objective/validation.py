from typing import Any, List

from pyapprox.optimization.minimize.objective.protocols import (
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
    Validate that the given objective satisfies one of the protocols in
    UnionOfObjectiveProtocols. The validation checks the protocols in order
    from most complex to least complex.
    Additionally, it checks that the objective has exactly one quantity of
    interest (nqoi == 1).

    Parameters
    ----------
    objective : Any
        The objective function to validate.

    Raises
    ------
    TypeError
        If the objective does not satisfy any of the protocols in
        UnionOfObjectiveProtocols.
    ValueError
        If the objective does not have exactly one quantity of interest (nqoi != 1).
    """
    # Check the instance against the protocols in order from most complex to least
    # complex
    if not isinstance(objective, ObjectiveProtocol):
        missing = _missing_protocol_methods(objective, ObjectiveProtocol)
        raise TypeError(
            f"Invalid objective type: expected an object implementing "
            f"ObjectiveProtocol (FunctionProtocol), got {type(objective).__name__}. "
            f"Missing or invalid methods: {missing}. "
            f"Required methods: bkd(), nvars(), nqoi(), __call__(samples)."
        )

    # Check that the objective has exactly one quantity of interest
    if objective.nqoi() != 1:
        raise ValueError(
            f"Invalid objective: expected nqoi=1, got nqoi={objective.nqoi()}."
        )

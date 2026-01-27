from typing import Sequence, Any, List

from pyapprox.typing.optimization.minimize.constraints.protocols import (
    NonlinearConstraintProtocol,
    NonlinearConstraintProtocolWithJacobian,
    NonlinearConstraintProtocolWithJacobianAndWHVP,
)
from pyapprox.typing.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
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


def validate_nonlinear_constraint(obj: object) -> None:
    """
    Validate that the given object satisfies one of the nonlinear constraint protocols.

    Parameters
    ----------
    obj : object
        The object to validate.

    Raises
    ------
    TypeError
        If the object does not satisfy any of the nonlinear constraint protocols.
    """
    if not isinstance(obj, NonlinearConstraintProtocol):
        missing = _missing_protocol_methods(obj, NonlinearConstraintProtocol)
        raise TypeError(
            f"The provided object must satisfy NonlinearConstraintProtocol. "
            f"Got an object of type {type(obj).__name__}. "
            f"Missing or invalid methods: {missing}. "
            f"Required methods: bkd(), nvars(), nqoi(), __call__(samples), lb(), ub()."
        )


def validate_linear_constraint(obj: object) -> None:
    """
    Validate that the given object is a linear constraint.

    Parameters
    ----------
    obj : object
        The object to validate.

    Raises
    ------
    TypeError
        If the object is not a PyApproxLinearConstraint.
    """
    if not isinstance(obj, PyApproxLinearConstraint):
        raise TypeError(
            "The provided object must be a PyApproxLinearConstraint. "
            f"Got an object of type {type(obj).__name__}."
        )


def validate_constraints(constraints: Sequence[Any]) -> None:
    """
    Validate that all objects in the list satisfy one of the constraint protocols
    (either nonlinear or linear).

    Parameters
    ----------
    constraints : List[object]
        The list of objects to validate.

    Raises
    ------
    TypeError
        If any object in the list does not satisfy the constraint protocols.
    """
    for idx, obj in enumerate(constraints):
        if not isinstance(
            obj,
            (NonlinearConstraintProtocol, PyApproxLinearConstraint),
        ):
            missing = _missing_protocol_methods(obj, NonlinearConstraintProtocol)
            raise TypeError(
                f"The constraint at index {idx} must satisfy either "
                f"NonlinearConstraintProtocol or PyApproxLinearConstraint. "
                f"Got an object of type {type(obj).__name__}. "
                f"For NonlinearConstraintProtocol, missing methods: {missing}. "
                f"Required methods: bkd(), nvars(), nqoi(), __call__(samples), lb(), ub()."
            )

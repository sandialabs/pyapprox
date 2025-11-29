from typing import Sequence, Any

from pyapprox.typing.optimization.minimize.constraints.protocols import (
    NonlinearConstraintProtocol,
    NonlinearConstraintProtocolWithJacobian,
    NonlinearConstraintProtocolWithJacobianAndWHVP,
)
from pyapprox.typing.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)


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
    if not isinstance(
        obj,
        (
            NonlinearConstraintProtocol,
            NonlinearConstraintProtocolWithJacobian,
            NonlinearConstraintProtocolWithJacobianAndWHVP,
        ),
    ):
        raise TypeError(
            "The provided object must satisfy one of the following nonlinear "
            "constraint protocols: "
            "'NonlinearConstraintProtocol', "
            "'NonlinearConstraintProtocolWithJacobian', or "
            "'NonlinearConstraintProtocolWithJacobianAndWHVP'. Got an object "
            f"of type {type(obj).__name__}."
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
            (
                NonlinearConstraintProtocol,
                NonlinearConstraintProtocolWithJacobian,
                NonlinearConstraintProtocolWithJacobianAndWHVP,
                PyApproxLinearConstraint,
            ),
        ):
            raise TypeError(
                f"The object at index {idx} must satisfy one of the "
                "following protocols: "
                "'NonlinearConstraintProtocol', "
                "'NonlinearConstraintProtocolWithJacobian', "
                "'NonlinearConstraintProtocolWithJacobianAndHVP', or "
                "'PyApproxLinearConstraint'. "
                f"Got an object of type {type(obj).__name__}."
            )

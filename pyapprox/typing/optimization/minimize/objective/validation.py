from typing import Any

from pyapprox.typing.optimization.minimize.objective.protocols import (
    ObjectiveProtocol,
    ObjectiveWithJacobianProtocol,
    ObjectiveWithJacobianAndHVPProtocol,
)


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
        If the objective does not satisfy any of the protocols in UnionOfObjectiveProtocols.
    ValueError
        If the objective does not have exactly one quantity of interest (nqoi != 1).
    """
    # Check the instance against the protocols in order from most complex to least complex
    if isinstance(objective, ObjectiveWithJacobianAndHVPProtocol):
        pass  # Valid: Objective satisfies ObjectiveWithJacobianAndHVPProtocol
    elif isinstance(objective, ObjectiveWithJacobianProtocol):
        pass  # Valid: Objective satisfies ObjectiveWithJacobianProtocol
    elif isinstance(objective, ObjectiveProtocol):
        pass  # Valid: Objective satisfies ObjectiveProtocol
    else:
        raise TypeError(
            "Invalid objective type: expected an object implementing one of "
            "the protocols in UnionOfObjectiveProtocols, got "
            f"{type(objective).__name__}."
        )

    # Check that the objective has exactly one quantity of interest
    if objective.nqoi() != 1:
        raise ValueError(
            f"Invalid objective: expected nqoi=1, got nqoi={objective.nqoi()}."
        )

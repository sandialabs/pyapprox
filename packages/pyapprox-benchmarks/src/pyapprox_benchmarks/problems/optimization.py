"""Function + bounded search space + constraints for constrained optimization.

Unconstrained optimization uses FunctionOverDomainProblem instead.
"""

from typing import Generic, Sequence, TypeVar

from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.util.backends.protocols import Array

from pyapprox_benchmarks.protocols import ConstraintProtocol, DomainProtocol

F = TypeVar("F", bound=FunctionProtocol)  # type: ignore[type-arg]


class ConstrainedOptimizationProblem(Generic[F, Array]):
    """Function + bounded search space + constraints.

    Unconstrained optimization uses FunctionOverDomainProblem instead.

    Parameters
    ----------
    name : str
        Problem name.
    function : F
        The objective function.
    domain : DomainProtocol[Array]
        The bounded search space.
    constraints : Sequence[ConstraintProtocol[Array]]
        Constraint functions.
    description : str
        Human-readable description.
    """

    def __init__(
        self,
        name: str,
        function: F,
        domain: DomainProtocol[Array],
        constraints: Sequence[ConstraintProtocol[Array]],
        description: str = "",
    ) -> None:
        self._name = name
        self._function = function
        self._domain = domain
        self._constraints = constraints
        self._description = description

    def name(self) -> str:
        """Return the problem name."""
        return self._name

    def function(self) -> F:
        """Return the objective function."""
        return self._function

    def domain(self) -> DomainProtocol[Array]:
        """Return the bounded search space."""
        return self._domain

    def constraints(self) -> Sequence[ConstraintProtocol[Array]]:
        """Return the constraint functions."""
        return self._constraints

    def description(self) -> str:
        """Return the problem description."""
        return self._description

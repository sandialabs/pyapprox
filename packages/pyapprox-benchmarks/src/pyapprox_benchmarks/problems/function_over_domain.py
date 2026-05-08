"""Single function over a bounded domain.

Used by quadrature benchmarks (integration domain) and unconstrained
optimization benchmarks (search-space domain). The domain's semantic
role is defined by the benchmark using this problem.
"""

from typing import Generic, TypeVar

from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.util.backends.protocols import Array

from pyapprox_benchmarks.protocols import DomainProtocol

F = TypeVar("F", bound=FunctionProtocol)  # type: ignore[type-arg]


class FunctionOverDomainProblem(Generic[F, Array]):
    """Single function over a bounded domain.

    Used by quadrature benchmarks (integration domain) and unconstrained
    optimization benchmarks (search-space domain). The domain's semantic
    role is defined by the benchmark using this problem.

    Parameters
    ----------
    name : str
        Problem name.
    function : F
        The function (satisfies FunctionProtocol or a subprotocol).
    domain : DomainProtocol[Array]
        The bounded domain.
    description : str
        Human-readable description.
    """

    def __init__(
        self,
        name: str,
        function: F,
        domain: DomainProtocol[Array],
        description: str = "",
    ) -> None:
        self._name = name
        self._function = function
        self._domain = domain
        self._description = description

    def name(self) -> str:
        """Return the problem name."""
        return self._name

    def function(self) -> F:
        """Return the function."""
        return self._function

    def domain(self) -> DomainProtocol[Array]:
        """Return the bounded domain."""
        return self._domain

    def description(self) -> str:
        """Return the problem description."""
        return self._description

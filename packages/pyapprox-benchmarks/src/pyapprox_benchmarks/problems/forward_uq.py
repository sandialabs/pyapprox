"""Function + prior for forward uncertainty quantification.

Prior defines input distribution; function propagates samples to outputs.
Domain (prior support) is reachable via the prior.
"""

from typing import Generic, TypeVar

from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.protocols import Array

F = TypeVar("F", bound=FunctionProtocol)  # type: ignore[type-arg]


class ForwardUQProblem(Generic[F, Array]):
    """Function + prior for forward uncertainty quantification.

    Prior defines input distribution; function propagates samples to outputs.
    Domain (prior support) is reachable via the prior.

    Parameters
    ----------
    name : str
        Problem name.
    function : F
        The function (satisfies FunctionProtocol or a subprotocol).
    prior : DistributionProtocol[Array]
        Prior distribution over inputs.
    description : str
        Human-readable description.
    """

    def __init__(
        self,
        name: str,
        function: F,
        prior: DistributionProtocol[Array],
        description: str = "",
    ) -> None:
        self._name = name
        self._function = function
        self._prior = prior
        self._description = description

    def name(self) -> str:
        """Return the problem name."""
        return self._name

    def function(self) -> F:
        """Return the function."""
        return self._function

    def prior(self) -> DistributionProtocol[Array]:
        """Return the prior distribution."""
        return self._prior

    def description(self) -> str:
        """Return the problem description."""
        return self._description

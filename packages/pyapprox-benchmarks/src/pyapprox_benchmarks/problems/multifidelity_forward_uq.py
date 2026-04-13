"""Multifidelity forward UQ problem — ensemble of models with prior."""

from typing import Any, Generic, List, Sequence, TypeVar

from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.protocols import Array

F = TypeVar("F", bound=FunctionProtocol[Any])


class MultifidelityForwardUQProblem(Generic[F, Array]):
    """Ensemble of models at different fidelities with a shared prior.

    The first model (index 0) is the highest-fidelity model.

    Parameters
    ----------
    name : str
        Problem name.
    models : Sequence[F]
        Ordered list of models (index 0 = highest fidelity).
    costs : Array
        Per-model evaluation costs, shape ``(nmodels,)``.
    prior : DistributionProtocol[Array]
        Prior distribution over inputs.
    description : str
        Human-readable description.
    """

    def __init__(
        self,
        name: str,
        models: Sequence[F],
        costs: Array,
        prior: DistributionProtocol[Array],
        description: str = "",
    ) -> None:
        self._name = name
        self._models = list(models)
        self._costs = costs
        self._prior = prior
        self._description = description

    def name(self) -> str:
        return self._name

    def models(self) -> List[F]:
        return self._models

    def nmodels(self) -> int:
        return len(self._models)

    def costs(self) -> Array:
        return self._costs

    def function(self) -> F:
        """Return the highest-fidelity model."""
        return self._models[0]

    def prior(self) -> DistributionProtocol[Array]:
        return self._prior

    def description(self) -> str:
        return self._description

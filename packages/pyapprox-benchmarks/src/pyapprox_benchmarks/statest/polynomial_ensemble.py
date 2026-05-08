"""Polynomial ensemble benchmark — analytical statistics."""

from typing import Generic, List

from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.functions.multifidelity.polynomial_ensemble import (
    PolynomialModelFunction,
)
from pyapprox_benchmarks.problems.multifidelity_forward_uq import (
    MultifidelityForwardUQProblem,
)


class PolynomialEnsembleBenchmark(Generic[Array]):
    """Polynomial ensemble benchmark — analytical statistics.

    Creates PolynomialModelFunction instances directly with degrees
    nmodels, nmodels-1, ..., 1 and geometric costs 1, 0.1, 0.01, ...

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nmodels : int
        Number of models in the ensemble (default 5).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nmodels: int = 5,
    ) -> None:
        self._bkd = bkd
        self._nmodels = nmodels
        self._models: List[PolynomialModelFunction[Array]] = [
            PolynomialModelFunction(bkd, degree=nmodels - k)
            for k in range(nmodels)
        ]
        costs = bkd.logspace(0, -nmodels + 1, nmodels)
        prior = IndependentJoint([UniformMarginal(0.0, 1.0, bkd)], bkd)
        self._problem = MultifidelityForwardUQProblem(
            f"polynomial_ensemble_{nmodels}model",
            list(self._models),
            costs,
            prior,
            description=f"{nmodels}-model polynomial ensemble",
        )

    def problem(
        self,
    ) -> MultifidelityForwardUQProblem[FunctionProtocol[Array], Array]:
        return self._problem  # type: ignore[return-value]

    def ensemble_means(self) -> Array:
        """Analytical means, shape (nmodels, 1).

        E[x^d] = 1/(d+1) for U[0,1].
        """
        means_1d = self._bkd.array([m.mean() for m in self._models])
        return self._bkd.reshape(means_1d, (-1, 1))

    def ensemble_covariance(self) -> Array:
        """Analytical covariance matrix, shape (nmodels, nmodels).

        Cov[x^d1, x^d2] = 1/(d1+d2+1) - 1/(d1+1)/(d2+1).
        """
        n = self._nmodels
        cov = self._bkd.zeros((n, n))
        for i in range(n):
            d1 = self._models[i].degree()
            for j in range(n):
                d2 = self._models[j].degree()
                cov[i, j] = 1.0 / (d1 + d2 + 1) - 1.0 / (d1 + 1) / (d2 + 1)
        return cov

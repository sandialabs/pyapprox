"""Composable fitter chains for multi-stage DGP optimization.

DGPFitterChain runs a sequence of DGP fitters, threading the fitted model
from each stage as input to the next. Each fitter internally clones the
model, preserving optimized parameters as initial values for the next stage.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    Hashable,
    List,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pyapprox.surrogates.gaussianprocess.fitters.results import (
    GPOptimizedFitResult,
)
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
        DeepGaussianProcess,
    )


@runtime_checkable
class DGPFitterProtocol(Protocol[Array]):
    """Protocol for fitters that operate on DeepGaussianProcess models."""

    def fit(
        self,
        dgp: DeepGaussianProcess[Array],
        data: Dict[Hashable, Tuple[Array, Array]],
    ) -> GPOptimizedFitResult[
        Array, DeepGaussianProcess[Array]
    ]: ...


class DGPChainedFitResult(Generic[Array]):
    """Result from a DGPFitterChain: wraps the final result with history.

    Delegates surrogate(), predict(), predict_std(), bkd(), __call__()
    to the final fitter's result. Provides access to intermediate
    results for diagnostics.

    Parameters
    ----------
    final_result : GPOptimizedFitResult
        Result from the last fitter in the chain.
    intermediate_results : List[GPOptimizedFitResult]
        Results from all fitters in order (including the final one).
    """

    def __init__(
        self,
        final_result: GPOptimizedFitResult[
            Array, DeepGaussianProcess[Array]
        ],
        intermediate_results: List[
            GPOptimizedFitResult[Array, DeepGaussianProcess[Array]]
        ],
    ) -> None:
        self._final = final_result
        self._intermediates = intermediate_results

    def surrogate(self) -> DeepGaussianProcess[Array]:
        return self._final.surrogate()

    def neg_log_marginal_likelihood(self) -> Array:
        return self._final.neg_log_marginal_likelihood()

    def initial_hyperparameters(self) -> Array:
        return self._intermediates[0].initial_hyperparameters()

    def optimized_hyperparameters(self) -> Array:
        return self._final.optimized_hyperparameters()

    def intermediate_results(
        self,
    ) -> List[
        GPOptimizedFitResult[Array, DeepGaussianProcess[Array]]
    ]:
        return list(self._intermediates)

    def bkd(self) -> Backend[Array]:
        return self._final.bkd()

    def __call__(self, X: Array) -> Array:
        result: Array = self._final(X)
        return result

    def predict(self, X: Array) -> Array:
        return self._final.predict(X)

    def predict_std(self, X: Array) -> Array:
        return self._final.predict_std(X)

    def __repr__(self) -> str:
        n_stages = len(self._intermediates)
        return (
            f"DGPChainedFitResult(stages={n_stages}, "
            f"final={type(self._final.surrogate()).__name__})"
        )


class DGPFitterChain(Generic[Array]):
    """Runs a sequence of DGP fitters, threading the model through each stage.

    Each fitter receives the previous fitter's fitted surrogate as input.
    The first fitter receives the original model. Fitters internally
    clone the model, so optimized parameters from stage N become initial
    values for stage N+1 via deep copy.

    Parameters
    ----------
    fitters : List[DGPFitterProtocol[Array]]
        Ordered list of fitters satisfying DGPFitterProtocol.

    Raises
    ------
    ValueError
        If fitters list is empty.
    """

    def __init__(self, fitters: List[DGPFitterProtocol[Array]]) -> None:
        if not fitters:
            raise ValueError("DGPFitterChain requires at least one fitter")
        self._fitters = fitters

    def fit(
        self,
        model: DeepGaussianProcess[Array],
        data: Dict[Hashable, Tuple[Array, Array]],
    ) -> DGPChainedFitResult[Array]:
        """Run all fitters in sequence.

        Parameters
        ----------
        model : DeepGaussianProcess[Array]
            Initial deep GP model.
        data : Dict[Hashable, Tuple[Array, Array]]
            Training data passed to each fitter.

        Returns
        -------
        DGPChainedFitResult
            Result wrapping the final fitted model with history.
        """
        current_model = model
        intermediates: List[
            GPOptimizedFitResult[Array, DeepGaussianProcess[Array]]
        ] = []

        for fitter in self._fitters:
            result = fitter.fit(current_model, data)
            intermediates.append(result)
            current_model = result.surrogate()

        return DGPChainedFitResult(
            final_result=intermediates[-1],
            intermediate_results=intermediates,
        )

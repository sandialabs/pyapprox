"""Fitter for DeepGaussianProcess.

Uses DGPELBOLoss / TorchDGPELBOLoss for hyperparameter optimization
via doubly-stochastic variational inference.
"""

from typing import Dict, Generic, Hashable, Optional, Tuple

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
    DGPELBOLoss,
    TorchDGPELBOLoss,
)
from pyapprox.surrogates.gaussianprocess.fitters.results import (
    GPOptimizedFitResult,
    PredictiveGPSurrogateProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class DGPMaximumLikelihoodFitter(Generic[Array]):
    """Deep GP fitter with maximum likelihood hyperparameter optimization.

    Optimizes active hyperparameters by minimizing the negative ELBO,
    then marks the DGP as fitted. Uses AdamOptimizer by default since
    DGP optimization requires gradient-based methods.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    optimizer : Optional[BindableOptimizerProtocol[Array]]
        Optimizer for hyperparameter tuning. If None, uses
        AdamOptimizer(lr=1e-2, maxiter=5000).
    n_propagation : int
        Number of propagation samples for the ELBO.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
        n_propagation: int = 10,
    ) -> None:
        self._bkd = bkd
        self._optimizer = optimizer
        self._n_propagation = n_propagation

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def fit(
        self,
        dgp: DeepGaussianProcess[Array],
        data: Dict[Hashable, Tuple[Array, Array]],
    ) -> GPOptimizedFitResult[Array, PredictiveGPSurrogateProtocol[Array]]:
        """Fit deep GP and optimize active hyperparameters.

        Parameters
        ----------
        dgp : DeepGaussianProcess[Array]
            The deep GP model.
        data : Dict[Hashable, Tuple[Array, Array]]
            Training data for each observed node: {node_id: (X, y)}.
            X has shape (nvars, N), y has shape (1, N).

        Returns
        -------
        GPOptimizedFitResult
            Result with fitted DGP and optimization metadata.
        """
        clone = dgp._clone_unfitted()

        initial_hyps = self._bkd.array(
            clone.hyp_list().get_active_values()
        )

        if clone.hyp_list().nactive_params() == 0:
            neg_elbo = DGPELBOLoss(clone, data, self._n_propagation)(
                clone.hyp_list().get_active_values()
            )
            clone.set_fitted()
            return GPOptimizedFitResult(
                surrogate=clone,
                neg_log_marginal_likelihood=neg_elbo,
                initial_hyperparameters=initial_hyps,
                optimized_hyperparameters=initial_hyps,
                optimization_result=None,
            )

        if isinstance(self._bkd, TorchBkd):
            loss = TorchDGPELBOLoss(clone, data, self._n_propagation)
        else:
            loss = DGPELBOLoss(clone, data, self._n_propagation)

        bounds = clone.hyp_list().get_active_bounds()

        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            from pyapprox.optimization.minimize.adam.adam_optimizer import (
                AdamOptimizer,
            )

            optimizer = AdamOptimizer(lr=1e-2, maxiter=5000, verbosity=0)

        optimizer.bind(loss, bounds)

        init_guess = clone.hyp_list().get_active_values()
        if len(init_guess.shape) == 1:
            init_guess = self._bkd.reshape(
                init_guess, (len(init_guess), 1)
            )

        opt_result = optimizer.minimize(init_guess)

        optimal_params = opt_result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        clone.hyp_list().set_active_values(optimal_params)

        neg_elbo = loss(optimal_params)
        optimized_hyps = clone.hyp_list().get_active_values()
        clone.set_fitted()

        return GPOptimizedFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=neg_elbo,
            initial_hyperparameters=initial_hyps,
            optimized_hyperparameters=optimized_hyps,
            optimization_result=opt_result,
        )

"""Fitters for MultiOutputGP.

MultiOutputGP has a different fit signature: _fit_internal takes
(X_train_list, y_train) where X_train_list is a list of arrays.
It also has no transforms.
"""

from typing import Generic, List, Optional, Union

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.surrogates.gaussianprocess.fitters.results import (
    GPFitResult,
    GPOptimizedFitResult,
)
from pyapprox.util.backends.protocols import Array, Backend


class MultiOutputGPFixedHyperparameterFitter(Generic[Array]):
    """Fixed hyperparameter fitter for MultiOutputGP.

    Computes Cholesky and alpha with fixed hyperparameters.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        gp,
        X_train_list: List[Array],
        y_train: Union[List[Array], Array],
    ) -> GPFitResult[Array]:
        """Fit multi-output GP without hyperparameter optimization.

        Parameters
        ----------
        gp : MultiOutputGP[Array]
            The multi-output GP model.
        X_train_list : List[Array]
            Training inputs for each output. Each array has shape
            (nvars, n_i).
        y_train : Union[List[Array], Array]
            Training outputs. List format (preferred): list of arrays
            each with shape (1, n_i). Stacked format: shape (sum(n_i), 1).

        Returns
        -------
        GPFitResult
            Result containing the fitted GP and NLL.
        """
        clone = gp._clone_unfitted()

        clone._fit_internal(X_train_list, y_train)

        nll = clone.neg_log_marginal_likelihood()

        return GPFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=nll,
        )


class MultiOutputGPMaximumLikelihoodFitter(Generic[Array]):
    """Multi-output GP fitter with maximum likelihood hyperparameter optimization.

    Optimizes active hyperparameters by minimizing the negative log
    marginal likelihood. Uses GPNegativeLogMarginalLikelihoodLoss.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    optimizer : Optional[BindableOptimizerProtocol[Array]]
        Optimizer for hyperparameter tuning. If None, uses default
        ScipyTrustConstrOptimizer with maxiter=1000.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._optimizer = optimizer

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        gp,
        X_train_list: List[Array],
        y_train: Union[List[Array], Array],
    ) -> GPOptimizedFitResult[Array]:
        """Fit multi-output GP and optimize active hyperparameters.

        Parameters
        ----------
        gp : MultiOutputGP[Array]
            The multi-output GP model.
        X_train_list : List[Array]
            Training inputs for each output.
        y_train : Union[List[Array], Array]
            Training outputs.

        Returns
        -------
        GPOptimizedFitResult
            Result with fitted GP and optimization metadata.
        """
        clone = gp._clone_unfitted()

        clone._fit_internal(X_train_list, y_train)

        initial_hyps = clone.hyp_list().get_active_values()
        initial_hyps = self._bkd.array(initial_hyps)

        if clone.hyp_list().nactive_params() == 0:
            nll = clone.neg_log_marginal_likelihood()
            return GPOptimizedFitResult(
                surrogate=clone,
                neg_log_marginal_likelihood=nll,
                initial_hyperparameters=initial_hyps,
                optimized_hyperparameters=initial_hyps,
                optimization_result=None,
            )

        from pyapprox.surrogates.gaussianprocess.gp_loss import (
            GPNegativeLogMarginalLikelihoodLoss,
        )

        loss = GPNegativeLogMarginalLikelihoodLoss(
            clone, (clone._X_train_list, clone._y_train_stacked)
        )
        clone._configure_loss(loss)

        bounds = clone.hyp_list().get_active_bounds()

        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            from pyapprox.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer,
            )

            optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=1000)

        optimizer.bind(loss, bounds)

        init_guess = clone.hyp_list().get_active_values()
        if len(init_guess.shape) == 1:
            init_guess = self._bkd.reshape(init_guess, (len(init_guess), 1))

        opt_result = optimizer.minimize(init_guess)

        optimal_params = opt_result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        clone.hyp_list().set_active_values(optimal_params)

        clone._fit_internal(X_train_list, y_train)

        nll = clone.neg_log_marginal_likelihood()
        optimized_hyps = clone.hyp_list().get_active_values()

        return GPOptimizedFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=nll,
            initial_hyperparameters=initial_hyps,
            optimized_hyperparameters=optimized_hyps,
            optimization_result=opt_result,
        )

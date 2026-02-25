"""Fitters for VariationalGaussianProcess.

Uses VariationalGPELBOLoss for hyperparameter optimization.
"""

from typing import Generic, Optional

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.surrogates.gaussianprocess.fitters.results import (
    GPFitResult,
    GPOptimizedFitResult,
)
from pyapprox.surrogates.gaussianprocess.input_transform import (
    IdentityInputTransform,
    InputAffineTransformProtocol,
)
from pyapprox.surrogates.gaussianprocess.output_transform import (
    OutputAffineTransformProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class VariationalGPFixedHyperparameterFitter(Generic[Array]):
    """Fixed hyperparameter fitter for VariationalGaussianProcess.

    Computes ELBO quantities and effective alpha/cholesky with fixed
    hyperparameters. Does NOT optimize hyperparameters.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    output_transform : Optional[OutputAffineTransformProtocol[Array]]
        If provided, y_train is in original space and will be scaled.
    input_transform : Optional[InputAffineTransformProtocol[Array]]
        If provided, X_train is in original space and will be scaled.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        output_transform: Optional[OutputAffineTransformProtocol[Array]] = None,
        input_transform: Optional[InputAffineTransformProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._output_transform = output_transform
        self._input_transform = input_transform

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        gp,
        X_train: Array,
        y_train: Array,
    ) -> GPFitResult:
        """Fit variational GP without hyperparameter optimization.

        Parameters
        ----------
        gp : VariationalGaussianProcess[Array]
            The variational GP model.
        X_train : Array
            Training input data, shape (nvars, n_train).
        y_train : Array
            Training output data, shape (nqoi, n_train).

        Returns
        -------
        GPFitResult
            Result containing the fitted GP and negative ELBO.
        """
        clone = gp._clone_unfitted()

        clone._output_transform = self._output_transform
        if self._input_transform is not None:
            clone._input_transform = self._input_transform
        else:
            clone._input_transform = IdentityInputTransform(gp.nvars(), self._bkd)

        X_scaled = clone._input_transform.transform(X_train)
        y_scaled = y_train
        if self._output_transform is not None:
            y_scaled = self._output_transform.inverse_transform(y_train)

        clone._fit_internal(X_scaled, y_scaled)

        neg_elbo = clone.neg_log_marginal_likelihood()

        return GPFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=neg_elbo,
        )


class VariationalGPMaximumLikelihoodFitter(Generic[Array]):
    """Variational GP fitter with maximum likelihood hyperparameter optimization.

    Optimizes active hyperparameters by minimizing the negative ELBO,
    then refits with optimal values. Uses VariationalGPELBOLoss.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    optimizer : Optional[BindableOptimizerProtocol[Array]]
        Optimizer for hyperparameter tuning. If None, uses default
        ScipyTrustConstrOptimizer with maxiter=1000.
    output_transform : Optional[OutputAffineTransformProtocol[Array]]
        If provided, y_train is in original space and will be scaled.
    input_transform : Optional[InputAffineTransformProtocol[Array]]
        If provided, X_train is in original space and will be scaled.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
        output_transform: Optional[OutputAffineTransformProtocol[Array]] = None,
        input_transform: Optional[InputAffineTransformProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._optimizer = optimizer
        self._output_transform = output_transform
        self._input_transform = input_transform

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        gp,
        X_train: Array,
        y_train: Array,
    ) -> GPOptimizedFitResult:
        """Fit variational GP and optimize active hyperparameters.

        Parameters
        ----------
        gp : VariationalGaussianProcess[Array]
            The variational GP model.
        X_train : Array
            Training input data, shape (nvars, n_train).
        y_train : Array
            Training output data, shape (nqoi, n_train).

        Returns
        -------
        GPOptimizedFitResult
            Result with fitted GP and optimization metadata.
        """
        clone = gp._clone_unfitted()

        clone._output_transform = self._output_transform
        if self._input_transform is not None:
            clone._input_transform = self._input_transform
        else:
            clone._input_transform = IdentityInputTransform(gp.nvars(), self._bkd)

        X_scaled = clone._input_transform.transform(X_train)
        y_scaled = y_train
        if self._output_transform is not None:
            y_scaled = self._output_transform.inverse_transform(y_train)

        clone._fit_internal(X_scaled, y_scaled)

        initial_hyps = clone.hyp_list().get_active_values()
        initial_hyps = self._bkd.array(initial_hyps)

        if clone.hyp_list().nactive_params() == 0:
            neg_elbo = clone.neg_log_marginal_likelihood()
            return GPOptimizedFitResult(
                surrogate=clone,
                neg_log_marginal_likelihood=neg_elbo,
                initial_hyperparameters=initial_hyps,
                optimized_hyperparameters=initial_hyps,
                optimization_result=None,
            )

        from pyapprox.surrogates.gaussianprocess.variational_loss import (
            VariationalGPELBOLoss,
        )

        loss = VariationalGPELBOLoss(clone, (clone._data.X(), clone._data.y()))
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

        clone._fit_internal(X_scaled, y_scaled)

        neg_elbo = clone.neg_log_marginal_likelihood()
        optimized_hyps = clone.hyp_list().get_active_values()

        return GPOptimizedFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=neg_elbo,
            initial_hyperparameters=initial_hyps,
            optimized_hyperparameters=optimized_hyps,
            optimization_result=opt_result,
        )

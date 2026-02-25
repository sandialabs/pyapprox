"""GP fitter with maximum likelihood hyperparameter optimization.

Optimizes hyperparameters by minimizing the negative log marginal
likelihood. Extracts the optimization logic from ExactGaussianProcess.fit().
"""

from typing import Generic, Optional

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.surrogates.gaussianprocess.fitters.results import (
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


class GPMaximumLikelihoodFitter(Generic[Array]):
    """GP fitter with maximum likelihood hyperparameter optimization.

    Optimizes active hyperparameters by minimizing the negative log
    marginal likelihood, then refits with optimal values. Uses the
    existing GPNegativeLogMarginalLikelihoodLoss.

    The original GP is not modified — a deep copy is created internally
    and returned in the result.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    optimizer : Optional[BindableOptimizerProtocol[Array]]
        Optimizer for hyperparameter tuning. If None, uses default
        ScipyTrustConstrOptimizer with maxiter=1000.
    output_transform : Optional[OutputAffineTransformProtocol[Array]]
        If provided, y_train is assumed to be in original space and
        will be scaled internally.
    input_transform : Optional[InputAffineTransformProtocol[Array]]
        If provided, X_train is assumed to be in original space and
        will be scaled internally. If None, uses identity transform.

    Examples
    --------
    >>> fitter = GPMaximumLikelihoodFitter(bkd)
    >>> result = fitter.fit(gp, X_train, y_train)
    >>> fitted_gp = result.surrogate()
    >>> print(result.neg_log_marginal_likelihood())
    >>> print(result.optimized_hyperparameters())
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
        """Fit GP to data and optimize active hyperparameters.

        1. Deep-copies the GP
        2. Applies transforms to data
        3. Computes initial Cholesky + alpha
        4. If active hyperparameters exist, optimizes them via NLL
        5. Refits with optimal hyperparameters
        6. Returns result with fitted GP and optimization metadata

        Parameters
        ----------
        gp : ExactGaussianProcess[Array]
            The GP model. Must have ``_clone_unfitted()``,
            ``_fit_internal()``, and ``_configure_loss()`` methods.
        X_train : Array
            Training input data, shape (nvars, n_train).
        y_train : Array
            Training output data, shape (nqoi, n_train).

        Returns
        -------
        GPOptimizedFitResult
            Result containing the fitted GP, NLL, and optimization metadata.
        """
        clone = gp._clone_unfitted()

        # Install transforms on clone
        clone._output_transform = self._output_transform
        if self._input_transform is not None:
            clone._input_transform = self._input_transform
        else:
            clone._input_transform = IdentityInputTransform(gp.nvars(), self._bkd)

        # Transform data
        X_scaled = clone._input_transform.transform(X_train)
        y_scaled = y_train
        if self._output_transform is not None:
            y_scaled = self._output_transform.inverse_transform(y_train)

        # Initial fit
        clone._fit_internal(X_scaled, y_scaled)

        # Store initial hyperparameters
        initial_hyps = clone.hyp_list().get_active_values()
        # Make a copy to avoid reference issues
        initial_hyps = self._bkd.array(initial_hyps)

        # Check if optimization is needed
        if clone.hyp_list().nactive_params() == 0:
            nll = clone.neg_log_marginal_likelihood()
            return GPOptimizedFitResult(
                surrogate=clone,
                neg_log_marginal_likelihood=nll,
                initial_hyperparameters=initial_hyps,
                optimized_hyperparameters=initial_hyps,
                optimization_result=None,
            )

        # Create loss function
        from pyapprox.surrogates.gaussianprocess.gp_loss import (
            GPNegativeLogMarginalLikelihoodLoss,
        )

        loss = GPNegativeLogMarginalLikelihoodLoss(
            clone, (clone._data.X(), clone._data.y())
        )
        clone._configure_loss(loss)

        # Get bounds for active hyperparameters
        bounds = clone.hyp_list().get_active_bounds()

        # Get optimizer (clone if user-provided to avoid shared state)
        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            from pyapprox.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer,
            )

            optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=1000)

        # Bind optimizer to loss and bounds
        optimizer.bind(loss, bounds)

        # Get initial guess from current hyperparameter values
        init_guess = clone.hyp_list().get_active_values()
        if len(init_guess.shape) == 1:
            init_guess = self._bkd.reshape(init_guess, (len(init_guess), 1))

        # Run optimization
        opt_result = optimizer.minimize(init_guess)

        # Update hyperparameters with optimal values
        optimal_params = opt_result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        clone.hyp_list().set_active_values(optimal_params)

        # Final refit with optimal hyperparameters
        clone._fit_internal(X_scaled, y_scaled)

        nll = clone.neg_log_marginal_likelihood()
        optimized_hyps = clone.hyp_list().get_active_values()

        return GPOptimizedFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=nll,
            initial_hyperparameters=initial_hyps,
            optimized_hyperparameters=optimized_hyps,
            optimization_result=opt_result,
        )

"""MSE-based gradient fitter for FunctionTrain.

Uses gradient-based optimization to minimize MSE loss.
Follows the GP optimizer pattern from gaussianprocess/exact.py.
"""

from typing import Generic, Optional

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.surrogates.functiontrain.fitters.results import (
    MSEFitterResult,
)
from pyapprox.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.surrogates.functiontrain.losses import FunctionTrainMSELoss
from pyapprox.util.backends.protocols import Array, Backend


class MSEFitter(Generic[Array]):
    """MSE-based gradient fitter for FunctionTrain.

    Minimizes the mean squared error loss using gradient-based optimization:

        Loss = (1/2n) sum_i ||y_i - f(x_i)||^2

    Follows the GP optimizer pattern: accepts a configured optimizer
    (BindableOptimizerProtocol) that is cloned during fit() to avoid
    shared state.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.optimization.minimize.scipy.trust_constr import (
    ...     ScipyTrustConstrOptimizer
    ... )
    >>> fitter = MSEFitter(bkd)
    >>> fitter.set_optimizer(ScipyTrustConstrOptimizer(maxiter=500, gtol=1e-8))
    >>> result = fitter.fit(ft, samples, values)
    >>> fitted_ft = result.surrogate()
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._optimizer: Optional[BindableOptimizerProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def set_optimizer(self, optimizer: BindableOptimizerProtocol[Array]) -> None:
        """Set custom optimizer.

        Parameters
        ----------
        optimizer : BindableOptimizerProtocol[Array]
            Configured optimizer (e.g., ScipyTrustConstrOptimizer).
            The optimizer is cloned during fit() to avoid shared state.
        """
        self._optimizer = optimizer

    def optimizer(self) -> Optional[BindableOptimizerProtocol[Array]]:
        """Return current optimizer (or None if using default)."""
        return self._optimizer

    def fit(
        self,
        surrogate: FunctionTrain[Array],
        samples: Array,
        values: Array,
        bounds: Optional[Array] = None,
    ) -> MSEFitterResult[Array]:
        """Fit FunctionTrain to data using gradient-based optimization.

        Parameters
        ----------
        surrogate : FunctionTrain[Array]
            The FunctionTrain to fit.
        samples : Array
            Training samples. Shape: (nvars, nsamples)
        values : Array
            Training values. Shape: (nqoi, nsamples) or (nsamples,)
        bounds : Optional[Array]
            Parameter bounds. Shape: (nparams, 2).
            If None, uses default bounds [-inf, inf].

        Returns
        -------
        MSEFitterResult[Array]
            Result containing fitted surrogate and diagnostics.

        Raises
        ------
        TypeError
            If surrogate is not a FunctionTrain.
        ValueError
            If dimensions don't match.
        """
        # Validate input type
        if not isinstance(surrogate, FunctionTrain):
            raise TypeError(
                f"MSEFitter only works with FunctionTrain, "
                f"got {type(surrogate).__name__}"
            )

        # Normalize values shape
        if values.ndim == 1:
            values = self._bkd.reshape(values, (1, -1))

        # Validate dimensions
        nvars = surrogate.nvars()
        nqoi = surrogate.nqoi()
        nsamples = samples.shape[1]

        if samples.shape[0] != nvars:
            raise ValueError(
                f"samples has {samples.shape[0]} variables, surrogate expects {nvars}"
            )
        if values.shape[0] != nqoi:
            raise ValueError(f"values has {values.shape[0]} QoIs, surrogate has {nqoi}")
        if values.shape[1] != nsamples:
            raise ValueError(
                f"values has {values.shape[1]} samples, samples has {nsamples}"
            )

        nparams = surrogate.nparams()

        # Handle case with no trainable parameters
        if nparams == 0:
            # Compute loss for the fixed surrogate
            loss_fn = FunctionTrainMSELoss(surrogate, samples, values, self._bkd)
            init_params = self._bkd.zeros((0, 1))
            final_loss = float(self._bkd.to_numpy(loss_fn(init_params)[0, 0]))
            # Create a dummy result
            from pyapprox.optimization.minimize.scipy.scipy_result import (
                ScipyOptimizerResultWrapper,
            )

            dummy_result = ScipyOptimizerResultWrapper(
                x=init_params,
                fun=self._bkd.asarray([[final_loss]]),
                success=True,
                nit=0,
                message="No parameters to optimize",
            )
            return MSEFitterResult(
                surrogate=surrogate,
                optimizer_result=dummy_result,
                final_loss=final_loss,
            )

        # Create loss function
        loss = FunctionTrainMSELoss(surrogate, samples, values, self._bkd)

        # Get or create optimizer (clone if user-provided)
        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=1000)

        # Get bounds (default to unbounded)
        if bounds is None:
            bounds = self._default_bounds(nparams)

        # Bind optimizer to loss and bounds
        optimizer.bind(loss, bounds)

        # Get initial guess from current parameters
        init_guess = surrogate._flatten_params()
        if init_guess.ndim == 1:
            init_guess = self._bkd.reshape(init_guess, (-1, 1))

        # Run optimization
        result = optimizer.minimize(init_guess)

        # Extract optimal parameters and create fitted surrogate
        optimal_params = result.optima()
        if optimal_params.ndim == 2:
            optimal_params = optimal_params[:, 0]
        fitted_surrogate = surrogate.with_params(optimal_params)

        # Compute final loss
        final_loss = float(self._bkd.to_numpy(loss(optimal_params)[0, 0]))

        return MSEFitterResult(
            surrogate=fitted_surrogate,
            optimizer_result=result,
            final_loss=final_loss,
        )

    def _default_bounds(self, nparams: int) -> Array:
        """Create default unbounded bounds.

        Parameters
        ----------
        nparams : int
            Number of parameters.

        Returns
        -------
        Array
            Bounds array. Shape: (nparams, 2)
        """
        import numpy as np

        bounds = np.full((nparams, 2), [-np.inf, np.inf])
        return self._bkd.asarray(bounds)

    def __repr__(self) -> str:
        opt_str = (
            type(self._optimizer).__name__
            if self._optimizer is not None
            else "ScipyTrustConstrOptimizer (default)"
        )
        return f"MSEFitter(optimizer={opt_str})"

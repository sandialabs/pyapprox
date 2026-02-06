"""GP fitter using fixed (current) hyperparameters without optimization.

Analogous to LeastSquaresFitter for basis expansions.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.gaussianprocess.input_transform import (
    InputAffineTransformProtocol,
    IdentityInputTransform,
)
from pyapprox.typing.surrogates.gaussianprocess.output_transform import (
    OutputAffineTransformProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.fitters.results import (
    GPFitResult,
)


class GPFixedHyperparameterFitter(Generic[Array]):
    """GP fitter using fixed (current) hyperparameters.

    Computes Cholesky factorization and precomputed weights (alpha)
    with fixed hyperparameters. Does NOT optimize hyperparameters.

    The original GP is not modified — a deep copy is created internally
    and returned in the result.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    output_transform : Optional[OutputAffineTransformProtocol[Array]]
        If provided, y_train is assumed to be in original space and
        will be scaled internally. Predictions will be returned in
        original space.
    input_transform : Optional[InputAffineTransformProtocol[Array]]
        If provided, X_train is assumed to be in original space and
        will be scaled internally. If None, uses identity transform.

    Examples
    --------
    >>> fitter = GPFixedHyperparameterFitter(bkd)
    >>> result = fitter.fit(gp, X_train, y_train)
    >>> fitted_gp = result.surrogate()
    >>> mean = fitted_gp.predict(X_test)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        output_transform: Optional[
            OutputAffineTransformProtocol[Array]
        ] = None,
        input_transform: Optional[
            InputAffineTransformProtocol[Array]
        ] = None,
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
        """Fit GP to data without hyperparameter optimization.

        Creates a deep copy of the GP, applies transforms, computes
        Cholesky and alpha, and returns the fitted copy in a result.

        Parameters
        ----------
        gp : ExactGaussianProcess[Array] or VariationalGaussianProcess[Array]
            The GP model. Must have ``_clone_unfitted()`` and
            ``_fit_internal()`` methods.
        X_train : Array
            Training input data, shape (nvars, n_train).
        y_train : Array
            Training output data, shape (nqoi, n_train).

        Returns
        -------
        GPFitResult
            Result containing the fitted GP and NLL.
        """
        clone = gp._clone_unfitted()

        # Install transforms on clone
        clone._output_transform = self._output_transform
        if self._input_transform is not None:
            clone._input_transform = self._input_transform
        else:
            clone._input_transform = IdentityInputTransform(
                gp.nvars(), self._bkd
            )

        # Transform data
        X_scaled = clone._input_transform.transform(X_train)
        y_scaled = y_train
        if self._output_transform is not None:
            y_scaled = self._output_transform.inverse_transform(y_train)

        # Fit (Cholesky + alpha)
        clone._fit_internal(X_scaled, y_scaled)

        nll = clone.neg_log_marginal_likelihood()

        return GPFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=nll,
        )

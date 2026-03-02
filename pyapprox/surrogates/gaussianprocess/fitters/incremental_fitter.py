"""GP fitter with rank-1 Cholesky update for incremental data.

When exactly one new point is added and a previously fitted GP is
available, this fitter performs a rank-1 Cholesky update in O(n²)
instead of recomputing the full Cholesky in O(n³).

Falls back to GPFixedHyperparameterFitter when the incremental
path is not applicable.
"""

from typing import Generic, Optional

from pyapprox.surrogates.gaussianprocess.data import GPTrainingData
from pyapprox.surrogates.gaussianprocess.fitters.results import (
    GPFitResult,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.cholesky_factor import CholeskyFactor


class GPIncrementalFitter(Generic[Array]):
    """GP fitter that uses rank-1 Cholesky updates when possible.

    Given a previously fitted GP with n training points and new data
    with n+1 training points (same first n points plus one new point),
    performs a rank-1 Cholesky update in O(n²) instead of full
    Cholesky factorization in O(n³).

    Falls back to full Cholesky (via GPFixedHyperparameterFitter)
    when the incremental path is not applicable.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        gp,
        X_train: Array,
        y_train: Array,
        prev_gp=None,
    ) -> GPFitResult:
        """Fit GP to data, using incremental update if possible.

        If prev_gp is provided and exactly one new point was added,
        performs a rank-1 Cholesky update. Otherwise falls back to
        full Cholesky factorization.

        Parameters
        ----------
        gp : ExactGaussianProcess[Array]
            GP template (unfitted, with current hyperparameters).
        X_train : Array
            Training inputs, shape (nvars, n_train).
        y_train : Array
            Training outputs, shape (nqoi, n_train).
        prev_gp : ExactGaussianProcess[Array], optional
            Previously fitted GP. If provided and exactly one new
            point was added, enables the incremental path.

        Returns
        -------
        GPFitResult
            Result containing the fitted GP and NLL.
        """
        if self._can_do_incremental(prev_gp, X_train):
            result = self._fit_incremental(gp, X_train, y_train, prev_gp)
            if result is not None:
                return result
        return self._fit_full(gp, X_train, y_train)

    def _can_do_incremental(self, prev_gp, X_train: Array) -> bool:
        """Check if incremental update is possible.

        Returns True if prev_gp is fitted and X_train has exactly
        one more sample than prev_gp's training data.
        """
        if prev_gp is None:
            return False
        if not prev_gp.is_fitted():
            return False
        n_prev = prev_gp.data().n_samples()
        n_new = X_train.shape[1]
        return n_new == n_prev + 1

    def _fit_incremental(self, gp, X_train, y_train, prev_gp) -> Optional[GPFitResult]:
        """Perform rank-1 Cholesky update.

        Returns None if the update fails (e.g., non-positive diagonal),
        signaling the caller to fall back to full Cholesky.
        """
        bkd = self._bkd
        clone = gp._clone_unfitted()

        # Install transforms from prev_gp
        clone._output_transform = prev_gp._output_transform
        clone._input_transform = prev_gp._input_transform

        # Scale full data using the same transforms
        X_scaled = clone._input_transform.transform(X_train)
        y_scaled = y_train
        if clone._output_transform is not None:
            y_scaled = clone._output_transform.inverse_transform(y_train)

        n_prev = prev_gp.data().n_samples()

        # Extract the new point (last column) in scaled space
        X_new_scaled = X_scaled[:, n_prev:]  # (nvars, 1)

        # Get previous Cholesky factor and scaled training data
        L_prev = prev_gp.cholesky().factor()  # (n_prev, n_prev)
        X_prev_scaled = prev_gp.data().X()  # (nvars, n_prev)

        # k_cross = kernel(X_new, X_prev) -> (1, n_prev)
        k_cross = clone._kernel(X_new_scaled, X_prev_scaled)

        # Forward substitution: v = L_prev^{-1} @ k_cross^T -> (n_prev, 1)
        v = bkd.solve_triangular(L_prev, k_cross.T, lower=True)

        # k_self = kernel(X_new, X_new) -> (1, 1)
        k_self = clone._kernel(X_new_scaled, X_new_scaled)

        # l_new_sq = k_self + nugget - v^T @ v
        l_new_sq = k_self[0, 0] + clone._nugget - bkd.sum(v * v)

        # Check positivity — fall back if not positive
        l_new_sq_val = float(bkd.to_numpy(bkd.reshape(l_new_sq, (1,)))[0])
        if l_new_sq_val <= 0:
            return None

        l_new = bkd.sqrt(bkd.reshape(l_new_sq, (1, 1)))

        # Build L_new = [[L_prev, 0], [v^T, l_new]]
        _n_total = n_prev + 1
        zeros_col = bkd.zeros((n_prev, 1))
        top_row = bkd.hstack([L_prev, zeros_col])  # (n_prev, n_total)
        bottom_row = bkd.hstack([v.T, l_new])  # (1, n_total)
        L_new = bkd.vstack([top_row, bottom_row])  # (n_total, n_total)

        cholesky = CholeskyFactor(L_new, bkd)

        # Store training data on clone
        clone._data = GPTrainingData(
            X_scaled,
            y_scaled,
            bkd,
            output_transform=clone._output_transform,
        )
        clone._cholesky = cholesky

        # Compute alpha = cholesky.solve(residual^T)^T
        mean_pred = clone._mean(X_scaled)
        residual = y_scaled - mean_pred
        clone._alpha = cholesky.solve(residual.T).T

        nll = clone.neg_log_marginal_likelihood()

        return GPFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=nll,
        )

    def _fit_full(self, gp, X_train, y_train) -> GPFitResult:
        """Fall back to full Cholesky factorization."""
        from pyapprox.surrogates.gaussianprocess.fitters.fixed_hyperparameter_fitter import (  # noqa: E501
            GPFixedHyperparameterFitter,
        )

        fitter = GPFixedHyperparameterFitter(self._bkd)
        return fitter.fit(gp, X_train, y_train)

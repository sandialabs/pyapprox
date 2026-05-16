"""Multi-output kernel latent regressor wrapping MultiOutputGP."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generic, List, Optional, Tuple

from pyapprox.surrogates.gaussianprocess.fitters import (
    MultiOutputGPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.gaussianprocess.multioutput import (
    MultiOutputGP,
)
from pyapprox.surrogates.kernels.multioutput.protocols import (
    MultiOutputKernelProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)

if TYPE_CHECKING:
    from pyapprox.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )


class MultiOutputKernelLatentRegressor(Generic[Array]):
    """Latent regressor using a multi-output kernel.

    Wraps MultiOutputGP. Converts flat code arrays to the list format
    expected by MultiOutputGP.

    Parameters
    ----------
    kernel : MultiOutputKernelProtocol[Array]
        Multi-output kernel with noutputs() == ncodes_out.
    ncodes_in : int
        Number of input codes.
    ncodes_out : int
        Number of output codes.
    bkd : Backend[Array]
        Computational backend.
    nugget : float
        Nugget for numerical stability.
    """

    def __init__(
        self,
        kernel: MultiOutputKernelProtocol[Array],
        ncodes_in: int,
        ncodes_out: int,
        bkd: Backend[Array],
        nugget: float = 1e-6,
    ) -> None:
        if kernel.noutputs() != ncodes_out:
            raise ValueError(
                f"kernel.noutputs() ({kernel.noutputs()}) must equal "
                f"ncodes_out ({ncodes_out})"
            )
        self._gp: MultiOutputGP[Array] = MultiOutputGP(
            kernel, nugget
        )
        self._ncodes_in = ncodes_in
        self._ncodes_out = ncodes_out
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ncodes_in(self) -> int:
        return self._ncodes_in

    def ncodes_out(self) -> int:
        return self._ncodes_out

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._gp.hyp_list()

    def is_fitted(self) -> bool:
        return self._gp._is_fitted

    def _to_list_format(
        self, U: Array, V: Array
    ) -> Tuple[List[Array], List[Array]]:
        X_list = [U] * self._ncodes_out
        y_list = [V[j : j + 1, :] for j in range(self._ncodes_out)]
        return X_list, y_list

    def fit_internal(self, U: Array, V: Array) -> None:
        """Fit via MultiOutputGP. U: (n, N), V: (m, N)."""
        X_list, y_list = self._to_list_format(U, V)
        self._gp._fit_internal(X_list, y_list)

    def predict(self, U_test: Array) -> Array:
        """Predict output codes. Returns (ncodes_out, N_test)."""
        X_list = [U_test] * self._ncodes_out
        pred_list = self._gp.predict(X_list)
        return self._bkd.concatenate(pred_list, axis=0)

    def predict_std(self, U_test: Array) -> Array:
        """Predict std of output codes. Returns (ncodes_out, N_test)."""
        X_list = [U_test] * self._ncodes_out
        _, std_list = self._gp.predict_with_uncertainty(X_list)
        return self._bkd.concatenate(std_list, axis=0)

    def neg_log_marginal_likelihood(self) -> Array:
        return self._gp.neg_log_marginal_likelihood()

    def clone_unfitted(self) -> MultiOutputKernelLatentRegressor[Array]:
        clone = copy.copy(self)
        clone._gp = self._gp._clone_unfitted()
        return clone

    def fit_with_optimizer(
        self,
        U: Array,
        V: Array,
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> Tuple[Array, Array, Optional[object]]:
        """Fit with hyperparameter optimization.

        Returns (initial_hyps, optimized_hyps, opt_result).
        """
        X_list, y_list = self._to_list_format(U, V)
        fitter = MultiOutputGPMaximumLikelihoodFitter(
            self._bkd, optimizer
        )
        gp_result = fitter.fit(self._gp, X_list, y_list)
        self._gp = gp_result.surrogate()  # type: ignore[assignment]
        return (
            gp_result.initial_hyperparameters(),
            gp_result.optimized_hyperparameters(),
            gp_result.optimization_result(),
        )

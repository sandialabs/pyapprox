"""Scalar-kernel latent regressor wrapping ExactGaussianProcess."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generic, Optional, Tuple

from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)

if TYPE_CHECKING:
    from pyapprox.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )


class ScalarKernelLatentRegressor(Generic[Array]):
    """Latent regressor using a scalar kernel with multi-RHS solve.

    Wraps ExactGaussianProcess. Implements the paper's default case
    Gamma = S * I_m: one N x N Cholesky shared across all m output
    codes via ExactGP's multi-RHS alpha computation.

    Parameters
    ----------
    kernel : Kernel[Array]
        Scalar kernel with nvars == ncodes_in.
    ncodes_in : int
        Number of input codes.
    ncodes_out : int
        Number of output codes.
    bkd : Backend[Array]
        Computational backend.
    nugget : float
        Nugget for numerical stability.
    mean_function : optional
        Mean function for the GP.
    """

    def __init__(
        self,
        kernel: object,
        ncodes_in: int,
        ncodes_out: int,
        bkd: Backend[Array],
        nugget: float = 1e-6,
        mean_function: object = None,
    ) -> None:
        self._gp = ExactGaussianProcess(
            kernel, ncodes_in, bkd, mean_function, nugget  # type: ignore[arg-type]
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
        return self._gp.is_fitted()

    def fit_internal(self, U: Array, V: Array) -> None:
        """Fit via ExactGP multi-RHS. U: (n, N), V: (m, N)."""
        self._gp._fit_internal(U, V)

    def predict(self, U_test: Array) -> Array:
        """Predict output codes. Returns (ncodes_out, N_test)."""
        return self._gp.predict(U_test)

    def predict_std(self, U_test: Array) -> Array:
        """Predict std of output codes. Returns (ncodes_out, N_test)."""
        return self._gp.predict_std(U_test)

    def neg_log_marginal_likelihood(self) -> Array:
        return self._gp.neg_log_marginal_likelihood()

    def clone_unfitted(self) -> ScalarKernelLatentRegressor[Array]:
        clone = copy.copy(self)
        clone._gp = self._gp._clone_unfitted()
        return clone

    def fit_with_optimizer(
        self,
        U: Array,
        V: Array,
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> Tuple[Array, Array, Optional[object]]:
        """Fit with hyperparameter optimization via GPMaximumLikelihoodFitter.

        Returns (initial_hyps, optimized_hyps, opt_result).
        """
        from pyapprox.surrogates.gaussianprocess.fitters.maximum_likelihood_fitter import (
            GPMaximumLikelihoodFitter,
        )

        fitter = GPMaximumLikelihoodFitter(self._bkd, optimizer)
        gp_result = fitter.fit(self._gp, U, V)
        self._gp = gp_result.surrogate()  # type: ignore[assignment]
        return (
            gp_result.initial_hyperparameters(),
            gp_result.optimized_hyperparameters(),
            gp_result.optimization_result(),
        )

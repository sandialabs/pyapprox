"""Loss functions for FunctionTrain fitting."""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.typing.surrogates.functiontrain.compute import (
    cache_basis_matrices,
    BasisCache,
)


class FunctionTrainMSELoss(Generic[Array]):
    """Mean squared error loss for FunctionTrain fitting.

    L(params) = (1/2n) ||f_params(X) - Y||^2

    Provides analytical jacobian for efficient optimization.
    Conforms to ObjectiveProtocol for use with ScipyTrustConstrOptimizer.

    Caches basis matrices at construction time since training samples are
    fixed. Also caches the with_params result between __call__ and jacobian
    since scipy calls both with the same params.

    Naming convention (to avoid confusion with FunctionTrain's samples):
    - self._train_samples: training input data (nvars, nsamples)
    - self._train_values: training target values (nqoi, nsamples)
    - params: optimization variable (the FT parameters being optimized)

    Parameters
    ----------
    surrogate : FunctionTrain[Array]
        FunctionTrain surrogate to fit.
    train_samples : Array
        Training samples. Shape: (nvars, nsamples)
    train_values : Array
        Target values. Shape: (nqoi, nsamples)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        surrogate: FunctionTrain[Array],
        train_samples: Array,
        train_values: Array,
        bkd: Backend[Array],
    ):
        self._surrogate = surrogate
        self._train_samples = train_samples
        self._train_values = train_values  # (nqoi, nsamples)
        self._bkd = bkd
        self._nsamples = train_samples.shape[1]
        self._nqoi = surrogate.nqoi()

        # Cache basis matrices (samples are fixed during training)
        self._basis_cache: BasisCache = cache_basis_matrices(
            surrogate.cores(), train_samples, bkd
        )

        # Param-caching: scipy calls fun(x) then jac(x) with same x
        self._cached_params: Optional[Array] = None
        self._cached_ft: Optional[FunctionTrain[Array]] = None
        self._cached_pred: Optional[Array] = None

    def _ensure_cached(self, params_flat: Array) -> None:
        """Ensure cached FT and predictions are up-to-date for params.

        Uses value-based caching: if params match the cached values, reuse
        the cached FT and predictions. This avoids redundant work when scipy
        calls fun(x) then jac(x) with the same x.

        Caching is skipped when params require gradients (torch autograd)
        since the computation graph must flow through the current params.
        """
        # Skip cache when autograd is active (requires_grad tensors)
        requires_grad = getattr(params_flat, "requires_grad", False)
        if not requires_grad and (
            self._cached_params is not None
            and self._bkd.allclose(
                params_flat, self._cached_params, rtol=0.0, atol=0.0
            )
        ):
            return

        if not requires_grad:
            self._cached_params = self._bkd.copy(params_flat)
        else:
            self._cached_params = None
        self._cached_ft = self._surrogate.with_params(params_flat)
        self._cached_pred = self._cached_ft.eval_cached(
            self._train_samples, self._basis_cache
        )

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of parameters to optimize."""
        return self._surrogate.nparams()

    def nqoi(self) -> int:
        """Loss is scalar."""
        return 1

    def __call__(self, params: Array) -> Array:
        """Compute loss.

        Parameters
        ----------
        params : Array
            FunctionTrain parameters. Shape: (nvars, 1) or (nvars,)

        Returns
        -------
        Array
            Loss value. Shape: (1, 1)
        """
        params_flat = self._bkd.flatten(params)
        self._ensure_cached(params_flat)

        # Compute MSE: (1/2n) ||pred - values||^2
        residual = self._cached_pred - self._train_values
        mse = 0.5 * self._bkd.sum(residual ** 2) / self._nsamples
        return self._bkd.reshape(mse, (1, 1))

    def jacobian(self, params: Array) -> Array:
        """Compute gradient of loss w.r.t. params.

        grad = (1/n) sum_i (f(x_i) - y_i) * df/d(params)

        Parameters
        ----------
        params : Array
            FunctionTrain parameters. Shape: (nvars, 1) or (nvars,)

        Returns
        -------
        Array
            Gradient. Shape: (1, nvars)
        """
        params_flat = self._bkd.flatten(params)
        self._ensure_cached(params_flat)

        # Compute residuals
        residual = self._cached_pred - self._train_values

        # Compute Jacobian using cached basis matrices
        jac = self._cached_ft.jacobian_wrt_params_cached(
            self._train_samples, self._basis_cache
        )

        # Chain rule: grad = (1/n) sum_s sum_q residual[q,s] * jac[s,q,:]
        grad = self._bkd.einsum("sq, sqp -> p", residual.T, jac)
        grad = grad / self._nsamples

        return self._bkd.reshape(grad, (1, -1))

    def __repr__(self) -> str:
        return (
            f"FunctionTrainMSELoss(nvars={self.nvars()}, "
            f"nsamples={self._nsamples}, nqoi={self._nqoi})"
        )

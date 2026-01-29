"""Loss functions for FunctionTrain fitting."""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.functiontrain.functiontrain import FunctionTrain


class FunctionTrainMSELoss(Generic[Array]):
    """Mean squared error loss for FunctionTrain fitting.

    L(params) = (1/2n) ||f_params(X) - Y||^2

    Provides analytical jacobian for efficient optimization.
    Conforms to ObjectiveProtocol for use with ScipyTrustConstrOptimizer.

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
        # Flatten params if needed
        params_flat = self._bkd.flatten(params)

        # Create FT with new params and evaluate
        ft = self._surrogate.with_params(params_flat)
        pred = ft(self._train_samples)  # (nqoi, nsamples)

        # Compute MSE: (1/2n) ||pred - values||^2
        residual = pred - self._train_values  # (nqoi, nsamples)
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
        # Flatten params if needed
        params_flat = self._bkd.flatten(params)

        # Create FT with new params
        ft = self._surrogate.with_params(params_flat)

        # Compute predictions and residuals
        pred = ft(self._train_samples)  # (nqoi, nsamples)
        residual = pred - self._train_values  # (nqoi, nsamples)

        # Compute Jacobian of FT output w.r.t. params
        # Shape: (nsamples, nqoi, nparams)
        jac = ft.jacobian_wrt_params(self._train_samples)

        # Compute gradient using chain rule:
        # grad = (1/n) sum_s sum_q residual[q,s] * jac[s,q,:]
        # Using einsum: "qs, sqp -> p" where residual is (nqoi, nsamples)
        # residual.T gives (nsamples, nqoi)
        grad = self._bkd.einsum("sq, sqp -> p", residual.T, jac)
        grad = grad / self._nsamples

        return self._bkd.reshape(grad, (1, -1))

    def __repr__(self) -> str:
        return (
            f"FunctionTrainMSELoss(nvars={self.nvars()}, "
            f"nsamples={self._nsamples}, nqoi={self._nqoi})"
        )

"""
ELBO loss function for Variational Gaussian Process optimization.

This module provides the VariationalGPELBOLoss class which wraps
the variational GP's negative ELBO for use with optimizers.
"""

from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class VariationalGPELBOLoss(Generic[Array]):
    """
    ELBO loss for variational GP hyperparameter optimization.

    Wraps the variational GP's neg_log_marginal_likelihood() (negative ELBO)
    for use with BindableOptimizerProtocol. No analytical jacobian is
    provided; use TorchVariationalGaussianProcess for autograd-based
    gradients.

    Parameters
    ----------
    gp : VariationalGaussianProcess
        The variational GP model.
    fit_args : tuple
        Arguments to pass to gp._fit_internal(): (X_train, y_train).
    """

    def __init__(
        self,
        gp,  # VariationalGaussianProcess
        fit_args: Tuple,
    ) -> None:
        self._gp = gp
        self._fit_args = fit_args
        self._bkd = gp.bkd()
        self._hyp_list = gp.hyp_list()

    def nvars(self) -> int:
        """Number of active hyperparameters."""
        return self._hyp_list.nactive_params()

    def nqoi(self) -> int:
        """Always 1 for scalar loss."""
        return 1

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list."""
        return self._hyp_list

    def __call__(self, params: Array) -> Array:
        """Compute negative ELBO.

        Parameters
        ----------
        params : Array
            Active hyperparameters, shape (nactive,) or (nactive, 1).

        Returns
        -------
        Array
            Negative ELBO, shape (1, 1).
        """
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        self._hyp_list.set_active_values(params)
        self._gp._fit_internal(*self._fit_args)
        neg_elbo = self._gp.neg_log_marginal_likelihood()

        neg_elbo_arr = (
            self._bkd.reshape(neg_elbo, (1,))
            if hasattr(neg_elbo, "shape")
            else self._bkd.array([neg_elbo])
        )
        return self._bkd.reshape(neg_elbo_arr, (1, 1))

    def __repr__(self) -> str:
        return (
            f"VariationalGPELBOLoss(nvars={self.nvars()}, "
            f"gp_type={type(self._gp).__name__})"
        )

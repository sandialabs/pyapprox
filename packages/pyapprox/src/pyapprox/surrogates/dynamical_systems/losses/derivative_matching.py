"""Derivative matching loss for dynamical systems learning.

L(eta) = (1/2N) sum_i ||dx_i/dt - F_eta(x_i)||^2
"""

from typing import Generic

from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.protocols import (
    LearnedFunctionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class DerivativeMatchingLoss(Generic[Array]):
    """Mean squared error between predicted and observed derivatives.

    L(eta) = (1/2N) sum_i ||dx_i/dt - F_eta(x_i)||^2

    Implements FunctionWithJacobianProtocol:
    - nvars() = nactive_params
    - nqoi() = 1
    - __call__(params: (nvars, 1)) -> (1, 1)
    - jacobian(params: (nvars, 1)) -> (1, nvars)

    Parameters
    ----------
    learned_function : LearnedFunctionProtocol[Array]
        Learned function to evaluate.
    dataset : SnapshotDataset[Array]
        Training data with states and derivatives.
    """

    def __init__(
        self,
        learned_function: LearnedFunctionProtocol[Array],
        dataset: SnapshotDataset[Array],
    ):
        if not isinstance(learned_function, LearnedFunctionProtocol):
            raise TypeError(
                f"learned_function must satisfy LearnedFunctionProtocol, "
                f"got {type(learned_function).__name__}"
            )
        if dataset.nstates_input() != learned_function.nvars():
            raise ValueError(
                f"dataset.nstates_input()={dataset.nstates_input()} != "
                f"learned_function.nvars()={learned_function.nvars()}"
            )
        if dataset.nstates_output() != learned_function.nqoi():
            raise ValueError(
                f"dataset.nstates_output()={dataset.nstates_output()} != "
                f"learned_function.nqoi()={learned_function.nqoi()}"
            )
        self._lf = learned_function
        self._dataset = dataset
        self._bkd = learned_function.bkd()
        self._states = dataset.states()
        self._derivs = dataset.derivatives()
        self._nsamples = dataset.nsamples()
        self.jacobian = self._jacobian

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._lf.hyp_list().nactive_params()

    def nqoi(self) -> int:
        return 1

    def __call__(self, params: Array) -> Array:
        """Evaluate loss L(eta).

        Parameters
        ----------
        params : Array
            Shape: (nvars, 1)

        Returns
        -------
        Array
            Shape: (1, 1)
        """
        self._lf.hyp_list().set_active_values(params[:, 0])
        residual = self._lf(self._states) - self._derivs
        loss = self._bkd.sum(residual * residual) / (2.0 * self._nsamples)
        return self._bkd.reshape(loss, (1, 1))

    def _jacobian(self, params: Array) -> Array:
        """Compute dL/d_eta.

        Parameters
        ----------
        params : Array
            Shape: (nvars, 1)

        Returns
        -------
        Array
            Shape: (1, nvars)
        """
        self._lf.hyp_list().set_active_values(params[:, 0])
        residual = self._lf(self._states) - self._derivs
        # residual: (nqoi, nsamples)
        # jacobian_wrt_params: (nsamples, nqoi, nactive)
        pjac = self._lf.jacobian_wrt_params(self._states)
        # einsum: "ji,ijk->k" sums over samples (j) and qoi (i)
        grad = self._bkd.einsum(
            "ji,ijk->k", residual, pjac
        ) / self._nsamples
        return self._bkd.reshape(grad, (1, -1))

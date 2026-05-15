"""Derivative matching loss for dynamical systems learning.

L(eta) = (1/2N) sum_i ||dx_i/dt - F_eta(x_i)||^2
"""

from typing import Generic

from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.protocols import (
    ParametricVectorFieldProtocol,
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

    jacobian is dynamically bound only if the vector field has
    param_jacobian.

    Parameters
    ----------
    vector_field : ParametricVectorFieldProtocol[Array]
        Parametric vector field to evaluate.
    dataset : SnapshotDataset[Array]
        Training data with states and derivatives.
    """

    def __init__(
        self,
        vector_field: ParametricVectorFieldProtocol[Array],
        dataset: SnapshotDataset[Array],
    ):
        if not isinstance(vector_field, ParametricVectorFieldProtocol):
            raise TypeError(
                f"vector_field must satisfy ParametricVectorFieldProtocol, "
                f"got {type(vector_field).__name__}"
            )
        self._vf = vector_field
        self._dataset = dataset
        self._bkd = vector_field.bkd()
        self._states = dataset.states()
        self._derivs = dataset.derivatives()
        self._nsamples = dataset.nsamples()
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        if hasattr(self._vf, "param_jacobian"):
            self.jacobian = self._jacobian

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._vf.hyp_list().nactive_params()

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
        self._vf.hyp_list().set_active_values(params[:, 0])
        residual = self._vf(self._states) - self._derivs
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
        self._vf.hyp_list().set_active_values(params[:, 0])
        residual = self._vf(self._states) - self._derivs
        # residual: (nstates, nsamples)
        # param_jacobian: (nsamples, nstates, nactive)
        pjac = self._vf.param_jacobian(self._states)
        # dL/d_eta = (1/N) sum_i residual_i^T @ pjac_i
        # residual.T: (nsamples, nstates) -> unsqueeze to (nsamples, 1, nstates)
        # pjac: (nsamples, nstates, nactive)
        # einsum: sum over samples and states
        grad = self._bkd.einsum(
            "ji,ijk->k", residual, pjac
        ) / self._nsamples
        return self._bkd.reshape(grad, (1, -1))

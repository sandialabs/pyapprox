"""Protocols for MFNet node models.

Defines the interface contracts that node models must satisfy to be
used within an MFNet network. Three protocol levels:

- NodeModelProtocol: minimum required for any node model
- NodeModelWithParamJacobianProtocol: enables analytical gradient computation
- LinearNodeModelProtocol: enables ALS direct solve via basis matrix
"""

from typing import Generic, runtime_checkable

from typing_extensions import Protocol

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class NodeModelProtocol(Protocol, Generic[Array]):
    """Protocol for models that can live inside an MFNet node.

    Any model satisfying this protocol can be placed in a node.
    The existing ``BasisExpansion`` satisfies this protocol.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate the model.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: ``(nvars, nsamples)``

        Returns
        -------
        Array
            Output values. Shape: ``(nqoi, nsamples)``
        """
        ...

    def hyp_list(self) -> HyperParameterList: ...


@runtime_checkable
class NodeModelWithParamJacobianProtocol(
    NodeModelProtocol[Array], Protocol, Generic[Array]
):
    """Extension for node models providing jacobian_wrt_params.

    Enables analytical gradient computation through the network.
    """

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """Jacobian of output w.r.t. active parameters.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: ``(nvars, nsamples)``

        Returns
        -------
        Array
            Jacobian. Shape: ``(nsamples, nqoi, nactive_params)``
        """
        ...


@runtime_checkable
class LinearNodeModelProtocol(NodeModelProtocol[Array], Protocol, Generic[Array]):
    """Extension for node models linear in their coefficients.

    Enables ALS direct solve via least squares on the basis matrix.
    The existing ``BasisExpansion`` satisfies this protocol.
    """

    def basis_matrix(self, samples: Array) -> Array:
        """Compute the design matrix.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: ``(nvars, nsamples)``

        Returns
        -------
        Array
            Basis matrix. Shape: ``(nsamples, nterms)``
        """
        ...

    def get_coefficients(self) -> Array:
        """Return coefficients. Shape: ``(nterms, nqoi)``."""
        ...

    def set_coefficients(self, coef: Array) -> None:
        """Set coefficients.

        Parameters
        ----------
        coef : Array
            Coefficients. Shape: ``(nterms, nqoi)``
        """
        ...

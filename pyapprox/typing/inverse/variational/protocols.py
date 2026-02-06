"""
Protocols for variational inference components.

Defines the interfaces for variational families, amortization functions,
and covariance parameterizations.
"""

from typing import Any, Generic, Optional, Protocol, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList


@runtime_checkable
class VariationalFamilyProtocol(Protocol, Generic[Array]):
    """Protocol for variational families.

    Methods accept an optional ``params`` argument of shape
    ``(nactive_params, nsamples)`` containing per-sample unconstrained
    active parameters.  When ``params`` is ``None``, the family uses
    its ``hyp_list()`` values (shared across all samples).  When
    provided, the family merges the per-sample active params with any
    fixed params from ``hyp_list()`` internally.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def hyp_list(self) -> HyperParameterList: ...

    def reparameterize(
        self, base_samples: Array, params: Optional[Array] = None
    ) -> Array: ...

    def logpdf(
        self, samples: Array, params: Optional[Array] = None
    ) -> Array: ...

    def kl_divergence(
        self, prior: Any, params: Optional[Array] = None
    ) -> Array: ...

    def base_distribution(self) -> Any: ...


@runtime_checkable
class AmortizationFunctionProtocol(Protocol, Generic[Array]):
    """Protocol for amortization functions.

    Maps label vectors to per-sample variational parameters.

    ``__call__(labels)`` takes ``labels`` of shape
    ``(nlabel_dims, nsamples)`` and returns ``(nparams_out, nsamples)``.
    """

    def bkd(self) -> Backend[Array]: ...

    def hyp_list(self) -> HyperParameterList: ...

    def __call__(self, labels: Array) -> Array: ...

    def nparams_out(self) -> int: ...


@runtime_checkable
class CovarianceParameterizationProtocol(Protocol, Generic[Array]):
    """Protocol for covariance parameterizations used by Gaussian copula.

    ``inv_quad_correlation(zeta)`` returns the *excess* quadratic form
    ``zeta^T (R^{-1} - I) zeta`` that appears in the Gaussian copula
    density:  ``log c_R(u) = -0.5 * log|R| - 0.5 * zeta^T (R^{-1} - I) zeta``.
    When ``R = I`` (independence), this returns 0.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def hyp_list(self) -> HyperParameterList: ...

    def nbase_samples(self) -> int: ...

    def transform_base_samples(self, epsilon: Array) -> Array: ...

    def diagonal(self) -> Array: ...

    def log_det_correlation(self) -> Array: ...

    def inv_quad_correlation(self, zeta: Array) -> Array: ...

"""
Protocols for variational inference components.

Defines the interface for covariance parameterizations used by
Gaussian copula variational families.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


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

    def hyp_list(self) -> HyperParameterList[Array]: ...

    def nbase_samples(self) -> int: ...

    def transform_base_samples(self, epsilon: Array) -> Array: ...

    def diagonal(self) -> Array: ...

    def log_det_correlation(self) -> Array: ...

    def inv_quad_correlation(self, zeta: Array) -> Array: ...


@runtime_checkable
class VariationalDistributionProtocol(Protocol, Generic[Array]):
    """Protocol for variational distributions used in VI diagnostics.

    Combines the conditional distribution interface with the
    reparameterization and hyperparameter methods needed for
    importance-weighted diagnostics.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def hyp_list(self) -> HyperParameterList[Array]: ...

    def logpdf(self, x: Array, y: Array) -> Array: ...

    def reparameterize(self, x: Array, base_samples: Array) -> Array: ...

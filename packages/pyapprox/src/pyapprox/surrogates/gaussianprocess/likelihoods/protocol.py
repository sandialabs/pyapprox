"""Protocol for likelihood functions."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class LikelihoodProtocol(Protocol, Generic[Array]):
    """Protocol for observation likelihoods p(y|f).

    Likelihoods connect latent GP function values f to observed data y.
    """

    def hyp_list(self) -> HyperParameterList[Array]: ...

    def log_prob(self, y: Array, f: Array) -> Array:
        """Pointwise log p(y_i | f_i), shape (1, n_points)."""
        ...

    def expected_log_prob(
        self, y: Array, f_mean: Array, f_var: Array
    ) -> Array:
        """E_{q(f)}[log p(y|f)] where q(f) = N(f_mean, f_var).

        Returns shape (1, n_points).
        """
        ...

    def bkd(self) -> Backend[Array]: ...

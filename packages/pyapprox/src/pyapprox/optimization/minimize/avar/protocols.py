"""
Protocols for AVaR (Average Value at Risk) optimization.

AVaR optimization minimizes the average of the worst (1-alpha) fraction of
outcomes across multiple objectives.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class AVaRSlackObjectiveProtocol(Protocol, Generic[Array]):
    """
    Protocol for AVaR slack-based objective functions.

    These objectives handle the transformation of AVaR optimization into
    a constrained problem with t (VaR) and s_i (excess) slack variables.
    """

    def bkd(self) -> Backend[Array]:
        """Get computational backend."""
        ...

    def nvars(self) -> int:
        """Total number of variables (slack + original)."""
        ...

    def nqoi(self) -> int:
        """Number of quantities of interest (always 1)."""
        ...

    def nslack(self) -> int:
        """Number of slack variables (1 + nscenarios for AVaR)."""
        ...

    def alpha(self) -> float:
        """Risk level in [0, 1)."""
        ...

    def __call__(self, sample: Array) -> Array:
        """Evaluate objective."""
        ...

    def jacobian(self, sample: Array) -> Array:
        """Jacobian of objective."""
        ...

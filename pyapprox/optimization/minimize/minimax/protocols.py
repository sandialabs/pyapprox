"""
Protocols for minimax optimization.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class MultiQoIObjectiveProtocol(Protocol, Generic[Array]):
    """
    Protocol for multi-output objective functions.

    These are objectives that return multiple values (nqoi > 1) and are
    candidates for minimax optimization.
    """

    def bkd(self) -> Backend[Array]:
        """Get computational backend."""
        ...

    def nvars(self) -> int:
        """Number of input variables."""
        ...

    def nqoi(self) -> int:
        """Number of quantities of interest (outputs)."""
        ...

    def __call__(self, sample: Array) -> Array:
        """
        Evaluate objective at sample.

        Parameters
        ----------
        sample : Array
            Input sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Objective values. Shape: (nqoi, 1)
        """
        ...

    def jacobian(self, sample: Array) -> Array:
        """
        Jacobian of objective.

        Parameters
        ----------
        sample : Array
            Input sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nqoi, nvars)
        """
        ...


@runtime_checkable
class SlackBasedObjectiveProtocol(Protocol, Generic[Array]):
    """
    Protocol for slack-based objective functions (minimax, AVaR).

    These objectives are used after transforming the original problem
    by introducing slack variables.
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
        """Number of slack variables."""
        ...

    def __call__(self, sample: Array) -> Array:
        """Evaluate objective."""
        ...

    def jacobian(self, sample: Array) -> Array:
        """Jacobian of objective."""
        ...

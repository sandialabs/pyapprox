"""
Protocols for OED objective functions.

The primary objective is the expected information gain (EIG):
    EIG = E_obs[log p(obs | design)] - E_obs,theta[log p(obs | theta, design)]
        = E_obs[log(evidence)] - E_obs[log(likelihood at true theta)]

This is the KL divergence between posterior and prior, averaged over data.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class OEDObjectiveProtocol(Protocol, Generic[Array]):
    """
    Protocol for OED objective functions.

    Function-shaped with a Derivatives bundle (see interface/functions).
    The objective takes design weights and returns a scalar value. The
    objective owns its derivative capability, declared via
    ``derivatives()``; an empty bundle is valid and leaves optimizers to
    their own finite differences on the value.

    For minimization, returns negative EIG.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of design variables (= nobs).
    nqoi()
        Number of quantities of interest (= 1 for scalar objective).
    __call__(design_weights)
        Evaluate objective at design weights.
    derivatives()
        Derivative bundle for the objective.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Number of design variables.

        Returns
        -------
        int
            Number of observation locations (nobs).
        """
        ...

    def nqoi(self) -> int:
        """
        Number of quantities of interest.

        Returns
        -------
        int
            Always 1 for scalar objective.
        """
        ...

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate objective at design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Objective value. Shape: (1, 1)
        """
        ...

    def derivatives(self) -> Derivatives[Array]:
        """
        Derivative bundle for the objective.

        Returns
        -------
        Derivatives[Array]
            The objective's derivative capability. May be empty.
        """
        ...


@runtime_checkable
class KLOEDObjectiveProtocol(Protocol, Generic[Array]):
    """
    Protocol for KL-based OED objective (expected information gain).

    Extends OEDObjectiveProtocol with methods specific to KL-OED:
    - Access to inner/outer sample counts
    - Expected information gain computation

    The KL-OED objective computes:
        -EIG = -(E_outer[log(evidence)] - E_outer[log(likelihood_true)])

    Negative sign converts maximization of EIG to minimization.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """Number of design variables (= nobs)."""
        ...

    def nqoi(self) -> int:
        """Number of quantities of interest (= 1)."""
        ...

    def __call__(self, design_weights: Array) -> Array:
        """Evaluate objective. Shape: (nobs, 1) -> (1, 1)"""
        ...

    def derivatives(self) -> Derivatives[Array]:
        """Derivative bundle for the objective. May be empty."""
        ...

    def ninner(self) -> int:
        """
        Number of inner (prior) samples for evidence integration.

        Returns
        -------
        int
            Number of inner samples.
        """
        ...

    def nouter(self) -> int:
        """
        Number of outer (observation) samples for expectation.

        Returns
        -------
        int
            Number of outer samples.
        """
        ...

    def expected_information_gain(self, design_weights: Array) -> float:
        """
        Compute expected information gain (positive value).

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        float
            Expected information gain.
        """
        ...

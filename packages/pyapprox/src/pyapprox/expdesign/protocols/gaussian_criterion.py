"""Protocol for design criteria on a relaxed linear-Gaussian observation.

A criterion scores one target seen through the relaxed observation at
some design weights, and gives the gradient of that score with respect to
the weights ``w`` and the independent-noise variances ``nu``. It reads
whatever it needs from the observation: the posterior covariance for
A- and D-optimality, the two log-determinants for the expected
information gain. The relaxation and the chain rule through ``nu(w)`` are
applied by the design objective, not here.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.inverse.joint_gaussian import LinearGaussianObservation
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class GaussianDesignCriterionProtocol(Protocol, Generic[Array]):
    """A criterion to minimize over designs.

    Methods
    -------
    value(observation)
        The criterion. Shape: (1,)
    gradient(observation)
        Its gradients with respect to ``w`` and ``nu``, each (d, 1).
    """

    def value(self, observation: LinearGaussianObservation[Array]) -> Array:
        """The criterion at the observation's weights. Shape: (1,)"""
        ...

    def gradient(
        self, observation: LinearGaussianObservation[Array]
    ) -> Tuple[Array, Array]:
        """Gradients with respect to ``w`` and ``nu``, each of shape (d, 1)."""
        ...

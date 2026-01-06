"""
Protocols for evidence computation in Bayesian OED.

The evidence (marginal likelihood) is:
    p(obs | design) = integral p(obs | theta, design) p(theta) d theta

For OED, we compute evidence for multiple potential observations (outer loop)
by integrating over prior samples (inner loop).
"""

from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class EvidenceProtocol(Protocol, Generic[Array]):
    """
    Protocol for evidence computation.

    Computes evidence for each outer sample by averaging likelihoods
    over inner samples (prior integration).

    Methods
    -------
    bkd()
        Get the computational backend.
    nouter()
        Number of outer (observation) samples.
    __call__(design_weights)
        Compute evidence for all outer samples.
    jacobian(design_weights)
        Jacobian of evidence w.r.t. design weights.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nouter(self) -> int:
        """Return the number of outer samples."""
        ...

    def __call__(self, design_weights: Array) -> Array:
        """
        Compute evidence for all outer samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Evidence values. Shape: (1, nouter)
        """
        ...

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of evidence w.r.t. design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nouter, nobs)
        """
        ...


@runtime_checkable
class LogEvidenceProtocol(Protocol, Generic[Array]):
    """
    Protocol for log-evidence computation.

    Computes log(evidence) with numerical stability using log-sum-exp.
    This is the primary quantity used in KL-OED objective.

    log p(obs | design) = log(sum_i w_i * exp(log p(obs | theta_i, design)))
                        = log_sum_exp(log_weights + log_likelihoods)

    Methods
    -------
    bkd()
        Get the computational backend.
    nouter()
        Number of outer (observation) samples.
    __call__(design_weights)
        Compute log-evidence for all outer samples.
    jacobian(design_weights)
        Jacobian of log-evidence w.r.t. design weights.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nouter(self) -> int:
        """Return the number of outer samples."""
        ...

    def __call__(self, design_weights: Array) -> Array:
        """
        Compute log-evidence for all outer samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Log-evidence values. Shape: (1, nouter)
        """
        ...

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of log-evidence w.r.t. design weights.

        Uses chain rule: d/dw log(E) = (1/E) * dE/dw

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nouter, nobs)
        """
        ...

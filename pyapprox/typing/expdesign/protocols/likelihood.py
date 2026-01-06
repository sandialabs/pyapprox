"""
Protocols for OED-specific likelihood functions.

These protocols extend the base likelihood protocols with methods needed
for optimal experimental design, including:
- Vectorized evaluation producing (ninner, nouter) likelihood matrices
- Jacobians with respect to design weights
- Latent sample management for reparameterization trick
"""

from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class OEDOuterLoopLikelihoodProtocol(Protocol, Generic[Array]):
    """
    Protocol for outer loop likelihood in Bayesian OED.

    The outer loop averages over potential observations (data realizations).
    This likelihood evaluates log p(obs | theta) where obs and shapes (means)
    have the same shape (nobs, nouter).

    The design weights control the noise covariance:
        Cov = diag(base_variance / weights)

    Methods
    -------
    bkd()
        Get the computational backend.
    nobs()
        Number of observation locations.
    set_shapes(shapes)
        Set model outputs (means for Gaussian).
    set_observations(obs)
        Set artificial observations.
    set_design_weights(weights)
        Set design weights.
    set_latent_samples(latent_samples)
        Set latent samples for reparameterization trick.
    __call__(design_weights)
        Evaluate log-likelihood for all outer samples.
    jacobian(design_weights)
        Jacobian w.r.t. design weights.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nobs(self) -> int:
        """Return the number of observation locations."""
        ...

    def set_shapes(self, shapes: Array) -> None:
        """
        Set model outputs (shapes/means for Gaussian).

        Parameters
        ----------
        shapes : Array
            Model outputs. Shape: (nobs, nouter)
        """
        ...

    def set_observations(self, obs: Array) -> None:
        """
        Set artificial observations.

        Parameters
        ----------
        obs : Array
            Artificial observations. Shape: (nobs, nouter)
            Must match shapes in shape.
        """
        ...

    def set_design_weights(self, weights: Array) -> None:
        """
        Set design weights.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)
        """
        ...

    def set_latent_samples(self, latent_samples: Array) -> None:
        """
        Set latent samples for reparameterization trick.

        For Gaussian noise: obs = shapes + sqrt(var/weights) * latent_samples
        where latent_samples ~ N(0, 1).

        Parameters
        ----------
        latent_samples : Array
            Latent samples. Shape: (nobs, nouter)
        """
        ...

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate log-likelihood for all outer samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Log-likelihood values. Shape: (1, nouter)
        """
        ...

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of log-likelihood w.r.t. design weights.

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
class OEDInnerLoopLikelihoodProtocol(Protocol, Generic[Array]):
    """
    Protocol for inner loop likelihood in Bayesian OED.

    The inner loop integrates over the prior (parameter uncertainty) to
    compute the evidence for each potential observation.

    This likelihood produces a full (ninner, nouter) matrix where:
    - ninner: number of prior samples (for evidence integration)
    - nouter: number of observation realizations

    Methods
    -------
    bkd()
        Get the computational backend.
    nobs()
        Number of observation locations.
    ninner()
        Number of inner (prior) samples.
    nouter()
        Number of outer (observation) samples.
    set_shapes(shapes)
        Set model outputs for inner samples.
    set_observations(obs)
        Set artificial observations for outer samples.
    set_design_weights(weights)
        Set design weights.
    set_latent_samples(latent_samples)
        Set latent samples for reparameterization.
    logpdf_matrix(design_weights)
        Compute full (ninner, nouter) log-likelihood matrix.
    jacobian_matrix(design_weights)
        Jacobian of likelihood matrix w.r.t. design weights.
    create_outer_loop_likelihood()
        Factory method for paired outer loop likelihood.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nobs(self) -> int:
        """Return the number of observation locations."""
        ...

    def ninner(self) -> int:
        """Return the number of inner (prior) samples."""
        ...

    def nouter(self) -> int:
        """Return the number of outer (observation) samples."""
        ...

    def set_shapes(self, shapes: Array) -> None:
        """
        Set model outputs for inner samples.

        Parameters
        ----------
        shapes : Array
            Model outputs. Shape: (nobs, ninner)
        """
        ...

    def set_observations(self, obs: Array) -> None:
        """
        Set artificial observations for outer samples.

        Parameters
        ----------
        obs : Array
            Artificial observations. Shape: (nobs, nouter)
        """
        ...

    def set_design_weights(self, weights: Array) -> None:
        """
        Set design weights.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)
        """
        ...

    def set_latent_samples(self, latent_samples: Array) -> None:
        """
        Set latent samples for reparameterization trick.

        Parameters
        ----------
        latent_samples : Array
            Latent samples. Shape: (nobs, nouter)
        """
        ...

    def logpdf_matrix(self, design_weights: Array) -> Array:
        """
        Compute full log-likelihood matrix.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Log-likelihood matrix. Shape: (ninner, nouter)
            Entry [i, j] = log p(obs_j | theta_i, design_weights)
        """
        ...

    def jacobian_matrix(self, design_weights: Array) -> Array:
        """
        Jacobian of log-likelihood matrix w.r.t. design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (ninner, nouter, nobs)
            Entry [i, j, k] = d/d(weight_k) log p(obs_j | theta_i)
        """
        ...

    def create_outer_loop_likelihood(self) -> OEDOuterLoopLikelihoodProtocol[Array]:
        """
        Create a paired outer loop likelihood.

        This ensures consistency between inner and outer loop likelihoods.

        Returns
        -------
        OEDOuterLoopLikelihoodProtocol
            Outer loop likelihood with same noise model.
        """
        ...

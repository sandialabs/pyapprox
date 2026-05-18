"""Protocols for dynamical systems learning.

Defines the contracts for learned functions (surrogates), encoders,
and related objects used throughout the module.

Protocol Hierarchy
------------------
LearnedFunctionProtocol
    Base tier: any surrogate that can be evaluated, differentiated w.r.t.
    inputs and parameters, and cloned with new parameters.
EncoderProtocol
    State-space encoder (linear or nonlinear).
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class LearnedFunctionProtocol(Protocol, Generic[Array]):
    """Protocol for learned function approximators F_eta(x): R^nvars -> R^nqoi.

    This is the base tier that any surrogate must satisfy for use with
    BatchedBoundODEResidual, DerivativeMatchingLoss, and gradient-based
    fitting. Supports non-square mappings (nvars != nqoi).
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def hyp_list(self) -> HyperParameterList[Array]: ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate F_eta at given samples.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Output values. Shape: (nqoi, nsamples)
        """
        ...

    def jacobian_batch(self, samples: Array) -> Array:
        """Compute dF/dx at each sample.

        Parameters
        ----------
        samples : Array
            Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, nqoi, nvars)
        """
        ...

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """Compute dF/d_eta (active params only) at each sample.

        Parameters
        ----------
        samples : Array
            Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, nqoi, nactive_params)
        """
        ...

    def with_params(self, params: Array) -> "LearnedFunctionProtocol[Array]":
        """Return a copy with new parameter values.

        Parameters
        ----------
        params : Array
            New parameter values. Shape depends on implementation
            (e.g., (nterms, nqoi) for BasisExpansion).

        Returns
        -------
        LearnedFunctionProtocol[Array]
            New instance with updated parameters.
        """
        ...


@runtime_checkable
class EncoderProtocol(Protocol, Generic[Array]):
    """Protocol for state-space encoders.

    Maps between full and reduced state spaces. Both linear (POD) and
    nonlinear (autoencoder) encoders satisfy this protocol.
    """

    def bkd(self) -> Backend[Array]: ...

    def full_dim(self) -> int: ...

    def latent_dim(self) -> int: ...

    def encode(self, states: Array) -> Array:
        """Encode full states to latent space.

        Parameters
        ----------
        states : Array
            Full states. Shape: (full_dim, nsamples)

        Returns
        -------
        Array
            Latent states. Shape: (latent_dim, nsamples)
        """
        ...

    def decode(self, latents: Array) -> Array:
        """Decode latent states to full space.

        Parameters
        ----------
        latents : Array
            Latent states. Shape: (latent_dim, nsamples)

        Returns
        -------
        Array
            Full states. Shape: (full_dim, nsamples)
        """
        ...

    def encode_jacobian(self) -> Array:
        """Return the Jacobian of the encoder.

        For linear encoders this is constant. For nonlinear encoders
        this would need to accept a state argument (Phase 3 extension).

        Returns
        -------
        Array
            Encoder Jacobian. Shape: (latent_dim, full_dim)
        """
        ...

"""Protocols for dynamical systems learning.

Defines the contracts for vector fields, encoders, and datasets used
throughout the module.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class ParametricVectorFieldProtocol(Protocol, Generic[Array]):
    """Protocol for parametric vector fields F_eta(x): R^n -> R^n.

    Models autonomous ODEs dx/dt = F_eta(x) where eta is a learnable
    parameter vector managed by HyperParameterList.

    Required Methods
    ----------------
    bkd()
        Get the computational backend.
    nstates()
        Number of state variables (= input and output dimension).
    hyp_list()
        HyperParameterList managing the learnable parameters.
    __call__(states)
        Evaluate F_eta at given states.

    Optional Methods (dynamically bound via hasattr)
    -------------------------------------------------
    state_jacobian(states)
        dF/dx at each sample. Shape: (nsamples, nstates, nstates).
    param_jacobian(states)
        dF/d_eta (active params only). Shape: (nsamples, nstates, nactive).
    """

    def bkd(self) -> Backend[Array]: ...

    def nstates(self) -> int: ...

    def hyp_list(self) -> HyperParameterList[Array]: ...

    def __call__(self, states: Array) -> Array:
        """Evaluate F_eta at given states.

        Parameters
        ----------
        states : Array
            State vectors. Shape: (nstates, nsamples)

        Returns
        -------
        Array
            Vector field values. Shape: (nstates, nsamples)
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

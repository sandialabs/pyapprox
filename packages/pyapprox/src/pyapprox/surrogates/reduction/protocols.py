"""Capability protocols for decoders.

A decoder maps latent coordinates back to full states. Consumers differ
in how much structure they need to see: reconstructing a state needs only
:meth:`decode`, while propagating a derivative through the decoder needs
either its tangent or enough structure to build one.

These protocols name those capabilities separately so a consumer depends
on the weakest one that carries it, and so an implementation that cannot
honestly provide a capability simply does not declare it. All are
``@runtime_checkable``, and the ``is_*`` helpers below wrap the
``isinstance`` check in a :class:`~typing.TypeGuard` so the narrowing
keeps the caller's ``Array`` binding, which a bare ``isinstance``
discards.

:class:`~pyapprox.surrogates.kerneloperator.protocols.FunctionEncoderProtocol`
remains the contract for a full bidirectional encoder; these describe the
decode direction alone.
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeGuard, runtime_checkable

from pyapprox.surrogates.reduction.feature_maps import (
    DifferentiableFeatureMap,
)
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class DecoderProtocol(Protocol, Generic[Array]):
    """Minimal decoder capability: latent coordinates to full states."""

    def latent_dim(self) -> int:
        """Dimension of the latent space."""
        ...

    def decode(self, latents: Array) -> Array:
        """Latent to full. ``(latent_dim, N) -> (full_dim, N)``."""
        ...


@runtime_checkable
class LinearDecoderProtocol(DecoderProtocol[Array], Protocol):
    """A decoder with an explicit linear basis."""

    def basis(self) -> Array:
        """The basis V. Shape: (full_dim, latent_dim)."""
        ...


@runtime_checkable
class ManifoldDecoderProtocol(LinearDecoderProtocol[Array], Protocol):
    """A linear basis plus a feature-map correction.

    Decodes as ``V z + W h(z)``. The tangent is available in closed form
    from the basis, the weights, and the feature map's own Jacobian, so a
    consumer needing the derivative does not have to difference
    :meth:`decode`.

    Deliberately not named for monomials: the closed-form tangent needs
    only that ``h`` supplies its Jacobian, so any
    :class:`DifferentiableFeatureMap` satisfies it. The choice of
    features belongs to the implementation --
    :class:`~pyapprox.surrogates.reduction.monomial_manifold.MonomialManifoldEncoder`
    names its own -- not to the structure consumers depend on.
    """

    def weights(self) -> Array:
        """The correction weights W. Shape: (full_dim, nterms)."""
        ...

    def feature_map(self) -> DifferentiableFeatureMap[Array]:
        """The feature map h, which must supply its own Jacobian."""
        ...


@runtime_checkable
class SelfJacobianDecoderProtocol(DecoderProtocol[Array], Protocol):
    """A decoder that supplies its own tangent directly."""

    def decode_jacobian(self, latents: Array) -> Array:
        """``d decode / d z`` at ``(latent_dim, 1)`` latents.

        Returns
        -------
        Array
            Shape: (full_dim, latent_dim).
        """
        ...


def is_self_jacobian_decoder(
    decoder: DecoderProtocol[Array],
) -> TypeGuard[SelfJacobianDecoderProtocol[Array]]:
    """Whether ``decoder`` supplies its own tangent.

    ``isinstance`` alone erases the generic parameter, narrowing to
    ``[Any]``; the TypeGuard carries the caller's ``[Array]`` binding
    through the narrowing.
    """
    return isinstance(decoder, SelfJacobianDecoderProtocol)


def is_manifold_decoder(
    decoder: DecoderProtocol[Array],
) -> TypeGuard[ManifoldDecoderProtocol[Array]]:
    """Whether ``decoder`` is a linear basis plus a correction."""
    return isinstance(decoder, ManifoldDecoderProtocol)


def is_linear_decoder(
    decoder: DecoderProtocol[Array],
) -> TypeGuard[LinearDecoderProtocol[Array]]:
    """Whether ``decoder`` exposes an explicit linear basis."""
    return isinstance(decoder, LinearDecoderProtocol)

"""Protocols for least-squares operator learning."""

from __future__ import annotations

from typing import Generic, Protocol, runtime_checkable

from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
)
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class IsometricEncoderProtocol(Protocol):
    r"""A function encoder that reports whether it preserves norms.

    Declares the one property least-squares operator learning needs
    from an *output* encoder beyond encoding and decoding: the error it
    minimizes is the Euclidean norm of coefficient residuals, and that
    equals the Bochner error only when the encoder is an isometry,
    :math:`\|f\|_Y = \|\mathrm{encode}(f)\|_2`.

    Kept separate from
    :class:`pyapprox.surrogates.kerneloperator.protocols.FunctionEncoderProtocol`
    so that encoders written for kernel operator learning, which does
    not need the property, remain usable unchanged.

    The property is declared rather than derived because verifying it
    numerically needs the Y-inner-product, which lives with the encoder
    rather than with its consumer. An encoder over an orthonormal basis
    is an isometry by construction; one over a non-orthonormal basis is
    not, until orthonormalized against the Gram matrix.
    """

    def is_isometry(self) -> bool:
        """Return whether encoding preserves the Y-norm."""
        ...


@runtime_checkable
class FieldEncoderProtocol(
    FunctionEncoderProtocol[Array], Protocol, Generic[Array]
):
    """A function encoder that also reports whether it is an isometry.

    Least-squares operator learning needs the encoding and decoding
    that :class:`FunctionEncoderProtocol` already declares, plus the
    isometry property, so it composes the two rather than restating
    them. Any encoder written for kernel operator learning satisfies
    this as soon as it can answer :meth:`is_isometry`.
    """

    def is_isometry(self) -> bool:
        """Return whether encoding preserves the Y-norm."""
        ...

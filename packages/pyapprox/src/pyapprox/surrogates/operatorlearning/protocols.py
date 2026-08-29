"""Protocols for least-squares operator learning."""

from __future__ import annotations

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


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
class FieldEncoderProtocol(Protocol, Generic[Array]):
    """Bidirectional map between field values on a grid and coefficients.

    Structurally identical to
    :class:`pyapprox.surrogates.kerneloperator.protocols.FunctionEncoderProtocol`
    with :class:`IsometricEncoderProtocol` folded in, so encoders from
    either module satisfy it.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def ncodes(self) -> int:
        """Return the number of coefficients."""
        ...

    def ngrid(self) -> int:
        """Return the number of grid points."""
        ...

    def encode(self, f_grid: Array) -> Array:
        """Encode grid values to coefficients. (ngrid, N) -> (ncodes, N)."""
        ...

    def decode(self, codes: Array) -> Array:
        """Decode coefficients to grid values. (ncodes, N) -> (ngrid, N)."""
        ...

    def is_isometry(self) -> bool:
        """Return whether encoding preserves the Y-norm."""
        ...

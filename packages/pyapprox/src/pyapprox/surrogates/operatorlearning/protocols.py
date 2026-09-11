"""Protocols for least-squares operator learning."""

from __future__ import annotations

from typing import Generic, Protocol, TypeGuard, runtime_checkable

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
    FunctionEncoderProtocol[Array],
    IsometricEncoderProtocol,
    Protocol,
    Generic[Array],
):
    """A function encoder that also reports whether it is an isometry.

    Least-squares operator learning needs the encoding and decoding
    that :class:`FunctionEncoderProtocol` already declares, plus the
    isometry property, so it composes the two rather than restating
    them. Any encoder written for kernel operator learning satisfies
    this as soon as it can answer :meth:`is_isometry`.

    The isometry half is inherited from
    :class:`IsometricEncoderProtocol` rather than redeclared. Redeclaring
    it left that protocol with no consumer and no relationship to this
    one, so the two were free to drift apart -- which is the failure
    composition exists to prevent, and the reason the sentence above was
    true of the intent and not of the code.
    """


@runtime_checkable
class LatentMapProtocol(Protocol, Generic[Array]):
    """Maps input codes to output codes.

    What an operator surrogate needs from the object between its two
    encoders. ``__call__`` is the whole forward path; ``nvars`` and
    ``nqoi`` exist so that assembling a surrogate whose widths disagree
    fails at construction rather than inside a matmul.

    Deliberately smaller than
    :class:`pyapprox.surrogates.kerneloperator.protocols.LatentRegressorProtocol`,
    which adds ``predict_std``, ``neg_log_marginal_likelihood`` and
    ``hyp_list`` -- meaningful for a GP, meaningless for a neural
    network or a function train. Requiring them here would force most
    implementations to ship methods whose only job is to raise.

    No ``bkd``. Every consumer in this package holds its own backend and
    none reads one off the map, so declaring it would oblige every
    implementation to expose an accessor nothing calls. The precedent is
    ``InnerProductProtocol``, which likewise omits a member all three of
    its implementations happen to have.
    """

    def nvars(self) -> int:
        """Number of input codes the map consumes."""
        ...

    def nqoi(self) -> int:
        """Number of output codes the map produces."""
        ...

    def __call__(self, coefs: Array) -> Array:
        """Map input codes to output codes.

        ``(nvars, nsamples) -> (nqoi, nsamples)``.
        """
        ...


@runtime_checkable
class LinearInParamsLatentMapProtocol(LatentMapProtocol[Array], Protocol):
    r"""A latent map linear in its parameters.

    .. math:: g(\hat f) = C^T \Phi(\hat f)

    with :math:`\Phi` supplied by :meth:`basis_matrix` and :math:`C` by
    :meth:`with_params`. This is what makes a fit a single linear least
    squares problem rather than an iterative optimization, so it is what
    :class:`WeightedLeastSquaresOperatorFitter` requires -- and the
    reason that fitter can solve in closed form while a neural latent
    map needs a gradient method.

    Separate from :class:`MultiIndexLatentMapProtocol` because linearity
    in the parameters and having a multi-index basis are independent. A
    function train is fittable by alternating least squares and carries
    no index set at all.
    """

    def basis_matrix(self, coefs: Array) -> Array:
        """Return :math:`\\Phi`. ``(nvars, nsamples) -> (nsamples, nterms)``.

        Note the transpose relative to this package's usual
        samples-are-columns convention: the design matrix is what a
        linear solver takes, and it is points-down-rows.
        """
        ...

    def with_params(
        self, params: Array
    ) -> "LinearInParamsLatentMapProtocol[Array]":
        """Return a copy carrying ``params``. Shape: (nterms, nqoi)."""
        ...


@runtime_checkable
class MultiIndexLatentMapProtocol(
    LinearInParamsLatentMapProtocol[Array], Protocol
):
    r"""A latent map whose basis is indexed by a multi-index set.

    Adds the index set itself, which is what lets a consumer ask
    structural questions about the basis -- whether every index is at
    most first order, and so whether the map is affine in its input
    codes. :class:`OperatorFitResult` needs exactly that to decide
    whether an operator matrix exists.

    ``PolynomialChaosExpansion`` satisfies this. A neural network
    satisfies none of the three levels beyond the first; a function
    train satisfies the middle one.
    """

    def get_indices(self) -> Array:
        """Return the index set. Shape: (nvars, nterms)."""
        ...


def is_linear_in_params(
    latent_map: LatentMapProtocol[Array],
) -> TypeGuard[LinearInParamsLatentMapProtocol[Array]]:
    """Whether ``latent_map`` can be fitted by linear least squares.

    ``isinstance`` alone erases the generic parameter, narrowing to
    ``[Any]``; the TypeGuard carries the caller's ``[Array]`` binding
    through the narrowing. Mirrors the helpers in
    :mod:`pyapprox.surrogates.reduction.protocols`, which exist for the
    same reason.
    """
    return isinstance(latent_map, LinearInParamsLatentMapProtocol)


def is_multi_index(
    latent_map: LatentMapProtocol[Array],
) -> TypeGuard[MultiIndexLatentMapProtocol[Array]]:
    """Whether ``latent_map`` has a multi-index basis to interrogate."""
    return isinstance(latent_map, MultiIndexLatentMapProtocol)

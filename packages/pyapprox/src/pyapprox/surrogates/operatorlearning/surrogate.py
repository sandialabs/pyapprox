r"""An operator surrogate between two function spaces."""

from __future__ import annotations

from typing import Generic

from pyapprox.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class OperatorSurrogate(Generic[Array]):
    r"""Approximates an operator :math:`G : U \to V` between field spaces.

    Encodes an input field to coefficients, maps those through a
    polynomial expansion, and decodes the result to an output field.

    Multiple input or output fields are handled by passing a
    :class:`ProductFieldEncoder`, so this class always sees exactly one
    encoder per side and never does offset arithmetic.

    Parameters
    ----------
    input_encoder : FieldEncoderProtocol[Array]
        Maps input fields to the coefficients the expansion takes as
        its variables.
    output_encoder : FieldEncoderProtocol[Array]
        Maps output fields to the coefficients the expansion predicts.
        Must be an isometry, or the least-squares error the fit
        minimizes is not the Bochner error of the fields.
    expansion : PolynomialChaosExpansion[Array]
        The fitted expansion, with ``nqoi`` equal to the number of
        output coefficients.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        input_encoder: FieldEncoderProtocol[Array],
        output_encoder: FieldEncoderProtocol[Array],
        expansion: PolynomialChaosExpansion[Array],
        bkd: Backend[Array],
    ) -> None:
        for name, encoder in (
            ("input_encoder", input_encoder),
            ("output_encoder", output_encoder),
        ):
            if not isinstance(encoder, FieldEncoderProtocol):
                raise TypeError(
                    f"{name} must satisfy FieldEncoderProtocol, got "
                    f"{type(encoder).__name__}"
                )
        if not output_encoder.is_isometry():
            raise ValueError(
                "output_encoder must be an isometry, otherwise the "
                "coefficient residuals the fit minimizes do not measure "
                "the Bochner error of the fields. Use "
                "orthonormalize_basis to correct the output basis."
            )
        if expansion.nqoi() != output_encoder.latent_dim():
            raise ValueError(
                f"expansion has nqoi {expansion.nqoi()} but "
                f"output_encoder has {output_encoder.latent_dim()} codes"
            )
        if expansion.nvars() != input_encoder.latent_dim():
            raise ValueError(
                f"expansion has nvars {expansion.nvars()} but "
                f"input_encoder has {input_encoder.latent_dim()} codes"
            )
        self._input_encoder = input_encoder
        self._output_encoder = output_encoder
        self._expansion = expansion
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def input_encoder(self) -> FieldEncoderProtocol[Array]:
        """Return the input encoder."""
        return self._input_encoder

    def output_encoder(self) -> FieldEncoderProtocol[Array]:
        """Return the output encoder."""
        return self._output_encoder

    def expansion(self) -> PolynomialChaosExpansion[Array]:
        """Return the underlying expansion."""
        return self._expansion

    def indices(self) -> Array:
        """Return the index set. Shape: (nvars, nterms)."""
        return self._expansion.get_indices()

    def is_affine(self) -> bool:
        r"""Return whether the surrogate is affine in its input coefficients.

        True when no index exceeds first order,
        :math:`|\lambda| \le 1`. Each first-order basis function is
        degree one in a single coordinate, and the zero index
        contributes a constant, so the surrogate reduces to
        :math:`\hat g = A \hat f + b`.

        Strict linearity would additionally require the constant term
        to be absent. It is not required here: an operator with a
        nonzero response to zero input is ordinary, and
        :meth:`OperatorFitResult.operator_matrix` keeps :math:`A` and
        :math:`b` separate so neither can be read as the other.
        """
        return bool(
            self._bkd.all_bool(self._bkd.sum(self.indices(), axis=0) <= 1)
        )

    def predict_coefficients(self, coefs: Array) -> Array:
        """Map input coefficients to output coefficients.

        The entry point for fitting and diagnostics, which work in
        coefficients throughout and never touch a grid.

        Parameters
        ----------
        coefs : Array
            Input coefficients. Shape: (ncodes_in, nsamples)

        Returns
        -------
        Array
            Output coefficients. Shape: (ncodes_out, nsamples)
        """
        return self._expansion(coefs)

    def __call__(self, fields: Array) -> Array:
        """Map input fields to output fields.

        Parameters
        ----------
        fields : Array
            Input field values on their grid. Shape: (ngrid_in, nsamples)

        Returns
        -------
        Array
            Output field values. Shape: (ngrid_out, nsamples)
        """
        coefs = self._input_encoder.encode(fields)
        return self._output_encoder.decode(self.predict_coefficients(coefs))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ncodes_in={self._input_encoder.latent_dim()}, "
            f"ncodes_out={self._output_encoder.latent_dim()})"
        )

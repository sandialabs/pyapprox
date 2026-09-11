r"""An operator surrogate between two function spaces."""

from __future__ import annotations

from typing import Generic

from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
)
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
    LatentMapProtocol,
    is_multi_index,
)
from pyapprox.util.backends.protocols import Array, Backend


class OperatorSurrogate(Generic[Array]):
    r"""Approximates an operator :math:`G : U \to V` between field spaces.

    Encodes an input field to coefficients, maps those through a latent
    map, and decodes the result to an output field.

    The latent map is any :class:`LatentMapProtocol` -- a polynomial
    chaos expansion, a neural network, a function train. Which one is
    chosen decides how the surrogate is *fitted*, not how it evaluates,
    so the forward path here is the same three steps either way.

    Multiple input or output fields are handled by passing a
    :class:`ProductFieldEncoder`, so this class always sees exactly one
    encoder per side and never does offset arithmetic.

    Parameters
    ----------
    input_encoder : FunctionEncoderProtocol[Array]
        Maps input fields to the coefficients the expansion takes as
        its variables. Deliberately the weaker protocol: the isometry
        is a property of the *output* side, where a coefficient
        residual stands in for a field error, and nothing here ever
        reads ``is_isometry`` on this encoder. Requiring it would
        exclude encoders that are otherwise perfectly usable as inputs.
    output_encoder : FieldEncoderProtocol[Array]
        Maps output fields to the coefficients the expansion predicts.
        Must be an isometry, or the least-squares error the fit
        minimizes is not the Bochner error of the fields.
    latent_map : LatentMapProtocol[Array]
        The fitted map from input codes to output codes, with ``nqoi``
        equal to the number of output coefficients and ``nvars`` to the
        number of input ones.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        input_encoder: FunctionEncoderProtocol[Array],
        output_encoder: FieldEncoderProtocol[Array],
        latent_map: LatentMapProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(latent_map, LatentMapProtocol):
            raise TypeError(
                f"latent_map must satisfy LatentMapProtocol, got "
                f"{type(latent_map).__name__}"
            )
        if not isinstance(input_encoder, FunctionEncoderProtocol):
            raise TypeError(
                f"input_encoder must satisfy FunctionEncoderProtocol, "
                f"got {type(input_encoder).__name__}"
            )
        if not isinstance(output_encoder, FieldEncoderProtocol):
            raise TypeError(
                f"output_encoder must satisfy FieldEncoderProtocol, got "
                f"{type(output_encoder).__name__}"
            )
        if not output_encoder.is_isometry():
            raise ValueError(
                "output_encoder must be an isometry, otherwise the "
                "coefficient residuals the fit minimizes do not measure "
                "the Bochner error of the fields. Use "
                "orthonormalize_basis to correct the output basis."
            )
        if latent_map.nqoi() != output_encoder.latent_dim():
            raise ValueError(
                f"latent_map has nqoi {latent_map.nqoi()} but "
                f"output_encoder has {output_encoder.latent_dim()} codes"
            )
        if latent_map.nvars() != input_encoder.latent_dim():
            raise ValueError(
                f"latent_map has nvars {latent_map.nvars()} but "
                f"input_encoder has {input_encoder.latent_dim()} codes"
            )
        self._input_encoder = input_encoder
        self._output_encoder = output_encoder
        self._latent_map = latent_map
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def input_encoder(self) -> FunctionEncoderProtocol[Array]:
        """Return the input encoder."""
        return self._input_encoder

    def output_encoder(self) -> FieldEncoderProtocol[Array]:
        """Return the output encoder."""
        return self._output_encoder

    def latent_map(self) -> LatentMapProtocol[Array]:
        """Return the map from input codes to output codes."""
        return self._latent_map

    def indices(self) -> Array:
        """Return the index set. Shape: (nvars, nterms).

        Raises
        ------
        TypeError
            If the latent map has no multi-index basis. A neural
            network or a function train has none, so the question has
            no answer rather than an empty one.
        """
        latent_map = self._latent_map
        if not is_multi_index(latent_map):
            raise TypeError(
                f"indices requires a latent map with a multi-index "
                f"basis, satisfying MultiIndexLatentMapProtocol, but "
                f"{type(latent_map).__name__} has no index set."
            )
        return latent_map.get_indices()

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

        Raises
        ------
        TypeError
            Via :meth:`indices`, if the latent map has no multi-index
            basis. Affineness is read off the index set, so a map
            without one cannot answer -- note this is different from
            answering False, which asserts the map *is* nonlinear.
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
        return self._latent_map(coefs)

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

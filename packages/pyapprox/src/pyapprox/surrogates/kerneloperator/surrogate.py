"""Kernel-based operator learning surrogate."""

from __future__ import annotations

import copy
from typing import Generic, List, Union

from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
    LatentRegressorProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class KernelOperatorSurrogate(Generic[Array]):
    """Kernel-based operator learning surrogate.

    Approximates an operator G : U -> V between function spaces via:
    1. Encode input functions to codes, concatenate
    2. Kernel regression in code space
    3. Split output codes, decode to output functions

    Parameters
    ----------
    input_encoders : List[FunctionEncoderProtocol[Array]]
        One encoder per input function.
    output_encoders : List[FunctionEncoderProtocol[Array]]
        One encoder per output function.
    latent_regressor : LatentRegressorProtocol[Array]
        Regressor mapping input codes to output codes.
    """

    def __init__(
        self,
        input_encoders: List[FunctionEncoderProtocol[Array]],
        output_encoders: List[FunctionEncoderProtocol[Array]],
        latent_regressor: LatentRegressorProtocol[Array],
    ) -> None:
        total_in = sum(enc.ncodes() for enc in input_encoders)
        total_out = sum(enc.ncodes() for enc in output_encoders)
        if total_in != latent_regressor.ncodes_in():
            raise ValueError(
                f"Sum of input encoder ncodes ({total_in}) must equal "
                f"regressor ncodes_in ({latent_regressor.ncodes_in()})"
            )
        if total_out != latent_regressor.ncodes_out():
            raise ValueError(
                f"Sum of output encoder ncodes ({total_out}) must equal "
                f"regressor ncodes_out ({latent_regressor.ncodes_out()})"
            )
        self._input_encoders = input_encoders
        self._output_encoders = output_encoders
        self._regressor = latent_regressor
        self._bkd = latent_regressor.bkd()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def input_encoders(self) -> List[FunctionEncoderProtocol[Array]]:
        return list(self._input_encoders)

    def output_encoders(self) -> List[FunctionEncoderProtocol[Array]]:
        return list(self._output_encoders)

    def latent_regressor(self) -> LatentRegressorProtocol[Array]:
        return self._regressor

    def is_fitted(self) -> bool:
        return self._regressor.is_fitted()

    def _encode_inputs(self, u_grids: List[Array]) -> Array:
        codes = [
            enc.encode(u) for enc, u in zip(self._input_encoders, u_grids)
        ]
        return self._bkd.concatenate(codes, axis=0)

    def _split_and_decode(self, V: Array) -> List[Array]:
        results = []
        offset = 0
        for enc in self._output_encoders:
            nc = enc.ncodes()
            results.append(enc.decode(V[offset : offset + nc, :]))
            offset += nc
        return results

    def _split_and_decode_std(self, V_std: Array) -> List[Array]:
        results = []
        offset = 0
        for enc in self._output_encoders:
            nc = enc.ncodes()
            results.append(enc.decode_std(V_std[offset : offset + nc, :]))
            offset += nc
        return results

    def predict(self, u_grids: List[Array]) -> List[Array]:
        """Predict output functions from input functions.

        Parameters
        ----------
        u_grids : List[Array]
            Input function values on grids. u_grids[i] has shape
            (ngrid_in_i, N).

        Returns
        -------
        List[Array]
            Output function values. v_grids[j] has shape (ngrid_out_j, N).
        """
        U = self._encode_inputs(u_grids)
        V = self._regressor.predict(U)
        return self._split_and_decode(V)

    def predict_std(self, u_grids: List[Array]) -> List[Array]:
        """Predict std of output functions.

        Uses decode_std (no mean shift) for correct std propagation.
        """
        U = self._encode_inputs(u_grids)
        V_std = self._regressor.predict_std(U)
        return self._split_and_decode_std(V_std)

    def __call__(
        self, u_grids: Union[Array, List[Array]]
    ) -> Union[Array, List[Array]]:
        """Convenience: accepts single Array or List[Array]."""
        if not isinstance(u_grids, list):
            result = self.predict([u_grids])
            return result[0]
        return self.predict(u_grids)

    def clone_unfitted(self) -> KernelOperatorSurrogate[Array]:
        return KernelOperatorSurrogate(
            input_encoders=copy.deepcopy(self._input_encoders),
            output_encoders=copy.deepcopy(self._output_encoders),
            latent_regressor=self._regressor.clone_unfitted(),
        )

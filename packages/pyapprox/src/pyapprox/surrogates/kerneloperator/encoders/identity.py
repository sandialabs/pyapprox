"""Identity function encoder — codes equal grid values."""

from __future__ import annotations

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class IdentityFunctionEncoder(Generic[Array]):
    """Identity encoder where ncodes == ngrid.

    Parameters
    ----------
    ngrid : int
        Number of grid points (and codes).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, ngrid: int, bkd: Backend[Array]) -> None:
        self._ngrid = ngrid
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ncodes(self) -> int:
        return self._ngrid

    def ngrid(self) -> int:
        return self._ngrid

    def encode(self, f_grid: Array) -> Array:
        return f_grid

    def decode(self, codes: Array) -> Array:
        return codes

    def decode_std(self, std_codes: Array) -> Array:
        return std_codes

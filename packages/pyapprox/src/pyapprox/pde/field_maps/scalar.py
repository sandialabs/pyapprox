"""ScalarAmplitude field map: field = p[0] * base_field."""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class ScalarAmplitude(Generic[Array]):
    """Scalar amplitude field map.

    Maps a single parameter to a spatial field:
        field(x) = params[0] * base_field(x)

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    base_field : Array
        Static spatial field. Shape: (npts,).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        base_field: Array,
    ) -> None:
        self._bkd = bkd
        self._base_field = base_field
        # Cache Jacobian: single column = base_field
        self._cached_jacobian = self._base_field[:, None]

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 1

    def __call__(self, params_1d: Array) -> Array:
        """Evaluate field map. Uses array slice to preserve autograd."""
        return params_1d[0:1] * self._base_field

    def jacobian(self, params_1d: Array) -> Array:
        """Return Jacobian d(field)/d(params). Shape: (npts, 1).

        Constant for linear map -- returns cached array.
        """
        return self._cached_jacobian

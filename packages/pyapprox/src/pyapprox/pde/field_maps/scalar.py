"""ScalarAmplitude field map: field = p[0] * base_field."""

from typing import Generic

from pyapprox.pde.field_maps.protocol import validate_params_1d
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
        validate_params_1d(params_1d, self.nvars())
        return params_1d[0:1] * self._base_field

    def jacobian(self, params_1d: Array) -> Array:
        """Return Jacobian d(field)/d(params). Shape: (npts, 1).

        Constant for linear map -- returns cached array.
        """
        return self._cached_jacobian

    def hvp(self, params_1d: Array, adj_state: Array, vvec: Array) -> Array:
        """Adjoint-weighted HVP: exactly zero (the map is linear).

        Declared so consumers keep exact second-order capability
        instead of degrading to first order. Shape: (nvars,).
        """
        return self._bkd.zeros((self.nvars(),))

    def is_linear(self) -> bool:
        """Linear in the parameters, so a temporal modulation may scale
        its jacobian columns. Declared because linearity cannot be
        detected, and composing a modulation with a nonlinear map
        silently describes a field the forward solve never evaluates."""
        return True

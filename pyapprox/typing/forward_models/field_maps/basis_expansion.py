"""BasisExpansion field map: D(x) = base + sum_i p_i * phi_i(x)."""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend


class BasisExpansion(Generic[Array]):
    """Linear basis expansion field map.

    Maps parameter vector to spatial field:
        field(x) = base_value + sum_i params[i] * basis_funs[i](x)

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    base_value : float
        Base (constant) value of the field.
    basis_funs : List[Array]
        Basis functions evaluated at nodes. Each shape: (npts,).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        base_value: float,
        basis_funs: List[Array],
    ) -> None:
        self._bkd = bkd
        self._base_value = base_value
        self._basis_funs = basis_funs
        # Cache constant Jacobian -- independent of params for linear map
        self._cached_jacobian = self._bkd.stack(self._basis_funs, axis=1)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return len(self._basis_funs)

    def __call__(self, params_1d: Array) -> Array:
        """Evaluate field map. Must NOT use float() -- called in Jacobian chain."""
        npts = self._basis_funs[0].shape[0]
        result = self._bkd.full((npts,), self._base_value)
        for i in range(self.nvars()):
            result = result + params_1d[i] * self._basis_funs[i]
        return result

    def jacobian(self, params_1d: Array) -> Array:
        """Return Jacobian d(field)/d(params). Shape: (npts, nvars).

        Constant for linear map -- returns cached array.
        """
        return self._cached_jacobian

"""BasisExpansion field map: D(x) = base + sum_i p_i * phi_i(x)."""

from typing import Generic, List, Union

from pyapprox.pde.field_maps.protocol import validate_params_1d
from pyapprox.util.backends.protocols import Array, Backend


class BasisExpansion(Generic[Array]):
    """Linear basis expansion field map.

    Maps parameter vector to spatial field:
        field(x) = base(x) + sum_i params[i] * basis_funs[i](x)

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    base_value : float or Array
        Base value of the field: a constant, or a nodal offset field of
        shape (npts,) — e.g. a fixed forcing the parameterized part is
        added to.
    basis_funs : List[Array]
        Basis functions evaluated at nodes. Each shape: (npts,).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        base_value: Union[float, Array],
        basis_funs: List[Array],
    ) -> None:
        self._bkd = bkd
        self._basis_funs = basis_funs
        npts = basis_funs[0].shape[0]
        if isinstance(base_value, (int, float)):
            self._base_field = bkd.full((npts,), float(base_value))
        else:
            if base_value.ndim != 1 or base_value.shape[0] != npts:
                raise ValueError(
                    f"array base_value must have shape ({npts},) to match "
                    f"the basis functions, got {base_value.shape}"
                )
            self._base_field = base_value
        # Cache constant Jacobian -- independent of params for linear map
        self._cached_jacobian = self._bkd.stack(self._basis_funs, axis=1)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return len(self._basis_funs)

    def base_field(self) -> Array:
        """Return the base field. Shape: (npts,)."""
        return self._base_field

    def __call__(self, params_1d: Array) -> Array:
        """Evaluate field map. Must NOT use float() -- called in Jacobian chain."""
        validate_params_1d(params_1d, self.nvars())
        result = self._base_field
        for i in range(self.nvars()):
            result = result + params_1d[i] * self._basis_funs[i]
        return result

    def jacobian(self, params_1d: Array) -> Array:
        """Return Jacobian d(field)/d(params). Shape: (npts, nvars).

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

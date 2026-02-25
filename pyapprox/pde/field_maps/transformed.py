"""TransformedFieldMap: applies pointwise transform to an inner field map."""

from typing import Callable, Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)


class TransformedFieldMap(Generic[Array]):
    """Pointwise-transformed field map.

    Computes: transform(inner_field_map(params))

    Parameters
    ----------
    inner : FieldMapProtocol
        Inner field map.
    transform : Callable[[Array], Array]
        Pointwise transform applied to inner field.
    transform_deriv : Callable[[Array], Array]
        Derivative of transform (pointwise).
    bkd : Backend
        Computational backend.
    transform_deriv2 : Callable[[Array], Array], optional
        Second derivative of transform (for HVP).
    """

    def __init__(
        self,
        inner: FieldMapProtocol[Array],
        transform: Callable[[Array], Array],
        transform_deriv: Callable[[Array], Array],
        bkd: Backend[Array],
        transform_deriv2: Optional[Callable[[Array], Array]] = None,
    ) -> None:
        if not isinstance(inner, FieldMapProtocol):
            raise TypeError(
                f"inner must satisfy FieldMapProtocol, "
                f"got {type(inner).__name__}"
            )
        self._inner = inner
        self._transform = transform
        self._transform_deriv = transform_deriv
        self._bkd = bkd
        self._transform_deriv2 = transform_deriv2

        # Dynamic binding: jacobian only if inner has jacobian
        if hasattr(self._inner, "jacobian"):
            self.jacobian = self._jacobian

        # Dynamic binding: hvp only if inner has jacobian AND deriv2 provided
        if hasattr(self._inner, "jacobian") and transform_deriv2 is not None:
            self.hvp = self._hvp

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._inner.nvars()

    def __call__(self, params_1d: Array) -> Array:
        """Evaluate transform(inner(params)). Must NOT use float()."""
        return self._transform(self._inner(params_1d))

    def _jacobian(self, params_1d: Array) -> Array:
        """Compute Jacobian via chain rule. Shape: (npts, nvars)."""
        inner_val = self._inner(params_1d)
        inner_jac = self._inner.jacobian(params_1d)
        t_deriv = self._transform_deriv(inner_val)
        return self._bkd.diag(t_deriv) @ inner_jac

    def _hvp(self, params_1d: Array, adj_state: Array, vvec: Array) -> Array:
        """Compute Hessian-vector product. Shape: (nvars,)."""
        inner_val = self._inner(params_1d)
        inner_jac = self._inner.jacobian(params_1d)
        t_deriv2 = self._transform_deriv2(inner_val)
        jac_v = inner_jac @ vvec[:, None]
        weighted = adj_state * t_deriv2 * jac_v[:, 0]
        return inner_jac.T @ weighted

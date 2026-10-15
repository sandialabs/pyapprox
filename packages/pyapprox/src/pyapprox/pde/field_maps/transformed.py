"""TransformedFieldMap: applies pointwise transform to an inner field map."""

from typing import Callable, Generic, Optional

from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
    FieldMapWithHVPProtocol,
    field_map_has_hvp,
)
from pyapprox.util.backends.protocols import Array, Backend


class TransformedFieldMap(Generic[Array]):
    """Pointwise-transformed field map.

    Computes: transform(inner_field_map(params))

    The adjoint-weighted HVP of :math:`f(p) = t(g(p))` is

    .. math::

        J_g^\\top \\operatorname{diag}(\\lambda \\odot t''(g)) J_g v
        + \\sum_i \\lambda_i t'(g_i) \\nabla^2 g_i \\, v,

    where the second (curvature) term is the inner map's own HVP
    contracted with the adjoint :math:`\\lambda \\odot t'(g)`. It is
    zero for a linear inner map, but nothing can detect linearity, so
    ``has_hvp`` is True only when ``transform_deriv2`` is given AND the
    inner map provides a usable HVP; otherwise the HVP is absent rather
    than approximated.

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
                f"inner must satisfy FieldMapProtocol, got {type(inner).__name__}"
            )
        self._inner = inner
        self._transform = transform
        self._transform_deriv = transform_deriv
        self._bkd = bkd
        self._transform_deriv2 = transform_deriv2
        # Recorded here (not isinstance-narrowed at the call site):
        # narrowing a Generic runtime protocol degrades its Array
        # parameter to Any under mypy.
        self._hvp_inner: Optional[FieldMapWithHVPProtocol[Array]] = None
        if field_map_has_hvp(inner) and isinstance(inner, FieldMapWithHVPProtocol):
            self._hvp_inner = inner

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._inner.nvars()

    def __call__(self, params_1d: Array) -> Array:
        """Evaluate transform(inner(params)). Must NOT use float()."""
        return self._transform(self._inner(params_1d))

    def has_hvp(self) -> bool:
        """Whether hvp is available.

        Requires ``transform_deriv2`` and an inner map that provides a
        usable adjoint-weighted HVP (its curvature term).
        """
        return self._transform_deriv2 is not None and self._hvp_inner is not None

    def jacobian(self, params_1d: Array) -> Array:
        """Compute Jacobian via chain rule. Shape: (npts, nvars)."""
        inner_val = self._inner(params_1d)
        inner_jac = self._inner.jacobian(params_1d)
        t_deriv = self._transform_deriv(inner_val)
        # Row scaling by broadcast: diag(t_deriv) @ inner_jac would form
        # an (npts, npts) dense matrix for an O(npts * nvars) operation.
        return t_deriv[:, None] * inner_jac

    def hvp(self, params_1d: Array, adj_state: Array, vvec: Array) -> Array:
        """Compute Hessian-vector product. Shape: (nvars,).

        Raises
        ------
        RuntimeError
            If transform_deriv2 was not provided or the inner map has no
            usable HVP; check has_hvp first.
        """
        deriv2 = self._transform_deriv2
        hvp_inner = self._hvp_inner
        if deriv2 is None or hvp_inner is None:
            raise RuntimeError(
                "hvp is unavailable (needs transform_deriv2 and an inner "
                "map with a usable hvp); check has_hvp before calling"
            )
        inner_val = self._inner(params_1d)
        inner_jac = self._inner.jacobian(params_1d)
        jac_v = inner_jac @ vvec[:, None]
        weighted = adj_state * deriv2(inner_val) * jac_v[:, 0]
        curvature = hvp_inner.hvp(
            params_1d, adj_state * self._transform_deriv(inner_val), vvec
        )
        return inner_jac.T @ weighted + curvature


class _ExpTransform(Generic[Array]):
    """Picklable exp transform for :class:`TransformedFieldMap`.

    One instance serves as ``transform``, ``transform_deriv``, and
    ``transform_deriv2`` for the exp map (since ``d/dx exp = exp``).
    Replaces the ``lambda x: bkd.exp(x)`` triple that previously broke
    pickling of the :class:`TransformedFieldMap` returned by the
    lognormal KLE factories (``pde.field_maps.kle_factory`` and
    ``pde.galerkin.kle_factory``).
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, x: Array) -> Array:
        return self._bkd.exp(x)

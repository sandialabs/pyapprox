"""Diffusivity-field parameterization for galerkin ADR physics.

Maps parameters through a field map (e.g. a lognormal KLE:
kappa = exp(K theta + mean)) onto the physics' ``NodalFieldDiffusion``
DOFs, with a second-order derivative bundle.

Assumptions (violate any -> this bundle is wrong for your physics):

1. Parameters enter ONLY through kappa; forcing/velocity/reaction/BCs
   and the IC are theta-independent (initial_param_jacobian = 0).
2. F is AFFINE in kappa, so dF/d(kappa) = B(u) is kappa-independent
   and all parameter curvature comes from the field map.
3. The kappa term is LINEAR in the state with B(0) = 0, and kappa
   multiplies a SELF-ADJOINT operator, making the mixed derivative
   tensor (state, residual)-symmetric — this is what lets the mixed
   HVPs reuse B at the adjoint/direction vectors.
4. kappa is state- and time-independent.
5. The field map's ``hvp`` is analytic (never finite differences).

All bundle outputs are RAW (no Dirichlet handling) — the BC-enforcing
wrappers own constraint rows.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

import numpy as np

from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
)
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
    FieldMapWithHVPProtocol,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class _GuardedHVPFieldMapProtocol(
    FieldMapWithHVPProtocol[Array], Protocol
):
    """Field map whose hvp availability is guarded (TransformedFieldMap
    exposes ``hvp`` structurally but honors it only when constructed
    with a second transform derivative)."""

    def has_hvp(self) -> bool: ...


def _field_map_has_hvp(field_map: FieldMapProtocol[Array]) -> bool:
    """Whether the field map provides a USABLE adjoint-weighted HVP."""
    if isinstance(field_map, _GuardedHVPFieldMapProtocol):
        return field_map.has_hvp()
    return isinstance(field_map, FieldMapWithHVPProtocol)


@runtime_checkable
class _DiffusivityFieldPhysicsProtocol(Protocol, Generic[Array]):
    """The physics members this parameterization consumes."""

    def nstates(self) -> int: ...

    def diffusion_function(self) -> object: ...

    def residual_diffusivity_jacobian(self, state: Array) -> Array:
        """dF/d(kappa DOFs); sparse at the skfem seam, typed Array per
        the repo's assembly convention."""
        ...


class AffineDiffusivityFieldParameterization(Generic[Array]):
    """Parameterize a galerkin physics' nodal diffusivity field.

    Parameters
    ----------
    physics : object
        Galerkin physics whose ``diffusion_function()`` is a
        ``NodalFieldDiffusion`` and which provides
        ``residual_diffusivity_jacobian``.
    field_map : FieldMapProtocol
        Map from parameters to nodal diffusivity values. When it also
        satisfies ``FieldMapWithHVPProtocol`` (with ``has_hvp()``) the
        bundle is second order; otherwise first order.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physics: _DiffusivityFieldPhysicsProtocol[Array],
        field_map: FieldMapProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, _DiffusivityFieldPhysicsProtocol):
            raise TypeError(
                "physics must provide diffusion_function/"
                "residual_diffusivity_jacobian/nstates, got "
                f"{type(physics).__name__}"
            )
        diffusion = physics.diffusion_function()
        if not isinstance(diffusion, NodalFieldDiffusion):
            raise TypeError(
                "physics.diffusion_function() must be a "
                "NodalFieldDiffusion (the differentiable "
                f"representation), got {type(diffusion).__name__}"
            )
        if not isinstance(field_map, FieldMapProtocol):
            raise TypeError(
                "field_map must satisfy FieldMapProtocol, got "
                f"{type(field_map).__name__}"
            )
        self._physics = physics
        self._diffusion = diffusion
        self._field_map = field_map
        self._bkd = bkd
        # Narrowed once here so HVP methods keep the typed reference.
        self._hvp_field_map: Optional[FieldMapWithHVPProtocol[Array]] = None
        if _field_map_has_hvp(field_map) and isinstance(
            field_map, FieldMapWithHVPProtocol
        ):
            self._hvp_field_map = field_map
            self._derivs: ParamDerivatives[Array] = (
                ParamDerivatives.second_order(
                    self._param_jacobian,
                    self.initial_param_jacobian,
                    self._param_param_hvp,
                    self._state_param_hvp,
                    self._param_state_hvp,
                )
            )
        else:
            self._derivs = ParamDerivatives.first_order(
                self._param_jacobian,
                self.initial_param_jacobian,
            )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def physics(self) -> _DiffusivityFieldPhysicsProtocol[Array]:
        """Return the bound physics instance."""
        return self._physics

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the derivative capability bundle."""
        return self._derivs

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._field_map.nvars()

    def apply(self, params_1d: Array) -> None:
        """Set the diffusivity DOFs to field_map(params).

        Bumps the diffusion function's version, invalidating the
        physics' stiffness cache.
        """
        self._diffusion.set_dofs(
            self._bkd.to_numpy(self._field_map(params_1d))
        )

    def _sensitivity(self, state: Array) -> Array:
        """B(state): the exact dF/d(kappa DOFs) mixed assembly."""
        return self._physics.residual_diffusivity_jacobian(state)

    def _field_jacobian_np(self, params_1d: Array) -> np.ndarray:
        """d(kappa)/d(theta) as numpy. Shape: (nfield, nparams)."""
        return np.asarray(
            self._bkd.to_numpy(self._field_map.jacobian(params_1d))
        )

    def _param_jacobian(
        self, state: Array, time: float, params_1d: Array
    ) -> Array:
        """dF/dp = B(u) @ d(kappa)/dp. Shape: (nstates, nparams)."""
        return self._bkd.asarray(
            self._sensitivity(state) @ self._field_jacobian_np(params_1d)
        )

    def _state_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2F/dy dp) v = B(adj) @ (d(kappa)/dp v).

        Valid because B is linear in the state with a symmetric
        state-derivative tensor (see module docstring).
        Shape: (nstates,).
        """
        kappa_dir = self._field_jacobian_np(params_1d) @ self._bkd.to_numpy(
            vvec
        )
        return self._bkd.asarray(self._sensitivity(adj_state) @ kappa_dir)

    def _param_state_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """adj^T (d^2F/dp dy) w = (d(kappa)/dp)^T B(w)^T adj.

        Transpose contraction of ``_state_param_hvp``.
        Shape: (nparams,).
        """
        weights = self._sensitivity(wvec).T @ self._bkd.to_numpy(adj_state)
        return self._bkd.asarray(
            self._field_jacobian_np(params_1d).T @ weights
        )

    def _param_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2F/dp^2) v via the field map's curvature.

        F is linear in kappa, so all parameter curvature is the field
        map's: the contraction is its adjoint-weighted HVP with weights
        ``B(u)^T adj``. Shape: (nparams,).
        """
        field_map = self._hvp_field_map
        if field_map is None:
            raise RuntimeError(
                "param_param_hvp is unavailable; check "
                "param_derivatives() before calling"
            )
        weights = self._bkd.asarray(
            self._sensitivity(state).T @ self._bkd.to_numpy(adj_state)
        )
        return field_map.hvp(params_1d, weights, vvec)

    def initial_param_jacobian(self, params_1d: Array) -> Array:
        """d(u_0)/dp = 0 (the IC does not depend on the diffusivity)."""
        return self._bkd.asarray(
            np.zeros((self._physics.nstates(), self.nparams()))
        )

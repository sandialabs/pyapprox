"""DiffusionParameterization: binds a FieldMap to diffusion coefficient."""

from typing import (
    Callable,
    Generic,
    List,
    Protocol,
    runtime_checkable,
)

from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.fields import ConstantInTimeField
from pyapprox.pde.parameterizations.protocol import (
    DerivativeMatrixBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class _CollocationDiffusionPhysicsProtocol(Protocol, Generic[Array]):
    """Physics members this parameterization calls (interim; the typed
    facades of the parameterization redesign replace it)."""

    def npts(self) -> int: ...

    def set_diffusion(self, diffusion: Callable[[float], Array]) -> None: ...

    def residual_diffusion_sensitivity(
        self,
        state: Array,
        time: float,
        delta_D: Array,
        grad_delta_D: List[Array],
    ) -> Array: ...


class DiffusionParameterization(Generic[Array]):
    """Parameterization that maps parameters to diffusion coefficient.

    The physics is bound at construction: one instance serves one
    physics (ensembles construct one parameterization per physics).

    Parameters
    ----------
    physics : _CollocationDiffusionPhysicsProtocol
        Collocation physics with ``set_diffusion``,
        ``residual_diffusion_sensitivity``, and ``npts`` members.
    field_map : FieldMapProtocol
        Maps parameter vector to diffusion field.
    derivative_matrices : List[Array]
        First-derivative matrices, one per spatial dimension.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        physics: _CollocationDiffusionPhysicsProtocol[Array],
        field_map: FieldMapProtocol[Array],
        derivative_matrices: List[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(physics, _CollocationDiffusionPhysicsProtocol):
            raise TypeError(
                f"physics must provide set_diffusion/"
                f"residual_diffusion_sensitivity/npts, "
                f"got {type(physics).__name__}"
            )
        if not isinstance(field_map, FieldMapProtocol):
            raise TypeError(
                f"field_map must satisfy FieldMapProtocol, "
                f"got {type(field_map).__name__}"
            )
        self._physics = physics
        self._field_map = field_map
        self._D_matrices = derivative_matrices
        self._bkd = bkd
        self._derivs: ParamDerivatives[Array] = ParamDerivatives.first_order(
            self.param_jacobian,
            self.initial_param_jacobian,
            bc_flux_param_sensitivity=self.bc_flux_param_sensitivity,
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def physics(self) -> _CollocationDiffusionPhysicsProtocol[Array]:
        """Return the bound physics instance."""
        return self._physics

    def param_derivatives(self) -> ParamDerivatives[Array]:
        return self._derivs

    def nparams(self) -> int:
        return self._field_map.nvars()

    def apply(self, params_1d: Array) -> None:
        """Apply parameterization: set diffusion field on physics."""
        field = self._field_map(params_1d)
        min_val = self._bkd.to_float(self._bkd.min(field))
        if min_val <= 0.0:
            min_idx = self._bkd.to_int(self._bkd.argmin(field))
            raise ValueError(
                f"Diffusion coefficient must be positive at all "
                f"collocation nodes; found min value {min_val:.2e} "
                f"at node {min_idx}"
            )
        self._physics.set_diffusion(ConstantInTimeField(field))

    def param_jacobian(
        self,
        state: Array,
        time: float,
        params_1d: Array,
    ) -> Array:
        """Compute d(residual)/d(params) via chain rule. Shape: (npts, nparams)."""
        fm_jac = self._field_map.jacobian(params_1d)  # (npts, nparams)
        nparams = self.nparams()
        npts = state.shape[0]
        result = self._bkd.zeros((npts, nparams))
        result = self._bkd.copy(result)
        for i in range(nparams):
            delta_D = fm_jac[:, i]
            grad_delta_D = [D_mat @ delta_D for D_mat in self._D_matrices]
            col = self._physics.residual_diffusion_sensitivity(
                state, time, delta_D, grad_delta_D
            )
            for j in range(npts):
                result[j, i] = col[j]
        return result

    def bc_flux_param_sensitivity(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """Compute d(flux·n)/dp at boundary nodes. Shape: (n_bc, n_params).

        For diffusion flux = -D*grad(u): d(flux·n)/dp = -(grad_u·n) * dD/dp.
        """
        dD_dp_all = self._field_map.jacobian(params_1d)
        dD_dp = dD_dp_all[bc_indices]  # (n_bc, nparams)
        ndim = len(self._D_matrices)
        nbnd = bc_indices.shape[0]
        grad_u_dot_n = self._bkd.zeros((nbnd,))
        for d in range(ndim):
            grad_u_d = self._D_matrices[d] @ state
            grad_u_dot_n = grad_u_dot_n + grad_u_d[bc_indices] * normals[:, d]
        return -grad_u_dot_n[:, None] * dD_dp

    def initial_param_jacobian(self, params_1d: Array) -> Array:
        """Return d(initial_state)/d(params). Shape: (nstates, nparams)."""
        npts = self._physics.npts()
        return self._bkd.zeros((npts, self.nparams()))


def create_diffusion_parameterization(
    physics: _CollocationDiffusionPhysicsProtocol[Array],
    bkd: Backend[Array],
    basis: DerivativeMatrixBasisProtocol[Array],
    field_map: FieldMapProtocol[Array],
) -> DiffusionParameterization[Array]:
    """Factory: create DiffusionParameterization extracting D matrices from basis.

    Parameters
    ----------
    physics : _CollocationDiffusionPhysicsProtocol
        Collocation physics to bind.
    bkd : Backend
        Computational backend.
    basis : DerivativeMatrixBasisProtocol
        Collocation basis.
    field_map : FieldMapProtocol
        Field map for diffusion.
    """
    D_matrices = [basis.derivative_matrix(1, dim) for dim in range(basis.ndim())]
    return DiffusionParameterization(physics, field_map, D_matrices, bkd)

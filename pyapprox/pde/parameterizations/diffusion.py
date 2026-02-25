"""DiffusionParameterization: binds a FieldMap to diffusion coefficient."""

from typing import Callable, Generic, List, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)


class DiffusionParameterization(Generic[Array]):
    """Parameterization that maps parameters to diffusion coefficient.

    Parameters
    ----------
    field_map : FieldMapProtocol
        Maps parameter vector to diffusion field.
    derivative_matrices : List[Array]
        First-derivative matrices, one per spatial dimension.
    bkd : Backend
        Computational backend.
    time_modulation : Callable[[float], float], optional
        Time modulation factor. Defaults to constant 1.0.
    """

    def __init__(
        self,
        field_map: FieldMapProtocol[Array],
        derivative_matrices: List[Array],
        bkd: Backend[Array],
        time_modulation: Optional[Callable[[float], float]] = None,
    ) -> None:
        if not isinstance(field_map, FieldMapProtocol):
            raise TypeError(
                f"field_map must satisfy FieldMapProtocol, "
                f"got {type(field_map).__name__}"
            )
        self._field_map = field_map
        self._D_matrices = derivative_matrices
        self._bkd = bkd
        self._time_mod = time_modulation if time_modulation is not None else lambda t: 1.0

        # Dynamic binding: param_jacobian only if field_map has jacobian
        if hasattr(self._field_map, "jacobian"):
            self.param_jacobian = self._param_jacobian
            self.initial_param_jacobian = self._initial_param_jacobian
            self.bc_flux_param_sensitivity = self._bc_flux_param_sensitivity

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nparams(self) -> int:
        return self._field_map.nvars()

    def apply(self, physics: object, params_1d: Array) -> None:
        """Apply parameterization: set diffusion field on physics."""
        field = self._field_map(params_1d)
        min_val = float(self._bkd.min(field))
        if min_val <= 0.0:
            min_idx = int(self._bkd.argmin(field))
            raise ValueError(
                f"Diffusion coefficient must be positive at all "
                f"collocation nodes; found min value {min_val:.2e} "
                f"at node {min_idx}"
            )
        physics.set_diffusion(lambda t, _f=field: _f * self._time_mod(t))

    def _param_jacobian(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
    ) -> Array:
        """Compute d(residual)/d(params) via chain rule. Shape: (npts, nparams)."""
        fm_jac = self._field_map.jacobian(params_1d)  # (npts, nparams)
        mod = self._time_mod(time)
        nparams = self.nparams()
        npts = state.shape[0]
        result = self._bkd.zeros((npts, nparams))
        result = self._bkd.copy(result)
        for i in range(nparams):
            delta_D = fm_jac[:, i] * mod
            grad_delta_D = [D_mat @ delta_D for D_mat in self._D_matrices]
            col = physics.residual_diffusion_sensitivity(
                state, time, delta_D, grad_delta_D
            )
            for j in range(npts):
                result[j, i] = col[j]
        return result

    def _bc_flux_param_sensitivity(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """Compute d(flux·n)/dp at boundary nodes. Shape: (n_bc, n_params).

        For diffusion flux = -D*grad(u): d(flux·n)/dp = -(grad_u·n) * dD/dp.
        """
        dD_dp_all = self._field_map.jacobian(params_1d) * self._time_mod(time)
        dD_dp = dD_dp_all[bc_indices]  # (n_bc, nparams)
        ndim = len(self._D_matrices)
        nbnd = bc_indices.shape[0]
        grad_u_dot_n = self._bkd.zeros((nbnd,))
        for d in range(ndim):
            grad_u_d = self._D_matrices[d] @ state
            grad_u_dot_n = grad_u_dot_n + grad_u_d[bc_indices] * normals[:, d]
        return -grad_u_dot_n[:, None] * dD_dp

    def _initial_param_jacobian(
        self, physics: object, params_1d: Array
    ) -> Array:
        """Return d(initial_state)/d(params). Shape: (nstates, nparams)."""
        npts = physics.npts()
        return self._bkd.zeros((npts, self.nparams()))


def create_diffusion_parameterization(
    bkd: Backend[Array],
    basis: TensorProductBasisProtocol[Array],
    field_map: FieldMapProtocol[Array],
    time_modulation: Optional[Callable[[float], float]] = None,
) -> DiffusionParameterization[Array]:
    """Factory: create DiffusionParameterization extracting D matrices from basis.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    basis : TensorProductBasisProtocol
        Collocation basis.
    field_map : FieldMapProtocol
        Field map for diffusion.
    time_modulation : Callable, optional
        Time modulation factor.
    """
    D_matrices = [
        basis.derivative_matrix(1, dim)
        for dim in range(basis.ndim())
    ]
    return DiffusionParameterization(
        field_map, D_matrices, bkd, time_modulation
    )

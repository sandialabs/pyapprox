"""YoungModulusParameterization: maps E field to Lame params for linear elasticity.

This class calls physics.residual_mu_sensitivity / residual_lamda_sensitivity,
which are implemented by both LinearElasticityPhysics and HyperelasticityPhysics.
Once HyperelasticityPhysics gains 2D sensitivity methods (Phase 9), this class
can replace HyperelasticYoungsModulusParameterization as the universal E-to-Lame
parameterization for all elasticity types.
"""

from typing import Generic, List

from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class YoungModulusParameterization(Generic[Array]):
    """Parameterization mapping Young's modulus E to Lame parameters.

    Converts E to (mu, lambda) using fixed Poisson ratio nu:
        mu = E / (2*(1+nu))
        lambda = E*nu / ((1+nu)*(1-2*nu))

    Works for 2D linear elasticity with vector state (2*npts DOFs).

    Parameters
    ----------
    field_map : FieldMapProtocol
        Maps parameter vector to Young's modulus field E(x).
    derivative_matrices : List[Array]
        First-derivative matrices, one per spatial dimension.
    bkd : Backend
        Computational backend.
    poisson_ratio : float
        Fixed Poisson ratio nu.
    """

    def __init__(
        self,
        field_map: FieldMapProtocol[Array],
        derivative_matrices: List[Array],
        bkd: Backend[Array],
        poisson_ratio: float,
    ) -> None:
        if not isinstance(field_map, FieldMapProtocol):
            raise TypeError(
                f"field_map must satisfy FieldMapProtocol, "
                f"got {type(field_map).__name__}"
            )
        self._field_map = field_map
        self._D_matrices = derivative_matrices
        self._bkd = bkd
        self._nu = poisson_ratio
        self._dmu_dE = 1.0 / (2.0 * (1.0 + poisson_ratio))
        self._dlam_dE = poisson_ratio / (
            (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
        )

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
        """Apply parameterization: convert E field to Lame params on physics."""
        E_field = self._field_map(params_1d)
        min_val = float(self._bkd.min(E_field))
        if min_val <= 0.0:
            raise ValueError(
                f"Young's modulus must be positive; found min {min_val:.2e}"
            )
        physics.set_mu(E_field * self._dmu_dE)
        physics.set_lamda(E_field * self._dlam_dE)

    def _param_jacobian(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
    ) -> Array:
        """Compute dR/dp via chain rule. Shape: (2*npts, nparams).

        dR/dp = (dR/dmu * dmu/dE + dR/dlam * dlam/dE) * dE/dp
        """
        fm_jac = self._field_map.jacobian(params_1d)  # (npts, nparams)
        nparams = self.nparams()
        nstates = physics.nstates()
        result = self._bkd.zeros((nstates, nparams))
        result = self._bkd.copy(result)
        for j in range(nparams):
            delta_E = fm_jac[:, j]
            delta_mu = delta_E * self._dmu_dE
            delta_lam = delta_E * self._dlam_dE
            col = physics.residual_mu_sensitivity(
                state, time, delta_mu
            ) + physics.residual_lamda_sensitivity(state, time, delta_lam)
            for k in range(nstates):
                result[k, j] = col[k]
        return result

    def _initial_param_jacobian(self, physics: object, params_1d: Array) -> Array:
        """Return d(initial_state)/d(params). Shape: (nstates, nparams)."""
        return self._bkd.zeros((physics.nstates(), self.nparams()))

    def _bc_flux_param_sensitivity(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """Compute d(traction)/dp at boundary nodes.

        Shape: (2*n_bc, nparams) for 2D vector physics (component-stacked).

        Traction: t_i = sigma_ij * n_j
        d(t_x)/dE = (2*exx*nx + 2*exy*ny)*dmu/dE + trace_e*nx*dlam/dE
        d(t_y)/dE = (2*exy*nx + 2*eyy*ny)*dmu/dE + trace_e*ny*dlam/dE
        """
        bkd = self._bkd
        fm_jac = self._field_map.jacobian(params_1d)  # (npts, nparams)
        npts = physics.npts()

        u = state[:npts]
        v = state[npts:]
        Dx = self._D_matrices[0]
        Dy = self._D_matrices[1]

        # Strain at boundary
        ux = (Dx @ u)[bc_indices]
        uy = (Dy @ u)[bc_indices]
        vx = (Dx @ v)[bc_indices]
        vy = (Dy @ v)[bc_indices]

        exx = ux
        exy = 0.5 * (uy + vx)
        eyy = vy
        trace_e = exx + eyy

        nx = normals[:, 0]
        ny = normals[:, 1]

        # d(traction)/dE at boundary
        dtx_dE = (
            2.0 * exx * nx + 2.0 * exy * ny
        ) * self._dmu_dE + trace_e * nx * self._dlam_dE
        dty_dE = (
            2.0 * exy * nx + 2.0 * eyy * ny
        ) * self._dmu_dE + trace_e * ny * self._dlam_dE

        # d(traction)/dp = d(traction)/dE * dE/dp
        dE_dp_bc = fm_jac[bc_indices, :]  # (n_bc, nparams)
        dtx_dp = dtx_dE[:, None] * dE_dp_bc  # (n_bc, nparams)
        dty_dp = dty_dE[:, None] * dE_dp_bc  # (n_bc, nparams)

        # Component-stacked: [tx_0,...,tx_n, ty_0,...,ty_n]
        return bkd.concatenate([dtx_dp, dty_dp], axis=0)


def create_youngs_modulus_parameterization(
    bkd: Backend[Array],
    basis: TensorProductBasisProtocol[Array],
    field_map: FieldMapProtocol[Array],
    poisson_ratio: float,
) -> YoungModulusParameterization[Array]:
    """Factory: create YoungModulusParameterization.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    basis : TensorProductBasisProtocol
        Collocation basis.
    field_map : FieldMapProtocol
        Field map for Young's modulus.
    poisson_ratio : float
        Fixed Poisson ratio.
    """
    D_matrices = [basis.derivative_matrix(1, dim) for dim in range(basis.ndim())]
    return YoungModulusParameterization(field_map, D_matrices, bkd, poisson_ratio)

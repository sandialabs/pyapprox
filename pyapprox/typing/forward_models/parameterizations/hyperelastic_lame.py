"""HyperelasticYoungsModulusParameterization: maps E field to Lame params.

Supports 1D and 2D hyperelastic physics. In 1D, uses stress model
sensitivities directly for efficiency. In 2D, delegates to the physics
residual_mu/lamda_sensitivity methods (same as YoungModulusParameterization).
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.forward_models.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.typing.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)


class HyperelasticYoungsModulusParameterization(Generic[Array]):
    """Parameterization mapping Young's modulus E to Lame parameters.

    Converts E to (mu, lambda) using fixed Poisson ratio nu:
        mu = E / (2*(1+nu))
        lambda = E*nu / ((1+nu)*(1-2*nu))

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
        """Compute dR/dp via chain rule. Shape: (nstates, nparams).

        1D: dR/dp = Dx @ dP/dp (direct stress sensitivity)
        2D: dR/dp via physics.residual_mu/lamda_sensitivity (universal)
        """
        fm_jac = self._field_map.jacobian(params_1d)  # (npts, nparams)
        ndim = len(self._D_matrices)
        if ndim == 1:
            Dx = self._D_matrices[0]
            F = 1.0 + Dx @ state
            stress_model = physics.stress_model()
            dP_dmu = stress_model.stress_sensitivity_mu_1d(F, self._bkd)
            dP_dlam = stress_model.stress_sensitivity_lamda_1d(F, self._bkd)
            dP_dp = (
                dP_dmu[:, None] * self._dmu_dE
                + dP_dlam[:, None] * self._dlam_dE
            ) * fm_jac
            return Dx @ dP_dp
        # 2D: delegate to physics sensitivity methods
        nparams = self.nparams()
        nstates = physics.nstates()
        bkd = self._bkd
        result = bkd.zeros((nstates, nparams))
        result = bkd.copy(result)
        for j in range(nparams):
            delta_E = fm_jac[:, j]
            delta_mu = delta_E * self._dmu_dE
            delta_lam = delta_E * self._dlam_dE
            col = (
                physics.residual_mu_sensitivity(state, time, delta_mu)
                + physics.residual_lamda_sensitivity(state, time, delta_lam)
            )
            for k in range(nstates):
                result[k, j] = col[k]
        return result

    def _initial_param_jacobian(
        self, physics: object, params_1d: Array
    ) -> Array:
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
        """Compute d(P*n)/dp at boundary nodes. Shape: (n_bc, nparams).

        1D: d(P*n)/dp = n * dP/dp at boundary nodes.
        2D: d(t_comp)/dp = sum_J n_J * dP_{comp,J}/dE * dE/dp at boundary.
        """
        fm_jac = self._field_map.jacobian(params_1d)  # (npts, nparams)
        ndim = len(self._D_matrices)
        bkd = self._bkd
        stress_model = physics.stress_model()
        if ndim == 1:
            Dx = self._D_matrices[0]
            F = 1.0 + Dx @ state
            dP_dmu = stress_model.stress_sensitivity_mu_1d(F, bkd)
            dP_dlam = stress_model.stress_sensitivity_lamda_1d(F, bkd)
            dP_dp = (
                dP_dmu[:, None] * self._dmu_dE
                + dP_dlam[:, None] * self._dlam_dE
            ) * fm_jac
            return normals[:, 0:1] * dP_dp[bc_indices, :]
        # 2D: bc_indices are state indices (mesh_idx + component*npts)
        npts = physics.npts()
        # Determine component from state indices
        comp = int(bc_indices[0]) // npts  # 0 or 1
        mesh_idx = bc_indices - comp * npts
        # Compute deformation gradient at boundary mesh points
        Dx, Dy = self._D_matrices[0], self._D_matrices[1]
        u = state[:npts]
        v = state[npts:]
        F11 = (1.0 + Dx @ u)[mesh_idx]
        F12 = (Dy @ u)[mesh_idx]
        F21 = (Dx @ v)[mesh_idx]
        F22 = (1.0 + Dy @ v)[mesh_idx]
        # PK1 stress sensitivity w.r.t. mu and lambda at boundary
        dP_mu = stress_model.stress_sensitivity_mu_2d(
            F11, F12, F21, F22, bkd,
        )  # (dP11, dP12, dP21, dP22)
        dP_lam = stress_model.stress_sensitivity_lamda_2d(
            F11, F12, F21, F22, bkd,
        )
        # dP_iJ/dE = dP_iJ/dmu * dmu/dE + dP_iJ/dlam * dlam/dE
        # For component comp: t_comp = P_{comp+1,1}*nx + P_{comp+1,2}*ny
        # Index into (dP11,dP12,dP21,dP22): comp=0 -> (0,1), comp=1 -> (2,3)
        i1 = 2 * comp      # P_{comp+1,1}
        i2 = 2 * comp + 1  # P_{comp+1,2}
        dP_i1_dE = dP_mu[i1] * self._dmu_dE + dP_lam[i1] * self._dlam_dE
        dP_i2_dE = dP_mu[i2] * self._dmu_dE + dP_lam[i2] * self._dlam_dE
        nx = normals[:, 0]
        ny = normals[:, 1]
        dt_dE = dP_i1_dE * nx + dP_i2_dE * ny  # (nbnd,)
        # Chain to params: dt/dp = dt/dE * dE/dp
        dE_dp_bc = fm_jac[mesh_idx, :]  # (nbnd, nparams)
        return dt_dE[:, None] * dE_dp_bc


def create_hyperelastic_youngs_modulus_parameterization(
    bkd: Backend[Array],
    basis: TensorProductBasisProtocol[Array],
    field_map: FieldMapProtocol[Array],
    poisson_ratio: float,
) -> HyperelasticYoungsModulusParameterization[Array]:
    """Factory: create HyperelasticYoungsModulusParameterization.

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
    D_matrices = [
        basis.derivative_matrix(1, dim)
        for dim in range(basis.ndim())
    ]
    return HyperelasticYoungsModulusParameterization(
        field_map, D_matrices, bkd, poisson_ratio
    )

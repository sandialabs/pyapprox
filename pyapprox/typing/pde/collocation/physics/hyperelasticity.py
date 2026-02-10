"""Hyperelasticity physics for spectral collocation.

Implements the equilibrium equation for hyperelastic materials:
    div(P) + f = 0

where P is the first Piola-Kirchhoff stress computed by a pluggable
stress model. Supports 1D, 2D, and 3D with any constitutive law that
satisfies StressModelProtocol.
"""

from typing import Callable, Generic, List, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.typing.pde.collocation.physics.base import AbstractVectorPhysics
from pyapprox.typing.pde.collocation.physics.stress_models.protocols import (
    StressModelProtocol,
)
from pyapprox.typing.pde.collocation.physics.stress_models.registry import (
    create_stress_model,
)


class HyperelasticityPhysics(AbstractVectorPhysics[Array], Generic[Array]):
    """Hyperelastic physics with pluggable stress model.

    Implements the equilibrium equation:
        div(P) + f = 0

    where P is the first Piola-Kirchhoff stress tensor computed by the
    provided stress model from the deformation gradient F = I + grad(u).

    Supports 1D, 2D, and 3D. The analytical Jacobian requires the stress
    model to provide a tangent modulus (compute_tangent methods).

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (1D, 2D, or 3D).
    bkd : Backend
        Computational backend.
    stress_model : StressModelProtocol
        Constitutive model for PK1 stress.
    forcing : Callable[[float], Array] or Array, optional
        Body force. If callable, takes time and returns shape (nstates,).
        If Array, shape (nstates,). Default: None (zero forcing).
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        stress_model: StressModelProtocol[Array],
        forcing: Optional[Callable[[float], Array]] = None,
    ):
        ndim = basis.ndim()
        if ndim not in (1, 2, 3):
            raise ValueError(f"Unsupported dimension: {ndim}")

        super().__init__(basis, bkd, ncomponents=ndim)

        self._stress_model = stress_model
        self._forcing_func = forcing
        self._ndim = ndim

        # Precompute first-order derivative matrices
        self._D: List[Array] = [
            basis.derivative_matrix(1, dim) for dim in range(ndim)
        ]

    def stress_model(self) -> StressModelProtocol[Array]:
        """Return the stress model."""
        return self._stress_model

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        nstates = self.nstates()
        if self._forcing_func is None:
            return self._bkd.zeros((nstates,))
        if callable(self._forcing_func):
            return self._forcing_func(time)
        return self._forcing_func

    def _extract_components(self, state: Array) -> Tuple[Array, ...]:
        """Extract displacement components from state vector.

        State is ordered as [u0_0, ..., u0_{n-1}, u1_0, ..., u1_{n-1}, ...]

        Parameters
        ----------
        state : Array
            Full state vector. Shape: (ndim * npts,)

        Returns
        -------
        Tuple[Array, ...]
            Displacement components, each shape (npts,).
        """
        npts = self.npts()
        return tuple(
            state[i * npts : (i + 1) * npts] for i in range(self._ndim)
        )

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------

    def residual(self, state: Array, time: float) -> Array:
        """Compute div(P) + f.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (nstates,)
        """
        if self._ndim == 1:
            return self._residual_1d(state, time)
        elif self._ndim == 2:
            return self._residual_2d(state, time)
        else:
            return self._residual_3d(state, time)

    def _residual_1d(self, state: Array, time: float) -> Array:
        """1D residual: dP/dx + f."""
        bkd = self._bkd
        Dx = self._D[0]
        u = state

        F = 1.0 + Dx @ u
        P = self._stress_model.compute_stress_1d(F, bkd)
        div_P = Dx @ P
        forcing = self._get_forcing(time)

        return div_P + forcing

    def _residual_2d(self, state: Array, time: float) -> Array:
        """2D residual: dP_iJ/dX_J + f_i."""
        bkd = self._bkd
        npts = self.npts()
        Dx, Dy = self._D

        u, v = self._extract_components(state)

        # Deformation gradient F = I + grad(u)
        F11 = 1.0 + Dx @ u
        F12 = Dy @ u
        F21 = Dx @ v
        F22 = 1.0 + Dy @ v

        P11, P12, P21, P22 = self._stress_model.compute_stress_2d(
            F11, F12, F21, F22, bkd
        )

        # div(P)_i = dP_i1/dx + dP_i2/dy
        div_P_x = Dx @ P11 + Dy @ P12
        div_P_y = Dx @ P21 + Dy @ P22

        forcing = self._get_forcing(time)
        fx = forcing[:npts]
        fy = forcing[npts:]

        return bkd.concatenate([div_P_x + fx, div_P_y + fy])

    def _residual_3d(self, state: Array, time: float) -> Array:
        """3D residual: dP_iJ/dX_J + f_i."""
        bkd = self._bkd
        npts = self.npts()
        Dx, Dy, Dz = self._D

        u, v, w = self._extract_components(state)

        # Deformation gradient F = I + grad(u)
        F = (
            (1.0 + Dx @ u, Dy @ u, Dz @ u),
            (Dx @ v, 1.0 + Dy @ v, Dz @ v),
            (Dx @ w, Dy @ w, 1.0 + Dz @ w),
        )

        P = self._stress_model.compute_stress_3d(F, bkd)
        P11, P12, P13 = P[0]
        P21, P22, P23 = P[1]
        P31, P32, P33 = P[2]

        # div(P)_i = dP_i1/dx + dP_i2/dy + dP_i3/dz
        div_P_x = Dx @ P11 + Dy @ P12 + Dz @ P13
        div_P_y = Dx @ P21 + Dy @ P22 + Dz @ P23
        div_P_z = Dx @ P31 + Dy @ P32 + Dz @ P33

        forcing = self._get_forcing(time)
        fx = forcing[:npts]
        fy = forcing[npts : 2 * npts]
        fz = forcing[2 * npts :]

        return bkd.concatenate([
            div_P_x + fx, div_P_y + fy, div_P_z + fz
        ])

    # ------------------------------------------------------------------
    # Jacobian computation
    # ------------------------------------------------------------------

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute d(residual)/d(state).

        Requires the stress model to provide analytical tangent modulus.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        if self._ndim == 1:
            return self._jacobian_1d(state)
        elif self._ndim == 2:
            return self._jacobian_2d(state)
        else:
            return self._jacobian_3d(state)

    def _jacobian_1d(self, state: Array) -> Array:
        """1D Jacobian: Dx @ diag(dP/dF) @ Dx."""
        bkd = self._bkd
        Dx = self._D[0]
        u = state

        F = 1.0 + Dx @ u

        if not hasattr(self._stress_model, "compute_tangent_1d"):
            raise NotImplementedError(
                "Stress model does not provide 1D tangent modulus."
            )
        A = self._stress_model.compute_tangent_1d(F, bkd)

        return Dx @ bkd.diag(A) @ Dx

    def _jacobian_2d(self, state: Array) -> Array:
        """2D Jacobian using tangent modulus A_iJkL.

        Block structure:
            J = [[J_uu, J_uv],
                 [J_vu, J_vv]]

        where J_ab = sum_{J,L} D_J @ diag(A_{aJ,bL}) @ D_L.

        Index mapping: a=1->u, a=2->v; J=1->x, J=2->y.
        The tangent A_iJkL relates stress row (i,J) to gradient (k,L),
        and the deformation gradient component F_kL is obtained from
        displacement component k via D_L.
        """
        bkd = self._bkd
        Dx, Dy = self._D

        u, v = self._extract_components(state)

        F11 = 1.0 + Dx @ u
        F12 = Dy @ u
        F21 = Dx @ v
        F22 = 1.0 + Dy @ v

        if not hasattr(self._stress_model, "compute_tangent_2d"):
            raise NotImplementedError(
                "Stress model does not provide 2D tangent modulus."
            )
        A = self._stress_model.compute_tangent_2d(
            F11, F12, F21, F22, bkd
        )

        D = [Dx, Dy]

        # Build block Jacobian
        # res_i = sum_J D_J @ P_iJ + f_i
        # P_iJ depends on F_kL = delta_{kL} + D_L @ u_k
        # d(res_i)/d(u_k) = sum_J sum_L D_J @ diag(A_{iJ,kL}) @ D_L
        #
        # Component indices: u->1, v->2; x->1, y->2
        # Block (a,b) computes d(res_a)/d(u_b)

        blocks = []
        for a in range(2):  # equation component (1-indexed: a+1)
            row_blocks = []
            for b in range(2):  # state component (1-indexed: b+1)
                block = bkd.zeros(
                    (self.npts(), self.npts())
                )
                for jj in range(2):  # stress column index J (1-indexed: jj+1)
                    for ll in range(2):  # gradient column index L
                        key = f"A_{a+1}{jj+1}{b+1}{ll+1}"
                        block = block + D[jj] @ bkd.diag(A[key]) @ D[ll]
                row_blocks.append(block)
            blocks.append(row_blocks)

        # Assemble [[J_uu, J_uv], [J_vu, J_vv]]
        top_row = bkd.concatenate(
            [blocks[0][0], blocks[0][1]], axis=1
        )
        bottom_row = bkd.concatenate(
            [blocks[1][0], blocks[1][1]], axis=1
        )
        return bkd.concatenate([top_row, bottom_row], axis=0)

    def _jacobian_3d(self, state: Array) -> Array:
        """3D Jacobian - not yet implemented."""
        raise NotImplementedError("3D Jacobian not yet implemented")


def create_hyperelasticity(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    stress_model_name: str = "neo_hookean",
    forcing: Optional[Callable[[float], Array]] = None,
    **stress_kwargs: float,
) -> HyperelasticityPhysics[Array]:
    """Create hyperelasticity physics with a named stress model.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (1D, 2D, or 3D).
    bkd : Backend
        Computational backend.
    stress_model_name : str
        Name of registered stress model. Default: "neo_hookean".
    forcing : Callable or Array, optional
        Body force term.
    **stress_kwargs
        Material parameters passed to the stress model factory.

    Returns
    -------
    HyperelasticityPhysics
        Hyperelasticity physics instance.
    """
    stress_model = create_stress_model(stress_model_name, **stress_kwargs)
    return HyperelasticityPhysics(basis, bkd, stress_model, forcing)

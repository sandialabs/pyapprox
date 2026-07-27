"""Linear elasticity physics for spectral collocation.

Implements the linear elasticity equations in 2D and 3D:
    -div(σ) + f = 0

where:
    σ = λ*tr(ε)*I + 2μ*ε  (stress tensor)
    ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)  (strain tensor)
    u = (u, v) or (u, v, w) is the displacement field
    λ, μ are Lamé parameters
"""

from typing import Callable, Generic, List, Optional, Tuple, Union

from pyapprox.pde.collocation.physics.base import AbstractVectorPhysics
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class LinearElasticityPhysics(AbstractVectorPhysics[Array], Generic[Array]):
    """Linear elasticity physics in 2D or 3D.

    Implements the equilibrium equation:
        -div(σ) + f = 0

    where the stress tensor is:
        σ = λ*tr(ε)*I + 2μ*ε

    and the strain tensor is:
        ε_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)

    The residual is formulated as:
        residual = div(σ) + f

    So that residual = 0 at equilibrium. The state stacks one
    displacement component per spatial dimension, each of length npts.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D or 3D collocation basis (provides nodes, derivative
        matrices). The number of displacement components equals
        ``basis.ndim()``.
    bkd : Backend
        Computational backend.
    lamda : float or Array
        Lamé's first parameter λ. If Array, shape: (npts,).
    mu : float or Array
        Shear modulus μ. If Array, shape: (npts,).
    forcing : Callable[[float], Array] or Array, optional
        Forcing term. If callable, takes time and returns
        (ndim*npts,) array with the force components stacked.
        If Array, shape: (ndim*npts,). Default: None (no forcing)
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        lamda: Union[float, Array],
        mu: Union[float, Array],
        forcing: Optional[Callable[[float], Array]] = None,
    ):
        ndim = basis.ndim()
        if ndim not in (2, 3):
            raise ValueError(
                "LinearElasticityPhysics requires a 2D or 3D basis"
            )

        super().__init__(basis, bkd, ncomponents=ndim)

        self._ndim = ndim
        npts = basis.npts()

        # Store Lamé parameters (the scalar mirrors are None for
        # per-point fields)
        self._lambda_value: Optional[float]
        self._mu_value: Optional[float]
        if isinstance(lamda, (int, float)):
            self._lambda_array = bkd.full((npts,), float(lamda))
            self._lambda_value = float(lamda)
        else:
            self._lambda_array = lamda
            self._lambda_value = None

        if isinstance(mu, (int, float)):
            self._mu_array = bkd.full((npts,), float(mu))
            self._mu_value = float(mu)
        else:
            self._mu_array = mu
            self._mu_value = None

        # Store forcing function
        self._forcing_func = forcing

        # Precompute first-derivative matrices per dimension
        self._D: List[Array] = [
            basis.derivative_matrix(1, dim) for dim in range(ndim)
        ]
        if ndim == 2:
            self._Dx = self._D[0]  # d/dx
            self._Dy = self._D[1]  # d/dy
            self._Dxx = basis.derivative_matrix(2, 0)  # d²/dx²
            self._Dyy = basis.derivative_matrix(2, 1)  # d²/dy²
            # Mixed derivatives: Dx@Dy != Dy@Dx on curvilinear domains
            self._DxDy = self._Dx @ self._Dy  # Dx applied after Dy
            self._DyDx = self._Dy @ self._Dx  # Dy applied after Dx

    # ------------------------------------------------------------------
    # Material property setters
    # ------------------------------------------------------------------

    def set_mu(self, mu_values: Union[float, Array]) -> None:
        """Set shear modulus (scalar or per-point array).

        Parameters
        ----------
        mu_values : float or Array
            Shear modulus values. Must be positive.
        """
        mu_arr = self._bkd.ravel(self._bkd.asarray(mu_values))
        min_val = self._bkd.to_float(self._bkd.min(mu_arr))
        if min_val <= 0.0:
            raise ValueError(f"mu must be positive; found min {min_val:.2e}")
        if isinstance(mu_values, (int, float)):
            npts = self.npts()
            self._mu_array = self._bkd.full((npts,), float(mu_values))
            self._mu_value = float(mu_values)
        else:
            self._mu_array = mu_values
            self._mu_value = None

    def set_lamda(self, lamda_values: Union[float, Array]) -> None:
        """Set Lame's first parameter (scalar or per-point array).

        Parameters
        ----------
        lamda_values : float or Array
            Lame's first parameter values. Must be non-negative.
        """
        lam_arr = self._bkd.ravel(self._bkd.asarray(lamda_values))
        min_val = self._bkd.to_float(self._bkd.min(lam_arr))
        if min_val < 0.0:
            raise ValueError(f"lamda must be non-negative; found min {min_val:.2e}")
        if isinstance(lamda_values, (int, float)):
            npts = self.npts()
            self._lambda_array = self._bkd.full((npts,), float(lamda_values))
            self._lambda_value = float(lamda_values)
        else:
            self._lambda_array = lamda_values
            self._lambda_value = None

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

        State is component-major: [u_0, ..., u_{n-1}, v_0, ..., v_{n-1}]
        in 2D, with a trailing w block in 3D.

        Parameters
        ----------
        state : Array
            Full state vector. Shape: (ndim*npts,)

        Returns
        -------
        Tuple[Array, ...]
            One component per spatial dimension, each shape (npts,)
        """
        npts = self.npts()
        return tuple(
            state[i * npts : (i + 1) * npts] for i in range(self._ndim)
        )

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        The residual is: div(σ) + forcing

        where σ = λ*tr(ε)*I + 2μ*ε

        Parameters
        ----------
        state : Array
            Solution state [u, v]. Shape: (2*npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual with one block per component. Shape: (ndim*npts,)
        """
        if self._ndim == 3:
            return self._residual_3d(state, time)
        bkd = self._bkd
        npts = self.npts()

        u, v = self._extract_components(state)

        # Compute strain components
        ux = self._Dx @ u  # du/dx
        uy = self._Dy @ u  # du/dy
        vx = self._Dx @ v  # dv/dx
        vy = self._Dy @ v  # dv/dy

        exx = ux
        exy = 0.5 * (uy + vx)
        eyy = vy

        # Trace of strain
        trace_e = exx + eyy

        # Stress tensor components: σ = λ*tr(ε)*I + 2μ*ε
        two_mu = 2.0 * self._mu_array
        sigma_xx = self._lambda_array * trace_e + two_mu * exx
        sigma_xy = two_mu * exy
        sigma_yy = self._lambda_array * trace_e + two_mu * eyy

        # Compute divergence of stress
        # div(σ)_x = ∂σ_xx/∂x + ∂σ_xy/∂y
        # div(σ)_y = ∂σ_xy/∂x + ∂σ_yy/∂y
        div_sigma_x = self._Dx @ sigma_xx + self._Dy @ sigma_xy
        div_sigma_y = self._Dx @ sigma_xy + self._Dy @ sigma_yy

        # Get forcing
        forcing = self._get_forcing(time)
        fx = forcing[:npts]
        fy = forcing[npts:]

        # Residual: div(σ) + f
        res_u = div_sigma_x + fx
        res_v = div_sigma_y + fy

        return bkd.concatenate([res_u, res_v])

    def _strains_3d(self, state: Array) -> List[List[Array]]:
        """Symmetric strain tensor fields eps[i][j] at all mesh points.

        Returns the full 3x3 nested list with eps[i][j] == eps[j][i],
        each entry shape (npts,).
        """
        comps = self._extract_components(state)
        grads = [
            [self._D[j] @ comps[i] for j in range(3)] for i in range(3)
        ]
        return [
            [
                grads[i][j]
                if i == j
                else 0.5 * (grads[i][j] + grads[j][i])
                for j in range(3)
            ]
            for i in range(3)
        ]

    def _residual_3d(self, state: Array, time: float) -> Array:
        """3D residual div(σ) + f with σ = λ*tr(ε)*I + 2μ*ε."""
        bkd = self._bkd
        npts = self.npts()
        eps = self._strains_3d(state)
        trace_e = eps[0][0] + eps[1][1] + eps[2][2]
        lam = self._lambda_array
        two_mu = 2.0 * self._mu_array
        forcing = self._get_forcing(time)
        residuals = []
        for i in range(3):
            div_sigma_i = bkd.zeros((npts,))
            for j in range(3):
                sigma_ij = two_mu * eps[i][j]
                if i == j:
                    sigma_ij = sigma_ij + lam * trace_e
                div_sigma_i = div_sigma_i + self._D[j] @ sigma_ij
            residuals.append(
                div_sigma_i + forcing[i * npts : (i + 1) * npts]
            )
        return bkd.concatenate(residuals)

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/d(u,v).

        For linear elasticity with constant Lamé parameters, the Jacobian
        is constant (independent of state).

        The Jacobian has block structure:
            J = [[J_uu, J_uv],
                 [J_vu, J_vv]]

        where:
            J_uu = (λ + 2μ)*D²_x + μ*D²_y
            J_uv = (λ + μ)*D_x @ D_y
            J_vu = (λ + μ)*D_y @ D_x
            J_vv = μ*D²_x + (λ + 2μ)*D²_y

        Parameters
        ----------
        state : Array
            Solution state. Shape: (2*npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (ndim*npts, ndim*npts)
        """
        if self._ndim == 3:
            return self._jacobian_3d(state, time)
        bkd = self._bkd
        self.npts()

        # For constant Lamé parameters, use scalar values
        if self._lambda_value is not None and self._mu_value is not None:
            lam = self._lambda_value
            mu = self._mu_value

            # Block Jacobians derived from residual:
            #   res_u = Dx @ sigma_xx + Dy @ sigma_xy + fx
            #   res_v = Dx @ sigma_xy + Dy @ sigma_yy + fy
            #
            # J_uu: d(res_u)/du = (λ+2μ)*Dxx + μ*Dyy
            J_uu = (lam + 2.0 * mu) * self._Dxx + mu * self._Dyy

            # J_uv: d(res_u)/dv = λ*(Dx@Dy) + μ*(Dy@Dx)
            # Note: Dx@Dy != Dy@Dx on curvilinear domains
            J_uv = lam * self._DxDy + mu * self._DyDx

            # J_vu: d(res_v)/du = μ*(Dx@Dy) + λ*(Dy@Dx)
            J_vu = mu * self._DxDy + lam * self._DyDx

            # J_vv: d(res_v)/dv = μ*Dxx + (λ+2μ)*Dyy
            J_vv = mu * self._Dxx + (lam + 2.0 * mu) * self._Dyy
        else:
            # Variable Lamé parameters: use Dx @ diag(coeff) @ Dx form
            # which automatically handles the product rule
            # d/dx(coeff * du/dx) = coeff * d²u/dx² + dcoeff/dx * du/dx
            lam_arr = self._lambda_array
            mu_arr = self._mu_array
            Dx, Dy = self._Dx, self._Dy

            diag_lam_2mu = bkd.diag(lam_arr + 2.0 * mu_arr)
            diag_lam = bkd.diag(lam_arr)
            diag_mu = bkd.diag(mu_arr)

            J_uu = Dx @ diag_lam_2mu @ Dx + Dy @ diag_mu @ Dy
            J_uv = Dx @ diag_lam @ Dy + Dy @ diag_mu @ Dx
            J_vu = Dx @ diag_mu @ Dy + Dy @ diag_lam @ Dx
            J_vv = Dx @ diag_mu @ Dx + Dy @ diag_lam_2mu @ Dy

        # Assemble block Jacobian
        # [[J_uu, J_uv], [J_vu, J_vv]]
        top_row = bkd.concatenate([J_uu, J_uv], axis=1)
        bottom_row = bkd.concatenate([J_vu, J_vv], axis=1)
        jacobian = bkd.concatenate([top_row, bottom_row], axis=0)

        return jacobian

    def _jacobian_3d(self, state: Array, time: float) -> Array:
        """3D block Jacobian in variable-Lamé D@diag@D form.

        Derived from R_i = sum_j D_j @ sigma_ij (divergence operator
        always the LEFT factor, so the ordering is correct on
        curvilinear meshes where D_i@D_j != D_j@D_i):

            J[i][i] = D_i diag(λ+2μ) D_i + sum_{k≠i} D_k diag(μ) D_k
            J[i][j] = D_i diag(λ) D_j + D_j diag(μ) D_i   (i ≠ j)

        The Lamé arrays are always populated (scalars are expanded at
        construction), so one form serves constant and field Lamé.
        """
        bkd = self._bkd
        diag_lam = bkd.diag(self._lambda_array)
        diag_mu = bkd.diag(self._mu_array)
        diag_lam_2mu = bkd.diag(self._lambda_array + 2.0 * self._mu_array)
        rows = []
        for i in range(3):
            blocks = []
            for j in range(3):
                if i == j:
                    block = self._D[i] @ diag_lam_2mu @ self._D[i]
                    for k in range(3):
                        if k != i:
                            block = (
                                block + self._D[k] @ diag_mu @ self._D[k]
                            )
                else:
                    block = (
                        self._D[i] @ diag_lam @ self._D[j]
                        + self._D[j] @ diag_mu @ self._D[i]
                    )
                blocks.append(block)
            rows.append(bkd.concatenate(blocks, axis=1))
        return bkd.concatenate(rows, axis=0)

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        """Compute lambda^T (d^2R/du^2) w = 0 (residual linear in state).

        Linear elasticity's residual is linear in the displacement, so
        the contraction is exactly zero.
        """
        return self._bkd.zeros((self.nstates(),))

    # -- full-matrix field-derivative assemblies (engine slots)

    def _strains(self, state: Array) -> Tuple[Array, Array, Array]:
        """Strain fields (exx, exy, eyy) at all mesh points."""
        u, v = self._extract_components(state)
        exx = self._Dx @ u
        exy = 0.5 * ((self._Dy @ u) + (self._Dx @ v))
        eyy = self._Dy @ v
        return exx, exy, eyy

    def residual_mu_jacobian(self, state: Array) -> Array:
        """Assemble :math:`S_\\mu(u) = \\partial R/\\partial
        \\mu_{\\text{field}}`.

        With :math:`\\partial\\sigma/\\partial\\mu = 2\\varepsilon`,

        .. math::

            S_\\mu = \\begin{bmatrix}
            D_x \\mathrm{diag}(2\\varepsilon_{xx})
            + D_y \\mathrm{diag}(2\\varepsilon_{xy}) \\\\
            D_x \\mathrm{diag}(2\\varepsilon_{xy})
            + D_y \\mathrm{diag}(2\\varepsilon_{yy})
            \\end{bmatrix}

        Parameters
        ----------
        state : Array
            Displacement state [u, v]. Shape: (2*npts,)

        Returns
        -------
        Array
            Assembly. Shape: (ndim*npts, npts)
        """
        bkd = self._bkd
        if self._ndim == 3:
            eps = self._strains_3d(state)
            row_blocks = []
            for i in range(3):
                block = self._D[0] @ bkd.diag(2.0 * eps[i][0])
                for j in range(1, 3):
                    block = block + self._D[j] @ bkd.diag(2.0 * eps[i][j])
                row_blocks.append(block)
            return bkd.concatenate(row_blocks, axis=0)
        exx, exy, eyy = self._strains(state)
        top = self._Dx @ bkd.diag(2.0 * exx) + self._Dy @ bkd.diag(
            2.0 * exy
        )
        bottom = self._Dx @ bkd.diag(2.0 * exy) + self._Dy @ bkd.diag(
            2.0 * eyy
        )
        return bkd.concatenate([top, bottom], axis=0)

    def residual_lamda_jacobian(self, state: Array) -> Array:
        """Assemble :math:`S_\\lambda(u) = \\partial R/\\partial
        \\lambda_{\\text{field}}`.

        With :math:`\\partial\\sigma/\\partial\\lambda =
        \\mathrm{tr}(\\varepsilon) I`,

        .. math::

            S_\\lambda = \\begin{bmatrix}
            D_x \\mathrm{diag}(\\mathrm{tr}\\,\\varepsilon) \\\\
            D_y \\mathrm{diag}(\\mathrm{tr}\\,\\varepsilon)
            \\end{bmatrix}

        Parameters
        ----------
        state : Array
            Displacement state [u, v]. Shape: (2*npts,)

        Returns
        -------
        Array
            Assembly. Shape: (ndim*npts, npts)
        """
        bkd = self._bkd
        if self._ndim == 3:
            eps = self._strains_3d(state)
            diag_trace = bkd.diag(eps[0][0] + eps[1][1] + eps[2][2])
            return bkd.concatenate(
                [self._D[i] @ diag_trace for i in range(3)], axis=0
            )
        exx, _, eyy = self._strains(state)
        diag_trace = bkd.diag(exx + eyy)
        return bkd.concatenate(
            [self._Dx @ diag_trace, self._Dy @ diag_trace], axis=0
        )

    def residual_mu_state_jacobian(
        self, delta: Array, state: Array
    ) -> Array:
        """Assemble the mixed operator :math:`A_\\mu(\\delta) =
        \\partial/\\partial(u,v)\\,[S_\\mu(u,v)\\,\\delta]`.

        The stress is bilinear in :math:`(\\varepsilon, \\mu)`, so the
        assembly is state-independent with blocks (:math:`\\delta` the
        diagonal of the field direction):

        .. math::

            A_\\mu = \\begin{bmatrix}
            D_x 2\\delta D_x + D_y \\delta D_y & D_y \\delta D_x \\\\
            D_x \\delta D_y & D_x \\delta D_x + D_y 2\\delta D_y
            \\end{bmatrix}

        Parameters
        ----------
        delta : Array
            Mu-field direction. Shape: (npts,)
        state : Array
            Displacement state (unused; kept for the engine's
            mixed-assembly signature).

        Returns
        -------
        Array
            Assembly. Shape: (ndim*npts, ndim*npts)
        """
        bkd = self._bkd
        if self._ndim == 3:
            diag_d = bkd.diag(delta)
            diag_2d = bkd.diag(2.0 * delta)
            rows = []
            for i in range(3):
                blocks = []
                for j in range(3):
                    if i == j:
                        block = self._D[i] @ diag_2d @ self._D[i]
                        for k in range(3):
                            if k != i:
                                block = (
                                    block
                                    + self._D[k] @ diag_d @ self._D[k]
                                )
                    else:
                        block = self._D[j] @ diag_d @ self._D[i]
                    blocks.append(block)
                rows.append(bkd.concatenate(blocks, axis=1))
            return bkd.concatenate(rows, axis=0)
        Dx, Dy = self._Dx, self._Dy
        diag_d = bkd.diag(delta)
        diag_2d = bkd.diag(2.0 * delta)
        a_uu = Dx @ diag_2d @ Dx + Dy @ diag_d @ Dy
        a_uv = Dy @ diag_d @ Dx
        a_vu = Dx @ diag_d @ Dy
        a_vv = Dx @ diag_d @ Dx + Dy @ diag_2d @ Dy
        top = bkd.concatenate([a_uu, a_uv], axis=1)
        bottom = bkd.concatenate([a_vu, a_vv], axis=1)
        return bkd.concatenate([top, bottom], axis=0)

    def residual_lamda_state_jacobian(
        self, delta: Array, state: Array
    ) -> Array:
        """Assemble the mixed operator :math:`A_\\lambda(\\delta) =
        \\partial/\\partial(u,v)\\,[S_\\lambda(u,v)\\,\\delta]`.

        .. math::

            A_\\lambda = \\begin{bmatrix}
            D_x \\delta D_x & D_x \\delta D_y \\\\
            D_y \\delta D_x & D_y \\delta D_y
            \\end{bmatrix}

        Parameters
        ----------
        delta : Array
            Lambda-field direction. Shape: (npts,)
        state : Array
            Displacement state (unused; kept for the engine's
            mixed-assembly signature).

        Returns
        -------
        Array
            Assembly. Shape: (ndim*npts, ndim*npts)
        """
        bkd = self._bkd
        if self._ndim == 3:
            diag_d = bkd.diag(delta)
            rows = [
                bkd.concatenate(
                    [
                        self._D[i] @ diag_d @ self._D[j]
                        for j in range(3)
                    ],
                    axis=1,
                )
                for i in range(3)
            ]
            return bkd.concatenate(rows, axis=0)
        Dx, Dy = self._Dx, self._Dy
        diag_d = bkd.diag(delta)
        top = bkd.concatenate(
            [Dx @ diag_d @ Dx, Dx @ diag_d @ Dy], axis=1
        )
        bottom = bkd.concatenate(
            [Dy @ diag_d @ Dx, Dy @ diag_d @ Dy], axis=1
        )
        return bkd.concatenate([top, bottom], axis=0)

    def boundary_traction_lame_jacobian(
        self,
        state: Array,
        time: float,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """Assemble :math:`\\partial t/\\partial [\\mu; \\lambda]` at
        the boundary rows of one traction BC.

        Consumer convention (the BC supplies its own row indices and
        normals): ``bc_indices`` are the BC's replaced STATE rows —
        one traction component per BC (``mesh_idx + comp * npts``).
        The traction at mesh point :math:`i` depends on the Lame
        fields only through their local values:

        .. math::

            \\partial t_x/\\partial\\mu_i
            = 2\\varepsilon_{xx} n_x + 2\\varepsilon_{xy} n_y,
            \\quad
            \\partial t_x/\\partial\\lambda_i
            = \\mathrm{tr}(\\varepsilon)\\, n_x

        (similarly for :math:`t_y`), so each row pairs one nonzero in
        the mu block with one in the lambda block.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (ndim*npts,)
        time : float
            Current time (unused; kept for signature uniformity).
        bc_indices : Array
            Replaced state-row indices of the BC. Shape: (n_bc,)
        normals : Array
            Outward unit normals. Shape: (n_bc, ndim)

        Returns
        -------
        Array
            Assembly over the stacked [mu; lambda] field.
            Shape: (n_bc, 2*npts)
        """
        bkd = self._bkd
        npts = self.npts()
        nbnd = bc_indices.shape[0]
        comp = bkd.to_int(bc_indices[0]) // npts
        mesh_idx = bc_indices - comp * npts
        if self._ndim == 3:
            eps = self._strains_3d(state)
            trace_b = (eps[0][0] + eps[1][1] + eps[2][2])[mesh_idx]
            dt_dmu = 2.0 * eps[comp][0][mesh_idx] * normals[:, 0]
            for j in range(1, 3):
                dt_dmu = (
                    dt_dmu + 2.0 * eps[comp][j][mesh_idx] * normals[:, j]
                )
            dt_dlam = trace_b * normals[:, comp]
            result = bkd.copy(bkd.zeros((nbnd, 2 * npts)))
            for i in range(nbnd):
                idx = bkd.to_int(mesh_idx[i])
                result[i, idx] = dt_dmu[i]
                result[i, npts + idx] = dt_dlam[i]
            return result
        exx, exy, eyy = self._strains(state)
        exx_b = exx[mesh_idx]
        exy_b = exy[mesh_idx]
        eyy_b = eyy[mesh_idx]
        trace_b = exx_b + eyy_b
        nx = normals[:, 0]
        ny = normals[:, 1]
        if comp == 0:
            dt_dmu = 2.0 * exx_b * nx + 2.0 * exy_b * ny
            dt_dlam = trace_b * nx
        else:
            dt_dmu = 2.0 * exy_b * nx + 2.0 * eyy_b * ny
            dt_dlam = trace_b * ny

        result = bkd.copy(bkd.zeros((nbnd, 2 * npts)))
        for i in range(nbnd):
            idx = bkd.to_int(mesh_idx[i])
            result[i, idx] = dt_dmu[i]
            result[i, npts + idx] = dt_dlam[i]
        return result

    def compute_interface_flux(
        self, state: Array, boundary_indices: Array, normal: Array
    ) -> Array:
        """Compute traction at boundary for DtN domain decomposition.

        Computes t = σ · n at the specified boundary points, where:
        - σ is the stress tensor
        - n is the outward unit normal

        Parameters
        ----------
        state : Array
            Solution state. Shape: (ndim*npts,)
        boundary_indices : Array
            Mesh indices at interface. Shape: (nboundary,)
        normal : Array
            Outward unit normal. Shape: (ndim,)

        Returns
        -------
        Array
            Traction components at boundary points.
            Shape: (ndim*nboundary,) with component-stacked ordering.
        """
        bkd = self._bkd
        if self._ndim == 3:
            eps = self._strains_3d(state)
            trace_b = (eps[0][0] + eps[1][1] + eps[2][2])[
                boundary_indices
            ]
            lam_b = self._lambda_array[boundary_indices]
            mu_b = self._mu_array[boundary_indices]
            tractions = []
            for i in range(3):
                t_i = (
                    lam_b * trace_b
                    + 2.0 * mu_b * eps[i][i][boundary_indices]
                ) * bkd.to_float(normal[i])
                for j in range(3):
                    if j != i:
                        t_i = t_i + (
                            2.0 * mu_b * eps[i][j][boundary_indices]
                        ) * bkd.to_float(normal[j])
                tractions.append(t_i)
            return bkd.concatenate(tractions)

        u, v = self._extract_components(state)

        # Compute strain components at boundary
        ux = (self._Dx @ u)[boundary_indices]
        uy = (self._Dy @ u)[boundary_indices]
        vx = (self._Dx @ v)[boundary_indices]
        vy = (self._Dy @ v)[boundary_indices]

        exx = ux
        exy = 0.5 * (uy + vx)
        eyy = vy
        trace_e = exx + eyy

        # Stress tensor at boundary
        lam = self._lambda_array[boundary_indices]
        mu = self._mu_array[boundary_indices]
        sigma_xx = lam * trace_e + 2.0 * mu * exx
        sigma_xy = 2.0 * mu * exy
        sigma_yy = lam * trace_e + 2.0 * mu * eyy

        # Traction t = σ · n
        nx = bkd.to_float(normal[0])
        ny = bkd.to_float(normal[1])
        traction_x = sigma_xx * nx + sigma_xy * ny
        traction_y = sigma_xy * nx + sigma_yy * ny

        # Component-stacked ordering: [t_x_0, ..., t_x_n, t_y_0, ..., t_y_n]
        return bkd.concatenate([traction_x, traction_y])


def create_linear_elasticity(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    lamda: Union[float, Array],
    mu: Union[float, Array],
    forcing: Optional[Callable[[float], Array]] = None,
) -> LinearElasticityPhysics[Array]:
    """Create linear elasticity physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D or 3D collocation basis.
    bkd : Backend
        Computational backend.
    lamda : float or Array
        Lamé's first parameter λ.
    mu : float or Array
        Shear modulus μ.
    forcing : Callable or Array, optional
        Source term.

    Returns
    -------
    LinearElasticityPhysics
        Linear elasticity physics instance.
    """
    return LinearElasticityPhysics(basis, bkd, lamda, mu, forcing)

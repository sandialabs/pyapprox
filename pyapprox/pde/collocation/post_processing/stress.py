"""Stress post-processing for 2D elasticity collocation solutions.

Provides ``StressPostProcessor2D`` for linear elasticity and
``HyperelasticStressPostProcessor2D`` for hyperelastic (Neo-Hookean)
materials. Both compute Cartesian stresses, hoop stress on polar domains,
strain energy density, and their state Jacobians.
"""

from typing import Callable, Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class StressPostProcessor2D(Generic[Array]):
    """Compute stress quantities from 2D linear elastic displacement fields.

    Given a displacement state vector ``[u_0,...,u_{n-1}, v_0,...,v_{n-1}]``
    (2*npts DOFs), computes Cartesian stresses, hoop stress on polar
    domains, and strain energy density.

    Parameters
    ----------
    Dx : Array, shape (npts, npts)
        Physical derivative matrix d/dx.
    Dy : Array, shape (npts, npts)
        Physical derivative matrix d/dy.
    get_lamda : callable
        Returns current Lame first parameter array, shape (npts,).
    get_mu : callable
        Returns current shear modulus array, shape (npts,).
    bkd : Backend
        Computational backend.
    curvilinear_basis : Array, optional
        Unit curvilinear basis vectors from ``PolarTransform``.
        Shape (npts, 2, 2). Required for ``hoop_stress``.
    """

    def __init__(
        self,
        Dx: Array,
        Dy: Array,
        get_lamda: Callable[[], Array],
        get_mu: Callable[[], Array],
        bkd: Backend[Array],
        curvilinear_basis: Optional[Array] = None,
    ) -> None:
        self._Dx = Dx
        self._Dy = Dy
        self._get_lamda = get_lamda
        self._get_mu = get_mu
        self._bkd = bkd
        self._npts = Dx.shape[0]
        self._curvilinear_basis = curvilinear_basis

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def npts(self) -> int:
        return self._npts

    def cartesian_stress(
        self, state: Array,
    ) -> Tuple[Array, Array, Array]:
        """Compute Cartesian stress components at all mesh points.

        Parameters
        ----------
        state : Array, shape (2*npts,)
            Displacement field [u, v].

        Returns
        -------
        sigma_xx, sigma_xy, sigma_yy : tuple of Array
            Each has shape (npts,).
        """
        npts = self._npts
        u = state[:npts]
        v = state[npts:]
        lamda = self._get_lamda()
        mu = self._get_mu()

        ux = self._Dx @ u
        uy = self._Dy @ u
        vx = self._Dx @ v
        vy = self._Dy @ v

        exx = ux
        exy = 0.5 * (uy + vx)
        eyy = vy
        trace_e = exx + eyy

        two_mu = 2.0 * mu
        sigma_xx = lamda * trace_e + two_mu * exx
        sigma_xy = two_mu * exy
        sigma_yy = lamda * trace_e + two_mu * eyy
        return sigma_xx, sigma_xy, sigma_yy

    def hoop_stress(self, state: Array) -> Array:
        """Compute circumferential (hoop) stress sigma_theta_theta.

        Requires ``curvilinear_basis`` to have been provided at construction.

        sigma_tt = e_theta^T . sigma . e_theta
                 = et_x^2 * sxx + 2 * et_x * et_y * sxy + et_y^2 * syy

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        Array, shape (npts,)
            Hoop stress at each mesh point.
        """
        if self._curvilinear_basis is None:
            raise ValueError(
                "curvilinear_basis required for hoop_stress computation"
            )
        sigma_xx, sigma_xy, sigma_yy = self.cartesian_stress(state)
        et_x = self._curvilinear_basis[:, 0, 1]  # -sin(theta)
        et_y = self._curvilinear_basis[:, 1, 1]  # cos(theta)

        return (
            et_x ** 2 * sigma_xx
            + 2.0 * et_x * et_y * sigma_xy
            + et_y ** 2 * sigma_yy
        )

    def radial_stress(self, state: Array) -> Array:
        """Compute radial stress sigma_rr.

        sigma_rr = e_r^T . sigma . e_r
                 = er_x^2 * sxx + 2 * er_x * er_y * sxy + er_y^2 * syy

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        Array, shape (npts,)
        """
        if self._curvilinear_basis is None:
            raise ValueError(
                "curvilinear_basis required for radial_stress computation"
            )
        sigma_xx, sigma_xy, sigma_yy = self.cartesian_stress(state)
        er_x = self._curvilinear_basis[:, 0, 0]  # cos(theta)
        er_y = self._curvilinear_basis[:, 1, 0]  # sin(theta)

        return (
            er_x ** 2 * sigma_xx
            + 2.0 * er_x * er_y * sigma_xy
            + er_y ** 2 * sigma_yy
        )

    def strain_energy_density(self, state: Array) -> Array:
        """Compute strain energy density psi = 0.5 * sigma : epsilon.

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        Array, shape (npts,)
        """
        npts = self._npts
        u = state[:npts]
        v = state[npts:]
        sigma_xx, sigma_xy, sigma_yy = self.cartesian_stress(state)

        exx = self._Dx @ u
        exy = 0.5 * (self._Dy @ u + self._Dx @ v)
        eyy = self._Dy @ v

        return 0.5 * (
            sigma_xx * exx + 2.0 * sigma_xy * exy + sigma_yy * eyy
        )

    def cartesian_stress_state_jacobian(
        self,
    ) -> Tuple[Array, Array, Array]:
        """Compute Jacobians d(sigma_ij)/d(state) for all stress components.

        For linear elasticity, the stress-state relationship is linear, so
        these Jacobians are state-independent.

        Returns
        -------
        dsxx, dsxy, dsyy : tuple of Array
            Each has shape (npts, 2*npts).
        """
        bkd = self._bkd
        npts = self._npts
        lamda = self._get_lamda()
        mu = self._get_mu()
        Dx, Dy = self._Dx, self._Dy

        diag_lam_2mu = bkd.diag(lamda + 2.0 * mu)
        diag_lam = bkd.diag(lamda)
        diag_mu = bkd.diag(mu)

        # dsxx/d[u, v] = [(lam+2mu)*Dx, lam*Dy]
        dsxx = bkd.concatenate(
            [diag_lam_2mu @ Dx, diag_lam @ Dy], axis=1,
        )
        # dsxy/d[u, v] = [mu*Dy, mu*Dx]
        dsxy = bkd.concatenate(
            [diag_mu @ Dy, diag_mu @ Dx], axis=1,
        )
        # dsyy/d[u, v] = [lam*Dx, (lam+2mu)*Dy]
        dsyy = bkd.concatenate(
            [diag_lam @ Dx, diag_lam_2mu @ Dy], axis=1,
        )
        return dsxx, dsxy, dsyy

    def hoop_stress_state_jacobian(self) -> Array:
        """Compute d(sigma_tt)/d(state) at all mesh points.

        State-independent for linear elasticity.

        Returns
        -------
        Array, shape (npts, 2*npts)
        """
        if self._curvilinear_basis is None:
            raise ValueError(
                "curvilinear_basis required for hoop_stress_state_jacobian"
            )
        bkd = self._bkd
        dsxx, dsxy, dsyy = self.cartesian_stress_state_jacobian()
        et_x = self._curvilinear_basis[:, 0, 1]
        et_y = self._curvilinear_basis[:, 1, 1]

        return (
            bkd.diag(et_x ** 2) @ dsxx
            + bkd.diag(2.0 * et_x * et_y) @ dsxy
            + bkd.diag(et_y ** 2) @ dsyy
        )

    def strain_energy_density_state_jacobian(self, state: Array) -> Array:
        """Compute d(psi)/d(state) at all mesh points.

        For linear elasticity, psi = 0.5 * sigma : epsilon is quadratic
        in state, so the Jacobian depends on state:
            d(psi)/d(state) = sigma : d(epsilon)/d(state)

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        Array, shape (npts, 2*npts)
        """
        bkd = self._bkd
        sigma_xx, sigma_xy, sigma_yy = self.cartesian_stress(state)
        Dx, Dy = self._Dx, self._Dy

        # d(psi)/d(u) = sigma_xx * Dx + sigma_xy * Dy
        # d(psi)/d(v) = sigma_xy * Dx + sigma_yy * Dy
        dpsi_du = bkd.diag(sigma_xx) @ Dx + bkd.diag(sigma_xy) @ Dy
        dpsi_dv = bkd.diag(sigma_xy) @ Dx + bkd.diag(sigma_yy) @ Dy

        return bkd.concatenate([dpsi_du, dpsi_dv], axis=1)


class HyperelasticStressPostProcessor2D(Generic[Array]):
    """Compute stress quantities from 2D hyperelastic displacement fields.

    Computes Cauchy stress from the first Piola-Kirchhoff stress:
        sigma = (1/J) * P * F^T

    where F = I + grad(u) is the deformation gradient, P is the PK1 stress
    from the constitutive model, and J = det(F).

    Parameters
    ----------
    Dx : Array, shape (npts, npts)
        Physical derivative matrix d/dx.
    Dy : Array, shape (npts, npts)
        Physical derivative matrix d/dy.
    stress_model
        Hyperelastic stress model providing ``compute_stress_2d``,
        ``compute_tangent_2d``, and material parameters ``_mu``, ``_lamda``.
    bkd : Backend
        Computational backend.
    curvilinear_basis : Array, optional
        Unit curvilinear basis vectors from ``PolarTransform``.
        Shape (npts, 2, 2). Required for ``hoop_stress``.
    """

    def __init__(
        self,
        Dx: Array,
        Dy: Array,
        stress_model,
        bkd: Backend[Array],
        curvilinear_basis: Optional[Array] = None,
    ) -> None:
        self._Dx = Dx
        self._Dy = Dy
        self._stress_model = stress_model
        self._bkd = bkd
        self._npts = Dx.shape[0]
        self._curvilinear_basis = curvilinear_basis

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def npts(self) -> int:
        return self._npts

    def _compute_F(
        self, state: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        """Compute deformation gradient components.

        Returns
        -------
        F11, F12, F21, F22 : tuple of Array
            Each has shape (npts,).
        """
        npts = self._npts
        u = state[:npts]
        v = state[npts:]
        F11 = 1.0 + self._Dx @ u
        F12 = self._Dy @ u
        F21 = self._Dx @ v
        F22 = 1.0 + self._Dy @ v
        return F11, F12, F21, F22

    def cartesian_stress(
        self, state: Array,
    ) -> Tuple[Array, Array, Array]:
        """Compute Cauchy stress sigma = (1/J)*P*F^T at all mesh points.

        Parameters
        ----------
        state : Array, shape (2*npts,)
            Displacement field [u, v].

        Returns
        -------
        sigma_xx, sigma_xy, sigma_yy : tuple of Array
            Each has shape (npts,).
        """
        bkd = self._bkd
        F11, F12, F21, F22 = self._compute_F(state)
        P11, P12, P21, P22 = self._stress_model.compute_stress_2d(
            F11, F12, F21, F22, bkd,
        )
        J = F11 * F22 - F12 * F21
        inv_J = 1.0 / J

        # sigma_ij = (1/J) * sum_K P_{iK} * F_{jK}
        sigma_xx = inv_J * (P11 * F11 + P12 * F12)
        sigma_xy = inv_J * (P11 * F21 + P12 * F22)
        sigma_yy = inv_J * (P21 * F21 + P22 * F22)
        return sigma_xx, sigma_xy, sigma_yy

    def hoop_stress(self, state: Array) -> Array:
        """Compute circumferential (hoop) stress sigma_theta_theta.

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        Array, shape (npts,)
        """
        if self._curvilinear_basis is None:
            raise ValueError(
                "curvilinear_basis required for hoop_stress computation"
            )
        sigma_xx, sigma_xy, sigma_yy = self.cartesian_stress(state)
        et_x = self._curvilinear_basis[:, 0, 1]
        et_y = self._curvilinear_basis[:, 1, 1]
        return (
            et_x ** 2 * sigma_xx
            + 2.0 * et_x * et_y * sigma_xy
            + et_y ** 2 * sigma_yy
        )

    def radial_stress(self, state: Array) -> Array:
        """Compute radial stress sigma_rr.

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        Array, shape (npts,)
        """
        if self._curvilinear_basis is None:
            raise ValueError(
                "curvilinear_basis required for radial_stress computation"
            )
        sigma_xx, sigma_xy, sigma_yy = self.cartesian_stress(state)
        er_x = self._curvilinear_basis[:, 0, 0]
        er_y = self._curvilinear_basis[:, 1, 0]
        return (
            er_x ** 2 * sigma_xx
            + 2.0 * er_x * er_y * sigma_xy
            + er_y ** 2 * sigma_yy
        )

    def strain_energy_density(self, state: Array) -> Array:
        """Compute Neo-Hookean stored energy density (plane strain).

        W = (mu/2)*(I1 - 3) - mu*ln(J) + (lambda/2)*(ln(J))^2

        where I1 = tr(F^T F) includes the out-of-plane F33=1 for plane
        strain, and J = det(F) (2D determinant, since F33=1 does not
        change the 3D determinant under plane strain).

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        Array, shape (npts,)
        """
        bkd = self._bkd
        F11, F12, F21, F22 = self._compute_F(state)
        J = F11 * F22 - F12 * F21
        ln_J = bkd.log(J)
        # Plane strain: F33=1, so I1 = F11^2+F12^2+F21^2+F22^2 + F33^2
        I1 = F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2 + 1.0
        mu = self._stress_model._mu
        lamda = self._stress_model._lamda
        return (
            0.5 * mu * (I1 - 3.0)
            - mu * ln_J
            + 0.5 * lamda * ln_J ** 2
        )

    def strain_energy_density_state_jacobian(
        self, state: Array,
    ) -> Array:
        """Compute d(W)/d(state) at all mesh points.

        Uses dW/dF = P (PK1 stress), so:
            dW/d(u) = diag(P11) @ Dx + diag(P12) @ Dy
            dW/d(v) = diag(P21) @ Dx + diag(P22) @ Dy

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        Array, shape (npts, 2*npts)
        """
        bkd = self._bkd
        F11, F12, F21, F22 = self._compute_F(state)
        P11, P12, P21, P22 = self._stress_model.compute_stress_2d(
            F11, F12, F21, F22, bkd,
        )
        Dx, Dy = self._Dx, self._Dy
        # dW/d(u) = P11*Dx + P12*Dy (since dF_{1L}/du = D_L)
        dW_du = bkd.diag(P11) @ Dx + bkd.diag(P12) @ Dy
        # dW/d(v) = P21*Dx + P22*Dy (since dF_{2L}/dv = D_L)
        dW_dv = bkd.diag(P21) @ Dx + bkd.diag(P22) @ Dy
        return bkd.concatenate([dW_du, dW_dv], axis=1)

    def cartesian_stress_state_jacobian(
        self, state: Array,
    ) -> Tuple[Array, Array, Array]:
        """Compute Jacobians d(sigma_ij)/d(state) for all stress components.

        State-dependent for hyperelastic materials.

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        dsxx, dsxy, dsyy : tuple of Array
            Each has shape (npts, 2*npts).
        """
        bkd = self._bkd
        npts = self._npts
        Dx, Dy = self._Dx, self._Dy
        D = [Dx, Dy]

        F11, F12, F21, F22 = self._compute_F(state)
        P11, P12, P21, P22 = self._stress_model.compute_stress_2d(
            F11, F12, F21, F22, bkd,
        )
        A = self._stress_model.compute_tangent_2d(
            F11, F12, F21, F22, bkd,
        )

        J = F11 * F22 - F12 * F21
        inv_J = 1.0 / J
        # dJ/d(state): dJ/d(u_m) = F22*Dx[m,:] - F21*Dy[m,:]
        #              dJ/d(v_m) = F11*Dy[m,:] - F12*Dx[m,:]
        dJ_du = bkd.diag(F22) @ Dx - bkd.diag(F21) @ Dy  # (npts, npts)
        dJ_dv = bkd.diag(F11) @ Dy - bkd.diag(F12) @ Dx  # (npts, npts)
        dJ = bkd.concatenate([dJ_du, dJ_dv], axis=1)  # (npts, 2*npts)

        # Build dP_iK/d(state) for each i,K
        # dP_iK/d(state) = [sum_L A_{iK1L}*D_L | sum_L A_{iK2L}*D_L]
        def _dP_block(i, K):
            """Compute dP_{iK}/d(state). Shape: (npts, 2*npts)."""
            du_block = bkd.zeros((npts, npts))
            du_block = bkd.copy(du_block)
            dv_block = bkd.zeros((npts, npts))
            dv_block = bkd.copy(dv_block)
            for L_idx in range(2):
                key_u = f"A_{i}{K}1{L_idx + 1}"
                key_v = f"A_{i}{K}2{L_idx + 1}"
                du_block = du_block + bkd.diag(A[key_u]) @ D[L_idx]
                dv_block = dv_block + bkd.diag(A[key_v]) @ D[L_idx]
            return bkd.concatenate([du_block, dv_block], axis=1)

        # dF_{jK}/d(state)
        # dF_{1,1}/d(state) = [Dx | 0], dF_{1,2}/d(state) = [Dy | 0]
        # dF_{2,1}/d(state) = [0 | Dx], dF_{2,2}/d(state) = [0 | Dy]
        Z = bkd.zeros((npts, npts))
        dF = {
            (1, 1): bkd.concatenate([Dx, Z], axis=1),
            (1, 2): bkd.concatenate([Dy, Z], axis=1),
            (2, 1): bkd.concatenate([Z, Dx], axis=1),
            (2, 2): bkd.concatenate([Z, Dy], axis=1),
        }

        # F values for matrix products
        F_vals = {(1, 1): F11, (1, 2): F12, (2, 1): F21, (2, 2): F22}
        P_vals = {(1, 1): P11, (1, 2): P12, (2, 1): P21, (2, 2): P22}

        # sigma_{ij} = (1/J) * sum_K P_{iK} * F_{jK}
        # d(sigma_{ij})/d(state) = -(1/J^2)*dJ * (sum_K P_{iK}*F_{jK})
        #   + (1/J) * sum_K [dP_{iK} * F_{jK} + P_{iK} * dF_{jK}]

        def _dsigma(i, j):
            """Compute d(sigma_{ij})/d(state). Shape: (npts, 2*npts)."""
            # sum_K P_{iK} * F_{jK}
            PFt_ij = sum(
                P_vals[(i, K)] * F_vals[(j, K)] for K in (1, 2)
            )
            # Term 1: -(1/J^2) * PFt_ij * dJ
            term1 = bkd.diag(-inv_J ** 2 * PFt_ij) @ dJ
            # Term 2: (1/J) * sum_K [dP_{iK} * F_{jK} + P_{iK} * dF_{jK}]
            term2 = bkd.zeros((npts, 2 * npts))
            term2 = bkd.copy(term2)
            for K in (1, 2):
                dP_iK = _dP_block(i, K)
                term2 = (
                    term2
                    + bkd.diag(F_vals[(j, K)]) @ dP_iK
                    + bkd.diag(P_vals[(i, K)]) @ dF[(j, K)]
                )
            term2 = bkd.diag(inv_J) @ term2
            return term1 + term2

        dsxx = _dsigma(1, 1)
        dsxy = _dsigma(1, 2)
        dsyy = _dsigma(2, 2)
        return dsxx, dsxy, dsyy

    def hoop_stress_state_jacobian(self, state: Array) -> Array:
        """Compute d(sigma_tt)/d(state) at all mesh points.

        State-dependent for hyperelastic materials.

        Parameters
        ----------
        state : Array, shape (2*npts,)

        Returns
        -------
        Array, shape (npts, 2*npts)
        """
        if self._curvilinear_basis is None:
            raise ValueError(
                "curvilinear_basis required for hoop_stress_state_jacobian"
            )
        bkd = self._bkd
        dsxx, dsxy, dsyy = self.cartesian_stress_state_jacobian(state)
        et_x = self._curvilinear_basis[:, 0, 1]
        et_y = self._curvilinear_basis[:, 1, 1]
        return (
            bkd.diag(et_x ** 2) @ dsxx
            + bkd.diag(2.0 * et_x * et_y) @ dsxy
            + bkd.diag(et_y ** 2) @ dsyy
        )

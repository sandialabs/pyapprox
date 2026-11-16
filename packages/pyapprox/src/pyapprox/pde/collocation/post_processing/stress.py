"""Stress post-processing for 2D elasticity collocation solutions.

Provides ``StressPostProcessor2D`` for linear elasticity and
``HyperelasticStressPostProcessor2D`` for hyperelastic (Neo-Hookean)
materials. Both compute Cartesian stresses, hoop stress on polar domains,
strain energy density, their state Jacobians, and their pointwise
derivatives with respect to the Lame parameters.

Every quantity is a function of the displacement and of the Lame
parameters ``mu`` and ``lamda`` (each shape (npts,)), passed on each
call: the post-processors hold no material, so a quantity of interest
built on them is a function of the state and the parameters alone.
"""

from typing import Generic, List, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


def _project(
    basis_x: Array, basis_y: Array, sxx: Array, sxy: Array, syy: Array
) -> Array:
    """``e^T sigma e`` for the unit vector ``e = (basis_x, basis_y)`` at
    every node; also applies row-wise to Jacobians when the components
    carry a trailing column axis."""
    return basis_x**2 * sxx + 2.0 * basis_x * basis_y * sxy + basis_y**2 * syy


class StressPostProcessor2D(Generic[Array]):
    """Compute stress quantities from 2D linear elastic displacement fields.

    Given a displacement state vector ``[u_0,...,u_{n-1}, v_0,...,v_{n-1}]``
    (2*npts DOFs), computes Cartesian stresses, hoop stress on polar
    domains, and strain energy density, for the Lame parameters passed
    on each call.

    Parameters
    ----------
    Dx : Array, shape (npts, npts)
        Physical derivative matrix d/dx.
    Dy : Array, shape (npts, npts)
        Physical derivative matrix d/dy.
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
        bkd: Backend[Array],
        curvilinear_basis: Optional[Array] = None,
    ) -> None:
        self._Dx = Dx
        self._Dy = Dy
        self._bkd = bkd
        self._npts = Dx.shape[0]
        self._curvilinear_basis = curvilinear_basis

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def npts(self) -> int:
        return self._npts

    def _strains(self, state: Array) -> Tuple[Array, Array, Array]:
        """``(exx, exy, eyy)`` at every node."""
        npts = self._npts
        u = state[:npts]
        v = state[npts:]
        exx = self._Dx @ u
        exy = 0.5 * (self._Dy @ u + self._Dx @ v)
        eyy = self._Dy @ v
        return exx, exy, eyy

    def _hoop_basis(self) -> Tuple[Array, Array]:
        if self._curvilinear_basis is None:
            raise ValueError("curvilinear_basis required for hoop stress")
        # e_theta = (-sin(theta), cos(theta))
        return self._curvilinear_basis[:, 0, 1], self._curvilinear_basis[:, 1, 1]

    def cartesian_stress(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array, Array]:
        """Compute Cartesian stress components at all mesh points.

        Parameters
        ----------
        state : Array, shape (2*npts,)
            Displacement field [u, v].
        mu, lamda : Array, shape (npts,)
            Lame parameters.

        Returns
        -------
        sigma_xx, sigma_xy, sigma_yy : tuple of Array
            Each has shape (npts,).
        """
        exx, exy, eyy = self._strains(state)
        trace_e = exx + eyy
        two_mu = 2.0 * mu
        sigma_xx = lamda * trace_e + two_mu * exx
        sigma_xy = two_mu * exy
        sigma_yy = lamda * trace_e + two_mu * eyy
        return sigma_xx, sigma_xy, sigma_yy

    def hoop_stress(self, state: Array, mu: Array, lamda: Array) -> Array:
        """Compute circumferential (hoop) stress sigma_theta_theta.

        Requires ``curvilinear_basis`` to have been provided at construction.

        sigma_tt = e_theta^T . sigma . e_theta
                 = et_x^2 * sxx + 2 * et_x * et_y * sxy + et_y^2 * syy

        Returns
        -------
        Array, shape (npts,)
            Hoop stress at each mesh point.
        """
        et_x, et_y = self._hoop_basis()
        return _project(et_x, et_y, *self.cartesian_stress(state, mu, lamda))

    def radial_stress(self, state: Array, mu: Array, lamda: Array) -> Array:
        """Compute radial stress sigma_rr.

        sigma_rr = e_r^T . sigma . e_r
                 = er_x^2 * sxx + 2 * er_x * er_y * sxy + er_y^2 * syy

        Returns
        -------
        Array, shape (npts,)
        """
        if self._curvilinear_basis is None:
            raise ValueError("curvilinear_basis required for radial_stress computation")
        er_x = self._curvilinear_basis[:, 0, 0]  # cos(theta)
        er_y = self._curvilinear_basis[:, 1, 0]  # sin(theta)
        return _project(er_x, er_y, *self.cartesian_stress(state, mu, lamda))

    def strain_energy_density(self, state: Array, mu: Array, lamda: Array) -> Array:
        """Compute strain energy density psi = 0.5 * sigma : epsilon.

        Returns
        -------
        Array, shape (npts,)
        """
        exx, exy, eyy = self._strains(state)
        sigma_xx, sigma_xy, sigma_yy = self.cartesian_stress(state, mu, lamda)
        return 0.5 * (sigma_xx * exx + 2.0 * sigma_xy * exy + sigma_yy * eyy)

    def cartesian_stress_state_jacobian(
        self, mu: Array, lamda: Array
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
        Dx, Dy = self._Dx, self._Dy
        lam_2mu = (lamda + 2.0 * mu)[:, None]
        lam = lamda[:, None]
        mu_col = mu[:, None]
        # dsxx/d[u, v] = [(lam+2mu)*Dx, lam*Dy]
        dsxx = bkd.concatenate([lam_2mu * Dx, lam * Dy], axis=1)
        # dsxy/d[u, v] = [mu*Dy, mu*Dx]
        dsxy = bkd.concatenate([mu_col * Dy, mu_col * Dx], axis=1)
        # dsyy/d[u, v] = [lam*Dx, (lam+2mu)*Dy]
        dsyy = bkd.concatenate([lam * Dx, lam_2mu * Dy], axis=1)
        return dsxx, dsxy, dsyy

    def hoop_stress_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array:
        """Compute d(sigma_tt)/d(state) at all mesh points.

        State-independent for linear elasticity; ``state`` is accepted
        so that linear and hyperelastic post-processors share one
        signature.

        Returns
        -------
        Array, shape (npts, 2*npts)
        """
        et_x, et_y = self._hoop_basis()
        dsxx, dsxy, dsyy = self.cartesian_stress_state_jacobian(mu, lamda)
        return _project(et_x[:, None], et_y[:, None], dsxx, dsxy, dsyy)

    def strain_energy_density_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array:
        """Compute d(psi)/d(state) at all mesh points.

        For linear elasticity, psi = 0.5 * sigma : epsilon is quadratic
        in state, so the Jacobian depends on state:
            d(psi)/d(state) = sigma : d(epsilon)/d(state)

        Returns
        -------
        Array, shape (npts, 2*npts)
        """
        bkd = self._bkd
        sigma_xx, sigma_xy, sigma_yy = self.cartesian_stress(state, mu, lamda)
        Dx, Dy = self._Dx, self._Dy

        # d(psi)/d(u) = sigma_xx * Dx + sigma_xy * Dy
        # d(psi)/d(v) = sigma_xy * Dx + sigma_yy * Dy
        dpsi_du = sigma_xx[:, None] * Dx + sigma_xy[:, None] * Dy
        dpsi_dv = sigma_xy[:, None] * Dx + sigma_yy[:, None] * Dy

        return bkd.concatenate([dpsi_du, dpsi_dv], axis=1)

    def hoop_stress_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]:
        """Pointwise :math:`(\\partial \\sigma_{tt}/\\partial \\mu,
        \\partial \\sigma_{tt}/\\partial \\lambda)`: :math:`2
        \\epsilon_{tt}` and :math:`\\operatorname{tr}\\epsilon`, each
        shape (npts,)."""
        et_x, et_y = self._hoop_basis()
        exx, exy, eyy = self._strains(state)
        trace_e = exx + eyy
        dmu = _project(et_x, et_y, 2.0 * exx, 2.0 * exy, 2.0 * eyy)
        dlamda = _project(et_x, et_y, trace_e, 0.0 * trace_e, trace_e)
        return dmu, dlamda

    def strain_energy_density_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]:
        """Pointwise :math:`(\\partial \\psi/\\partial \\mu, \\partial
        \\psi/\\partial \\lambda) = (\\epsilon:\\epsilon,
        \\tfrac12 (\\operatorname{tr}\\epsilon)^2)`, each shape (npts,)."""
        exx, exy, eyy = self._strains(state)
        trace_e = exx + eyy
        return exx**2 + 2.0 * exy**2 + eyy**2, 0.5 * trace_e**2


@runtime_checkable
class PostProcessingLawProtocol(Protocol, Generic[Array]):
    """What the hyperelastic post-processor needs of the constitutive
    law: the PK1 stress, its tangent and its Lame sensitivities from the
    nested deformation gradient ``F[i][J]`` and the Lame parameters at
    the same points."""

    def stress(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]: ...

    def tangent(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[List[List[Array]]]]: ...

    def stress_mu_sensitivity(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]: ...

    def stress_lamda_sensitivity(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]: ...


class HyperelasticStressPostProcessor2D(Generic[Array]):
    """Compute stress quantities from 2D hyperelastic displacement fields.

    Computes Cauchy stress from the first Piola-Kirchhoff stress:
        sigma = (1/J) * P * F^T

    where F = I + grad(u) is the deformation gradient, P is the PK1 stress
    the law gives for the Lame parameters passed on each call, and
    J = det(F). The strain energy density is the plane-strain neo-Hookean
    one, so the law should be neo-Hookean.

    Parameters
    ----------
    Dx : Array, shape (npts, npts)
        Physical derivative matrix d/dx.
    Dy : Array, shape (npts, npts)
        Physical derivative matrix d/dy.
    law : PostProcessingLawProtocol
        The constitutive law, e.g. ``NeoHookeanLaw``.
    bkd : Backend
        Computational backend.
    curvilinear_basis : Array, optional
        Unit curvilinear basis vectors from ``PolarTransform``.
        Shape (npts, 2, 2). Required for ``hoop_stress``.

    Raises
    ------
    TypeError
        If ``law`` does not satisfy ``PostProcessingLawProtocol``.
    """

    def __init__(
        self,
        Dx: Array,
        Dy: Array,
        law: PostProcessingLawProtocol[Array],
        bkd: Backend[Array],
        curvilinear_basis: Optional[Array] = None,
    ) -> None:
        if not isinstance(law, PostProcessingLawProtocol):
            raise TypeError(
                "law must satisfy PostProcessingLawProtocol, got "
                f"{type(law).__name__}"
            )
        self._Dx = Dx
        self._Dy = Dy
        self._law = law
        self._bkd = bkd
        self._npts = Dx.shape[0]
        self._curvilinear_basis = curvilinear_basis

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def npts(self) -> int:
        return self._npts

    def _compute_F(self, state: Array) -> List[List[Array]]:
        """Deformation gradient ``F[i][J]``, each entry shape (npts,)."""
        npts = self._npts
        u = state[:npts]
        v = state[npts:]
        return [
            [1.0 + self._Dx @ u, self._Dy @ u],
            [self._Dx @ v, 1.0 + self._Dy @ v],
        ]

    def _hoop_basis(self) -> Tuple[Array, Array]:
        if self._curvilinear_basis is None:
            raise ValueError("curvilinear_basis required for hoop stress")
        return self._curvilinear_basis[:, 0, 1], self._curvilinear_basis[:, 1, 1]

    @staticmethod
    def _cauchy(
        P: List[List[Array]], F: List[List[Array]], inv_J: Array
    ) -> Tuple[Array, Array, Array]:
        """sigma_ij = (1/J) * sum_K P_iK * F_jK, as (xx, xy, yy)."""

        def entry(i: int, j: int) -> Array:
            return inv_J * (P[i][0] * F[j][0] + P[i][1] * F[j][1])

        return entry(0, 0), entry(0, 1), entry(1, 1)

    def cartesian_stress(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array, Array]:
        """Compute Cauchy stress sigma = (1/J)*P*F^T at all mesh points.

        Parameters
        ----------
        state : Array, shape (2*npts,)
            Displacement field [u, v].
        mu, lamda : Array, shape (npts,)
            Lame parameters.

        Returns
        -------
        sigma_xx, sigma_xy, sigma_yy : tuple of Array
            Each has shape (npts,).
        """
        F = self._compute_F(state)
        P = self._law.stress(F, mu, lamda)
        J = F[0][0] * F[1][1] - F[0][1] * F[1][0]
        return self._cauchy(P, F, 1.0 / J)

    def hoop_stress(self, state: Array, mu: Array, lamda: Array) -> Array:
        """Compute circumferential (hoop) stress sigma_theta_theta.

        Returns
        -------
        Array, shape (npts,)
        """
        et_x, et_y = self._hoop_basis()
        return _project(et_x, et_y, *self.cartesian_stress(state, mu, lamda))

    def radial_stress(self, state: Array, mu: Array, lamda: Array) -> Array:
        """Compute radial stress sigma_rr.

        Returns
        -------
        Array, shape (npts,)
        """
        if self._curvilinear_basis is None:
            raise ValueError("curvilinear_basis required for radial_stress computation")
        er_x = self._curvilinear_basis[:, 0, 0]
        er_y = self._curvilinear_basis[:, 1, 0]
        return _project(er_x, er_y, *self.cartesian_stress(state, mu, lamda))

    def strain_energy_density(self, state: Array, mu: Array, lamda: Array) -> Array:
        """Compute Neo-Hookean stored energy density (plane strain).

        W = (mu/2)*(I1 - 3) - mu*ln(J) + (lambda/2)*(ln(J))^2

        where I1 = tr(F^T F) includes the out-of-plane F33=1 for plane
        strain, and J = det(F) (2D determinant, since F33=1 does not
        change the 3D determinant under plane strain).

        Returns
        -------
        Array, shape (npts,)
        """
        dW_dmu, dW_dlamda = self.strain_energy_density_lame_jacobian(
            state, mu, lamda
        )
        # W is linear in (mu, lamda).
        return mu * dW_dmu + lamda * dW_dlamda

    def strain_energy_density_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array:
        """Compute d(W)/d(state) at all mesh points.

        Uses dW/dF = P (PK1 stress), so:
            dW/d(u) = P11 * Dx + P12 * Dy
            dW/d(v) = P21 * Dx + P22 * Dy

        Returns
        -------
        Array, shape (npts, 2*npts)
        """
        bkd = self._bkd
        P = self._law.stress(self._compute_F(state), mu, lamda)
        Dx, Dy = self._Dx, self._Dy
        # dW/d(u) = P11*Dx + P12*Dy (since dF_{1L}/du = D_L)
        dW_du = P[0][0][:, None] * Dx + P[0][1][:, None] * Dy
        # dW/d(v) = P21*Dx + P22*Dy (since dF_{2L}/dv = D_L)
        dW_dv = P[1][0][:, None] * Dx + P[1][1][:, None] * Dy
        return bkd.concatenate([dW_du, dW_dv], axis=1)

    def cartesian_stress_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array, Array]:
        """Compute Jacobians d(sigma_ij)/d(state) for all stress components.

        State-dependent for hyperelastic materials.

        Returns
        -------
        dsxx, dsxy, dsyy : tuple of Array
            Each has shape (npts, 2*npts).
        """
        bkd = self._bkd
        npts = self._npts
        Dx, Dy = self._Dx, self._Dy
        D = [Dx, Dy]

        F = self._compute_F(state)
        P = self._law.stress(F, mu, lamda)
        A = self._law.tangent(F, mu, lamda)

        J = F[0][0] * F[1][1] - F[0][1] * F[1][0]
        inv_J = 1.0 / J
        # dJ/d(state): dJ/d(u_m) = F22*Dx[m,:] - F21*Dy[m,:]
        #              dJ/d(v_m) = F11*Dy[m,:] - F12*Dx[m,:]
        dJ_du = F[1][1][:, None] * Dx - F[1][0][:, None] * Dy
        dJ_dv = F[0][0][:, None] * Dy - F[0][1][:, None] * Dx
        dJ = bkd.concatenate([dJ_du, dJ_dv], axis=1)  # (npts, 2*npts)

        # dP_iK/d(state) = [sum_L A_{iK1L}*D_L | sum_L A_{iK2L}*D_L]
        def _dP_block(i: int, K: int) -> Array:
            """Compute dP_{iK}/d(state). Shape: (npts, 2*npts)."""
            blocks = [
                A[i][K][k][0][:, None] * D[0] + A[i][K][k][1][:, None] * D[1]
                for k in range(2)
            ]
            return bkd.concatenate(blocks, axis=1)

        # dF_{jK}/d(state)
        # dF_{1,1}/d(state) = [Dx | 0], dF_{1,2}/d(state) = [Dy | 0]
        # dF_{2,1}/d(state) = [0 | Dx], dF_{2,2}/d(state) = [0 | Dy]
        Z = bkd.zeros((npts, npts))
        dF = [
            [bkd.concatenate([Dx, Z], axis=1), bkd.concatenate([Dy, Z], axis=1)],
            [bkd.concatenate([Z, Dx], axis=1), bkd.concatenate([Z, Dy], axis=1)],
        ]

        # sigma_{ij} = (1/J) * sum_K P_{iK} * F_{jK}
        # d(sigma_{ij})/d(state) = -(1/J^2)*dJ * (sum_K P_{iK}*F_{jK})
        #   + (1/J) * sum_K [dP_{iK} * F_{jK} + P_{iK} * dF_{jK}]

        def _dsigma(i: int, j: int) -> Array:
            """Compute d(sigma_{ij})/d(state). Shape: (npts, 2*npts)."""
            # sum_K P_{iK} * F_{jK}
            PFt_ij = P[i][0] * F[j][0] + P[i][1] * F[j][1]
            # Term 1: -(1/J^2) * PFt_ij * dJ
            term1 = (-(inv_J**2) * PFt_ij)[:, None] * dJ
            # Term 2: (1/J) * sum_K [dP_{iK} * F_{jK} + P_{iK} * dF_{jK}]
            term2 = sum(
                F[j][K][:, None] * _dP_block(i, K) + P[i][K][:, None] * dF[j][K]
                for K in range(2)
            )
            return term1 + inv_J[:, None] * term2

        return _dsigma(0, 0), _dsigma(0, 1), _dsigma(1, 1)

    def hoop_stress_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array:
        """Compute d(sigma_tt)/d(state) at all mesh points.

        State-dependent for hyperelastic materials.

        Returns
        -------
        Array, shape (npts, 2*npts)
        """
        et_x, et_y = self._hoop_basis()
        dsxx, dsxy, dsyy = self.cartesian_stress_state_jacobian(state, mu, lamda)
        return _project(et_x[:, None], et_y[:, None], dsxx, dsxy, dsyy)

    def hoop_stress_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]:
        """Pointwise :math:`(\\partial \\sigma_{tt}/\\partial \\mu,
        \\partial \\sigma_{tt}/\\partial \\lambda)`, from
        :math:`\\partial \\sigma/\\partial c = J^{-1} (\\partial P/
        \\partial c) F^T` at fixed F; each shape (npts,)."""
        et_x, et_y = self._hoop_basis()
        F = self._compute_F(state)
        inv_J = 1.0 / (F[0][0] * F[1][1] - F[0][1] * F[1][0])
        dmu = self._law.stress_mu_sensitivity(F, mu, lamda)
        dlamda = self._law.stress_lamda_sensitivity(F, mu, lamda)
        return (
            _project(et_x, et_y, *self._cauchy(dmu, F, inv_J)),
            _project(et_x, et_y, *self._cauchy(dlamda, F, inv_J)),
        )

    def strain_energy_density_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]:
        """Pointwise :math:`(\\partial W/\\partial \\mu, \\partial
        W/\\partial \\lambda) = (\\tfrac12 (I_1 - 3) - \\ln J,
        \\tfrac12 (\\ln J)^2)`, each shape (npts,)."""
        F = self._compute_F(state)
        J = F[0][0] * F[1][1] - F[0][1] * F[1][0]
        ln_J = self._bkd.log(J)
        # Plane strain: F33=1, so I1 = F11^2+F12^2+F21^2+F22^2 + F33^2
        I1 = F[0][0] ** 2 + F[0][1] ** 2 + F[1][0] ** 2 + F[1][1] ** 2 + 1.0
        return 0.5 * (I1 - 3.0) - ln_J, 0.5 * ln_J**2

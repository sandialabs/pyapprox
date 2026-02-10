"""Neo-Hookean hyperelastic stress model.

Compressible Neo-Hookean constitutive law for 1D, 2D, and 3D:

    P = mu * F + (lamda * ln(J) - mu) * F^{-T}

where F is the deformation gradient, J = det(F), and lamda, mu are
Lame parameters.

Implements StressModelProtocol, StressModelWithTangentProtocol (1D/2D),
and SymbolicStressModelProtocol.
"""

from typing import Dict, Generic, Tuple

import sympy as sp

from pyapprox.typing.util.backends.protocols import Array, Backend


class NeoHookeanStress(Generic[Array]):
    """Isotropic compressible Neo-Hookean stress model.

    First Piola-Kirchhoff stress:
        P = mu * F + (lamda * ln(J) - mu) * F^{-T}

    where J = det(F) is the Jacobian determinant.

    Parameters
    ----------
    lamda : float
        Lame's first parameter.
    mu : float
        Shear modulus.
    """

    def __init__(self, lamda: float, mu: float):
        self._lamda = lamda
        self._mu = mu

    # ------------------------------------------------------------------
    # Numerical stress computation (StressModelProtocol)
    # ------------------------------------------------------------------

    def compute_stress_1d(
        self, F: Array, bkd: Backend[Array]
    ) -> Array:
        """Compute 1D PK1 stress P = mu*F + (lamda*ln(J) - mu)/J.

        In 1D, J = F and F^{-T} = 1/F.
        """
        J = F
        ln_J = bkd.log(J)
        return self._mu * F + (self._lamda * ln_J - self._mu) / J

    def compute_stress_2d(
        self,
        F11: Array,
        F12: Array,
        F21: Array,
        F22: Array,
        bkd: Backend[Array],
    ) -> Tuple[Array, Array, Array, Array]:
        """Compute 2D PK1 stress components.

        F^{-T} = (1/J) * [[F22, -F21], [-F12, F11]]
        """
        J = F11 * F22 - F12 * F21
        ln_J = bkd.log(J)
        coef = self._lamda * ln_J - self._mu

        # F^{-T} components
        Finv_T_11 = F22 / J
        Finv_T_12 = -F21 / J
        Finv_T_21 = -F12 / J
        Finv_T_22 = F11 / J

        P11 = self._mu * F11 + coef * Finv_T_11
        P12 = self._mu * F12 + coef * Finv_T_12
        P21 = self._mu * F21 + coef * Finv_T_21
        P22 = self._mu * F22 + coef * Finv_T_22

        return P11, P12, P21, P22

    def compute_stress_3d(
        self,
        F: Tuple[Tuple[Array, ...], ...],
        bkd: Backend[Array],
    ) -> Tuple[Tuple[Array, ...], ...]:
        """Compute 3D PK1 stress components.

        Uses cofactor matrix for F^{-T} = cof(F)^T / J.
        """
        F11, F12, F13 = F[0]
        F21, F22, F23 = F[1]
        F31, F32, F33 = F[2]

        # Determinant
        J = (
            F11 * (F22 * F33 - F23 * F32)
            - F12 * (F21 * F33 - F23 * F31)
            + F13 * (F21 * F32 - F22 * F31)
        )
        ln_J = bkd.log(J)
        coef = self._lamda * ln_J - self._mu

        # Cofactor matrix entries
        cof11 = F22 * F33 - F23 * F32
        cof12 = -(F21 * F33 - F23 * F31)
        cof13 = F21 * F32 - F22 * F31
        cof21 = -(F12 * F33 - F13 * F32)
        cof22 = F11 * F33 - F13 * F31
        cof23 = -(F11 * F32 - F12 * F31)
        cof31 = F12 * F23 - F13 * F22
        cof32 = -(F11 * F23 - F13 * F21)
        cof33 = F11 * F22 - F12 * F21

        # F^{-T}_{ij} = cof_{ji} / J
        Finv_T_11 = cof11 / J
        Finv_T_12 = cof21 / J
        Finv_T_13 = cof31 / J
        Finv_T_21 = cof12 / J
        Finv_T_22 = cof22 / J
        Finv_T_23 = cof32 / J
        Finv_T_31 = cof13 / J
        Finv_T_32 = cof23 / J
        Finv_T_33 = cof33 / J

        P11 = self._mu * F11 + coef * Finv_T_11
        P12 = self._mu * F12 + coef * Finv_T_12
        P13 = self._mu * F13 + coef * Finv_T_13
        P21 = self._mu * F21 + coef * Finv_T_21
        P22 = self._mu * F22 + coef * Finv_T_22
        P23 = self._mu * F23 + coef * Finv_T_23
        P31 = self._mu * F31 + coef * Finv_T_31
        P32 = self._mu * F32 + coef * Finv_T_32
        P33 = self._mu * F33 + coef * Finv_T_33

        return (
            (P11, P12, P13),
            (P21, P22, P23),
            (P31, P32, P33),
        )

    # ------------------------------------------------------------------
    # Tangent modulus (StressModelWithTangentProtocol)
    # ------------------------------------------------------------------

    def compute_tangent_1d(
        self, F: Array, bkd: Backend[Array]
    ) -> Array:
        """Compute 1D tangent dP/dF.

        dP/dF = mu + (mu + lamda*(1 - ln J)) / J^2
        """
        J = F
        ln_J = bkd.log(J)
        return self._mu + (
            self._mu + self._lamda * (1.0 - ln_J)
        ) / (J ** 2)

    def compute_tangent_2d(
        self,
        F11: Array,
        F12: Array,
        F21: Array,
        F22: Array,
        bkd: Backend[Array],
    ) -> Dict[str, Array]:
        """Compute 2D tangent modulus A_iJkL = dP_iJ/dF_kL.

        Uses the beta/gamma parameterization:
            beta  = (lamda*ln(J) - mu) / J
            gamma = (mu + lamda*(1 - ln(J))) / J^2
        """
        J = F11 * F22 - F12 * F21
        ln_J = bkd.log(J)

        beta = (self._lamda * ln_J - self._mu) / J
        gamma = (self._mu + self._lamda * (1.0 - ln_J)) / (J ** 2)

        return {
            # dP_11/dF_kL
            "A_1111": self._mu + gamma * F22 ** 2,
            "A_1112": -gamma * F21 * F22,
            "A_1121": -gamma * F12 * F22,
            "A_1122": beta + gamma * F11 * F22,
            # dP_12/dF_kL
            "A_1211": -gamma * F22 * F21,
            "A_1212": self._mu + gamma * F21 ** 2,
            "A_1221": -beta + gamma * F12 * F21,
            "A_1222": -gamma * F11 * F21,
            # dP_21/dF_kL
            "A_2111": -gamma * F22 * F12,
            "A_2112": -beta + gamma * F21 * F12,
            "A_2121": self._mu + gamma * F12 ** 2,
            "A_2122": -gamma * F11 * F12,
            # dP_22/dF_kL
            "A_2211": beta + gamma * F22 * F11,
            "A_2212": -gamma * F21 * F11,
            "A_2221": -gamma * F12 * F11,
            "A_2222": self._mu + gamma * F11 ** 2,
        }

    # ------------------------------------------------------------------
    # Symbolic expressions (SymbolicStressModelProtocol)
    # ------------------------------------------------------------------

    def sympy_stress_1d(self, F_expr: sp.Expr) -> sp.Expr:
        """Return symbolic 1D PK1 stress."""
        lamda = sp.Rational(self._lamda) if self._lamda == int(self._lamda) \
            else sp.Float(self._lamda)
        mu = sp.Rational(self._mu) if self._mu == int(self._mu) \
            else sp.Float(self._mu)
        J = F_expr
        return mu * F_expr + (lamda * sp.log(J) - mu) / J

    def sympy_stress_2d(
        self,
        F11: sp.Expr,
        F12: sp.Expr,
        F21: sp.Expr,
        F22: sp.Expr,
    ) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
        """Return symbolic 2D PK1 stress expressions."""
        lamda = sp.Rational(self._lamda) if self._lamda == int(self._lamda) \
            else sp.Float(self._lamda)
        mu = sp.Rational(self._mu) if self._mu == int(self._mu) \
            else sp.Float(self._mu)

        J = F11 * F22 - F12 * F21
        coef = lamda * sp.log(J) - mu

        Finv_T_11 = F22 / J
        Finv_T_12 = -F21 / J
        Finv_T_21 = -F12 / J
        Finv_T_22 = F11 / J

        P11 = mu * F11 + coef * Finv_T_11
        P12 = mu * F12 + coef * Finv_T_12
        P21 = mu * F21 + coef * Finv_T_21
        P22 = mu * F22 + coef * Finv_T_22

        return P11, P12, P21, P22

    def sympy_stress_3d(
        self,
        F: Tuple[Tuple[sp.Expr, ...], ...],
    ) -> Tuple[Tuple[sp.Expr, ...], ...]:
        """Return symbolic 3D PK1 stress expressions."""
        lamda = sp.Rational(self._lamda) if self._lamda == int(self._lamda) \
            else sp.Float(self._lamda)
        mu = sp.Rational(self._mu) if self._mu == int(self._mu) \
            else sp.Float(self._mu)

        F11, F12, F13 = F[0]
        F21, F22, F23 = F[1]
        F31, F32, F33 = F[2]

        J = (
            F11 * (F22 * F33 - F23 * F32)
            - F12 * (F21 * F33 - F23 * F31)
            + F13 * (F21 * F32 - F22 * F31)
        )
        coef = lamda * sp.log(J) - mu

        # Cofactor matrix
        cof11 = F22 * F33 - F23 * F32
        cof12 = -(F21 * F33 - F23 * F31)
        cof13 = F21 * F32 - F22 * F31
        cof21 = -(F12 * F33 - F13 * F32)
        cof22 = F11 * F33 - F13 * F31
        cof23 = -(F11 * F32 - F12 * F31)
        cof31 = F12 * F23 - F13 * F22
        cof32 = -(F11 * F23 - F13 * F21)
        cof33 = F11 * F22 - F12 * F21

        # F^{-T}_{ij} = cof_{ji} / J
        P11 = mu * F11 + coef * cof11 / J
        P12 = mu * F12 + coef * cof21 / J
        P13 = mu * F13 + coef * cof31 / J
        P21 = mu * F21 + coef * cof12 / J
        P22 = mu * F22 + coef * cof22 / J
        P23 = mu * F23 + coef * cof32 / J
        P31 = mu * F31 + coef * cof13 / J
        P32 = mu * F32 + coef * cof23 / J
        P33 = mu * F33 + coef * cof33 / J

        return (
            (P11, P12, P13),
            (P21, P22, P23),
            (P31, P32, P33),
        )

"""Protocols for hyperelastic stress models.

Defines a 3-level protocol hierarchy for constitutive models:
1. StressModelProtocol - numerical PK1 stress computation
2. StressModelWithTangentProtocol - adds analytical tangent modulus dP/dF
3. SymbolicStressModelProtocol - adds sympy expressions for MMS

All protocols use the full (potentially non-symmetric) first Piola-Kirchhoff
stress tensor P_iJ, supporting both isotropic and anisotropic materials.
"""

from typing import Dict, Generic, Protocol, Tuple, runtime_checkable

import sympy as sp

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class StressModelProtocol(Protocol, Generic[Array]):
    """Protocol for hyperelastic constitutive models.

    Computes the first Piola-Kirchhoff stress P from the deformation
    gradient F. The full tensor P_iJ is returned with no symmetry assumed,
    supporting both isotropic and anisotropic materials.

    Each array argument and return value has shape (npts,), representing
    pointwise evaluation at collocation nodes.
    """

    def compute_stress_1d(
        self, F: Array, bkd: Backend[Array]
    ) -> Array:
        """Compute 1D PK1 stress.

        Parameters
        ----------
        F : Array
            Deformation gradient (scalar). Shape: (npts,).
        bkd : Backend
            Computational backend.

        Returns
        -------
        Array
            PK1 stress P. Shape: (npts,).
        """
        ...

    def compute_stress_2d(
        self,
        F11: Array,
        F12: Array,
        F21: Array,
        F22: Array,
        bkd: Backend[Array],
    ) -> Tuple[Array, Array, Array, Array]:
        """Compute 2D PK1 stress.

        Parameters
        ----------
        F11, F12, F21, F22 : Array
            Deformation gradient components. Each shape: (npts,).
        bkd : Backend
            Computational backend.

        Returns
        -------
        Tuple[Array, Array, Array, Array]
            PK1 stress components (P11, P12, P21, P22). Each shape: (npts,).
        """
        ...

    def compute_stress_3d(
        self,
        F: Tuple[Tuple[Array, ...], ...],
        bkd: Backend[Array],
    ) -> Tuple[Tuple[Array, ...], ...]:
        """Compute 3D PK1 stress.

        Parameters
        ----------
        F : Tuple[Tuple[Array, ...], ...]
            Deformation gradient as 3x3 nested tuple. F[i][J] has shape (npts,).
        bkd : Backend
            Computational backend.

        Returns
        -------
        Tuple[Tuple[Array, ...], ...]
            PK1 stress as 3x3 nested tuple. P[i][J] has shape (npts,).
        """
        ...


@runtime_checkable
class StressModelWithTangentProtocol(Protocol, Generic[Array]):
    """Extended protocol adding analytical tangent modulus dP/dF.

    The tangent modulus A_iJkL = dP_iJ/dF_kL is used to compute the
    analytical Jacobian of the hyperelasticity residual. Without this,
    the physics class cannot provide an analytical Jacobian.

    Implementations should also satisfy StressModelProtocol.
    """

    def compute_stress_1d(
        self, F: Array, bkd: Backend[Array]
    ) -> Array: ...

    def compute_stress_2d(
        self,
        F11: Array,
        F12: Array,
        F21: Array,
        F22: Array,
        bkd: Backend[Array],
    ) -> Tuple[Array, Array, Array, Array]: ...

    def compute_stress_3d(
        self,
        F: Tuple[Tuple[Array, ...], ...],
        bkd: Backend[Array],
    ) -> Tuple[Tuple[Array, ...], ...]: ...

    def compute_tangent_1d(
        self, F: Array, bkd: Backend[Array]
    ) -> Array:
        """Compute 1D tangent modulus dP/dF.

        Parameters
        ----------
        F : Array
            Deformation gradient. Shape: (npts,).
        bkd : Backend
            Computational backend.

        Returns
        -------
        Array
            Tangent modulus dP/dF. Shape: (npts,).
        """
        ...

    def compute_tangent_2d(
        self,
        F11: Array,
        F12: Array,
        F21: Array,
        F22: Array,
        bkd: Backend[Array],
    ) -> Dict[str, Array]:
        """Compute 2D tangent modulus components A_iJkL = dP_iJ/dF_kL.

        The returned dict uses keys 'A_iJkL' where i,J,k,L are 1-indexed.
        For example, 'A_1111' is dP_11/dF_11. There are 16 components total
        (4 stress components x 4 gradient components).

        Parameters
        ----------
        F11, F12, F21, F22 : Array
            Deformation gradient components. Each shape: (npts,).
        bkd : Backend
            Computational backend.

        Returns
        -------
        Dict[str, Array]
            Tangent modulus components. Each value has shape: (npts,).
        """
        ...


@runtime_checkable
class SymbolicStressModelProtocol(Protocol):
    """Protocol for stress models with sympy expression support.

    Used by ManufacturedHyperelasticityEquations to symbolically compute
    forcing terms for the Method of Manufactured Solutions (MMS).
    """

    def sympy_stress_1d(self, F_expr: sp.Expr) -> sp.Expr:
        """Return symbolic 1D PK1 stress expression.

        Parameters
        ----------
        F_expr : sp.Expr
            Symbolic deformation gradient (scalar).

        Returns
        -------
        sp.Expr
            Symbolic PK1 stress.
        """
        ...

    def sympy_stress_2d(
        self,
        F11: sp.Expr,
        F12: sp.Expr,
        F21: sp.Expr,
        F22: sp.Expr,
    ) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
        """Return symbolic 2D PK1 stress expressions.

        Parameters
        ----------
        F11, F12, F21, F22 : sp.Expr
            Symbolic deformation gradient components.

        Returns
        -------
        Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]
            Symbolic (P11, P12, P21, P22).
        """
        ...

    def sympy_stress_3d(
        self,
        F: Tuple[Tuple[sp.Expr, ...], ...],
    ) -> Tuple[Tuple[sp.Expr, ...], ...]:
        """Return symbolic 3D PK1 stress expressions.

        Parameters
        ----------
        F : Tuple[Tuple[sp.Expr, ...], ...]
            Symbolic 3x3 deformation gradient.

        Returns
        -------
        Tuple[Tuple[sp.Expr, ...], ...]
            Symbolic 3x3 PK1 stress.
        """
        ...

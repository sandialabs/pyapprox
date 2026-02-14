"""Strain energy functional for 1D collocation-based PDE solutions.

Computes W(u) = integral psi(grad_u) dx where the strain energy density
psi depends on the constitutive model. The derivative matrix Dx maps the
state vector to its spatial gradient, and the chain rule gives
dW/du = (w * dpsi) @ Dx.
"""

from typing import Callable, Generic, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.typing.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)
from pyapprox.typing.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)


class StrainEnergyFunctional1D(Generic[Array]):
    """Strain energy functional for 1D bars.

    Computes W(u) = integral_a^b psi(arg(u)) dx where arg depends on
    the ``deformation_gradient`` flag:

    - ``deformation_gradient=False``: arg = du/dx (strain, for linear elastic)
    - ``deformation_gradient=True``: arg = 1 + du/dx (deformation gradient,
      for hyperelastic)

    The energy density callable returns both psi and its derivative dpsi/d_arg.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    nparams : int
        Number of parameters in the forward model.
    bkd : Backend
        Computational backend.
    energy_density : callable
        ``energy_density(arg, bkd) -> (psi, dpsi_d_arg)`` where both
        arrays have shape ``(npts,)``.
    a_sub : float, optional
        Left endpoint of integration domain (physical coordinates).
        If None, integrates over the full domain.
    b_sub : float, optional
        Right endpoint of integration domain (physical coordinates).
        If None, integrates over the full domain.
    deformation_gradient : bool
        If True, passes F = 1 + du/dx to the energy density callable.
        If False, passes epsilon = du/dx.
    """

    def __init__(
        self,
        basis: ChebyshevBasis1D[Array],
        nparams: int,
        bkd: Backend[Array],
        energy_density: Callable[
            [Array, Backend[Array]], Tuple[Array, Array]
        ],
        a_sub: Optional[float] = None,
        b_sub: Optional[float] = None,
        deformation_gradient: bool = False,
    ) -> None:
        self._bkd = bkd
        self._nparams = nparams
        self._energy_density = energy_density
        self._deformation_gradient = deformation_gradient

        # Physical derivative matrix
        self._Dx = basis.derivative_matrix(1, 0)  # shape (npts, npts)
        self._nstates = self._Dx.shape[0]

        # Quadrature weights
        quad = CollocationQuadrature1D(basis, bkd)
        if a_sub is None and b_sub is None:
            self._weights = quad.full_domain_weights()
        else:
            if a_sub is None or b_sub is None:
                raise ValueError(
                    "Both a_sub and b_sub must be provided, or both None"
                )
            self._weights = quad.weights(a_sub, b_sub)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return self._nstates

    def nparams(self) -> int:
        return self._nparams

    def nunique_params(self) -> int:
        return 0

    def _compute_arg(self, state: Array) -> Array:
        """Compute the argument to the energy density from state."""
        u = state[:, 0]  # shape (npts,)
        grad_u = self._Dx @ u  # du/dx, shape (npts,)
        if self._deformation_gradient:
            return 1.0 + grad_u
        return grad_u

    def __call__(self, state: Array, param: Array) -> Array:
        """Evaluate strain energy W(u).

        Parameters
        ----------
        state : Array, shape (nstates, 1)
            Displacement field at collocation nodes.
        param : Array, shape (nparams, 1)
            Parameters (unused).

        Returns
        -------
        Array, shape (1, 1)
            Total strain energy.
        """
        bkd = self._bkd
        arg = self._compute_arg(state)
        psi, _ = self._energy_density(arg, bkd)
        W = bkd.sum(self._weights * psi)
        return bkd.reshape(W, (1, 1))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dW/du via chain rule: (w * dpsi) @ Dx.

        Returns
        -------
        Array, shape (1, nstates)
        """
        bkd = self._bkd
        arg = self._compute_arg(state)
        _, dpsi = self._energy_density(arg, bkd)
        # dW/du_j = sum_i w_i * dpsi_i * Dx_{ij}
        return (self._weights * dpsi)[None, :] @ self._Dx

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Return dW/dp = 0 (no parameter dependence).

        Returns
        -------
        Array, shape (1, nparams)
        """
        return self._bkd.zeros((1, self._nparams))

    def __repr__(self) -> str:
        mode = "deformation_gradient" if self._deformation_gradient else "strain"
        return (
            f"{self.__class__.__name__}("
            f"mode={mode}, "
            f"nstates={self._nstates}, "
            f"nparams={self._nparams}, "
            f"bkd={type(self._bkd).__name__})"
        )


def create_linear_strain_energy_1d(
    basis: ChebyshevBasis1D[Array],
    nparams: int,
    bkd: Backend[Array],
    E: float,
    a_sub: Optional[float] = None,
    b_sub: Optional[float] = None,
) -> StrainEnergyFunctional1D[Array]:
    """Create strain energy functional for a linear elastic 1D bar.

    Computes W = integral (1/2)*E*epsilon^2 dx where epsilon = du/dx.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    nparams : int
        Number of parameters in the forward model.
    bkd : Backend
        Computational backend.
    E : float
        Young's modulus (constant).
    a_sub, b_sub : float, optional
        Integration bounds. None for full domain.
    """
    def energy_density(
        epsilon: Array, bkd: Backend[Array]
    ) -> Tuple[Array, Array]:
        psi = 0.5 * E * epsilon ** 2
        dpsi = E * epsilon
        return psi, dpsi

    return StrainEnergyFunctional1D(
        basis, nparams, bkd, energy_density,
        a_sub=a_sub, b_sub=b_sub, deformation_gradient=False,
    )


def create_neo_hookean_strain_energy_1d(
    basis: ChebyshevBasis1D[Array],
    nparams: int,
    bkd: Backend[Array],
    lamda: float,
    mu: float,
    a_sub: Optional[float] = None,
    b_sub: Optional[float] = None,
) -> StrainEnergyFunctional1D[Array]:
    """Create strain energy functional for a 1D Neo-Hookean bar.

    Computes W = integral psi(F) dx where F = 1 + du/dx and
    psi(F) = mu/2*(F^2 - 1 - 2*ln(F)) + lamda/2*(ln(F))^2.
    The derivative dpsi/dF = P(F) is the first Piola-Kirchhoff stress.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    nparams : int
        Number of parameters in the forward model.
    bkd : Backend
        Computational backend.
    lamda : float
        Lame's first parameter.
    mu : float
        Shear modulus.
    a_sub, b_sub : float, optional
        Integration bounds. None for full domain.
    """
    stress_model = NeoHookeanStress(lamda=lamda, mu=mu)

    def energy_density(
        F: Array, bkd: Backend[Array]
    ) -> Tuple[Array, Array]:
        ln_F = bkd.log(F)
        psi = mu / 2.0 * (F ** 2 - 1.0 - 2.0 * ln_F) + lamda / 2.0 * ln_F ** 2
        P = stress_model.compute_stress_1d(F, bkd)
        return psi, P

    return StrainEnergyFunctional1D(
        basis, nparams, bkd, energy_density,
        a_sub=a_sub, b_sub=b_sub, deformation_gradient=True,
    )

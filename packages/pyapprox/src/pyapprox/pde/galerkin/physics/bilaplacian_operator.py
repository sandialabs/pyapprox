"""The diffusion-reaction operator of the bi-Laplacian prior.

``A u = -div(gamma T grad u) + delta u``, the operator whose inverse,
applied twice to white noise, gives a Whittle-Matern field. It is an
ordinary Galerkin physics: it holds no boundary conditions and is
composed with them (typically Robin, to damp the boundary artifact) by
``compose_galerkin_system``.
"""

from typing import TYPE_CHECKING, Any, Generic, Optional

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.galerkin.physics.helpers import ScalarMassAssembler
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import BilinearForm, asm
    from skfem.helpers import dot, grad, mul
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class BiLaplacianOperator(Generic[Array]):
    r"""Interior operator :math:`F_\Omega(u) = -K u` of the bi-Laplacian
    prior, with :math:`K_{ij} = \int \gamma (T \nabla\phi_j)\cdot
    \nabla\phi_i + \delta \phi_j \phi_i`.

    Linear in the state and time-invariant. The ``delta`` term uses the
    consistent mass: it is part of the operator, not the noise weighting.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Scalar finite element basis.
    gamma : float
        Diffusion scaling; :math:`\gamma/\delta` sets the correlation
        length and :math:`\gamma\delta` the variance.
    delta : float
        Reaction coefficient.
    bkd : Backend
        Computational backend.
    anisotropic_tensor : np.ndarray, optional
        Anisotropy tensor :math:`T`, shape ``(ndim, ndim)``, setting
        directional correlation lengths. Default: identity.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        gamma: float,
        delta: float,
        bkd: Backend[Array],
        anisotropic_tensor: Optional[NDArray[np.floating[Any]]] = None,
    ) -> None:
        ndim = basis.mesh().ndim()
        tensor = (
            np.eye(ndim)
            if anisotropic_tensor is None
            else np.asarray(anisotropic_tensor, dtype=float)
        )
        if tensor.shape != (ndim, ndim):
            raise ValueError(
                f"anisotropic_tensor has incorrect shape {tensor.shape}, "
                f"expected ({ndim}, {ndim})"
            )
        self._basis = basis
        self._gamma = gamma
        self._delta = delta
        self._bkd = bkd
        self._diffusion_tensor = gamma * tensor
        self._stiffness: Optional[Array] = None
        self._mass: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def basis(self) -> GalerkinBasisProtocol[Array]:
        """Return the finite element basis."""
        return self._basis

    def nstates(self) -> int:
        return self._basis.ndofs()

    def gamma(self) -> float:
        """Return the diffusion scaling parameter."""
        return self._gamma

    def delta(self) -> float:
        """Return the reaction coefficient."""
        return self._delta

    def stiffness_matrix(self) -> Array:
        """Return :math:`K` (sparse), assembled once."""
        if self._stiffness is None:
            tensor = self._diffusion_tensor
            delta = self._delta

            def form(
                u: "DiscreteField", v: "DiscreteField", w: "FormExtraParams"
            ) -> np.ndarray:
                ret: NDArray[np.floating[Any]] = (
                    dot(mul(tensor, grad(u)), grad(v)) + delta * u * v
                )
                return ret

            self._stiffness = asm(BilinearForm(form), self._basis.skfem_basis())
        return self._stiffness

    def interior_residual(self, state: Array, time: float) -> Array:
        """Return :math:`-K u`."""
        residual: Array = -(self.stiffness_matrix() @ state)
        return residual

    def interior_jacobian(self, state: Array, time: float) -> Array:
        """Return :math:`-K`."""
        jacobian: Array = -self.stiffness_matrix()
        return jacobian

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Exact zero curvature: the operator is linear in u."""
        return StateDerivatives.linear(self._bkd)

    def interior_is_time_invariant(self) -> bool:
        """Always: no coefficient depends on time."""
        return True

    def mass_matrix(self) -> Array:
        """Return the consistent mass matrix."""
        if self._mass is None:
            self._mass = ScalarMassAssembler(
                self._basis, self._bkd
            ).mass_matrix()
        return self._mass

    def __repr__(self) -> str:
        return (
            f"BiLaplacianOperator(ndofs={self.nstates()}, "
            f"gamma={self._gamma}, delta={self._delta})"
        )

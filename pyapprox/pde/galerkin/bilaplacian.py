"""BiLaplacian prior for random field generation via Galerkin FEM.

Generates random field samples by solving a diffusion-reaction equation
with Robin boundary conditions. The covariance structure is controlled by:
- gamma * delta -> variance of the prior
- gamma / delta -> correlation length
- anisotropic_tensor -> directional correlation lengths

The bilaplacian prior is used as a Gaussian process approximation for
Bayesian inverse problems.
"""

from typing import Any, TYPE_CHECKING, Generic, List, Optional

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.galerkin.boundary.implementations import RobinBC
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.boundary import RobinBCProtocol
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import BilinearForm, asm, condense, solve
    from skfem.helpers import dot, grad, mul
    from skfem.models.poisson import mass
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class BiLaplacianPrior(Generic[Array]):
    r"""BiLaplacian prior for Gaussian random field generation.

    Generates samples from a Gaussian random field by solving a
    diffusion-reaction equation with Robin boundary conditions:

        K u = sqrt(M_lumped) * white_noise

    where K is the stiffness matrix assembled from:
        dot(mul(K_tensor, grad(u)), grad(v)) + delta * u * v

    plus Robin BC contributions.

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Finite element basis.
    gamma : float
        Diffusion scaling parameter.
        :math:`\delta \gamma` controls the variance of the prior.
    delta : float
        Reaction coefficient.
        :math:`\gamma / \delta` controls the correlation length.
    bkd : Backend[Array]
        Computational backend.
    boundary_conditions : List[RobinBCProtocol[Array]]
        Robin boundary conditions applied to all desired boundaries.
    anisotropic_tensor : np.ndarray, optional
        Anisotropy tensor of shape ``(ndim, ndim)``. Controls directional
        correlation lengths. Default: identity matrix (isotropic).
        Stored internally as ``gamma * tensor``.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        gamma: float,
        delta: float,
        bkd: Backend[Array],
        boundary_conditions: List[RobinBCProtocol[Array]],
        anisotropic_tensor: Optional[np.ndarray] = None,
    ):
        self._basis = basis
        self._gamma = gamma
        self._delta = delta
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions

        ndim = basis.mesh().ndim()
        if anisotropic_tensor is None:
            self._anisotropic_tensor = np.eye(ndim) * gamma
        else:
            anisotropic_tensor = np.asarray(anisotropic_tensor, dtype=float)
            if anisotropic_tensor.shape != (ndim, ndim):
                raise ValueError(
                    f"anisotropic_tensor has incorrect shape "
                    f"{anisotropic_tensor.shape}, expected ({ndim}, {ndim})"
                )
            self._anisotropic_tensor = anisotropic_tensor * gamma

        self._stiffness: Optional[Array] = None
        self._lumped_mass: Optional[NDArray[np.floating[Any]]] = None

    @classmethod
    def with_uniform_robin(
        cls,
        basis: GalerkinBasisProtocol[Array],
        gamma: float,
        delta: float,
        bkd: Backend[Array],
        anisotropic_tensor: Optional[np.ndarray] = None,
        robin_alpha: Optional[float] = None,
    ) -> "BiLaplacianPrior[Array]":
        """Create prior with uniform Robin BCs on all boundaries.

        Parameters
        ----------
        basis : GalerkinBasisProtocol[Array]
            Finite element basis.
        gamma : float
            Diffusion scaling parameter.
        delta : float
            Reaction coefficient.
        bkd : Backend[Array]
            Computational backend.
        anisotropic_tensor : np.ndarray, optional
            Anisotropy tensor. Default: identity.
        robin_alpha : float, optional
            Robin BC coefficient. Default: ``sqrt(gamma * delta) * 1.42``.

        Returns
        -------
        BiLaplacianPrior[Array]
            Constructed prior.
        """
        if robin_alpha is None:
            robin_alpha = np.sqrt(gamma * delta) * 1.42
        boundaries = list(basis.skfem_basis().mesh.boundaries.keys())
        robin_bcs: List[RobinBCProtocol[Array]] = [
            RobinBC(basis, name, alpha=robin_alpha, value_func=0.0, bkd=bkd)
            for name in boundaries
        ]
        return cls(basis, gamma, delta, bkd, robin_bcs, anisotropic_tensor)

    def _assemble_system(self) -> None:
        """Lazily assemble stiffness matrix and lumped mass vector."""
        if self._stiffness is not None:
            return

        skfem_basis = self._basis.skfem_basis()
        K_tensor = self._anisotropic_tensor
        delta = self._delta

        def bilinear_form(u: "DiscreteField", v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            return dot(mul(K_tensor, grad(u)), grad(v)) + delta * u * v

        stiffness = asm(BilinearForm(bilinear_form), skfem_basis)

        # Apply Robin BC contributions
        for bc in self._boundary_conditions:
            alpha = bc.alpha()
            bndry_basis = skfem_basis.boundary(bc.boundary_name())

            def robin_bilinear(
                u: object, v: object, w: object, _alpha: object = alpha
            ) -> object:
                return _alpha * u * v

            stiffness += asm(BilinearForm(robin_bilinear), bndry_basis)

        self._stiffness = stiffness

        # Lumped mass: row sums of consistent mass matrix
        mass_mat = asm(mass, skfem_basis)
        self._lumped_mass = np.asarray(mass_mat.sum(axis=1))[:, 0]

    def rvs(
        self,
        nsamples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Array:
        """Generate random field samples.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.
        rng : np.random.Generator, optional
            Random number generator. If None, uses the global numpy RNG
            (``np.random.normal``).

        Returns
        -------
        Array
            Random field samples. Shape: ``(ndofs, nsamples)``.
        """
        self._assemble_system()
        if self._lumped_mass is None or self._stiffness is None:
            raise RuntimeError("Assembly failed")

        ndofs = self._lumped_mass.shape[0]
        if rng is not None:
            white_noise = rng.standard_normal((ndofs, nsamples))
        else:
            white_noise = np.random.normal(0, 1, (ndofs, nsamples))

        sqrt_lumped = np.sqrt(self._lumped_mass)
        samples = np.empty((ndofs, nsamples))
        empty_D = np.empty(0, dtype=int)
        for ii in range(nsamples):
            rhs = sqrt_lumped * white_noise[:, ii]
            samples[:, ii] = solve(*condense(self._stiffness, rhs, D=empty_D))

        return self._bkd.asarray(samples.astype(np.float64))

    def stiffness_matrix(self) -> Array:
        """Return the assembled stiffness matrix as a backend array.

        Returns
        -------
        Array
            Stiffness matrix. Shape: ``(ndofs, ndofs)``.
        """
        self._assemble_system()
        if self._stiffness is None:
            raise RuntimeError("Assembly failed")
        return self._stiffness

    def lumped_mass(self) -> Array:
        """Return the lumped mass vector.

        Returns
        -------
        Array
            Lumped mass. Shape: ``(ndofs,)``.
        """
        self._assemble_system()
        if self._lumped_mass is None:
            raise RuntimeError("Assembly failed")
        return self._bkd.asarray(self._lumped_mass.astype(np.float64))

    def __repr__(self) -> str:
        ndofs = self._basis.ndofs()
        return (
            f"BiLaplacianPrior(ndofs={ndofs}, gamma={self._gamma}, delta={self._delta})"
        )

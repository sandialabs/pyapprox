"""BiLaplacian prior for random field generation via Galerkin FEM.

Generates random field samples by solving a diffusion-reaction equation
with Robin boundary conditions. The covariance structure is controlled by:
- gamma * delta -> variance of the prior
- gamma / delta -> correlation length
- anisotropic_tensor -> directional correlation lengths

The bilaplacian prior is used as a Gaussian process approximation for
Bayesian inverse problems.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    List,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import eigsh, splu

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.boundary import NaturalBCOperator, WeakFormBCProtocol
from pyapprox.pde.galerkin.boundary.implementations import RobinBC
from pyapprox.pde.galerkin.noise_mass import (
    LumpedNoiseMass,
    NoiseMassProtocol,
)
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.spatial_operator import ComposedSpatialOperator
from pyapprox.surrogates.kle.utils import adjust_sign_eig
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import BilinearForm, asm
    from skfem.helpers import dot, grad, mul
    from skfem.models.poisson import mass
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


def _dense(matrix: Any) -> NDArray[np.floating[Any]]:
    """A sparse or dense matrix as a dense NumPy array."""
    if issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _sparse(matrix: Any) -> Any:
    """A sparse or dense matrix as scipy CSC, for the sparse solvers."""
    if issparse(matrix):
        return matrix.tocsc()
    return csc_matrix(np.asarray(matrix, dtype=np.float64))


class _LinearInterior(Generic[Array]):
    """Interior operator ``F_Omega(u) = -K u`` for a fixed matrix ``K``.

    Lets an assembled linear operator be composed with natural-BC terms
    by ``ComposedSpatialOperator``, like any physics. Module-level, so it
    pickles.
    """

    def __init__(self, stiffness: Any, bkd: Backend[Array]) -> None:
        self._stiffness = stiffness
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return int(self._stiffness.shape[0])

    def interior_residual(self, state: Array, time: float) -> Array:
        residual: Array = -(self._stiffness @ state)
        return residual

    def interior_jacobian(self, state: Array, time: float) -> Array:
        jacobian: Array = -self._stiffness
        return jacobian

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        return StateDerivatives.linear(self._bkd)

    def interior_is_time_invariant(self) -> bool:
        return True


class BiLaplacianPrior(Generic[Array]):
    r"""BiLaplacian prior for Gaussian random field generation.

    Generates samples from a Gaussian random field by solving a
    diffusion-reaction equation with Robin boundary conditions:

    .. math::

        K u = B \xi, \qquad \xi \sim N(0, I), \qquad B B^\top = M,

    so the covariance is :math:`K^{-1} M K^{-1}`. ``M`` is the white-noise
    mass, supplied by an injected ``NoiseMassProtocol`` (default: the
    row-sum lumped mass). K is the stiffness matrix assembled from:
        dot(mul(K_tensor, grad(u)), grad(v)) + delta * u * v

    plus minus the Jacobian of each natural-BC term (``a`` times the
    boundary mass for Robin), composed by ``ComposedSpatialOperator``.
    The ``delta * u * v`` term always uses the consistent mass: it is
    part of the operator's discretization, not the noise weighting.

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
    boundary_conditions : List[WeakFormBCProtocol[Array]]
        Natural boundary-condition terms (typically Robin, to damp the
        boundary artifact). Only their Jacobian enters the precision
        operator; their data does not.
    anisotropic_tensor : np.ndarray, optional
        Anisotropy tensor of shape ``(ndim, ndim)``. Controls directional
        correlation lengths. Default: identity matrix (isotropic).
        Stored internally as ``gamma * tensor``.
    noise_mass : NoiseMassProtocol, optional
        The white-noise mass ``M`` and its factor ``B``. Default: the
        row-sum lumped mass (``LumpedNoiseMass``).
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        gamma: float,
        delta: float,
        bkd: Backend[Array],
        boundary_conditions: List[WeakFormBCProtocol[Array]],
        anisotropic_tensor: Optional[np.ndarray] = None,
        noise_mass: Optional[NoiseMassProtocol[Array]] = None,
    ):
        if noise_mass is not None and not isinstance(
            noise_mass, NoiseMassProtocol
        ):
            raise TypeError(
                "noise_mass must satisfy NoiseMassProtocol, got "
                f"{type(noise_mass).__name__}"
            )
        self._basis = basis
        self._gamma = gamma
        self._delta = delta
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions
        self._noise_mass: Optional[NoiseMassProtocol[Array]] = noise_mass

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
        # Sparse LU of the stiffness, shared by rvs and apply_covariance.
        self._stiffness_lu: Optional[Any] = None

    @classmethod
    def with_uniform_robin(
        cls,
        basis: GalerkinBasisProtocol[Array],
        gamma: float,
        delta: float,
        bkd: Backend[Array],
        anisotropic_tensor: Optional[np.ndarray] = None,
        robin_alpha: Optional[float] = None,
        noise_mass: Optional[NoiseMassProtocol[Array]] = None,
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
        noise_mass : NoiseMassProtocol, optional
            The white-noise mass. Default: the row-sum lumped mass.

        Returns
        -------
        BiLaplacianPrior[Array]
            Constructed prior.
        """
        if robin_alpha is None:
            robin_alpha = np.sqrt(gamma * delta) * 1.42
        boundaries = list(basis.skfem_basis().mesh.boundaries.keys())
        robin_bcs: List[WeakFormBCProtocol[Array]] = [
            RobinBC(basis, name, alpha=robin_alpha, value_func=0.0, bkd=bkd)
            for name in boundaries
        ]
        return cls(
            basis,
            gamma,
            delta,
            bkd,
            robin_bcs,
            anisotropic_tensor,
            noise_mass=noise_mass,
        )

    def _assemble_system(self) -> None:
        """Lazily assemble stiffness matrix and lumped mass vector."""
        if self._stiffness is not None:
            return

        skfem_basis = self._basis.skfem_basis()
        K_tensor = self._anisotropic_tensor
        delta = self._delta

        def bilinear_form(
            u: "DiscreteField",
            v: "DiscreteField",
            w: "FormExtraParams",
        ) -> np.ndarray:
            ret: NDArray[np.floating[Any]] = (
                dot(mul(K_tensor, grad(u)), grad(v))
                + delta * u * v
            )
            return ret

        interior_stiffness = asm(BilinearForm(bilinear_form), skfem_basis)

        # The precision operator is minus the Jacobian of the composed
        # operator F = -(K + delta M) u + sum_k c_k, so the boundary terms
        # come from the same composition every physics uses.
        composed = ComposedSpatialOperator(
            _LinearInterior(interior_stiffness, self._bkd),
            NaturalBCOperator(self._boundary_conditions),
        )
        zero_state = self._bkd.zeros((composed.nstates(),))
        stiffness = -composed.spatial_jacobian(zero_state, 0.0)

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
        if self._stiffness is None:
            raise RuntimeError("Assembly failed")

        noise_mass = self.noise_mass()
        nnoise = noise_mass.nnoise()
        if rng is not None:
            white_noise = rng.standard_normal((nnoise, nsamples))
        else:
            white_noise = np.random.normal(0, 1, (nnoise, nsamples))

        # The right-hand sides B xi, solved together with one cached
        # factorization of K; the solve stays at the scipy seam.
        rhs = self._bkd.to_numpy(
            noise_mass.apply_factor(self._bkd.asarray(white_noise))
        )
        samples = self._stiffness_factor().solve(
            np.asarray(rhs, dtype=np.float64)
        )
        return self._bkd.asarray(np.asarray(samples, dtype=np.float64))

    def _stiffness_factor(self) -> Any:
        """The sparse LU factorization of ``K``, computed once."""
        if self._stiffness_lu is None:
            self._assemble_system()
            self._stiffness_lu = splu(_sparse(self._stiffness))
        return self._stiffness_lu

    def noise_mass(self) -> NoiseMassProtocol[Array]:
        """Return the white-noise mass (the row-sum lumped mass by
        default)."""
        if self._noise_mass is None:
            self._noise_mass = LumpedNoiseMass(self._basis, self._bkd)
        return self._noise_mass

    def nnoise(self) -> int:
        """Return the length of the white-noise vector each sample uses.

        The number of inputs of a field map built on
        ``covariance_factor()``.
        """
        return self.noise_mass().nnoise()

    def _dense_stiffness(self) -> NDArray[np.floating[Any]]:
        """``K`` as a dense NumPy array, for the dense covariance methods."""
        self._assemble_system()
        return _dense(self._stiffness)

    def covariance(self) -> Array:
        r"""Return the covariance :math:`K^{-1} M K^{-1}`.

        Dense, shape ``(ndofs, ndofs)``: for moderate meshes only.
        """
        stiffness = self._dense_stiffness()
        mass = _dense(self.noise_mass().mass_matrix())
        half = np.linalg.solve(stiffness, mass)
        covariance = np.linalg.solve(stiffness, half.T)
        return self._bkd.asarray(covariance)

    def apply_covariance(self, vectors: Array) -> Array:
        r"""Return :math:`K^{-1} M K^{-1} v` without forming the covariance.

        Two sparse solves with one cached factorization of ``K``, so it
        scales to large meshes.

        Parameters
        ----------
        vectors : Array
            Shape: ``(ndofs, ncols)``.

        Returns
        -------
        Array
            Shape: ``(ndofs, ncols)``.
        """
        ndofs = self._basis.ndofs()
        if vectors.ndim != 2 or vectors.shape[0] != ndofs:
            raise ValueError(
                f"vectors must have shape ({ndofs}, ncols), got "
                f"{tuple(vectors.shape)}"
            )
        lu = self._stiffness_factor()
        mass = self.noise_mass().mass_matrix()
        inner = lu.solve(self._bkd.to_numpy(vectors).astype(np.float64))
        result = lu.solve(np.asarray(mass @ inner, dtype=np.float64))
        return self._bkd.asarray(result)

    def covariance_factor(self) -> Array:
        r"""Return :math:`W = K^{-1} B`, so that :math:`W W^\top` is the
        covariance and ``mean + W xi`` is a sample for ``xi ~ N(0, I)``.

        Dense, shape ``(ndofs, nnoise)``; ``nnoise`` may exceed ``ndofs``
        (e.g. for ``ConsistentNoiseMass``), so ``W`` need not be square.
        """
        noise_mass = self.noise_mass()
        factor = _dense(
            noise_mass.apply_factor(self._bkd.eye(noise_mass.nnoise()))
        )
        return self._bkd.asarray(
            np.linalg.solve(self._dense_stiffness(), factor)
        )

    def generalized_eigenpairs(self, rank: int) -> Tuple[Array, Array]:
        r"""Return the ``rank`` smallest eigenpairs of
        :math:`K \phi = \mu M \phi`, with :math:`\phi^\top M \phi = I`.

        Uses sparse shift-invert Lanczos for ``rank < ndofs``, and a dense
        generalized eigensolve for all ``ndofs`` (which Lanczos cannot
        return). Signs follow the repository's eigenvector convention
        (``adjust_sign_eig``).

        Returns
        -------
        eigenvalues : Array
            Ascending, shape ``(rank,)``.
        eigenvectors : Array
            Shape ``(ndofs, rank)``.
        """
        self._assemble_system()
        ndofs = self._basis.ndofs()
        if not 1 <= rank <= ndofs:
            raise ValueError(f"rank must be in [1, {ndofs}], got {rank}")
        mass = self.noise_mass().mass_matrix()
        if rank < ndofs:
            # A fixed start vector makes each call reproducible: Lanczos
            # otherwise starts from a random vector, and a mode whose
            # extreme entries tie to rounding can then change sign
            # between calls. It must be generic, not e.g. ones, which is
            # orthogonal to every antisymmetric mode.
            start = np.random.default_rng(0).standard_normal(ndofs)
            eigenvalues, modes = eigsh(
                _sparse(self._stiffness),
                k=rank,
                M=_sparse(mass),
                sigma=0.0,
                which="LM",
                v0=start,
            )
        else:
            eigenvalues, modes = eigh(self._dense_stiffness(), _dense(mass))
        order = np.argsort(eigenvalues)
        eigenvectors = adjust_sign_eig(
            self._bkd.asarray(np.ascontiguousarray(modes[:, order])), self._bkd
        )
        return self._bkd.asarray(eigenvalues[order]), eigenvectors

    def truncated_covariance_factor(self, rank: int) -> Array:
        r"""Return :math:`\Phi_r \mathrm{diag}(1/\mu_r)`, shape
        ``(ndofs, rank)``, from ``generalized_eigenpairs(rank)``.

        Since :math:`K^{-1} M K^{-1} = \Phi \mathrm{diag}(\mu^{-2})
        \Phi^\top`, the full expansion (``rank = ndofs``) equals
        ``covariance()`` exactly, for any mass.
        """
        eigenvalues, eigenvectors = self.generalized_eigenpairs(rank)
        return eigenvectors / eigenvalues

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

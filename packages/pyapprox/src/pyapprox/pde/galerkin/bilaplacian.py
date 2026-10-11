"""BiLaplacian prior for random field generation via Galerkin FEM.

Generates random field samples by solving a diffusion-reaction equation
with Robin boundary conditions. The covariance structure is controlled by:
- gamma * delta -> variance of the prior
- gamma / delta -> correlation length
- anisotropic_tensor -> directional correlation lengths

The bilaplacian prior is used as a Gaussian process approximation for
Bayesian inverse problems.
"""

from math import gamma as gamma_func
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    List,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    pass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import eigsh, splu

from pyapprox.pde.boundary import BoundaryConditionRole
from pyapprox.pde.galerkin.boundary.implementations import RobinBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.noise_mass import (
    LumpedNoiseMass,
    NoiseMassProtocol,
)
from pyapprox.pde.galerkin.physics.bilaplacian_operator import (
    BiLaplacianOperator,
)
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinSteadySystemProtocol,
)
from pyapprox.surrogates.kle.utils import adjust_sign_eig
from pyapprox.util.backends.protocols import Array, Backend


def bilaplacian_stationary_variance(
    gamma: float, delta: float, ndim: int
) -> float:
    r"""Return the marginal variance of the bi-Laplacian prior in free space.

    The prior solves :math:`\gamma(\kappa^2 - \Delta) u = W` with
    :math:`\kappa^2 = \delta/\gamma`, a Whittle-Matern field of
    smoothness :math:`\nu = 2 - d/2`, whose variance away from any
    boundary is

    .. math::

        \sigma^2 = \frac{\Gamma(\nu)}
                        {\Gamma(\nu + d/2)\,(4\pi)^{d/2}\,
                         \kappa^{2\nu}\,\gamma^2}.

    In 2D this is :math:`1/(4\pi\gamma\delta)`. The discrete field
    approaches it in the interior as the mesh is refined; boundary
    conditions change it near the boundary.

    Parameters
    ----------
    gamma : float
        Diffusion scaling parameter.
    delta : float
        Reaction coefficient.
    ndim : int
        Spatial dimension, 1, 2 or 3 (the field is defined for
        :math:`\nu > 0`).

    Returns
    -------
    float
        The stationary marginal variance.
    """
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2 or 3, got {ndim}")
    nu = 2.0 - ndim / 2.0
    kappa_sq = delta / gamma
    return float(
        gamma_func(nu)
        / (
            gamma_func(nu + ndim / 2.0)
            * (4.0 * np.pi) ** (ndim / 2.0)
            * kappa_sq**nu
            * gamma**2
        )
    )


def default_robin_coefficient(gamma: float, delta: float) -> float:
    r"""Return the default Robin coefficient :math:`\sqrt{\gamma\delta}/1.42`.

    Without a boundary condition that mimics the unbounded domain, the
    prior's variance is distorted near the boundary. This coefficient
    keeps it close to the interior value in 2D: on the unit square
    (40 x 40 tri P1, correlation length 0.1 and 0.25, either noise mass)
    the std at an edge midpoint is 0.93-0.96 and at a corner 0.90-0.93
    of the free-space value, against 0.70 and 0.56 for
    :math:`1.42\sqrt{\gamma\delta}`. The constant is tuned for 2D; in 1D
    the endpoint std is 0.83 of the interior value.

    Parameters
    ----------
    gamma : float
        Diffusion scaling parameter.
    delta : float
        Reaction coefficient.

    Returns
    -------
    float
        The Robin coefficient.
    """
    return float(np.sqrt(gamma * delta) / 1.42)


def bilaplacian_correlation_parameters(
    length: float, std: float, ndim: int
) -> Tuple[float, float]:
    r"""Return the :math:`(\gamma, \delta)` of a given correlation length
    and stationary standard deviation.

    :math:`\delta = \gamma/\ell^2`, and :math:`\gamma` is the value for
    which ``bilaplacian_stationary_variance`` equals :math:`\sigma^2`.
    Scaling :math:`(\gamma, \delta)` together keeps :math:`\ell` and
    divides the variance by the square of the scale, so the pair is
    unique. In 2D, :math:`\gamma\delta = 1/(4\pi\sigma^2)`.

    Parameters
    ----------
    length : float
        Correlation length :math:`\ell = \sqrt{\gamma/\delta}` (the
        inverse of the Matern :math:`\kappa`, not the practical range
        :math:`\sqrt{8\nu}\,\ell`).
    std : float
        Stationary marginal standard deviation :math:`\sigma`.
    ndim : int
        Spatial dimension, 1, 2 or 3.

    Returns
    -------
    gamma, delta : float
    """
    if length <= 0.0 or std <= 0.0:
        raise ValueError(
            f"length and std must be positive, got {length} and {std}"
        )
    # At gamma = 1 the variance is v1; it scales as 1/gamma^2 at fixed
    # kappa = 1/length.
    unit_variance = bilaplacian_stationary_variance(1.0, 1.0 / length**2, ndim)
    gamma = float(np.sqrt(unit_variance)) / std
    return gamma, gamma / length**2


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


class BiLaplacianPrior(Generic[Array]):
    r"""BiLaplacian prior for Gaussian random field generation.

    Generates samples from a Gaussian random field by solving a
    diffusion-reaction equation with Robin boundary conditions:

    .. math::

        K u = B \xi, \qquad \xi \sim N(0, I), \qquad B B^\top = M,

    so the covariance is :math:`K^{-1} M K^{-1}`. ``K`` is minus the
    Jacobian of a composed system: a ``BiLaplacianOperator`` (the
    interior :math:`-\nabla\cdot(\gamma T\nabla u) + \delta u`) plus its
    natural boundary terms, typically Robin to damp the boundary artifact,
    composed by ``compose_galerkin_system`` like any physics. ``M`` is the
    white-noise mass, supplied by an injected ``NoiseMassProtocol``.

    Most callers use ``with_uniform_robin`` or ``from_correlation_length``,
    which build the system and default the noise mass to the row-sum
    lumped mass.

    Parameters
    ----------
    system : GalerkinSteadySystemProtocol
        The composed precision operator, declared time-invariant. It must
        have no essential (Dirichlet) constraints: only natural terms can
        enter a precision operator.
    noise_mass : NoiseMassProtocol
        The white-noise mass ``M`` and its factor ``B``, on the system's
        DOFs.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If the system has essential constraints, or the noise mass is on
        a different number of DOFs.
    """

    def __init__(
        self,
        system: GalerkinSteadySystemProtocol[Array],
        noise_mass: NoiseMassProtocol[Array],
    ):
        if not isinstance(system, GalerkinSteadySystemProtocol):
            raise TypeError(
                "system must satisfy GalerkinSteadySystemProtocol, got "
                f"{type(system).__name__}"
            )
        if not isinstance(noise_mass, NoiseMassProtocol):
            raise TypeError(
                "noise_mass must satisfy NoiseMassProtocol, got "
                f"{type(noise_mass).__name__}"
            )
        if system.constraint_set().ndofs() > 0:
            raise ValueError(
                "a precision operator takes only natural boundary terms; "
                f"the system constrains {system.constraint_set().ndofs()} "
                "DOFs"
            )
        if noise_mass.mass_matrix().shape[0] != system.nstates():
            raise ValueError(
                f"noise mass is on {noise_mass.mass_matrix().shape[0]} DOFs, "
                f"the system on {system.nstates()}"
            )
        self._system = system
        self._bkd = system.bkd()
        self._noise_mass = noise_mass
        self._stiffness: Optional[Array] = None
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
            Robin BC coefficient. Default:
            ``default_robin_coefficient(gamma, delta)``.
        noise_mass : NoiseMassProtocol, optional
            The white-noise mass. Default: the row-sum lumped mass.

        Returns
        -------
        BiLaplacianPrior[Array]
            Constructed prior.
        """
        if robin_alpha is None:
            robin_alpha = default_robin_coefficient(gamma, delta)
        boundaries = list(basis.skfem_basis().mesh.boundaries.keys())
        robin_bcs: List[BoundaryConditionRole[Array]] = [
            RobinBC(basis, name, alpha=robin_alpha, value_func=0.0, bkd=bkd)
            for name in boundaries
        ]
        operator = BiLaplacianOperator(
            basis, gamma, delta, bkd, anisotropic_tensor=anisotropic_tensor
        )
        return cls(
            compose_galerkin_system(operator, robin_bcs),
            LumpedNoiseMass(basis, bkd) if noise_mass is None else noise_mass,
        )

    @classmethod
    def from_correlation_length(
        cls,
        basis: GalerkinBasisProtocol[Array],
        length: float,
        std: float,
        bkd: Backend[Array],
        robin_alpha: Optional[float] = None,
        noise_mass: Optional[NoiseMassProtocol[Array]] = None,
    ) -> "BiLaplacianPrior[Array]":
        r"""Create an isotropic prior with a given correlation length and
        standard deviation, with uniform Robin BCs on all boundaries.

        :math:`(\gamma, \delta)` come from
        ``bilaplacian_correlation_parameters``.

        The standard deviation is the free-space value: the discrete
        field approaches it in the interior as the mesh is refined, and
        the Robin coefficient changes it near the boundary.

        Parameters
        ----------
        basis : GalerkinBasisProtocol[Array]
            Finite element basis.
        length : float
            Correlation length :math:`\ell = \sqrt{\gamma/\delta}` (the
            inverse of the Matern :math:`\kappa`, not the practical range
            :math:`\sqrt{8\nu}\,\ell`).
        std : float
            Stationary marginal standard deviation :math:`\sigma`.
        bkd : Backend[Array]
            Computational backend.
        robin_alpha : float, optional
            Robin BC coefficient. Default:
            ``default_robin_coefficient(gamma, delta)``.
        noise_mass : NoiseMassProtocol, optional
            The white-noise mass. Default: the row-sum lumped mass.

        Returns
        -------
        BiLaplacianPrior[Array]
            Constructed prior.
        """
        gamma, delta = bilaplacian_correlation_parameters(
            length, std, basis.mesh().ndim()
        )
        return cls.with_uniform_robin(
            basis,
            gamma=gamma,
            delta=delta,
            bkd=bkd,
            robin_alpha=robin_alpha,
            noise_mass=noise_mass,
        )

    def system(self) -> GalerkinSteadySystemProtocol[Array]:
        """Return the composed precision operator."""
        return self._system

    def _assemble_system(self) -> None:
        """Take ``K`` once, as minus the system's Jacobian (the system is
        linear, so any state gives it)."""
        if self._stiffness is not None:
            return
        zero_state = self._bkd.zeros((self._system.nstates(),))
        self._stiffness = -self._system.steady().steady_jacobian(zero_state)

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
        """Return the white-noise mass."""
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
        ndofs = self._system.nstates()
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
        ndofs = self._system.nstates()
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

    def __repr__(self) -> str:
        return (
            f"BiLaplacianPrior(ndofs={self._system.nstates()}, "
            f"noise_mass={type(self._noise_mass).__name__})"
        )

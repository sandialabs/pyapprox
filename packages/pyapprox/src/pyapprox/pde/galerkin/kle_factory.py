r"""SPDE-based Matern KLE factories (FEM/bilaplacian-backed).

These factories construct Karhunen-Loeve expansions via the SPDE
(Whittle-Matern) representation, assembling the sparse precision
operator with :class:`~pyapprox.pde.galerkin.bilaplacian.BiLaplacianPrior`
(Robin boundary conditions, skfem assembly). They live in
``pde.galerkin`` — not ``pde.field_maps`` — because the construction
depends on the galerkin solver's assembly and boundary-condition layers;
the general field-map layer must stay below the solvers. The *products*
(:class:`SPDEMaternKLE`, :class:`TransformedFieldMap`) are general
objects from the layers below.

Memory: the SPDE approach uses only sparse matrices and a partial
eigensolve, giving O(N) memory — prefer it over the dense kernel-based
factories in :mod:`pyapprox.pde.field_maps.kle_factory` for large
meshes.
"""

from __future__ import annotations

from typing import Optional, Union

from pyapprox.pde.field_maps.mesh_kle_field_map import (
    MeshKLEFieldMap,
)
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.bilaplacian import (
    BiLaplacianPrior,
    bilaplacian_stationary_variance,
    default_robin_coefficient,
)
from pyapprox.pde.galerkin.noise_mass import (
    ConsistentNoiseMass,
    NoiseMassProtocol,
)
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.surrogates.kle.spde_kle import SPDEMaternKLE
from pyapprox.surrogates.kle.utils import (
    adjust_sign_eig,
    sort_eigenpairs,
)
from pyapprox.util.backends.protocols import Array, Backend


def create_spde_matern_kle(
    basis: GalerkinBasisProtocol[Array],
    n_modes: int,
    gamma: float,
    delta: float,
    sigma: float,
    bkd: Backend[Array],
    xi: Optional[float] = None,
    mean_field: Union[float, Array] = 0.0,
    noise_mass: Optional[NoiseMassProtocol[Array]] = None,
) -> SPDEMaternKLE[Array]:
    r"""Create a KLE via the SPDE representation of a Matern random field.

    Uses :class:`BiLaplacianPrior` to assemble the sparse precision
    operator with Robin boundary conditions, then solves

    .. math::

        A\,\phi_k = \mu_k\,M\,\phi_k

    for the smallest eigenvalues :math:`\mu_k`.  The covariance
    :math:`A^{-1} M A^{-1} = \Phi\,\mathrm{diag}(\mu^{-2})\,\Phi^\top` has
    the stationary marginal variance :math:`v(\gamma, \delta)` of
    ``bilaplacian_stationary_variance``, so the KLE eigenvalues
    :math:`\lambda_k = 1/(v\,\mu_k^2)` are those of the unit-variance
    field. ``SPDEMaternKLE`` then scales the field by :math:`\sigma`, so
    the marginal variance is :math:`\sigma^2`. This makes the SPDE
    eigenvalues match the kernel-based unit-variance eigenvalues
    mode-by-mode (up to discretization and boundary effects).

    With the same Robin coefficient and noise mass, a full-rank KLE
    therefore has the covariance of
    ``BiLaplacianPrior.from_correlation_length(basis, sqrt(gamma/delta),
    sigma)`` exactly.

    This uses only sparse matrices and a partial eigensolve, giving
    O(N) memory instead of the O(N^2) of kernel-based methods.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        FEM basis (e.g. ``LagrangeBasis(mesh, degree=1)``).
    n_modes : int
        Number of KLE modes to compute.
    gamma : float
        Diffusion coefficient.  Controls correlation length via
        :math:`\ell_c = \sqrt{\gamma/\delta}`.
    delta : float
        Reaction coefficient.
    sigma : float
        Target marginal standard deviation.
    bkd : Backend[Array]
        Computational backend.
    xi : float, optional
        Robin BC coefficient.  Default:
        ``default_robin_coefficient(gamma, delta)``, the prior's.
    mean_field : float or Array, optional
        Mean field.  Scalar is broadcast to all nodes.  Default: 0.
    noise_mass : NoiseMassProtocol, optional
        The mass :math:`M` of the white noise, hence of the covariance
        :math:`A^{-1} M A^{-1}` and the eigenproblem. Default:
        ``ConsistentNoiseMass(basis, bkd)``. ``LumpedNoiseMass`` gives
        the KLE of the prior's default (lumped) samples; the two agree
        as the mesh is refined.

    Returns
    -------
    SPDEMaternKLE
        KLE with M-orthonormal eigenvectors and scaled eigenvalues.
    """
    if xi is None:
        xi = default_robin_coefficient(gamma, delta)
    if noise_mass is None:
        noise_mass = ConsistentNoiseMass(basis, bkd)

    # The precision operator A and the mass M, both from the prior; it
    # solves A phi = mu M phi for the n_modes smallest mu.
    prior = BiLaplacianPrior.with_uniform_robin(
        basis,
        gamma=gamma,
        delta=delta,
        bkd=bkd,
        robin_alpha=xi,
        noise_mass=noise_mass,
    )
    mu_array, phi_array = prior.generalized_eigenpairs(n_modes)
    mu_vals = bkd.to_numpy(mu_array)

    # Unit-variance eigenvalues: A^{-1} M A^{-1} divided by its stationary
    # variance. SPDEMaternKLE scales by sigma.
    variance = bilaplacian_stationary_variance(
        gamma, delta, basis.mesh().ndim()
    )
    lambda_vals = 1.0 / (variance * mu_vals**2)

    eig_vals = bkd.asarray(lambda_vals)
    eig_vecs = phi_array

    # Sort descending and fix sign convention
    eig_vals, eig_vecs, _ = sort_eigenpairs(eig_vals, eig_vecs, n_modes, bkd)
    eig_vecs = adjust_sign_eig(eig_vecs, bkd)

    return SPDEMaternKLE(
        eigenvalues=eig_vals,
        eigenvectors=eig_vecs,
        sigma=sigma,
        mean_field=mean_field,
        bkd=bkd,
        gamma=gamma,
        delta=delta,
        xi=xi,
    )


def create_spde_lognormal_kle_field_map(
    basis: GalerkinBasisProtocol[Array],
    mean_log_field: Array,
    bkd: Backend[Array],
    n_modes: int,
    gamma: float,
    delta: float,
    sigma: float,
    xi: Optional[float] = None,
    noise_mass: Optional[NoiseMassProtocol[Array]] = None,
) -> TransformedFieldMap[Array]:
    r"""Create a lognormal field map using the SPDE-based Matern KLE.

    Composes ``create_spde_matern_kle`` -> ``MeshKLEFieldMap`` ->
    ``TransformedFieldMap(exp)``.  Same pattern as
    :func:`pyapprox.pde.field_maps.kle_factory.create_lognormal_kle_field_map`
    but uses the sparse SPDE approach instead of dense kernel matrices.

    Result: ``field(x) = exp(mean_log_field(x) + W @ params)``.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        FEM basis.
    mean_log_field : Array, shape (nnodes,)
        Mean of the log-field at mesh nodes.
    bkd : Backend[Array]
        Computational backend.
    n_modes : int
        Number of KLE modes.
    gamma : float
        Diffusion coefficient.
    delta : float
        Reaction coefficient.
    sigma : float
        Standard deviation of the log-field.
    xi : float, optional
        Robin BC coefficient.  Default:
        ``default_robin_coefficient(gamma, delta)``, the prior's.
    noise_mass : NoiseMassProtocol, optional
        White-noise mass, passed to ``create_spde_matern_kle``.
        Default: ``ConsistentNoiseMass(basis, bkd)``.

    Returns
    -------
    TransformedFieldMap
        Composed field map: exp(mean_log_field + W @ params).
    """
    spde_kle = create_spde_matern_kle(
        basis,
        n_modes=n_modes,
        gamma=gamma,
        delta=delta,
        sigma=sigma,
        bkd=bkd,
        xi=xi,
        noise_mass=noise_mass,
    )

    inner = MeshKLEFieldMap(
        bkd,
        mean_log_field,
        spde_kle.weighted_eigenvectors(),
    )

    exp_transform = _ExpTransform(bkd)
    return TransformedFieldMap(
        inner,
        transform=exp_transform,
        transform_deriv=exp_transform,
        bkd=bkd,
        transform_deriv2=exp_transform,
    )

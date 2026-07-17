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

skfem imports are function-local because skfem is an optional
dependency (the convention throughout ``pde.galerkin``).
"""

from __future__ import annotations

from math import gamma as gamma_func
from typing import Optional, Union

import numpy as np
from scipy.sparse.linalg import eigsh

from pyapprox.pde.field_maps.mesh_kle_field_map import (
    MeshKLEFieldMap,
)
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.bilaplacian import BiLaplacianPrior
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.surrogates.kle.spde_kle import SPDEMaternKLE
from pyapprox.surrogates.kle.utils import (
    adjust_sign_eig,
    sort_eigenpairs,
)
from pyapprox.util.backends.protocols import Array, Backend


def _compute_spde_tau_squared(
    sigma: float,
    gamma: float,
    delta: float,
    d: int,
    alpha: int = 2,
) -> float:
    r"""Compute :math:`\tau^2` from the SPDE-Matern variance formula.

    The SPDE covariance is :math:`\Sigma = \tau^{-2} A^{-1} M A^{-1}`.
    The parameter :math:`\tau` is determined by requiring the marginal
    variance to equal :math:`\sigma^2`:

    .. math::

        \sigma^2 = \frac{\Gamma(\nu)}
                        {\Gamma(\nu + d/2)\,(4\pi)^{d/2}\,
                         \kappa^{2\nu}\,\tau^2}

    where :math:`\kappa = \sqrt{\delta/\gamma}` and
    :math:`\nu = \alpha - d/2`.

    Parameters
    ----------
    sigma : float
        Target marginal standard deviation.
    gamma : float
        Diffusion coefficient.
    delta : float
        Reaction coefficient.
    d : int
        Spatial dimension.
    alpha : int
        SPDE order.  Default: 2 (bilaplacian).

    Returns
    -------
    float
        :math:`\tau^2`.
    """
    nu = alpha - d / 2.0
    kappa = np.sqrt(delta / gamma)
    tau_sq = gamma_func(nu) / (
        gamma_func(nu + d / 2.0)
        * (4 * np.pi) ** (d / 2.0)
        * kappa ** (2 * nu)
        * sigma**2
    )
    return float(tau_sq)


def create_spde_matern_kle(
    basis: GalerkinBasisProtocol[Array],
    n_modes: int,
    gamma: float,
    delta: float,
    sigma: float,
    bkd: Backend[Array],
    xi: Optional[float] = None,
    mean_field: Union[float, Array] = 0.0,
) -> SPDEMaternKLE[Array]:
    r"""Create a KLE via the SPDE representation of a Matern random field.

    Uses :class:`BiLaplacianPrior` to assemble the sparse precision
    operator with Robin boundary conditions, then solves

    .. math::

        A\,\phi_k = \mu_k\,M\,\phi_k

    for the smallest eigenvalues :math:`\mu_k`.  The KLE eigenvalues are
    :math:`\lambda_k = \gamma^2/(\tau^2 \mu_k^2)` (the :math:`\gamma^2`
    arises because :math:`A = \gamma L_h`), where :math:`\tau` is computed
    analytically from the SPDE-Matern variance formula:

    .. math::

        \sigma^2 = \frac{\Gamma(\nu)}
                        {\Gamma(\nu + d/2)\,(4\pi)^{d/2}\,
                         \kappa^{2\nu}\,\tau^2}

    with :math:`\kappa = \sqrt{\delta/\gamma}` and
    :math:`\nu = \alpha - d/2`.  This ensures the SPDE eigenvalues
    match the kernel-based eigenvalues mode-by-mode (up to
    discretization and boundary effects).

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
        Robin BC coefficient.  Default: ``sqrt(gamma * delta)``.
    mean_field : float or Array, optional
        Mean field.  Scalar is broadcast to all nodes.  Default: 0.

    Returns
    -------
    SPDEMaternKLE
        KLE with M-orthonormal eigenvectors and scaled eigenvalues.
    """
    from skfem import asm
    from skfem.models.poisson import mass

    if xi is None:
        xi = np.sqrt(gamma * delta)

    # Use BiLaplacianPrior to assemble the precision operator A
    prior = BiLaplacianPrior.with_uniform_robin(
        basis,
        gamma=gamma,
        delta=delta,
        bkd=bkd,
        robin_alpha=xi,
    )
    A = prior.stiffness_matrix()

    # Assemble consistent mass matrix M
    M = asm(mass, basis.skfem_basis())

    # Solve generalized eigenvalue problem A phi = mu M phi
    # for the n_modes smallest eigenvalues (shift-invert with sigma=0)
    mu_vals, phi_vecs = eigsh(A, k=n_modes, M=M, sigma=0.0, which="LM")

    # Compute tau^2 analytically from the SPDE-Matern variance formula
    d = basis.mesh().ndim()
    tau_sq = _compute_spde_tau_squared(sigma, gamma, delta, d)

    # KLE eigenvalues: lambda_k = gamma^2 / (tau^2 * mu_k^2)
    # The gamma^2 factor arises because A = gamma * L_h where L_h is the
    # SPDE operator, so A^{-1} = (1/gamma) * L_h^{-1} and the covariance
    # C = tau^{-2} L_h^{-1} M L_h^{-1} = tau^{-2} gamma^2 A^{-1} M A^{-1}
    lambda_vals = gamma**2 / (tau_sq * mu_vals**2)

    # Convert to backend arrays
    eig_vals = bkd.asarray(lambda_vals)
    eig_vecs = bkd.asarray(phi_vecs)

    # Sort descending and fix sign convention
    eig_vals, eig_vecs = sort_eigenpairs(eig_vals, eig_vecs, n_modes, bkd)
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
        Robin BC coefficient.  Default: ``sqrt(gamma * delta)``.

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

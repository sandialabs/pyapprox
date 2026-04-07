"""Factory for lognormal KLE field maps and FEM-aware KLE construction.

Memory considerations for large meshes
---------------------------------------
The Nystrom KLE methods build a dense kernel matrix K(x_i, x_j) whose
size scales as O(N^2) where N is the number of collocation points.
For FEM meshes with many elements and multiple quadrature points per
element, this can quickly become prohibitive:

- ``create_fem_nystrom_quadrature_kle``: N = nelems * nquad.
  A mesh with 10K elements and 4 quad points yields a 40K x 40K
  matrix (~12 GB in float64). Avoid for large meshes.

- ``create_fem_nystrom_nodes_kle``: N = nnodes.
  Same mesh typically has ~10K nodes, yielding a 10K x 10K matrix
  (~800 MB). Usually manageable.

- ``create_fem_galerkin_kle``: projects through FEM basis functions,
  so the eigensolve operates on an ndofs x ndofs covariance matrix.
  However, the kernel is still evaluated at all quadrature points
  during projection, so peak memory is similar to the quadrature
  Nystrom variant.

- ``create_spde_matern_kle``: uses the SPDE representation of the
  Matern field, solving a sparse eigenvalue problem.  Memory is
  O(N) with sparse matrices.  Best for large meshes.

For large meshes, prefer ``create_spde_matern_kle`` (Matern fields)
or ``create_fem_nystrom_nodes_kle`` (arbitrary kernels).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy.typing as npt

if TYPE_CHECKING:
    from skfem.assembly.basis.cell_basis import CellBasis

    from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol

import numpy as np

from pyapprox.pde.field_maps.mesh_kle_field_map import (
    MeshKLEFieldMap,
)
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
)
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.surrogates.kle.galerkin_kle import GalerkinKLE
from pyapprox.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.surrogates.kle.spde_kle import SPDEMaternKLE
from pyapprox.util.backends.protocols import Array, Backend


def create_lognormal_kle_field_map(
    mesh_coords: Array,
    mean_log_field: Array,
    bkd: Backend[Array],
    kernel: Optional[KernelProtocol[Array]] = None,
    correlation_length: float = 0.3,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    quad_weights: Optional[Array] = None,
) -> TransformedFieldMap[Array]:
    """Create a lognormal KLE field map: MeshKLE -> MeshKLEFieldMap -> exp.

    Convention: mean_log_field is mu_ln(x) (mean of the log-field).
    Result: field(x) = exp(mu_ln(x) + sigma * sum_k sqrt(lam_k)*phi_k(x)*xi_k).

    Parameters
    ----------
    mesh_coords : Array, shape (nphys_vars, ncoords)
        Spatial coordinates of the mesh points.
    mean_log_field : Array, shape (ncoords,)
        Mean of the log-field at mesh nodes.
    bkd : Backend
        Computational backend.
    kernel : KernelProtocol, optional
        Covariance kernel. If None, uses SquaredExponentialKernel with
        isotropic correlation_length.
    correlation_length : float
        Isotropic correlation length (used only if kernel is None).
    num_kle_terms : int
        Number of KLE terms.
    sigma : float
        Standard deviation of the log-field.
    quad_weights : Array, shape (ncoords,), optional
        Quadrature weights for weighted eigendecomposition. When
        mesh_coords are element centroids, use element areas/volumes.
        If None, uses unweighted eigendecomposition.

    Returns
    -------
    TransformedFieldMap
        Composed field map: exp(mean_log_field + W @ params).
    """
    ndim = mesh_coords.shape[0]
    if kernel is None:
        lenscale = bkd.full((ndim,), correlation_length)
        kernel = SquaredExponentialKernel(
            lenscale,
            (0.01, 10.0),
            ndim,
            bkd,
        )

    mesh_kle = MeshKLE(
        mesh_coords,
        kernel,
        sigma=sigma,
        mean_field=0.0,
        nterms=num_kle_terms,
        quad_weights=quad_weights,
        bkd=bkd,
    )

    inner = MeshKLEFieldMap(bkd, mean_log_field, mesh_kle.weighted_eigenvectors())

    return TransformedFieldMap(
        inner,
        transform=lambda x: bkd.exp(x),
        transform_deriv=lambda x: bkd.exp(x),
        bkd=bkd,
        transform_deriv2=lambda x: bkd.exp(x),
    )


def _build_phi_matrix(skfem_basis: CellBasis) -> npt.NDArray[np.floating[Any]]:
    """Build global shape-function matrix at quadrature points.

    Parameters
    ----------
    skfem_basis : skfem CellBasis
        Scalar FEM basis (not vector).

    Returns
    -------
    Phi : np.ndarray, shape (nelems * nquad, ndofs)
        Phi[e*nquad + q, j] = value of global shape function j at
        quadrature point q in element e.
    """
    nelems = skfem_basis.mesh.nelements
    nquad = skfem_basis.dx.shape[1]
    ndofs = skfem_basis.N
    ndofs_per_elem = skfem_basis.element_dofs.shape[0]

    Phi = np.zeros((nelems * nquad, ndofs))
    for k in range(ndofs_per_elem):
        vals = skfem_basis.basis[k][0]  # (nelems, nquad)
        cols = skfem_basis.element_dofs[k, :]  # (nelems,)
        rows = np.arange(nelems)[:, None] * nquad + np.arange(nquad)[None, :]
        Phi[rows.ravel(), np.repeat(cols, nquad)] += vals.ravel()
    return Phi


def create_fem_galerkin_kle(
    skfem_basis: CellBasis,
    kernel: KernelProtocol[Array],
    nterms: int,
    sigma: float,
    bkd: Backend[Array],
) -> GalerkinKLE[Array]:
    """Create a KLE via Galerkin projection: C_h Phi = M Phi Lambda.

    Assembles the covariance matrix C_h by evaluating the kernel at
    quadrature points and projecting through the FEM basis:
        C_h = Phi^T diag(dx) K_q diag(dx) Phi
    where Phi[q,j] = shape function j at quadrature point q.

    Eigenvectors are nodal FEM coefficients. Evaluation at arbitrary
    points requires FEM interpolation.

    Parameters
    ----------
    skfem_basis : skfem CellBasis
        Scalar FEM basis (e.g. ``Basis(mesh, ElementQuad1())``).
    kernel : KernelProtocol
        Covariance kernel.
    nterms : int
        Number of KLE terms.
    sigma : float
        Standard deviation scaling factor.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    GalerkinKLE
        KLE with nodal eigenvectors.
    """
    from skfem import asm
    from skfem.models.poisson import mass

    M = asm(mass, skfem_basis)

    coords = skfem_basis.mapping.F(skfem_basis.X)  # (ndim, nelems, nquad)
    ndim, nelems, nquad = coords.shape
    coords_flat = coords.reshape(ndim, -1)  # (ndim, nelems*nquad)
    dx_flat = skfem_basis.dx.ravel()  # (nelems*nquad,)

    K_q = bkd.to_numpy(kernel(bkd.asarray(coords_flat), bkd.asarray(coords_flat)))

    Phi = _build_phi_matrix(skfem_basis)

    # C_h = Phi^T @ diag(dx) @ K_q @ diag(dx) @ Phi
    dx_K = dx_flat[:, None] * K_q * dx_flat[None, :]
    C_h = Phi.T @ dx_K @ Phi

    return GalerkinKLE(
        cov_matrix=bkd.asarray(C_h),
        mass_matrix=bkd.asarray(M.toarray()),
        nterms=nterms,
        sigma=sigma,
        bkd=bkd,
    )


def create_fem_nystrom_nodes_kle(
    skfem_basis: CellBasis,
    kernel: KernelProtocol[Array],
    nterms: int,
    sigma: float,
    bkd: Backend[Array],
) -> MeshKLE[Array]:
    """Create a KLE via Nystrom method at FEM nodes with lumped mass weights.

    Uses the mesh nodal coordinates as collocation points and lumped mass
    matrix row sums as quadrature weights for the Nystrom method.

    Parameters
    ----------
    skfem_basis : skfem CellBasis
        Scalar FEM basis.
    kernel : KernelProtocol
        Covariance kernel.
    nterms : int
        Number of KLE terms.
    sigma : float
        Standard deviation scaling factor.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    MeshKLE
        KLE with nodal eigenvectors.
    """
    from skfem import asm
    from skfem.models.poisson import mass

    M = asm(mass, skfem_basis)
    w = np.array(M.sum(axis=1)).ravel()

    mesh_coords = skfem_basis.mesh.p  # (ndim, nnodes)

    return MeshKLE(
        bkd.asarray(mesh_coords),
        kernel,
        sigma=sigma,
        mean_field=0.0,
        nterms=nterms,
        quad_weights=bkd.asarray(w),
        bkd=bkd,
    )


def create_fem_nystrom_quadrature_kle(
    skfem_basis: CellBasis,
    kernel: KernelProtocol[Array],
    nterms: int,
    sigma: float,
    bkd: Backend[Array],
) -> MeshKLE[Array]:
    """Create a KLE via Nystrom method at Gauss quadrature points.

    Uses the FEM quadrature points as collocation points and the
    integration weights ``basis.dx`` as quadrature weights.

    Eigenvectors have shape ``(nelems * nquad, nterms)``. For use
    with FEM physics, reshape to ``(nelems, nquad, nterms)`` or
    ``(nelems, nquad)`` per sample.

    Parameters
    ----------
    skfem_basis : skfem CellBasis
        Scalar FEM basis.
    kernel : KernelProtocol
        Covariance kernel.
    nterms : int
        Number of KLE terms.
    sigma : float
        Standard deviation scaling factor.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    MeshKLE
        KLE at quadrature points.
    """
    coords = skfem_basis.mapping.F(skfem_basis.X)  # (ndim, nelems, nquad)
    ndim = coords.shape[0]
    coords_flat = coords.reshape(ndim, -1)  # (ndim, nelems*nquad)
    weights_flat = skfem_basis.dx.ravel()  # (nelems*nquad,)

    return MeshKLE(
        bkd.asarray(coords_flat),
        kernel,
        sigma=sigma,
        mean_field=0.0,
        nterms=nterms,
        quad_weights=bkd.asarray(weights_flat),
        bkd=bkd,
    )


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
    from math import gamma as gamma_func

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
    from scipy.sparse.linalg import eigsh
    from skfem import asm
    from skfem.models.poisson import mass

    from pyapprox.pde.galerkin.bilaplacian import (
        BiLaplacianPrior,
    )
    from pyapprox.surrogates.kle.utils import (
        adjust_sign_eig,
        sort_eigenpairs,
    )

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
    :func:`create_lognormal_kle_field_map` but uses the sparse SPDE
    approach instead of dense kernel matrices.

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

    return TransformedFieldMap(
        inner,
        transform=lambda x: bkd.exp(x),
        transform_deriv=lambda x: bkd.exp(x),
        bkd=bkd,
        transform_deriv2=lambda x: bkd.exp(x),
    )

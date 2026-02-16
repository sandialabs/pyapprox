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

For large meshes, prefer ``create_fem_nystrom_nodes_kle`` or build
the KLE on element centroids with element-area weights via
``create_lognormal_kle_field_map``.
"""

from typing import Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.kernels.protocols import KernelProtocol
from pyapprox.typing.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.typing.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.typing.surrogates.kle.galerkin_kle import GalerkinKLE
from pyapprox.typing.pde.field_maps.mesh_kle_field_map import (
    MeshKLEFieldMap,
)
from pyapprox.typing.pde.field_maps.transformed import (
    TransformedFieldMap,
)


def create_lognormal_kle_field_map(
    mesh_coords: Array,
    mean_log_field: Array,
    bkd: Backend[Array],
    kernel: Optional[KernelProtocol] = None,
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
            lenscale, (0.01, 10.0), ndim, bkd,
        )

    mesh_kle = MeshKLE(
        mesh_coords, kernel,
        sigma=sigma, mean_field=0.0,
        nterms=num_kle_terms, quad_weights=quad_weights, bkd=bkd,
    )

    inner = MeshKLEFieldMap(bkd, mean_log_field, mesh_kle.weighted_eigenvectors())

    return TransformedFieldMap(
        inner,
        transform=lambda x: bkd.exp(x),
        transform_deriv=lambda x: bkd.exp(x),
        bkd=bkd,
        transform_deriv2=lambda x: bkd.exp(x),
    )


def _build_phi_matrix(skfem_basis) -> np.ndarray:
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
        rows = (
            np.arange(nelems)[:, None] * nquad
            + np.arange(nquad)[None, :]
        )
        Phi[rows.ravel(), np.repeat(cols, nquad)] += vals.ravel()
    return Phi


def create_fem_galerkin_kle(
    skfem_basis,
    kernel: KernelProtocol,
    nterms: int,
    sigma: float,
    bkd: Backend[Array],
) -> GalerkinKLE:
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
    skfem_basis,
    kernel: KernelProtocol,
    nterms: int,
    sigma: float,
    bkd: Backend[Array],
) -> MeshKLE:
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
        bkd.asarray(mesh_coords), kernel,
        sigma=sigma, mean_field=0.0,
        nterms=nterms, quad_weights=bkd.asarray(w), bkd=bkd,
    )


def create_fem_nystrom_quadrature_kle(
    skfem_basis,
    kernel: KernelProtocol,
    nterms: int,
    sigma: float,
    bkd: Backend[Array],
) -> MeshKLE:
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
        bkd.asarray(coords_flat), kernel,
        sigma=sigma, mean_field=0.0,
        nterms=nterms, quad_weights=bkd.asarray(weights_flat), bkd=bkd,
    )

"""Factory for lognormal KLE field maps."""

from typing import Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.kernels.protocols import KernelProtocol
from pyapprox.typing.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.typing.surrogates.kle.mesh_kle import MeshKLE
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
        nterms=num_kle_terms, bkd=bkd,
    )

    inner = MeshKLEFieldMap(bkd, mean_log_field, mesh_kle.weighted_eigenvectors())

    return TransformedFieldMap(
        inner,
        transform=lambda x: bkd.exp(x),
        transform_deriv=lambda x: bkd.exp(x),
        bkd=bkd,
        transform_deriv2=lambda x: bkd.exp(x),
    )

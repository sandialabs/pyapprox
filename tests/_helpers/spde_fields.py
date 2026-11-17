"""SPDE Matern random fields on a Galerkin basis, for tests.

Composes three pieces that live in different packages, none of which
imports another: the bi-Laplacian precision and its generalized
eigenpairs (``pde.galerkin.bilaplacian``), the KLE built from those
eigenpairs (``surrogates.kle.spde_kle``), and the lognormal field map
(``pde.field_maps``). :func:`spde_matern_kle` returns the KLE;
:func:`spde_lognormal_field_map` its exponentiated field map.

With the same Robin coefficient and noise mass, a full-rank KLE has the
covariance of ``BiLaplacianPrior.from_correlation_length(basis,
sqrt(gamma / delta), sigma)`` exactly.

Requires skfem; import only after skipping when it is absent.
"""

from typing import Optional, Union

from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    exponentiated_field_map,
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
from pyapprox.surrogates.kle.spde_kle import (
    SPDEMaternKLE,
    spde_matern_kle_from_precision_eigenpairs,
)
from pyapprox.util.backends.protocols import Array, Backend


def spde_matern_kle(
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
    r"""The ``n_modes``-term KLE of the bi-Laplacian Matern field.

    Solves :math:`A\phi_k = \mu_k M \phi_k` for the smallest
    :math:`\mu_k`, with :math:`A` the prior's precision (Robin
    coefficient ``xi``, default ``default_robin_coefficient(gamma,
    delta)``) and :math:`M` the white-noise mass (default
    ``ConsistentNoiseMass(basis, bkd)``; ``LumpedNoiseMass`` gives the KLE
    of the prior's default samples).
    """
    if xi is None:
        xi = default_robin_coefficient(gamma, delta)
    if noise_mass is None:
        noise_mass = ConsistentNoiseMass(basis, bkd)
    prior = BiLaplacianPrior.with_uniform_robin(
        basis,
        gamma=gamma,
        delta=delta,
        bkd=bkd,
        robin_alpha=xi,
        noise_mass=noise_mass,
    )
    mu, phi = prior.generalized_eigenpairs(n_modes)
    return spde_matern_kle_from_precision_eigenpairs(
        mu,
        phi,
        bilaplacian_stationary_variance(gamma, delta, basis.mesh().ndim()),
        sigma=sigma,
        bkd=bkd,
        gamma=gamma,
        delta=delta,
        xi=xi,
        mean_field=mean_field,
    )


def spde_lognormal_field_map(
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
    """The field map ``exp(mean_log_field + W @ params)``, with ``W`` the
    weighted eigenvectors of :func:`spde_matern_kle`."""
    kle = spde_matern_kle(
        basis,
        n_modes=n_modes,
        gamma=gamma,
        delta=delta,
        sigma=sigma,
        bkd=bkd,
        xi=xi,
        noise_mass=noise_mass,
    )
    inner = MeshKLEFieldMap(bkd, mean_log_field, kle.weighted_eigenvectors())
    return exponentiated_field_map(inner, bkd)

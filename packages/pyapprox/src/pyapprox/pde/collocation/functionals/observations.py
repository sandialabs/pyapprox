r"""Linear observations of a 1D collocation state.

A point value off the nodes and a subdomain integral are each a weight
column :math:`w` over the nodes, so the observation is :math:`H = W^\top`
for the stacked columns. Each factory returns a
:class:`~pyapprox.implicitfunction.observation_terms.RowWeights` reading
every node, which satisfies
:class:`~pyapprox.implicitfunction.observation_terms.StateObservationProtocol`;
the generic terms then give the functionals: ``ObservedValue(H)`` or
``ObservationMisfit(bkd, H, data)`` of a steady state, ``EndpointTerm(H)``
or ``MisfitTerm`` of a trajectory.
"""

from typing import Optional, Sequence, Tuple

from pyapprox.implicitfunction.observation_terms import RowWeights
from pyapprox.pde.collocation.basis.chebyshev.basis_1d import (
    ChebyshevBasis1D,
)
from pyapprox.pde.collocation.functionals.point_evaluation import (
    point_interpolation_weights,
)
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet, backend_dtype_name
from pyapprox.util.linalg.operators import SpaceDescriptor


def _every_node(bkd: Backend[Array], npts: int) -> BackendRowSet[Array]:
    """The row set of every node of a scalar state of ``npts`` nodes."""
    descriptor = SpaceDescriptor("state", npts, backend_dtype_name(bkd))
    return BackendRowSet(descriptor, range(npts), bkd)


def point_observation(
    basis: ChebyshevBasis1D[Array], points: Sequence[float], bkd: Backend[Array]
) -> RowWeights[Array]:
    r"""The observation of a scalar state's interpolant at ``points``:
    entry :math:`k` is :math:`u(x_k)`.

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    points : sequence of float
        The points :math:`x_k`, in physical coordinates, each in the
        domain.
    bkd : Backend
        Computational backend.

    Raises
    ------
    ValueError
        If ``points`` is empty or a point lies outside the domain.
    """
    weights = point_interpolation_weights(basis, points, bkd)
    return RowWeights(bkd, _every_node(bkd, basis.npts()), weights)


def subdomain_integral_observation(
    basis: ChebyshevBasis1D[Array],
    bkd: Backend[Array],
    coefficient: Optional[Array] = None,
    bounds: Optional[Tuple[float, float]] = None,
) -> RowWeights[Array]:
    r"""The observation :math:`\int_a^b c(x)\, u(x)\, dx` of a scalar state,
    one entry, by the collocation quadrature (exact for :math:`c u` of
    degree below the number of nodes).

    Parameters
    ----------
    basis : ChebyshevBasis1D
        The 1D Chebyshev collocation basis.
    bkd : Backend
        Computational backend.
    coefficient : Array, optional
        :math:`c` at the nodes, shape ``(npts,)``; default 1.
    bounds : tuple of float, optional
        :math:`(a, b)` in physical coordinates, :math:`a < b`; default the
        whole domain.

    Raises
    ------
    ValueError
        If ``coefficient`` is not of shape ``(npts,)`` or ``bounds`` are not
        increasing.
    """
    npts = basis.npts()
    quadrature = CollocationQuadrature1D(basis, bkd)
    if bounds is None:
        weights = quadrature.full_domain_weights()
    else:
        lower, upper = bounds
        if not lower < upper:
            raise ValueError(f"bounds must be increasing, got {bounds}")
        weights = quadrature.weights(lower, upper)
    if coefficient is not None:
        if tuple(coefficient.shape) != (npts,):
            raise ValueError(
                f"coefficient must have shape ({npts},), got "
                f"{tuple(coefficient.shape)}"
            )
        weights = weights * coefficient
    return RowWeights(bkd, _every_node(bkd, npts), bkd.reshape(weights, (npts, 1)))

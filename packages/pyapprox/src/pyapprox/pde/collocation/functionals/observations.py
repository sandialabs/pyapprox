r"""Linear observations of a collocation state.

A point value off the nodes, a subdomain integral and an average radial
displacement are each a weight
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


def _every_node(bkd: Backend[Array], nstates: int) -> BackendRowSet[Array]:
    """The row set of every entry of a state of ``nstates`` entries."""
    descriptor = SpaceDescriptor("state", nstates, backend_dtype_name(bkd))
    return BackendRowSet(descriptor, range(nstates), bkd)


def _selection(
    bkd: Backend[Array], indices: Sequence[int], offset: int, nstates: int
) -> Array:
    """The ``(nstates, len(indices))`` 0/1 matrix whose column ``k``
    selects entry ``indices[k] + offset``, built from plain integers
    (constant data)."""
    return bkd.asarray(
        [
            [1.0 if row == index + offset else 0.0 for index in indices]
            for row in range(nstates)
        ]
    )


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


def average_radial_displacement_observation(
    bkd: Backend[Array],
    mesh_indices: Sequence[int],
    cos_theta: Array,
    sin_theta: Array,
    npts: int,
) -> RowWeights[Array]:
    r"""The average radial displacement
    :math:`\frac{1}{n} \sum_k (u_{x,k} \cos\theta_k + u_{y,k} \sin\theta_k)`
    over ``n`` nodes of a component-stacked 2D displacement
    :math:`[u_x; u_y]` (``2 * npts`` entries).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_indices : sequence of int
        The nodes, each in ``[0, npts)``.
    cos_theta, sin_theta : Array
        :math:`\cos\theta_k` and :math:`\sin\theta_k`, each shape ``(n,)``.
    npts : int
        The number of mesh nodes.

    Raises
    ------
    ValueError
        If ``mesh_indices`` is empty or out of range, or an angle array is
        not of shape ``(n,)``.
    """
    indices = [int(index) for index in mesh_indices]
    if not indices:
        raise ValueError("an average needs at least one node")
    if any(index < 0 or index >= npts for index in indices):
        raise ValueError(f"node indices {indices} are not all in [0, {npts})")
    count = len(indices)
    for name, angles in (("cos_theta", cos_theta), ("sin_theta", sin_theta)):
        if tuple(angles.shape) != (count,):
            raise ValueError(
                f"{name} must have shape ({count},), got {tuple(angles.shape)}"
            )
    nstates = 2 * npts
    weights = (
        _selection(bkd, indices, 0, nstates) @ cos_theta
        + _selection(bkd, indices, npts, nstates) @ sin_theta
    ) / count
    return RowWeights(
        bkd, _every_node(bkd, nstates), bkd.reshape(weights, (nstates, 1))
    )

r"""Flux moments of pointwise-flux models, on NumPy and Torch.

Checks: a flux in the trace space has moments :math:`M_p q`, against the
closed-form linear-hat mass; k = 2 takes each component's moments
separately; a point trace's moments are the values themselves; a 1D
chain with one closed-form piece behind :class:`PointwiseFluxResponse`
matches the exact interface values; the response's tangent is the
model's, and absent when the model has none; a torch graph through the
values survives.
"""

from typing import Any, Generic, Sequence

import numpy as np
import pytest
import torch

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.coupling.analytic import KirchhoffResponse1D
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.pointwise_flux import PointwiseFluxResponse, flux_moments
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.responses import (
    LocalResponseProtocol,
    PortQuantity,
    ResponseDerivatives,
    ResponsePointProtocol,
    TimeWindow,
)
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    DenseSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.coupling.traces import LinearSegmentTrace, PointPortTrace
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.rootfinding.nonlinear import Newton

# Port order deliberately not the order along the curve.
_YS = [0.71, 0.0, 0.4, 1.0, 0.13, 0.9, 0.55]
_STEADY = TimeWindow.steady()


def _trace(bkd: Backend[Array]) -> LinearSegmentTrace[Array]:
    y = np.asarray(_YS)
    points = bkd.asarray(np.vstack([np.full(y.shape, 0.5), y]))
    return LinearSegmentTrace(points, [int(i) for i in np.argsort(y)], 3, bkd)


def _hat_mass() -> Any:
    """The linear-hat mass on ``_YS``, from h/6 [[2, 1], [1, 2]] per
    segment."""
    order = np.argsort(_YS)
    mass = np.zeros((len(_YS), len(_YS)))
    for a, b in zip(order[:-1], order[1:]):
        h = abs(_YS[b] - _YS[a])
        mass[np.ix_([a, b], [a, b])] += h / 6.0 * np.asarray([[2.0, 1.0], [1.0, 2.0]])
    return mass


def _in_trace_space(trace: LinearSegmentTrace[Array], coefs: Array) -> Array:
    """The function with nodal values ``coefs`` (nnodes, ncols) at the
    trace's points, shape (npoints, ncols)."""
    bkd = trace.bkd()
    nodes = trace.element_nodes()
    local = bkd.stack([coefs[nodes[:, a]] for a in range(nodes.shape[1])], axis=1)
    values = bkd.einsum("eqa,eaj->eqj", trace.values(), local)
    return bkd.reshape(values, (-1, int(coefs.shape[1])))


def test_a_flux_in_the_trace_space_has_its_mass_moments(bkd: Backend[Array]) -> None:
    trace = _trace(bkd)
    coefs = bkd.asarray(np.random.default_rng(0).normal(size=(len(_YS), 2)))
    moments = flux_moments(trace, _in_trace_space(trace, coefs))
    bkd.assert_allclose(moments, bkd.asarray(_hat_mass()) @ coefs, atol=1e-14)


def test_components_take_their_moments_separately(bkd: Backend[Array]) -> None:
    trace = _trace(bkd)
    npoints = int(trace.points().shape[1])
    values = bkd.asarray(np.random.default_rng(1).normal(size=(2 * npoints, 3)))
    both = flux_moments(trace, values, 2)
    for c in range(2):
        bkd.assert_allclose(
            both[c::2], flux_moments(trace, values[c::2]), rtol=1e-13, atol=1e-15
        )


def test_a_point_trace_moments_are_the_values(bkd: Backend[Array]) -> None:
    trace = PointPortTrace(bkd.asarray([[0.25, 0.5, 0.75]]), bkd)
    values = bkd.asarray(np.random.default_rng(2).normal(size=(3, 2)))
    bkd.assert_allclose(flux_moments(trace, values), values, rtol=1e-15)


def _exact_piecewise_constant(conductivities: Sequence[float], f: float) -> list[float]:
    """Interface values of -(k u')' = f on [0, 1], u(0) = u(1) = 0, with
    constant k_i on [i/n, (i+1)/n], from q' = f alone (q = -k u')."""
    n = len(conductivities)
    ends = [(i / n, (i + 1) / n) for i in range(n)]
    q0 = -sum(f * (b * b - a * a) / (2 * k) for (a, b), k in zip(ends, conductivities))
    q0 /= sum((b - a) / k for (a, b), k in zip(ends, conductivities))
    values, u = [], 0.0
    for (a, b), k in zip(ends[:-1], conductivities[:-1]):
        u -= (q0 * (b - a) + f * (b * b - a * a) / 2) / k
        values.append(u)
    return values


def test_chain_with_a_pointwise_flux_piece_matches_the_exact_values(
    bkd: Backend[Array],
) -> None:
    conductivities, f = [2.0, 0.5, 3.0, 1.0], 1.7
    n = len(conductivities)
    topo = TensorBoxPartition((n,))
    bases = UniformBases(lobatto_lagrange_basis(0, 1, bkd))
    skeleton = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    space = BackendVectorSpace(skeleton.descriptor(), bkd)
    subdomains = []
    for s, k in enumerate(conductivities):
        left = 0.0 if s == 0 else None
        right = 0.0 if s == n - 1 else None
        piece: LocalResponseProtocol[Array] = KirchhoffResponse1D(
            1.0 / n, f, (k,), left, right, bkd
        )
        if s == 1:
            # Its port ends, left first, are the trace's points.
            ends = bkd.asarray([[s / n, (s + 1) / n]])
            piece = PointwiseFluxResponse(piece, PointPortTrace(ends, bkd))
        rows = BackendPortRows(
            skeleton.descriptor(), subdomain_port_dofs(topo, skeleton, s), bkd
        )
        subdomains.append(CoupledSubdomain(piece, MatchingNodalTransfer(rows)))
    residual: SkeletonResidual[Array, Array, Array, DenseFormat[Array]] = (
        SkeletonResidual(space, subdomains, DenseSchur(bkd))
    )
    setups: RecomputeAlways[Array, DenseFormat[Array]] = RecomputeAlways(
        BackendDenseLU(bkd)
    )
    point = SteadySystem(residual, space, setups).solve(
        Newton(atol=1e-11, rtol=1e-11), space.zeros(1)
    )
    assert point.converged()
    exact = _exact_piecewise_constant(conductivities, f)
    bkd.assert_allclose(point.solution(), bkd.asarray([[v] for v in exact]), rtol=1e-12)


class _NoJvp(Generic[Array]):
    """A Kirchhoff piece whose points drop their trace derivatives."""

    def __init__(self, inner: KirchhoffResponse1D[Array]) -> None:
        self._inner = inner

    def bkd(self) -> Backend[Array]:
        return self._inner.bkd()

    def nport(self) -> int:
        return self._inner.nport()

    def receives(self) -> PortQuantity:
        return self._inner.receives()

    def nparams(self) -> int:
        return 0

    def nqoi(self) -> int:
        return 0

    def linearize(
        self, port_data: Array, params: Array, window: TimeWindow
    ) -> ResponsePointProtocol[Array]:
        return _Underived(self._inner.linearize(port_data, params, window))


class _Underived(Generic[Array]):
    """A point without derivatives."""

    def __init__(self, point: ResponsePointProtocol[Array]) -> None:
        self._point = point

    def output(self) -> Array:
        return self._point.output()

    def qoi(self) -> Array:
        return self._point.qoi()

    def derivatives(self) -> ResponseDerivatives[Array]:
        return ResponseDerivatives.none()


def test_tangent_follows_the_model(bkd: Backend[Array]) -> None:
    inner = KirchhoffResponse1D(0.5, 1.0, (1.0, 0.5), None, None, bkd)
    trace = PointPortTrace(bkd.asarray([[0.0, 0.5]]), bkd)
    data = bkd.asarray([[0.2], [0.4]])
    params = bkd.zeros((0, 1))
    with_jvp = PointwiseFluxResponse(inner, trace).linearize(data, params, _STEADY)
    jvp = with_jvp.derivatives().jvp_trace
    assert jvp is not None
    directions = bkd.asarray([[1.0, 0.0], [0.5, 1.0]])
    expected = inner.linearize(data, params, _STEADY).derivatives().jvp_trace
    assert expected is not None
    bkd.assert_allclose(jvp(directions), expected(directions), rtol=1e-15)
    without = PointwiseFluxResponse(_NoJvp(inner), trace).linearize(
        data, params, _STEADY
    )
    assert without.derivatives().jvp_trace is None


def test_a_graph_through_the_values_survives(torch_bkd: TorchBkd) -> None:
    trace = _trace(torch_bkd)
    npoints = int(trace.points().shape[1])
    values = torch.ones((npoints, 1), dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(flux_moments(trace, values).sum(), [values])
    # The hats sum to 1, so each point's value enters with its weight.
    torch_bkd.assert_allclose(
        grad, torch_bkd.reshape(trace.weights(), (npoints, 1)), rtol=1e-14
    )


def test_refusals(bkd: Backend[Array]) -> None:
    trace = _trace(bkd)
    npoints = int(trace.points().shape[1])
    with pytest.raises(ValueError):
        flux_moments(trace, bkd.zeros((npoints, 1)), 0)
    with pytest.raises(ValueError):
        flux_moments(trace, bkd.zeros((npoints + 1, 1)))
    inner = KirchhoffResponse1D(0.5, 1.0, (1.0,), None, None, bkd)
    with pytest.raises(ValueError):
        PointwiseFluxResponse(inner, trace)
    with pytest.raises(TypeError):
        PointwiseFluxResponse(object(), trace)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PointPortTrace(bkd.zeros((1, 0)), bkd)

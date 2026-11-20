r"""A collocation subdomain built by :func:`collocation_pinned_response`;
NumPy and Torch.

- 1D, the setup of ``test_normal_flux.py``: [0, 1], 8 nodes,
  :math:`2 u'' + 4 = 0`, u(0) = 1, the right end a port. At port value 1
  the response returns the exact outward flux 2 (unit weight at a point)
  and its state is :math:`1 + x - x^2` at the nodes, to 1e-12.
- 2D: [0, 1] x [0, 2], 6 x 7 nodes, :math:`2 \Delta u + f = 0` for
  :math:`u = 1 + x + x y + y^2`, Dirichlet on the left, bottom and top
  sides except at the port nodes, which are the whole right side. Given
  the exact trace, the response returns the side's weights times the
  exact outward flux :math:`-2(1 + y)` (1e-10), and these sum to the
  exact flux integral, -8; its trace tangent passes central differences.
  With a Gauss side rule it returns the flux tested against each node's
  Lagrange polynomial in y, against a 20-point rule (1e-10).
- A corner on two port sides (right and top) is one port node, tested on
  both: the outputs sum to the exact flux over both sides, -17, under
  either rule (1e-10).
- Refusals: no port node; a node off its side; a side given twice; a
  side repeating a node; a mesh outside the protocol.
"""

from typing import Generic, Optional

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJVPFromCallable,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform1D,
    AffineTransform2D,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.quadrature.boundary_quadrature import (
    CollocationNodeRule,
    GaussLegendreSideRule,
    SideRuleProtocol,
    boundary_quadrature_weights,
)
from pyapprox.pde.coupling.pinned_response import SteadyPinnedResponse
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.stacks.collocation import (
    PortSide,
    collocation_pinned_response,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.polynomials.univariate.lagrange import univariate_lagrange_polynomial
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton

_STEADY = TimeWindow.steady()
_D = 2.0
_LEFT, _RIGHT, _BOTTOM, _TOP = 0, 1, 2, 3


def _newton() -> Newton[Array, Array]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=30)


def _no_params(bkd: Backend[Array]) -> Array:
    return bkd.zeros((0, 1))


def _nodes(bkd: Backend[Array], indices: Array) -> list[int]:
    return [bkd.to_int(indices[k]) for k in range(indices.shape[0])]


def test_a_1d_end_returns_the_exact_flux(bkd: Backend[Array]) -> None:
    mesh = TransformedMesh1D(8, bkd, AffineTransform1D((0.0, 1.0), bkd))
    physics = AdvectionDiffusionReaction(
        ChebyshevBasis1D(mesh, bkd),
        bkd,
        diffusion=_D,
        forcing=NodalField(bkd.full((8,), 4.0)),
    )
    essential = DirichletRows(bkd, 8, 1, [DirichletPart(mesh.boundary_indices(0), 1.0)])
    response = collocation_pinned_response(
        CollocationSystem(physics, essential),
        physics,
        mesh,
        [PortSide(_RIGHT, _nodes(bkd, mesh.boundary_indices(_RIGHT)))],
        CollocationNodeRule(),
        RecomputeAlways(BackendDenseLU(bkd)),
        _newton(),
    )
    point = response.linearize(bkd.asarray([[1.0]]), _no_params(bkd), _STEADY)
    bkd.assert_allclose(point.output(), bkd.asarray([[2.0]]), rtol=1e-12)
    x = mesh.points()[0]
    bkd.assert_allclose(point.state()[:, 0], 1.0 + x - x**2, rtol=1e-12, atol=1e-13)


def _exact(points: Array) -> Array:
    x, y = points[0], points[1]
    return 1.0 + x + x * y + y**2


class _Rectangle(Generic[Array]):
    """[0, 1] x [0, 2] with the whole right side a port."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._mesh = TransformedMesh2D(
            6, 7, bkd, AffineTransform2D((0.0, 1.0, 0.0, 2.0), bkd)
        )
        npts = self._mesh.npts()
        # Laplacian of u is 2, so f = -2 D.
        self._physics = AdvectionDiffusionReaction(
            ChebyshevBasis2D(self._mesh, bkd),
            bkd,
            diffusion=_D,
            forcing=NodalField(bkd.full((npts,), -2.0 * _D)),
        )
        self._port = _nodes(bkd, self._mesh.boundary_indices(_RIGHT))

    def mesh(self) -> TransformedMesh2D[Array]:
        return self._mesh

    def physics(self) -> AdvectionDiffusionReaction[Array]:
        return self._physics

    def port(self) -> list[int]:
        return self._port

    def system(self, free: Optional[set[int]] = None) -> CollocationSystem[Array]:
        """Dirichlet on the left, bottom and top sides, except at the
        nodes ``free`` (default: the right side's)."""
        bkd = self._bkd
        fixed = sorted(
            {
                node
                for side in (_LEFT, _BOTTOM, _TOP)
                for node in _nodes(bkd, self._mesh.boundary_indices(side))
            }
            - (set(self._port) if free is None else free)
        )
        index = bkd.asarray(fixed, dtype=bkd.int64_dtype())
        part = DirichletPart(index, NodalField(_exact(self._mesh.points()[:, index])))
        return CollocationSystem(
            self._physics, DirichletRows(bkd, self._mesh.npts(), 1, [part])
        )

    def response(
        self,
        sides: list[PortSide],
        rule: Optional[SideRuleProtocol] = None,
    ) -> SteadyPinnedResponse[Array, Array, DenseFormat[Array]]:
        free = {node for side in sides for node in side.nodes} | set(self._port)
        return collocation_pinned_response(
            self.system(free),
            self._physics,
            self._mesh,
            sides,
            CollocationNodeRule() if rule is None else rule,
            RecomputeAlways(BackendDenseLU(self._bkd)),
            _newton(),
        )

    def trace(self) -> Array:
        """The exact trace at the port nodes. Shape: (nport, 1)"""
        values = _exact(self._mesh.points()[:, self._port])
        return self._bkd.reshape(values, (-1, 1))


def test_a_2d_side_returns_the_weighted_exact_flux(bkd: Backend[Array]) -> None:
    rect = _Rectangle(bkd)
    response = rect.response([PortSide(_RIGHT, rect.port())])
    point = response.linearize(rect.trace(), _no_params(bkd), _STEADY)
    y = rect.mesh().points()[1, rect.port()]
    weights = boundary_quadrature_weights(rect.mesh(), _RIGHT)
    expected = weights * (-_D * (1.0 + y))
    bkd.assert_allclose(point.output()[:, 0], expected, rtol=1e-10, atol=1e-12)
    bkd.assert_allclose(
        bkd.reshape(bkd.sum(point.output()), (1,)),
        bkd.asarray([-8.0]),
        rtol=1e-10,
    )


def test_a_gauss_rule_tests_the_flux_against_the_lagrange_polynomials(
    bkd: Backend[Array],
) -> None:
    rect = _Rectangle(bkd)
    response = rect.response([PortSide(_RIGHT, rect.port())], GaussLegendreSideRule())
    point = response.linearize(rect.trace(), _no_params(bkd), _STEADY)
    # int_0^2 -2 (1 + y) l_a(y) dy by a 20-point rule on y = 1 + t.
    t, w = GaussLegendreSideRule(13).rule(bkd.zeros((7,)), bkd)
    nodes = rect.mesh().reference_nodes_1d(1)
    expected = (w * (-_D * (2.0 + t))) @ univariate_lagrange_polynomial(nodes, t, bkd)
    bkd.assert_allclose(point.output()[:, 0], expected, rtol=1e-10, atol=1e-12)
    bkd.assert_allclose(
        bkd.reshape(bkd.sum(point.output()), (1,)), bkd.asarray([-8.0]), rtol=1e-10
    )


@pytest.mark.parametrize("rule", [CollocationNodeRule(), GaussLegendreSideRule()])
def test_a_corner_on_two_port_sides_is_tested_on_both(
    bkd: Backend[Array], rule: SideRuleProtocol
) -> None:
    """The right and top sides as ports, sharing the corner (1, 2): one
    port node there, and the outputs sum to the exact outward flux over
    both sides, -8 - 9."""
    rect = _Rectangle(bkd)
    top = _nodes(bkd, rect.mesh().boundary_indices(_TOP))
    sides = [PortSide(_RIGHT, rect.port()), PortSide(_TOP, top)]
    response = rect.response(sides, rule)
    ports = list(dict.fromkeys(rect.port() + top))
    assert response.nport() == len(ports) == len(rect.port()) + len(top) - 1
    trace = bkd.reshape(_exact(rect.mesh().points()[:, ports]), (-1, 1))
    point = response.linearize(trace, _no_params(bkd), _STEADY)
    bkd.assert_allclose(
        bkd.reshape(bkd.sum(point.output()), (1,)), bkd.asarray([-17.0]), rtol=1e-10
    )


def test_the_2d_trace_tangent_passes_finite_differences(bkd: Backend[Array]) -> None:
    rect = _Rectangle(bkd)
    response = rect.response([PortSide(_RIGHT, rect.port())])
    nport = len(rect.port())

    def fun(sample: Array) -> Array:
        return response.linearize(sample, _no_params(bkd), _STEADY).output()

    def jvp(sample: Array, direction: Array) -> Array:
        return response.linearize(sample, _no_params(bkd), _STEADY).jvp_trace(direction)

    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=nport, nvars=nport, fun=fun, jvp=jvp, bkd=bkd)
    ).check_derivatives(1.3 * rect.trace(), relative=True, central=True)
    assert bkd.to_float(bkd.min(errors[0])) < 1e-6


def test_refusals(bkd: Backend[Array]) -> None:
    rect = _Rectangle(bkd)
    with pytest.raises(ValueError, match="at least one port node"):
        rect.response([PortSide(_RIGHT, [])])
    top = _nodes(bkd, rect.mesh().boundary_indices(_TOP))
    interior = sorted(set(range(rect.mesh().npts())) - set(rect.port()) - set(top))
    with pytest.raises(ValueError, match="not on side 1"):
        rect.response([PortSide(_RIGHT, interior[:1])])
    with pytest.raises(ValueError, match="appears more than once"):
        rect.response([PortSide(_RIGHT, rect.port()), PortSide(_RIGHT, rect.port())])
    with pytest.raises(ValueError, match="repeats a node"):
        rect.response([PortSide(_RIGHT, rect.port() + rect.port()[:1])])
    with pytest.raises(TypeError, match="PortMeshProtocol"):
        collocation_pinned_response(
            rect.system(),
            rect.physics(),
            object(),  # type: ignore[arg-type]
            [PortSide(_RIGHT, rect.port())],
            CollocationNodeRule(),
            RecomputeAlways(BackendDenseLU(bkd)),
            _newton(),
        )

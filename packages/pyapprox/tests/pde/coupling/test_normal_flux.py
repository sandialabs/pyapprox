r"""A collocation subdomain as a Dirichlet-to-Neumann response whose flux
is the pointwise normal flux, integrated against the port's trace
functions; NumPy and Torch.

The subdomain is [0, 1] with 8 Chebyshev–Gauss–Lobatto nodes:
:math:`D u'' + f = 0`, D = 2, f = 4, u(0) = 1, and the right end a port.
The exact solution :math:`u = 1 + x - x^2` is a polynomial of the basis
degree, so the collocation solve reproduces it at the nodes.

- At the port value u(1) = 1 the response returns the exact outward flux
  :math:`-D u'(1) = 2` (unit weight at a point port) to 1e-12, and its
  state is u at the nodes to 1e-12.
- The trace tangent passes central finite differences (smallest relative
  error below 1e-6) at :math:`\lambda = 1.3`.
- Wrong-model control: the operator built with the inward normal returns
  -2, so the outward sign is the normal's, not the response's.
- Refusals: no parts; parts selecting a different number of nodes than
  the trace has points; an operator outside the protocol.
"""

from typing import Generic

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import FluxNormalOperator
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.mesh.transforms.affine import AffineTransform1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.coupling.normal_flux import (
    NormalFluxMoments,
    NormalFluxPart,
    selecting,
)
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import SteadyPinnedResponse
from pyapprox.pde.coupling.ports import BackendPortRows
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.traces import LumpedNodalTrace
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
    backend_dtype_name,
)
from pyapprox.util.linalg.operators import SpaceDescriptor
from pyapprox.util.rootfinding.nonlinear import Newton

_N = 8
_D = 2.0
_STEADY = TimeWindow(0.0, 0.0)


def _exact(x: Array) -> Array:
    return 1.0 + x - x**2


class _Piece(Generic[Array]):
    """The subdomain [0, 1], its physics and its port at x = 1."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._mesh = TransformedMesh1D(_N, bkd, AffineTransform1D((0.0, 1.0), bkd))
        basis = ChebyshevBasis1D(self._mesh, bkd)
        self._physics = AdvectionDiffusionReaction(
            basis, bkd, diffusion=_D, forcing=NodalField(bkd.full((_N,), 4.0))
        )
        self._right = self._mesh.boundary_indices(1)
        self._port = [int(self._right[0])]

    def points(self) -> Array:
        return self._mesh.points()

    def operator(self, normal: float) -> FluxNormalOperator[Array]:
        normals = self._bkd.asarray([[normal]])
        return FluxNormalOperator(self._bkd, self._right, normals, self._physics)

    def flux(self, operator: FluxNormalOperator[Array]) -> NormalFluxMoments[Array]:
        bkd = self._bkd
        trace = LumpedNodalTrace(self.points()[:, self._port], bkd.ones((1,)), bkd)
        return NormalFluxMoments(
            [NormalFluxPart([operator], selecting([0], 1, bkd))], trace, 0.0
        )

    def response(
        self, normal: float = 1.0
    ) -> SteadyPinnedResponse[Array, Array, DenseFormat[Array]]:
        bkd = self._bkd
        left = self._mesh.boundary_indices(0)
        essential = DirichletRows(bkd, _N, 1, [DirichletPart(left, 1.0)])
        pinned = collocation_steady_residual(
            CollocationSystem(self._physics, essential)
        )
        states = backend_state_space(bkd, _N)
        theta = BackendVectorSpace(
            SpaceDescriptor("params", 1, backend_dtype_name(bkd)), bkd
        )
        ports: PortPinnedData[Array, Array] = PortPinnedData(
            BackendRowSet(states.descriptor(), self._port, bkd),
            BackendPortRows(states.descriptor(), self._port, bkd),
            BackendPortRows(theta.descriptor(), [0], bkd),
            states,
            theta,
        )
        newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        return SteadyPinnedResponse(
            pinned,
            ports,
            self.flux(self.operator(normal)),
            states,
            RecomputeAlways(BackendDenseLU(bkd)),
            newton,
            bkd,
        )


def _no_params(bkd: Backend[Array]) -> Array:
    return bkd.zeros((0, 1))


def test_returns_the_exact_outward_flux(bkd: Backend[Array]) -> None:
    piece = _Piece(bkd)
    point = piece.response().linearize(bkd.asarray([[1.0]]), _no_params(bkd), _STEADY)
    bkd.assert_allclose(point.output(), bkd.asarray([[2.0]]), rtol=1e-12)
    bkd.assert_allclose(
        point.state()[:, 0], _exact(piece.points()[0]), rtol=1e-12, atol=1e-13
    )


def test_trace_tangent_passes_finite_differences(bkd: Backend[Array]) -> None:
    response = _Piece(bkd).response()

    def fun(sample: Array) -> Array:
        return response.linearize(sample, _no_params(bkd), _STEADY).output()

    def jvp(sample: Array, direction: Array) -> Array:
        point = response.linearize(sample, _no_params(bkd), _STEADY)
        return point.jvp_trace(direction)

    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=1, nvars=1, fun=fun, jvp=jvp, bkd=bkd)
    ).check_derivatives(bkd.asarray([[1.3]]), relative=True, central=True)
    assert bkd.to_float(bkd.min(errors[0])) < 1e-6


def test_the_sign_is_the_normals(bkd: Backend[Array]) -> None:
    inward = _Piece(bkd).response(normal=-1.0)
    point = inward.linearize(bkd.asarray([[1.0]]), _no_params(bkd), _STEADY)
    bkd.assert_allclose(point.output(), bkd.asarray([[-2.0]]), rtol=1e-12)


def test_refusals(bkd: Backend[Array]) -> None:
    piece = _Piece(bkd)
    operator = piece.operator(1.0)
    trace = LumpedNodalTrace(piece.points()[:, :1], bkd.ones((1,)), bkd)
    with pytest.raises(ValueError, match="at least one part"):
        NormalFluxMoments([], trace, 0.0)
    with pytest.raises(ValueError, match="sample 2 points"):
        NormalFluxMoments(
            [NormalFluxPart([operator], selecting([0, 0], 1, bkd))], trace, 0.0
        )
    with pytest.raises(ValueError, match="sampling matrix must have shape"):
        NormalFluxMoments([NormalFluxPart([operator], bkd.ones((1,)))], trace, 0.0)
    with pytest.raises(TypeError, match="NormalFluxOperatorProtocol"):
        NormalFluxMoments(
            [NormalFluxPart([object()], selecting([0], 1, bkd))],  # type: ignore[list-item]
            trace,
            0.0,
        )
    with pytest.raises(ValueError, match="at least one operator"):
        NormalFluxMoments([NormalFluxPart([], selecting([0], 1, bkd))], trace, 0.0)
    one = NormalFluxPart([operator], selecting([0], 1, bkd))
    two = NormalFluxPart([operator, operator], selecting([0], 1, bkd))
    with pytest.raises(ValueError, match="every part needs 1 operators"):
        NormalFluxMoments([one, two], trace, 0.0)

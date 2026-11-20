r"""A chain of three collocation subdomains coupled through their interface
values, against a single-domain collocation solve; NumPy and Torch.

The problem is :math:`D u'' - (v u)' + r u + f = 0` on [0, 1] with
constant coefficients, Dirichlet data at both ends, split at x = 0.3 and
x = 0.65. Each subdomain is pinned to the skeleton values at its interface
ends and returns there its outward flux :math:`-D u' + v u`, with unit
weight at a point port; the skeleton residual is the flux balance at each
interface. Two coefficient sets: diffusion alone (D = 2) and
advection-diffusion-reaction (D = 2, v = 0.7, r = -1.5), with f
manufactured from the exact solution.

- A cubic exact solution lies in every subdomain space (6 nodes) and in
  the single-domain one (8 nodes): the chain's subdomain states equal the
  single-domain solution interpolated to their nodes, and the exact
  values, to 1e-10, and each interface flux is the exact one.
- For :math:`u = e^x \sin 3x` the chain's largest nodal error decays
  spectrally in the per-subdomain node count.
- At the chain's solution the skeleton residual, the summed outward fluxes
  at each interface, is below the solver tolerance.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
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
from pyapprox.pde.constitutive.advection import ConservativeAdvection
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.normal_flux import NormalFluxMoments, NormalFluxPart
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import SteadyPinnedResponse
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    DenseSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.coupling.traces import LumpedNodalTrace
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.polynomials.univariate.lagrange import univariate_lagrange_polynomial
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
from tests._helpers.physics_test_utils import solve_steady_dirichlet

_ENDS = (0.0, 0.3, 0.65, 1.0)
_NPIECES = len(_ENDS) - 1
_STEADY = TimeWindow.steady()
_ATOL = 1e-11

# u, u' and u'' at the given points.
Exact = Callable[[Array, Backend[Array]], tuple[Array, Array, Array]]


def _cubic(x: Array, bkd: Backend[Array]) -> tuple[Array, Array, Array]:
    return 1.0 + x - 2.0 * x**2 + x**3, 1.0 - 4.0 * x + 3.0 * x**2, -4.0 + 6.0 * x


def _smooth(x: Array, bkd: Backend[Array]) -> tuple[Array, Array, Array]:
    ex, s, c = bkd.exp(x), bkd.sin(3.0 * x), bkd.cos(3.0 * x)
    return ex * s, ex * (s + 3.0 * c), ex * (6.0 * c - 8.0 * s)


@dataclass(frozen=True)
class _Coefficients:
    diffusion: float
    velocity: Optional[float]
    reaction: Optional[float]


_DIFFUSION = _Coefficients(2.0, None, None)
_ADR = _Coefficients(2.0, 0.7, -1.5)


class _Interval(Generic[Array]):
    """The physics on [a, b] with ``n`` nodes, its forcing manufactured
    from ``exact``."""

    def __init__(
        self,
        bkd: Backend[Array],
        a: float,
        b: float,
        n: int,
        coefs: _Coefficients,
        exact: Exact[Array],
    ) -> None:
        self._bkd, self._n, self._exact = bkd, n, exact
        self._mesh = TransformedMesh1D(n, bkd, AffineTransform1D((a, b), bkd))
        u, du, d2u = exact(self.points(), bkd)
        forcing = -coefs.diffusion * d2u
        velocity = None
        if coefs.velocity is not None:
            forcing = forcing + coefs.velocity * du
            velocity = [bkd.full((n,), coefs.velocity)]
        if coefs.reaction is not None:
            forcing = forcing - coefs.reaction * u
        self._physics = AdvectionDiffusionReaction(
            ChebyshevBasis1D(self._mesh, bkd),
            bkd,
            diffusion=coefs.diffusion,
            velocity=velocity,
            reaction=coefs.reaction,
            forcing=NodalField(forcing),
            advection_form=ConservativeAdvection(),
        )

    def physics(self) -> AdvectionDiffusionReaction[Array]:
        return self._physics

    def points(self) -> Array:
        """Shape: (n,)"""
        return self._mesh.points()[0]

    def end(self, side: int) -> Array:
        """The node index at the left (0) or right (1) end. Shape: (1,)"""
        return self._mesh.boundary_indices(side)

    def dirichlet(self, side: int) -> DirichletPart[Array]:
        """The exact value at an end as Dirichlet data."""
        index = self.end(side)
        return DirichletPart(
            index, NodalField(self._exact(self.points()[index], self._bkd)[0])
        )


class _Piece(Generic[Array]):
    """Subdomain ``s`` of the chain: physical Dirichlet data at a domain
    end, a port at each interface end, left before right (the skeleton's
    increasing DOF order)."""

    def __init__(
        self,
        bkd: Backend[Array],
        s: int,
        n: int,
        coefs: _Coefficients,
        exact: Exact[Array],
    ) -> None:
        self._bkd, self._n = bkd, n
        self._interval = _Interval(bkd, _ENDS[s], _ENDS[s + 1], n, coefs, exact)
        physical: list[DirichletPart[Array]] = []
        parts: list[NormalFluxPart[Array]] = []
        self._port: list[int] = []
        self._normals: list[float] = []
        for side, normal, is_physical in (
            (0, -1.0, s == 0),
            (1, 1.0, s == _NPIECES - 1),
        ):
            if is_physical:
                physical.append(self._interval.dirichlet(side))
                continue
            index = self._interval.end(side)
            operator = FluxNormalOperator(
                bkd, index, bkd.asarray([[normal]]), self._interval.physics()
            )
            parts.append(NormalFluxPart(operator, [0]))
            self._port.append(int(index[0]))
            self._normals.append(normal)
        self._physical = physical
        trace = LumpedNodalTrace(
            self._interval.points()[None, self._port],
            bkd.ones((len(self._port),)),
            bkd,
        )
        self._flux = NormalFluxMoments(parts, trace, 0.0)

    def points(self) -> Array:
        """Shape: (n,)"""
        return self._interval.points()

    def port_points(self) -> Array:
        """Shape: (nport,)"""
        return self._interval.points()[self._port]

    def port_normals(self) -> Array:
        """Shape: (nport,)"""
        return self._bkd.asarray(self._normals)

    def response(self) -> SteadyPinnedResponse[Array, Array, DenseFormat[Array]]:
        bkd, n = self._bkd, self._n
        # The middle piece has no physical boundary: no essential rows.
        essential = DirichletRows(bkd, n, 1, self._physical) if self._physical else None
        pinned = collocation_steady_residual(
            CollocationSystem(self._interval.physics(), essential)
        )
        states = backend_state_space(bkd, n)
        nport = len(self._port)
        theta = BackendVectorSpace(
            SpaceDescriptor("params", nport, backend_dtype_name(bkd)), bkd
        )
        ports: PortPinnedData[Array, Array] = PortPinnedData(
            BackendRowSet(states.descriptor(), self._port, bkd),
            BackendPortRows(states.descriptor(), self._port, bkd),
            BackendPortRows(theta.descriptor(), list(range(nport)), bkd),
            states,
            theta,
        )
        newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        return SteadyPinnedResponse(
            pinned,
            ports,
            self._flux,
            states,
            RecomputeAlways(BackendDenseLU(bkd)),
            newton,
            bkd,
        )


@dataclass(frozen=True)
class _PieceSolution(Generic[Array]):
    points: Array
    state: Array
    port_points: Array
    port_normals: Array
    flux: Array


@dataclass(frozen=True)
class _ChainSolution(Generic[Array]):
    pieces: list[_PieceSolution[Array]]
    balance: Array


def _solve_chain(
    bkd: Backend[Array], n: int, coefs: _Coefficients, exact: Exact[Array]
) -> _ChainSolution[Array]:
    topo = TensorBoxPartition((_NPIECES,))
    bases = UniformBases(lobatto_lagrange_basis(0, 1, bkd))
    skeleton = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    space = BackendVectorSpace(skeleton.descriptor(), bkd)
    pieces = [_Piece(bkd, s, n, coefs, exact) for s in range(_NPIECES)]
    subdomains = [
        CoupledSubdomain(
            piece.response(),
            MatchingNodalTransfer(
                BackendPortRows(
                    skeleton.descriptor(), subdomain_port_dofs(topo, skeleton, s), bkd
                )
            ),
        )
        for s, piece in enumerate(pieces)
    ]
    residual: SkeletonResidual[Array, Array, Array, DenseFormat[Array]] = (
        SkeletonResidual(space, subdomains, DenseSchur(bkd))
    )
    setups: RecomputeAlways[Array, DenseFormat[Array]] = RecomputeAlways(
        BackendDenseLU(bkd)
    )
    point = SteadySystem(residual, space, setups).solve(
        Newton(atol=_ATOL, rtol=0.0), space.zeros(1)
    )
    assert point.converged()
    solution = point.solution()
    solved = []
    for piece, sd in zip(pieces, subdomains):
        local = piece.response().linearize(
            sd.transfer.scatter(solution), bkd.zeros((0, 1)), _STEADY
        )
        solved.append(
            _PieceSolution(
                piece.points(),
                local.state()[:, 0],
                piece.port_points(),
                piece.port_normals(),
                local.output(),
            )
        )
    return _ChainSolution(solved, residual(solution))


def _solve_single(
    bkd: Backend[Array], n: int, coefs: _Coefficients, exact: Exact[Array]
) -> tuple[Array, Array]:
    """The single-domain collocation solution on [0, 1]: its nodes and
    nodal values."""
    whole = _Interval(bkd, _ENDS[0], _ENDS[-1], n, coefs, exact)
    state = solve_steady_dirichlet(
        whole.physics(),
        1,
        [whole.dirichlet(0), whole.dirichlet(1)],
        bkd.zeros((n,)),
        atol=_ATOL,
    )
    return whole.points(), state


def _max_error(
    bkd: Backend[Array], chain: _ChainSolution[Array], exact: Exact[Array]
) -> float:
    return max(
        bkd.to_float(bkd.max(bkd.abs(piece.state - exact(piece.points, bkd)[0])))
        for piece in chain.pieces
    )


@pytest.mark.parametrize("coefs", [_DIFFUSION, _ADR], ids=["diffusion", "adr"])
def test_a_cubic_equals_the_single_domain_solve(
    bkd: Backend[Array], coefs: _Coefficients
) -> None:
    chain = _solve_chain(bkd, 6, coefs, _cubic)
    nodes, values = _solve_single(bkd, 8, coefs, _cubic)
    for piece in chain.pieces:
        interpolated = univariate_lagrange_polynomial(nodes, piece.points, bkd) @ values
        bkd.assert_allclose(piece.state, interpolated, rtol=1e-10, atol=1e-10)
        bkd.assert_allclose(
            piece.state, _cubic(piece.points, bkd)[0], rtol=1e-10, atol=1e-10
        )
    for piece in chain.pieces:
        u, du, _ = _cubic(piece.port_points, bkd)
        flux = -coefs.diffusion * du
        if coefs.velocity is not None:
            flux = flux + coefs.velocity * u
        bkd.assert_allclose(
            piece.flux[:, 0], piece.port_normals * flux, rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("coefs", [_DIFFUSION, _ADR], ids=["diffusion", "adr"])
def test_a_smooth_solution_converges_spectrally(
    bkd: Backend[Array], coefs: _Coefficients
) -> None:
    # Measured: 1.1e-2, 2.3e-5, 9.4e-8, 1.5e-10, 8.1e-14 (diffusion).
    errors = [
        _max_error(bkd, _solve_chain(bkd, n, coefs, _smooth), _smooth)
        for n in (4, 6, 8, 10, 12)
    ]
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine < coarse / 100.0
    assert errors[-1] < 1e-12


@pytest.mark.parametrize("coefs", [_DIFFUSION, _ADR], ids=["diffusion", "adr"])
def test_the_interface_fluxes_balance(
    bkd: Backend[Array], coefs: _Coefficients
) -> None:
    chain = _solve_chain(bkd, 8, coefs, _smooth)
    assert bkd.to_float(bkd.max(bkd.abs(chain.balance))) < _ATOL

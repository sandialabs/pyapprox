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
from typing import Generic

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
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.constitutive.advection import ConservativeAdvection
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.normal_flux import (
    NormalFluxMoments,
    NormalFluxPart,
    selecting,
)
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
from pyapprox.pde.expressions.evaluation import PointExpressions
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
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
from tests._helpers.coupling_collocation import (
    Coefficients,
    adr_problem,
    diffusion_at,
    forcing_at,
    solution_at,
)
from tests._helpers.physics_test_utils import solve_steady_dirichlet

_ENDS = (0.0, 0.3, 0.65, 1.0)
_NPIECES = len(_ENDS) - 1
_STEADY = TimeWindow.steady()
_ATOL = 1e-11

_CUBIC = "1 + x - 2*x**2 + x**3"
_SMOOTH = "exp(x)*sin(3*x)"
_DIFFUSION = Coefficients(2.0, None, None)
_ADR = Coefficients(2.0, (0.7,), -1.5)


def _problem(coefs: Coefficients, solution: str) -> ManufacturedProblem:
    """The strong form on [0, 1], the forcing derived from ``solution``."""
    return adr_problem(coefs, solution, Box([(0.0, 1.0)]))


class _Interval(Generic[Array]):
    """The physics on [a, b] with ``n`` nodes, forced by ``problem``."""

    def __init__(
        self,
        bkd: Backend[Array],
        a: float,
        b: float,
        n: int,
        coefs: Coefficients,
        problem: ManufacturedProblem,
    ) -> None:
        self._bkd, self._n, self._problem = bkd, n, problem
        self._mesh = TransformedMesh1D(n, bkd, AffineTransform1D((a, b), bkd))
        velocity = (
            None if coefs.velocity is None else [bkd.full((n,), coefs.velocity[0])]
        )
        self._physics = AdvectionDiffusionReaction(
            ChebyshevBasis1D(self._mesh, bkd),
            bkd,
            diffusion=diffusion_at(coefs, problem, self._mesh.points(), bkd),
            velocity=velocity,
            reaction=coefs.reaction,
            forcing=NodalField(forcing_at(problem, self._mesh.points(), bkd)[0]),
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
        points = self._mesh.points()[:, index]
        return DirichletPart(
            index, NodalField(solution_at(self._problem, points, self._bkd)[0])
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
        coefs: Coefficients,
        problem: ManufacturedProblem,
    ) -> None:
        self._bkd, self._n = bkd, n
        self._interval = _Interval(bkd, _ENDS[s], _ENDS[s + 1], n, coefs, problem)
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
            parts.append(NormalFluxPart([operator], selecting([0], 1, bkd)))
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
        essential = (
            DirichletRows(bkd, StateLayout.uniform(n, 1), self._physical)
            if self._physical
            else None
        )
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
    bkd: Backend[Array], n: int, coefs: Coefficients, problem: ManufacturedProblem
) -> _ChainSolution[Array]:
    topo = TensorBoxPartition((_NPIECES,))
    bases = UniformBases(lobatto_lagrange_basis(0, 1, bkd))
    skeleton = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    space = BackendVectorSpace(skeleton.descriptor(), bkd)
    pieces = [_Piece(bkd, s, n, coefs, problem) for s in range(_NPIECES)]
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
    bkd: Backend[Array], n: int, coefs: Coefficients, problem: ManufacturedProblem
) -> tuple[Array, Array]:
    """The single-domain collocation solution on [0, 1]: its nodes and
    nodal values."""
    whole = _Interval(bkd, _ENDS[0], _ENDS[-1], n, coefs, problem)
    state = solve_steady_dirichlet(
        whole.physics(),
        1,
        [whole.dirichlet(0), whole.dirichlet(1)],
        bkd.zeros((n,)),
        atol=_ATOL,
    )
    return whole.points(), state


def _exact(bkd: Backend[Array], problem: ManufacturedProblem, x: Array) -> Array:
    """The exact solution at the 1D points ``x``. Shape: (n,)"""
    return solution_at(problem, x[None, :], bkd)[0]


def _max_error(
    bkd: Backend[Array], chain: _ChainSolution[Array], problem: ManufacturedProblem
) -> float:
    return max(
        bkd.to_float(bkd.max(bkd.abs(piece.state - _exact(bkd, problem, piece.points))))
        for piece in chain.pieces
    )


@pytest.mark.parametrize("coefs", [_DIFFUSION, _ADR], ids=["diffusion", "adr"])
def test_a_cubic_equals_the_single_domain_solve(
    bkd: Backend[Array], coefs: Coefficients
) -> None:
    problem = _problem(coefs, _CUBIC)
    chain = _solve_chain(bkd, 6, coefs, problem)
    nodes, values = _solve_single(bkd, 8, coefs, problem)
    # The exact flux -D u' + v u, from the core.
    flux = problem.flux("u")
    if flux is None:
        raise ValueError("the strong form has no flux")
    exact_flux = PointExpressions(
        [flux[0]], problem.names(), problem.coordinates(), bkd
    )
    for piece in chain.pieces:
        interpolated = univariate_lagrange_polynomial(nodes, piece.points, bkd) @ values
        bkd.assert_allclose(piece.state, interpolated, rtol=1e-10, atol=1e-10)
        bkd.assert_allclose(
            piece.state, _exact(bkd, problem, piece.points), rtol=1e-10, atol=1e-10
        )
        expected = piece.port_normals * exact_flux(piece.port_points[None, :])[0]
        bkd.assert_allclose(piece.flux[:, 0], expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("coefs", [_DIFFUSION, _ADR], ids=["diffusion", "adr"])
def test_a_smooth_solution_converges_spectrally(
    bkd: Backend[Array], coefs: Coefficients
) -> None:
    # Measured: 1.1e-2, 2.3e-5, 9.4e-8, 1.5e-10, 8.1e-14 (diffusion).
    problem = _problem(coefs, _SMOOTH)
    errors = [
        _max_error(bkd, _solve_chain(bkd, n, coefs, problem), problem)
        for n in (4, 6, 8, 10, 12)
    ]
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine < coarse / 100.0
    assert errors[-1] < 1e-12


@pytest.mark.parametrize("coefs", [_DIFFUSION, _ADR], ids=["diffusion", "adr"])
def test_the_interface_fluxes_balance(bkd: Backend[Array], coefs: Coefficients) -> None:
    chain = _solve_chain(bkd, 8, coefs, _problem(coefs, _SMOOTH))
    assert bkd.to_float(bkd.max(bkd.abs(chain.balance))) < _ATOL

r"""Two-dimensional skfem subdomains coupled through the skeleton, against
the single-domain skfem solve. NumPy only: skfem is.

Matching nodal transfer needs each subdomain mesh's interface nodes on
the skeleton's face nodes, which are Gauss-Lobatto points. Every
subdomain is meshed by Q1 quadrilaterals on the tensor grid of
:data:`_NPOINTS` Lobatto points per direction of its partition cell, so
the single-domain mesh on the union of those grids has the same
elements: the coupled solve and the single-domain solve are the same
discrete equations, equal to solver tolerance.

- Two subdomains (cut at x = 1/2), :math:`-\Delta u = f`, u = 1 on the
  left edge, u = 2 on the right, natural on top and bottom.
- Four subdomains (2 x 2, one cross point),
  :math:`-\Delta u + u [+ u^3] = f`, natural all round.

Interface ends lie on natural boundaries, where they are unknowns of
the skeleton like any interface node. Checks: the skeleton solution and
every subdomain's state equal the single-domain solve at the same
points (1e-10), and the gathered interface fluxes balance (1e-10).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from skfem import MeshQuad

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    ConstantDiffusion,
    TimeIndependent,
)
from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry, skeleton_dof_points
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import (
    SteadyPinnedPoint,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    MatrixFreeSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import (
    galerkin_steady_residual,
    solve_galerkin_steady,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol, MatrixFree
from pyapprox.util.linalg.operators.krylov import GMRES
from pyapprox.util.linalg.scipy_stack import NumpyRowSet, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]
_NPOINTS = 5
_STEADY = TimeWindow.steady()
_Reaction = Callable[[_Np, _Np], _Np]
_Piece = SteadyPinnedResponse[_Np, _Np, Any]


class _TensorQuadMesh:
    """Q1 quadrilaterals on the tensor grid of ``xs`` by ``ys``, with
    boundaries named left, right, bottom and top."""

    def __init__(self, xs: _Np, ys: _Np, bkd: NumpyBkd) -> None:
        self._bkd = bkd
        self._lower = (float(xs[0]), float(ys[0]))
        self._upper = (float(xs[-1]), float(ys[-1]))
        self._mesh = MeshQuad.init_tensor(xs, ys)
        self._nodes: _Np = np.asarray(self._mesh.p, dtype=np.float64)

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def ndim(self) -> int:
        return 2

    def nelements(self) -> int:
        return int(self._mesh.nelements)

    def nnodes(self) -> int:
        return int(self._mesh.nvertices)

    def nodes(self) -> _Np:
        return self._nodes

    def elements(self) -> _Np:
        return np.asarray(self._mesh.t, dtype=np.int64)

    def skfem_mesh(self) -> MeshQuad:
        return self._mesh

    def boundary_nodes(self, boundary_id: str) -> _Np:
        axis, value = {
            "left": (0, self._lower[0]),
            "right": (0, self._upper[0]),
            "bottom": (1, self._lower[1]),
            "top": (1, self._upper[1]),
        }[boundary_id]
        return np.where(np.abs(self._nodes[axis] - value) < 1e-12)[0]


def _forcing(x: _Np) -> _Np:
    return np.asarray(1.0 + x[0] + 2.0 * x[1])


def _no_sink(x: _Np, u: _Np) -> _Np:
    return np.zeros_like(u)


def _no_sink_deriv(x: _Np, u: _Np) -> _Np:
    return np.zeros_like(u)


# The skfem physics adds R(u) as a source, so a sink is negative.
def _linear_sink(x: _Np, u: _Np) -> _Np:
    return -u


def _linear_sink_deriv(x: _Np, u: _Np) -> _Np:
    return -np.ones_like(u)


def _cubic_sink(x: _Np, u: _Np) -> _Np:
    sink: _Np = -(u + u**3)
    return sink


def _cubic_sink_deriv(x: _Np, u: _Np) -> _Np:
    return -(1.0 + 3.0 * u**2)


def _newton() -> Newton[_Np, _Np]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=30)


class _Problem:
    """A partition of the unit square, its skeleton, and the PDE data."""

    def __init__(
        self,
        ncells: tuple[int, int],
        sink: tuple[_Reaction, _Reaction],
        essentials: Sequence[tuple[str, float]],
    ) -> None:
        self.bkd = NumpyBkd()
        self.topo = TensorBoxPartition(ncells)
        basis = TensorLagrangeBasis(1, _NPOINTS, self.bkd)
        self._bases = UniformBases(basis)
        self.skeleton = ContinuousSkeletonSpace(
            self.topo, self._bases, VertexWeightNumbering(self.topo, self._bases)
        )
        self.sink, self.essentials = sink, essentials
        self._ref_nodes = np.asarray(basis.nodes()[0])

    def cell_coords(self, direction: int, cell: int) -> _Np:
        """The Lobatto points of ``cell`` along ``direction``."""
        width = 1.0 / self.topo.ncells()[direction]
        return cell * width + 0.5 * (self._ref_nodes + 1.0) * width

    def coords(self, direction: int) -> _Np:
        """The union of the cells' points along ``direction``."""
        cells = [
            self.cell_coords(direction, c) for c in range(self.topo.ncells()[direction])
        ]
        return np.concatenate([cells[0]] + [c[1:] for c in cells[1:]])

    def skeleton_coords(self) -> _Np:
        """The point of each skeleton DOF. Shape: (2, ndofs)."""
        return skeleton_dof_points(
            self.skeleton, TensorBoxGeometry(self.topo, self.bkd), self._bases
        )

    def system(self, xs: _Np, ys: _Np, sides: Sequence[tuple[str, float]]) -> Any:
        mesh = _TensorQuadMesh(xs, ys, self.bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=ConstantDiffusion(1.0),
            bkd=self.bkd,
            reaction=CallableReaction(*self.sink),
            forcing=TimeIndependent(_forcing),
        )
        conditions = [DirichletBC(basis, side, v, self.bkd) for side, v in sides]
        return compose_galerkin_system(physics, conditions), mesh

    def piece(self, subdomain: int) -> tuple[_Piece, _Np, tuple[int, ...]]:
        """Subdomain ``subdomain``'s response, its node points and its
        skeleton DOFs."""
        ix, iy = divmod(subdomain, self.topo.ncells()[1])
        xs, ys = self.cell_coords(0, ix), self.cell_coords(1, iy)
        # Only the outer edges a subdomain touches carry its essentials.
        touched = {
            "left": ix == 0,
            "right": ix == self.topo.ncells()[0] - 1,
            "bottom": iy == 0,
            "top": iy == self.topo.ncells()[1] - 1,
        }
        sides = [(s, v) for s, v in self.essentials if touched[s]]
        system, mesh = self.system(xs, ys, sides)
        disc = skfem_discretization(system)
        dofs = subdomain_port_dofs(self.topo, self.skeleton, subdomain)
        skel = self.skeleton_coords()
        ports = [_node_at(mesh.nodes(), skel[:, d]) for d in dofs]
        states, theta = disc.space(), disc.parameter_space(len(ports))
        # The stack's space has an inner product; its declared type does not.
        assert isinstance(states, InnerProductSpaceProtocol)
        data: PortPinnedData[_Np, _Np] = PortPinnedData(
            NumpyRowSet(states.descriptor(), ports),
            BackendPortRows(states.descriptor(), ports, self.bkd),
            BackendPortRows(theta.descriptor(), list(range(len(ports))), self.bkd),
            states,
            theta,
        )
        response: _Piece = SteadyPinnedResponse(
            galerkin_steady_residual(disc),
            data,
            states,
            RecomputeAlways(ScipySparseLU()),
            _newton(),
            self.bkd,
        )
        return response, mesh.nodes(), dofs

    def single_domain(self) -> tuple[_Np, _Np]:
        """The single-domain solution and its node points."""
        system, mesh = self.system(self.coords(0), self.coords(1), self.essentials)
        _, point = solve_galerkin_steady(
            skfem_discretization(system), RecomputeAlways(ScipySparseLU()), _newton()
        )
        assert point.converged()
        return point.state(), mesh.nodes()

    def solve_skeleton(self, pieces: Sequence[_Piece]) -> _Np:
        bkd, space = self.bkd, self.skeleton.vector_space()
        subdomains = [
            CoupledSubdomain(
                piece,
                MatchingNodalTransfer(
                    BackendPortRows(
                        self.skeleton.descriptor(),
                        subdomain_port_dofs(self.topo, self.skeleton, s),
                        bkd,
                    )
                ),
            )
            for s, piece in enumerate(pieces)
        ]
        schur: MatrixFreeSchur[_Np, _Np] = MatrixFreeSchur()
        residual = SkeletonResidual(space, subdomains, schur)
        setups: RecomputeAlways[_Np, MatrixFree] = RecomputeAlways(
            GMRES(space, bkd, tol=1e-13)
        )
        newton: Newton[_Np, _Np] = Newton(atol=1e-11, rtol=1e-11, maxiters=20)
        point = SteadySystem(residual, space, setups).solve(newton, space.zeros(1))
        assert point.converged()
        return point.solution()


def _node_at(nodes: _Np, point: _Np) -> int:
    distances = np.linalg.norm(nodes - point[:, None], axis=0)
    node = int(np.argmin(distances))
    if distances[node] > 1e-12:
        raise ValueError(f"no mesh node at {point}")
    return node


def _check(problem: _Problem) -> None:
    bkd = problem.bkd
    reference, ref_nodes = problem.single_domain()
    built = [problem.piece(s) for s in range(problem.topo.nsubdomains())]
    lam = problem.solve_skeleton([piece for piece, _, _ in built])
    skel = problem.skeleton_coords()
    at_skeleton = [_node_at(ref_nodes, skel[:, d]) for d in range(skel.shape[1])]
    bkd.assert_allclose(lam, reference[at_skeleton], atol=1e-10)
    balance = np.zeros_like(lam)
    for piece, nodes, dofs in built:
        point = piece.linearize(lam[list(dofs)], bkd.zeros((0, 1)), _STEADY)
        assert isinstance(point, SteadyPinnedPoint)
        np.add.at(balance, list(dofs), point.output())
        rows = [_node_at(ref_nodes, nodes[:, k]) for k in range(nodes.shape[1])]
        bkd.assert_allclose(point.state(), reference[rows], atol=1e-10)
    bkd.assert_allclose(balance, bkd.zeros(balance.shape), atol=1e-10)


def test_two_subdomains_equal_the_single_domain() -> None:
    _check(
        _Problem((2, 1), (_no_sink, _no_sink_deriv), [("left", 1.0), ("right", 2.0)])
    )


@pytest.mark.parametrize("cubic", [False, True])
def test_four_subdomains_equal_the_single_domain(cubic: bool) -> None:
    sink = (_cubic_sink, _cubic_sink_deriv) if cubic else (
        _linear_sink, _linear_sink_deriv
    )
    problem = _Problem((2, 2), sink, [])
    assert len(problem.skeleton_coords()[0]) == 4 * (_NPOINTS - 2) + 5
    _check(problem)

r"""Nested substructuring of skfem subdomains, against the single-domain
skfem solve. NumPy only: skfem is.

Groups of subdomains become single responses
(:func:`~pyapprox.pde.coupling.nesting.skeleton_group_response`) on the
:mod:`tests._helpers.coupling_skfem` problems, whose flat skeleton
solves equal the single domain (``test_pinned_response_skfem_2d``):

- the 2 x 2 partition grouped as two columns: each column solves its
  horizontal interface, and the top skeleton is the vertical interface
  with the cross point; :math:`-\Delta u + u [+ u^3] = f`, natural all
  round, matrix-free (GMRES inside and out) and dense (LU inside and
  out);
- the same with u = g on the whole boundary: each horizontal cut's
  outer end is pinned by its column, the vertical cut's ends by the top;
- a 4 x 2 partition grouped twice: columns of two, then pairs of columns,
  so a group's members are groups.

A group's DOFs are its members' ports; its outer DOFs are those a
subdomain outside it also reads. Krylov tolerances loosen tenfold per
level above the subdomains, since an operator applied through inner
Krylov solves is only as exact as they are.
Checks: the top skeleton and every group's inner skeleton equal the
single-domain solution at their points (1e-10); a group's trace tangent
passes central finite differences.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Union

import numpy as np

from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJVPFromCallable,
)
from pyapprox.pde.boundary import ExceptDofs
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    ConstantDiffusion,
    TimeIndependent,
)
from pyapprox.pde.coupling.nesting import (
    FixedValues,
    GroupDofs,
    group_dofs,
    skeleton_group_response,
)
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import (
    SteadyPinnedPoint,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.pinning import PinnedSkeletonResidual
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    DenseSchur,
    MatrixFreeSchur,
    SkeletonResidual,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendRowSet,
    BackendVectorSpace,
)
from pyapprox.util.linalg.operators import SpaceDescriptor
from pyapprox.util.linalg.operators.krylov import GMRES
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling_skfem import (
    STEADY,
    CoupledProblem,
    Np,
    Piece,
    SkeletonPins,
    TensorQuadMesh,
    Value,
    box_problem,
    node_at,
)
from tests._helpers.markers import slow_test, slower_test, slowest_test

_Reaction = Callable[[Np, Np], Np]
_Group = SteadyPinnedResponse[Np, Np, Any]
# A group's members: leaf subdomains or groups.
_Tree = Union[int, Sequence["_Tree"]]


def _forcing(x: Np) -> Np:
    return np.asarray(1.0 + x[0] + 2.0 * x[1])


# The skfem physics adds R(u) as a source, so a sink is negative.
def _linear_sink(x: Np, u: Np) -> Np:
    return -u


def _linear_sink_deriv(x: Np, u: Np) -> Np:
    return -np.ones_like(u)


def _cubic_sink(x: Np, u: Np) -> Np:
    sink: Np = -(u + u**3)
    return sink


def _cubic_sink_deriv(x: Np, u: Np) -> Np:
    return -(1.0 + 3.0 * u**2)


def _sink(cubic: bool) -> tuple[_Reaction, _Reaction]:
    if cubic:
        return _cubic_sink, _cubic_sink_deriv
    return _linear_sink, _linear_sink_deriv


def _poisson(sink: tuple[_Reaction, _Reaction]) -> Callable[..., Any]:
    def build(
        mesh: TensorQuadMesh, sides: Sequence[tuple[str, Value]], exempt: set[int]
    ) -> Any:
        bkd = mesh.bkd()
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=ConstantDiffusion(1.0),
            bkd=bkd,
            reaction=CallableReaction(*sink),
            forcing=TimeIndependent(_forcing),
        )
        conditions = []
        for side, v in sides:
            bc = DirichletBC(
                basis, side, v if isinstance(v, float) else TimeIndependent(v), bkd
            )
            on_side = exempt & {int(d) for d in bc.constrained_dofs()}
            conditions.append(ExceptDofs(bc, sorted(on_side)))
        return compose_galerkin_system(physics, conditions)

    return build


def _boundary_data(x: Np) -> Np:
    data: Np = 1.0 + x[0] * x[1] + 0.5 * x[1] ** 2
    return data


def _everywhere(x: Np) -> Np:
    return np.ones(x.shape[1], dtype=bool)


@dataclass
class _Node:
    """A built child of a skeleton: a response, its port DOFs in the
    parent skeleton's numbering, and, for a group, its DOFs and members."""

    response: Any
    ports: tuple[int, ...]
    group: Union[GroupDofs, None] = None
    members: Sequence["_Node"] = ()
    height: int = 0


class _Nested:
    """Builds groups over a problem's subdomains and solves the top
    skeleton; ``dense`` uses dense Schur builders and LU at every level,
    otherwise matrix-free ones and GMRES."""

    def __init__(self, problem: CoupledProblem, dense: bool) -> None:
        self.problem, self.bkd, self._dense = problem, problem.bkd, dense
        nsub = problem.topo.nsubdomains()
        self.leaf_ports = [
            subdomain_port_dofs(problem.topo, problem.space, s) for s in range(nsub)
        ]
        self.leaves: list[Piece] = [problem.piece(s)[0] for s in range(nsub)]
        pinned = problem.pinned()
        coords = problem.skeleton_coords()
        self._pinned = (
            dict(zip(pinned, _boundary_data(coords[:, pinned]))) if pinned else {}
        )

    def _leaves_of(self, tree: _Tree) -> list[int]:
        if isinstance(tree, int):
            return [tree]
        return [leaf for child in tree for leaf in self._leaves_of(child)]

    def _schur(self) -> Any:
        return DenseSchur(self.bkd) if self._dense else MatrixFreeSchur()

    def _setups(self, space: BackendVectorSpace[Np], height: int) -> Any:
        """The level's linear solver. A Krylov solve whose operator acts
        through inner Krylov solves is only as exact as they are, so its
        tolerance loosens tenfold per level above the subdomains."""
        if self._dense:
            return RecomputeAlways(BackendDenseLU(self.bkd))
        return RecomputeAlways(GMRES(space, self.bkd, tol=10.0 ** (height - 14)))

    def _members(
        self, children: Sequence[_Node], numbering: GroupDofs, descriptor: Any
    ) -> list[CoupledSubdomain[Np, Np]]:
        return [
            CoupledSubdomain(
                child.response,
                MatchingNodalTransfer(
                    BackendPortRows(descriptor, numbering.local(child.ports), self.bkd)
                ),
            )
            for child in children
        ]

    def build(self, tree: _Tree) -> _Node:
        """The subdomain or group ``tree`` as a child of its parent."""
        if isinstance(tree, int):
            return _Node(self.leaves[tree], self.leaf_ports[tree])
        bkd = self.bkd
        children = [self.build(child) for child in tree]
        height = 1 + max(child.height for child in children)
        inside = set(self._leaves_of(tree))
        numbering = group_dofs(
            [child.ports for child in children],
            [p for s, p in enumerate(self.leaf_ports) if s not in inside],
        )
        descriptor = SpaceDescriptor("skeleton", numbering.size())
        space = BackendVectorSpace(descriptor, bkd)
        residual = SkeletonResidual(
            space, self._members(children, numbering, descriptor), self._schur()
        )
        own = [k for k in numbering.inner() if numbering.dofs[k] in self._pinned]
        values = bkd.zeros((numbering.size(), 1))
        for k in own:
            values[k, 0] = self._pinned[numbering.dofs[k]]
        own_rows = BackendRowSet(descriptor, own, bkd)
        essential = FixedValues(own_rows, values, space)
        nport = len(numbering.outer)
        theta = BackendVectorSpace(SpaceDescriptor("ports", nport), bkd)
        ports: PortPinnedData[Np, Np] = PortPinnedData(
            BackendRowSet(descriptor, list(numbering.outer), bkd),
            BackendPortRows(descriptor, list(numbering.outer), bkd),
            BackendPortRows(theta.descriptor(), list(range(nport)), bkd),
            space,
            theta,
        )
        response = skeleton_group_response(
            residual,
            space,
            ports,
            essential,
            self._setups(space, height),
            Newton(atol=1e-12, rtol=0.0, maxiters=30),
            bkd,
        )
        return _Node(response, numbering.outer_parent(), numbering, children, height)

    def solve(self, top: Sequence[_Node]) -> tuple[GroupDofs, Np]:
        """The top skeleton, on the DOFs the top children read, and its
        solution."""
        bkd = self.bkd
        numbering = GroupDofs(tuple(sorted({d for c in top for d in c.ports})), ())
        descriptor = SpaceDescriptor("skeleton", numbering.size())
        space = BackendVectorSpace(descriptor, bkd)
        residual: ResidualProtocol[Np, Any] = SkeletonResidual(
            space, self._members(top, numbering, descriptor), self._schur()
        )
        pinned = [k for k, d in enumerate(numbering.dofs) if d in self._pinned]
        if pinned:
            values = bkd.zeros((numbering.size(), 1))
            for k in pinned:
                values[k, 0] = self._pinned[numbering.dofs[k]]
            residual = PinnedSkeletonResidual(
                residual, space, BackendRowSet(descriptor, pinned, bkd), values
            )
        outer: Newton[Np, Np] = Newton(atol=1e-11, rtol=1e-11, maxiters=20)
        height = 1 + max(child.height for child in top)
        point = SteadySystem(residual, space, self._setups(space, height)).solve(
            outer, space.zeros(1)
        )
        assert point.converged(), point.info().message
        return numbering, point.solution()


def _check_group(
    node: _Node, lam: Np, expected: Callable[[Sequence[int]], Np], bkd: NumpyBkd
) -> None:
    """``node``, given ``lam`` on its ports, holds the reference skeleton
    on all its DOFs, and so do its member groups."""
    if node.group is None:
        return
    point = node.response.linearize(lam, bkd.zeros((0, 1)), STEADY)
    assert isinstance(point, SteadyPinnedPoint)
    inner = point.state()
    bkd.assert_allclose(inner, expected(node.group.dofs), atol=1e-10)
    for member in node.members:
        local = node.group.local(member.ports)
        _check_group(member, inner[list(local)], expected, bkd)


def _check(problem: CoupledProblem, tree: Sequence[_Tree], dense: bool) -> None:
    bkd = problem.bkd
    reference, ref_nodes = problem.single_domain()
    skel = problem.skeleton_coords()

    def expected(dofs: Sequence[int]) -> Np:
        rows = [node_at(ref_nodes, skel[:, d]) for d in dofs]
        return np.asarray(reference[rows])

    nested = _Nested(problem, dense)
    top = [nested.build(child) for child in tree]
    numbering, lam = nested.solve(top)
    bkd.assert_allclose(lam, expected(numbering.dofs), atol=1e-10)
    for child in top:
        _check_group(child, lam[list(numbering.local(child.ports))], expected, bkd)


@pytest.mark.parametrize(
    "cubic,dense",
    [
        (False, False),
        (False, True),
        pytest.param(True, False, marks=slow_test),
        (True, True),
    ],
)
def test_columns_of_a_two_by_two_partition(cubic: bool, dense: bool) -> None:
    problem = box_problem((2, 2), _poisson(_sink(cubic)), [])
    _check(problem, [[0, 1], [2, 3]], dense)


@pytest.mark.parametrize("cubic", [False, True])
def test_columns_with_dirichlet_closure_points(cubic: bool) -> None:
    sides = [(s, _boundary_data) for s in ("left", "right", "bottom", "top")]
    pins = SkeletonPins(_everywhere, lambda x: _boundary_data(x)[None, :])
    problem = box_problem((2, 2), _poisson(_sink(cubic)), sides, pins=pins)
    nested = _Nested(problem, dense=False)
    ports = nested.leaf_ports
    left = group_dofs([ports[0], ports[1]], [ports[2], ports[3]])
    # The horizontal cut's left end is the left column's own pin; the
    # vertical cut's ends are its ports, pinned by the top skeleton.
    own = [d for d in (left.dofs[k] for k in left.inner()) if d in problem.pinned()]
    assert len(own) == 1
    assert len(set(left.outer_parent()) & set(problem.pinned())) == 2
    _check(problem, [[0, 1], [2, 3]], dense=False)


@pytest.mark.parametrize(
    "cubic",
    [pytest.param(False, marks=slower_test), pytest.param(True, marks=slowest_test)],
)
def test_two_level_grouping_of_a_four_by_two_partition(cubic: bool) -> None:
    problem = box_problem((4, 2), _poisson(_sink(cubic)), [])
    # Subdomain s is cell (s // 2, s % 2): columns of two, then pairs.
    _check(problem, [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dense=False)


def test_group_trace_tangent_passes_finite_differences() -> None:
    problem = box_problem((2, 2), _poisson(_sink(True)), [])
    nested = _Nested(problem, dense=False)
    node = nested.build([0, 1])
    assert node.group is not None
    bkd, nport = problem.bkd, len(node.ports)
    response = node.response
    # A smooth trace on the vertical cut.
    y = problem.skeleton_coords()[1, list(node.ports)]
    sample = np.asarray((1.0 + 0.3 * y + 0.2 * y**2)[:, None])

    def fun(lam: Np) -> Np:
        out: Np = response.linearize(lam, bkd.zeros((0, 1)), STEADY).output()
        return out

    def jvp(lam: Np, direction: Np) -> Np:
        point = response.linearize(lam, bkd.zeros((0, 1)), STEADY)
        assert isinstance(point, SteadyPinnedPoint)
        tangent: Np = point.jvp_trace(direction)
        return tangent

    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=nport, nvars=nport, fun=fun, jvp=jvp, bkd=bkd)
    ).check_derivatives(sample, relative=True, central=True)
    assert bkd.to_float(bkd.min(errors[0])) < 1e-6

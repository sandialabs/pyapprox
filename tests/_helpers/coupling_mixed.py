r"""Harness for a collocation subdomain beside an skfem one on the unit
square, coupled through a polynomial skeleton, against a manufactured
solution.

The square is cut at x = 1/2 (a 2 x 1 :class:`~pyapprox.pde.coupling.
topology.TensorBoxPartition`). The skeleton's face basis is the Lagrange
basis on ``nskel`` Chebyshev-Gauss-Lobatto nodes, as in the collocation
harness; its two DOFs at the interface ends lie on the outer boundary and
are held at the exact solution.

- Collocation on [0, 1/2] x [0, 1], ``ncol`` Chebyshev nodes per
  direction, Dirichlet on its three outer sides. Its ports are its
  interface nodes off those sides; they read the skeleton by
  interpolation and test their flux exactly (Gauss side rule).
- skfem on [1/2, 1] x [0, 1], ``ncells`` Q``degree`` cells per
  direction, Dirichlet on the right and the exact Neumann data on the
  top and bottom, so every interface DOF, ends included, is a port node
  and the trace space keeps the constants. The ports read the skeleton
  by :math:`L^2` projection onto the interface trace
  (:class:`~pyapprox.pde.coupling.traces.LagrangeSegmentTrace`) and
  return the reaction.

Both sides return the outward diffusive flux :math:`-D \nabla u \cdot
n`: a Galerkin reaction is the diffusive flux in either advection form,
so the collocation ports are given the non-conservative form's flux.
With the velocity constant and u continuous, matching it is the
transmission condition of the conservative equation too.

The physics is :math:`D \Delta u - \nabla\cdot(v u) + r u + f = 0` with
constant v. Each side's problem is a verification-core
:class:`~pyapprox.pde.verification.problem.ManufacturedProblem` from the
same strong form, so the forcings agree; the skfem side's groups only
:math:`-D \nabla u` inside the divergence, so its Neumann data are the
diffusive flux, as the Galerkin physics reads them.

NumPy only: skfem is. Importers must skip when skfem is not installed.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.assembly.backend_dense_sink import BackendDenseMatrixAssembly
from pyapprox.pde.assembly.scipy_stack import ScipyAssemblyStack
from pyapprox.pde.collocation.quadrature.boundary_quadrature import (
    GaussLegendreSideRule,
    SideRuleProtocol,
)
from pyapprox.pde.constitutive.advection import (
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry, skeleton_dof_points
from pyapprox.pde.coupling.pinned_response import (
    SteadyPinnedPoint,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.pinning import PinnedSkeletonResidual
from pyapprox.pde.coupling.ports import (
    AverageIncidentFaces,
    BackendPortRows,
    PortAssemblyProtocol,
    PortMatrixTransfer,
    PortTransferProtocol,
    port_interpolation,
    port_projection,
    port_trace_coupling,
    port_trace_mass,
    port_trace_mortar_coupling,
    port_trace_pairing,
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
from pyapprox.pde.coupling.traces import LagrangeSegmentTrace, mortar_segment_traces
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import NeumannBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh.structured import StructuredMesh2D
from pyapprox.pde.galerkin.physics import (
    AdvectionDiffusionReaction as GalerkinAdr,
)
from pyapprox.pde.models.galerkin.verification import (
    GalerkinApproximation,
    boundary_conditions,
    field,
    forcing,
    state_law,
    vector_field,
)
from pyapprox.pde.verification.conditions import ConditionProtocol, Dirichlet, Neumann
from pyapprox.pde.verification.domains import Box as VerificationBox
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, Source, TermProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendRowSet,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators.algebra import Product
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling_collocation import (
    Adr,
    Box,
    Coefficients,
    UnitBox,
    box_response,
    interior_port_sides,
    reference_nodes,
    solution_at,
)
from tests._helpers.coupling_skfem import pinned_piece

Np = NDArray[Any]
STEADY = TimeWindow.steady()
_BKD = NumpyBkd()


def galerkin_conditions(dirichlet_ends: bool) -> dict[str, ConditionProtocol]:
    """The skfem half's conditions: Dirichlet on the right, and on the top
    and bottom Dirichlet or the exact Neumann data. The interface side
    gets none (its DOFs are ports), so its entry only completes the
    problem's parts."""
    ends: ConditionProtocol = Dirichlet() if dirichlet_ends else Neumann()
    return {"left": Neumann(), "right": Dirichlet(), "bottom": ends, "top": ends}


def galerkin_adr_problem(
    coefs: Coefficients,
    solution: str,
    region: VerificationBox,
    conditions: dict[str, ConditionProtocol],
) -> ManufacturedProblem:
    r""":math:`D \Delta u - \nabla\cdot(v u) + r u + f = 0` on ``region``
    with natural data the diffusive flux: only :math:`-D \nabla u` inside
    the divergence, the transport :math:`v \cdot \nabla u` a source (v is
    constant, so it equals :math:`\nabla\cdot(v u)`)."""
    coords = region.coordinates()
    terms: list[TermProtocol] = []
    if coefs.velocity is not None:
        transport = [f"({v})*u_{c}" for v, c in zip(coefs.velocity, coords)]
        terms.append(Source("u", " + ".join(transport)))
    terms.append(Divergence("u", [f"-({coefs.diffusion})*u_{c}" for c in coords]))
    if coefs.reaction is not None:
        terms.append(Source("u", f"-({coefs.reaction})*u"))
    return ManufacturedProblem({"u": solution}, terms, region, conditions)


class NegatedFlux:
    """A transfer whose gathered flux is negated: the wrong-normal
    control, a port returning its flux with the inward normal."""

    def __init__(self, transfer: PortTransferProtocol[Np, Np]) -> None:
        self._transfer = transfer

    def nport(self) -> int:
        return self._transfer.nport()

    def scatter(self, lam: Np) -> Np:
        return self._transfer.scatter(lam)

    def gather_add(self, block: Np, y: Np) -> Np:
        return self._transfer.gather_add(-block, y)


@dataclass(frozen=True)
class Skeleton:
    """The skeleton a subdomain's ports read: its space and geometry, the
    faces the subdomain's port lies on, its skeleton DOFs, and the
    skeleton's nodes per face."""

    space: ContinuousSkeletonSpace[Np]
    geometry: TensorBoxGeometry[Np]
    faces: list[int]
    dofs: list[int]
    nskel: int


@dataclass(frozen=True)
class SkfemAdrPiece:
    """An skfem advection-diffusion-reaction subdomain on [1/2, 1] x [0, 1]
    with its port on x = 1/2: its response and transfer, its problem,
    basis and DOF points."""

    response: SteadyPinnedResponse[Np, Np, Any]
    transfer: PortTransferProtocol[Np, Np]
    problem: ManufacturedProblem
    basis: LagrangeBasis[Np]
    dof_points: Np


def _skfem_map(
    skeleton: Skeleton,
    points: Np,
    along: list[int],
    elements: list[list[int]],
    nquad: int,
    dirichlet_ends: bool,
    mortar: bool,
) -> tuple[Product[Np], list[int]]:
    """The skfem side's local map and its port DOFs: every interface DOF,
    or with Dirichlet ends those between them."""
    bkd = _BKD
    stack: PortAssemblyProtocol[Np, CSRFormat, Np, Np] = ScipyAssemblyStack(np.float64)
    s = skeleton
    if not dirichlet_ends:
        full = LagrangeSegmentTrace(points, elements, nquad, bkd)
        mass = port_trace_mass(full, 1, stack)
        coupling = port_trace_coupling(
            s.space, full, s.geometry, s.faces, s.dofs, AverageIncidentFaces(), stack
        )
        return port_projection(mass, coupling, ScipySparseLU()), along
    traces = mortar_segment_traces(points, elements, [0, len(along) - 1], nquad, bkd)
    ports = [along[n] for n in traces.interior]
    if mortar:
        mass = port_trace_pairing(traces.test, traces.trial, 1, stack)
        coupling = port_trace_mortar_coupling(
            s.space,
            traces.test,
            traces.ends,
            traces.end_points,
            s.geometry,
            s.faces,
            s.dofs,
            AverageIncidentFaces(),
            stack,
        )
    else:
        mass = port_trace_mass(traces.trial, 1, stack)
        coupling = port_trace_coupling(
            s.space,
            traces.trial,
            s.geometry,
            s.faces,
            s.dofs,
            AverageIncidentFaces(),
            stack,
        )
    return port_projection(mass, coupling, ScipySparseLU()), ports


def skfem_adr_piece(
    coefs: Coefficients,
    solution: str,
    skeleton: Skeleton,
    ncells: int,
    degree: int = 2,
    dirichlet_ends: bool = False,
    mortar: bool = True,
) -> SkfemAdrPiece:
    """The skfem side of the strip (module docstring): Q``degree`` on
    ``ncells`` x ``ncells`` cells, the physics ``coefs`` with exact
    solution ``solution``, its ports reading ``skeleton`` by L2 projection
    (mortar with ``dirichlet_ends``; plain projection onto the port nodes'
    functions with ``mortar`` False)."""
    bkd = _BKD
    region = VerificationBox([(0.5, 1.0), (0.0, 1.0)])
    problem = galerkin_adr_problem(
        coefs, solution, region, galerkin_conditions(dirichlet_ends)
    )
    mesh = StructuredMesh2D(ncells, ncells, [(0.5, 1.0), (0.0, 1.0)], bkd)
    basis = LagrangeBasis(mesh, degree=degree)
    physics = GalerkinAdr(
        basis=basis,
        diffusivity=field(problem, str(coefs.diffusion)),
        bkd=bkd,
        velocity=(
            None
            if coefs.velocity is None
            else vector_field(problem, [str(v) for v in coefs.velocity])
        ),
        reaction=(
            None
            if coefs.reaction is None
            else state_law(problem, f"({coefs.reaction})*u", "u")
        ),
        forcing=forcing(problem, "u"),
        advection_form=ConservativeAdvection(),
    )
    conditions = [
        bc
        for bc in boundary_conditions(problem, basis, "u")
        if not (isinstance(bc, NeumannBC) and bc.boundary_name() == "left")
    ]
    system = compose_galerkin_system(physics, conditions)
    dof_points: Np = np.asarray(basis.dof_coordinates())
    left = [int(d) for d in np.asarray(basis.get_dofs("left"))]
    # The interface DOFs in order along it, so element e is the run of
    # degree + 1 of them from degree * e.
    along = sorted(left, key=lambda d: float(dof_points[1, d]))
    if len(along) != degree * ncells + 1:
        raise ValueError(f"expected {degree * ncells + 1} interface DOFs")
    elements = [list(range(degree * e, degree * e + degree + 1)) for e in range(ncells)]
    # Exact for a trace function times a skeleton function.
    nquad = (degree + skeleton.nskel + 1) // 2 + 1
    local, ports = _skfem_map(
        skeleton,
        dof_points[:, along],
        along,
        elements,
        nquad,
        dirichlet_ends,
        mortar,
    )
    transfer: PortTransferProtocol[Np, Np] = PortMatrixTransfer(
        BackendPortRows(skeleton.space.descriptor(), skeleton.dofs, bkd), local
    )
    return SkfemAdrPiece(
        pinned_piece(system, ports), transfer, problem, basis, dof_points
    )


class MixedStrip:
    """The coupled problem, its skeleton solve and each side's state.

    Parameters
    ----------
    coefs, solution
        The physics' coefficients and the exact solution.
    ncol
        Collocation nodes per direction.
    ncells
        skfem cells per direction.
    nskel
        Skeleton nodes per face.
    degree
        The skfem Lagrange degree.
    reverse_galerkin_normal
        Negate the skfem side's returned flux (the wrong-normal control).
    rule
        Tests the collocation side's flux; exact (Gauss) by default.
    dirichlet_ends
        Dirichlet on the skfem top and bottom too: the interface ends are
        then essential rows, the ports the interface DOFs between them.
    mortar
        With ``dirichlet_ends``, read the skeleton by the mortar projection
        (:func:`~pyapprox.pde.coupling.traces.mortar_segment_traces`);
        False projects onto the port nodes' own functions alone (the
        control).
    galerkin
        The skfem side's own coefficients and exact solution, for
        different physics per side; omitted, ``coefs`` and ``solution``.
        The two solutions must agree on the interface and their diffusive
        fluxes balance there.
    """

    def __init__(
        self,
        coefs: Coefficients,
        solution: str,
        ncol: int,
        ncells: int,
        nskel: int,
        degree: int = 2,
        reverse_galerkin_normal: bool = False,
        rule: Optional[SideRuleProtocol] = None,
        dirichlet_ends: bool = False,
        mortar: bool = True,
        galerkin: Optional[tuple[Coefficients, str]] = None,
    ) -> None:
        bkd = _BKD
        self._coefs, self._solution = coefs, solution
        self._galerkin_coefs, self._galerkin_solution = (
            (coefs, solution) if galerkin is None else galerkin
        )
        self._topo = TensorBoxPartition((2, 1))
        bases = UniformBases(
            TensorLagrangeBasis(1, reference_nodes(nskel, bkd), True, bkd)
        )
        self._space = ContinuousSkeletonSpace(
            self._topo, bases, VertexWeightNumbering(self._topo, bases)
        )
        self._geometry = TensorBoxGeometry(self._topo, bkd)
        self._coords = skeleton_dof_points(self._space, self._geometry, bases)
        self._nskel = nskel
        self._collocation(ncol, GaussLegendreSideRule() if rule is None else rule)
        self._dirichlet_ends, self._mortar = dirichlet_ends, mortar
        self._galerkin(ncells, degree, reverse_galerkin_normal)

    def _faces_and_dofs(self, subdomain: int) -> tuple[list[int], list[int]]:
        faces = [
            f
            for f in self._topo.subdomain_entities(subdomain)
            if f in self._space.faces()
        ]
        dofs = list(subdomain_port_dofs(self._topo, self._space, subdomain))
        return faces, dofs

    def _collocation(self, ncol: int, rule: SideRuleProtocol) -> None:
        bkd = _BKD
        model: Adr[Np] = Adr(self._coefs, self._solution, NonConservativeAdvection())
        problem = model.problem(UnitBox[Np]().region(2))
        self._box = Box(bkd, [(0.0, 0.5), (0.0, 1.0)], ncol, model, problem, UnitBox())
        interface = [(0, True)]
        outer = [(0, False), (1, False), (1, True)]
        sides = interior_port_sides(self._box, interface, outer)
        ports = [node for side in sides for node in side.nodes]
        faces, dofs = self._faces_and_dofs(0)
        matrix = port_interpolation(
            self._space,
            self._box.parameters()[:, ports],
            self._geometry,
            faces,
            dofs,
            AverageIncidentFaces(),
            BackendDenseMatrixAssembly(bkd),
        )
        self._collocation_transfer: PortTransferProtocol[Np, Np] = PortMatrixTransfer(
            BackendPortRows(self._space.descriptor(), dofs, bkd), matrix
        )
        self._collocation_response = box_response(self._box, sides, outer, set(), rule)

    def _galerkin(self, ncells: int, degree: int, reverse: bool) -> None:
        faces, dofs = self._faces_and_dofs(1)
        piece = skfem_adr_piece(
            self._galerkin_coefs,
            self._galerkin_solution,
            Skeleton(self._space, self._geometry, faces, dofs, self._nskel),
            ncells,
            degree,
            self._dirichlet_ends,
            self._mortar,
        )
        self._galerkin_problem, self._basis = piece.problem, piece.basis
        self._dof_points: Np = piece.dof_points
        self._galerkin_transfer = (
            NegatedFlux(piece.transfer) if reverse else piece.transfer
        )
        self._galerkin_response = piece.response

    def _pieces(
        self,
    ) -> list[tuple[SteadyPinnedResponse[Np, Np, Any], PortTransferProtocol[Np, Np]]]:
        return [
            (self._collocation_response, self._collocation_transfer),
            (self._galerkin_response, self._galerkin_transfer),
        ]

    def exact(self, points: Np) -> Np:
        """The collocation side's exact solution at ``points``, which is the
        interface's too. Shape: (n,)"""
        problem = self._box_problem()
        return np.asarray(solution_at(problem, points, _BKD)[0])

    def galerkin_exact(self, points: Np) -> Np:
        """The skfem side's exact solution at ``points``. Shape: (n,)"""
        return np.asarray(solution_at(self._galerkin_problem, points, _BKD)[0])

    def _box_problem(self) -> ManufacturedProblem:
        model: Adr[Np] = Adr(self._coefs, self._solution)
        return model.problem(UnitBox[Np]().region(2))

    def pinned(self) -> list[int]:
        """The skeleton DOFs on the outer boundary: the interface ends."""
        return sorted(
            dof
            for entity in range(self._topo.nentities())
            if self._topo.on_boundary(entity)
            for dof in self._space.entity_dofs(entity)
        )

    def solve_skeleton(self) -> Np:
        """The skeleton values, the interface ends held at the exact
        solution. Shape: (ndofs, 1)"""
        bkd, descriptor = _BKD, self._space.descriptor()
        space = BackendVectorSpace(descriptor, bkd)
        pinned = self.pinned()
        values = np.zeros((descriptor.size, 1))
        values[pinned, 0] = self.exact(self._coords[:, pinned])
        skeleton: SkeletonResidual[Np, Np, Np, DenseFormat[Np]] = SkeletonResidual(
            space,
            [CoupledSubdomain(r, t) for r, t in self._pieces()],
            DenseSchur(bkd),
        )
        residual: PinnedSkeletonResidual[Np, DenseFormat[Np]] = PinnedSkeletonResidual(
            skeleton, space, BackendRowSet(descriptor, pinned, bkd), values
        )
        lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(BackendDenseLU(bkd))
        outer: Newton[Np, Np] = Newton(atol=1e-10, rtol=0.0, maxiters=20)
        point = SteadySystem(residual, space, lu).solve(outer, space.zeros(1))
        if not point.converged():
            raise RuntimeError("the skeleton solve did not converge")
        return point.solution()

    def _point(
        self,
        response: SteadyPinnedResponse[Np, Np, Any],
        transfer: PortTransferProtocol[Np, Np],
        lam: Np,
    ) -> SteadyPinnedPoint[Np, Any]:
        point = response.linearize(transfer.scatter(lam), _BKD.zeros((0, 1)), STEADY)
        if not isinstance(point, SteadyPinnedPoint):
            raise TypeError("expected a SteadyPinnedPoint")
        return point

    def errors(self, lam: Np) -> tuple[float, float]:
        """The largest nodal errors of the collocation and skfem states for
        the skeleton values ``lam``."""
        collocation = self._point(
            self._collocation_response, self._collocation_transfer, lam
        ).state()[:, 0]
        galerkin = self._point(
            self._galerkin_response, self._galerkin_transfer, lam
        ).state()[:, 0]
        points = np.asarray(self._box.points())
        return (
            float(np.abs(collocation - self.exact(points)).max()),
            float(np.abs(galerkin - self.galerkin_exact(self._dof_points)).max()),
        )

    def gathered(self, lam: Np) -> tuple[Np, Np]:
        """Each side's returned flux gathered onto the skeleton, the
        collocation side's first. Shape of each: (ndofs, 1)"""
        size = self._space.descriptor().size
        out = []
        for response, transfer in self._pieces():
            point = self._point(response, transfer, lam)
            out.append(transfer.gather_add(point.output(), np.zeros((size, 1))))
        return out[0], out[1]

    def galerkin_total_flux(self, lam: Np) -> float:
        """The skfem side's outward flux integrated over the interface: the
        sum of its tested fluxes, its trace functions summing to one."""
        point = self._point(self._galerkin_response, self._galerkin_transfer, lam)
        return float(np.sum(point.output()))

    def galerkin_l2_error(self, lam: Np) -> float:
        r"""The skfem state's :math:`L^2` error against the exact solution,
        integrated by skfem quadrature of order 8."""
        point = self._point(self._galerkin_response, self._galerkin_transfer, lam)
        approximation = GalerkinApproximation(
            self._galerkin_problem, ["u"], self._basis, point.state()[:, 0], 8
        )
        return approximation.l2_error()

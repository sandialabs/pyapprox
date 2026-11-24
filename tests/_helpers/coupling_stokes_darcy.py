r"""Harness for Stokes beside Darcy on the unit square, coupled through a
scalar interface pressure, against a manufactured solution.

The square is cut at x = 1/2 (a 2 x 1 :class:`~pyapprox.pde.coupling.
topology.TensorBoxPartition`); the skeleton's face basis is the Lagrange
basis on ``nskel`` Chebyshev-Gauss-Lobatto nodes, its unknown
:math:`\lambda` the interface pressure. Both sides are natural on the top
and bottom, so the interface ends are ordinary port nodes and skeleton
unknowns.

- Stokes on [0, 1/2] x [0, 1], Taylor-Hood Q2-Q1 on ``ncells`` x
  ``ncells`` cells: :math:`-\nu \Delta u + \nabla p = f`,
  :math:`\nabla\cdot u = f_p`; velocity Dirichlet on the left, the exact
  pseudo-traction :math:`t = \nu \partial_n u - p n` on the top and
  bottom. On the interface the normal pseudo-traction is the port load
  :math:`-\lambda n` and the tangential one obeys Beavers-Joseph-Saffman,
  :math:`t \cdot \tau = -\beta\, u \cdot \tau`, a Robin condition with
  coefficients (0, beta) per component. The port is every interface
  velocity DOF, node-major; it reads the skeleton by
  :class:`~pyapprox.pde.coupling.ports.NormalProjectionTransfer` with
  sign -1 over the tested coupling :math:`B = \int \psi \phi`, so the
  load is :math:`-B\lambda\, n` and the gathered output :math:`-B^\top
  (n \cdot u)`: a port-loaded response (:func:`~pyapprox.pde.coupling.
  stacks.galerkin.skfem_loaded_response`) returning the velocity there.
- Darcy on [1/2, 1] x [0, 1], primal form :math:`-\nabla\cdot(K \nabla
  p) = f_D`, Q2: an advection-diffusion-reaction piece of the mixed
  harness (:func:`~tests._helpers.coupling_mixed.skfem_adr_piece`)
  pinned to :math:`\lambda` by L2 projection, returning :math:`K
  \partial p / \partial n = -u_D \cdot n_D` tested; Dirichlet on the
  right, the exact flux on the top and bottom.

Both responses declare that they receive an :class:`InterfacePressure`
and return a :class:`NormalVelocity` (the Darcy one through
:class:`~pyapprox.pde.coupling.responses.DeclaredKinds`), so the kind
layout gives one unknown per skeleton DOF and the residual is the plain
sum :math:`-\int \phi\, (u_S \cdot n_S + u_D \cdot n_D)`.

Each manufactured :class:`Solution` (:data:`SMOOTH`, :data:`POLYNOMIAL`)
satisfies the three interface conditions at x = 1/2 for every y, with
:math:`\nu = K = \beta = 1`: mass, :math:`u_S = -\partial_x p_D`; normal
stress, :math:`\partial_x u_S - p_S = -p_D`; BJS, :math:`\partial_x v_S
+ v_S = 0`, with a nonzero tangential traction, so dropping BJS is
wrong. :data:`POLYNOMIAL` lies in the discrete spaces, its interface
pressure of degree 2, so the coupled discretization reproduces it.

NumPy only: skfem is. Importers must skip when skfem is not installed.
"""

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.assembly.scipy_stack import ScipyAssemblyStack
from pyapprox.pde.boundary.protocols import BoundaryConditionRole
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry, skeleton_dof_points
from pyapprox.pde.coupling.kind_layout import KindLayout, kind_layout
from pyapprox.pde.coupling.pinned_response import SteadyPinnedPoint
from pyapprox.pde.coupling.ports import (
    AverageIncidentFaces,
    BackendPortRows,
    NormalProjectionTransfer,
    PortAssemblyProtocol,
    PortMatrixTransfer,
    PortTransferProtocol,
    port_trace_coupling,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.responses import (
    DeclaredKinds,
    LocalResponseProtocol,
    PortQuantity,
    TimeWindow,
)
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    DenseSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.stacks.galerkin import skfem_loaded_response
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.coupling.traces import LagrangeSegmentTrace
from pyapprox.pde.galerkin.basis import MixedBasis
from pyapprox.pde.galerkin.boundary.implementations import NeumannBC, RobinBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh.structured import StructuredMesh2D
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.galerkin.protocols.basis import ComponentDofsBasisProtocol
from pyapprox.pde.models.galerkin.verification import (
    GalerkinApproximation,
    TaylorHood,
    forcing,
    vector_boundary_conditions,
    vector_forcing,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling_collocation import (
    Coefficients,
    reference_nodes,
    solution_at,
)
from tests._helpers.coupling_mixed import Skeleton, skfem_adr_piece
from tests._helpers.coupling_skfem import newton

Np = NDArray[Any]
STEADY = TimeWindow.steady()
_BKD = NumpyBkd()
NU, K, BETA = 1.0, 1.0, 1.0


@dataclass(frozen=True)
class Solution:
    """A manufactured solution: the Stokes velocity (u, v) and pressure on
    [0, 1/2] x [0, 1], the Darcy pressure on [1/2, 1] x [0, 1]."""

    stokes: tuple[str, str, str]
    darcy: str


SMOOTH = Solution(
    (
        "sin(1/2)*cos(y) - 1 + (x - 1/2)*cos(2*y)",
        "sin(y)*(3/2 - x)",
        "cos(1/2)*cos(y) + 1/2 + cos(2*y) + (x - 1/2)*sin(y)",
    ),
    "cos(x)*cos(y) + x",
)
# With s = x - 1/2: u = -b + s d + s^2, v = m (1 - s) + s^2 y, p = g + s h,
# p_D = a + s b + s^2 y, b = 1 + y^2, d = y - y^2, m = 1 + y + y^2,
# g = 1 + y, h = 2 - y, a = g - d. Velocity and Darcy pressure in Q2, the
# Stokes pressure in Q1, every interface trace of degree 2, so the
# discretization reproduces it.
POLYNOMIAL = Solution(
    (
        "-(1 + y**2) + (x - 1/2)*(y - y**2) + (x - 1/2)**2",
        "(1 + y + y**2)*(3/2 - x) + (x - 1/2)**2*y",
        "1 + y + (x - 1/2)*(2 - y)",
    ),
    "1 + y**2 + (x - 1/2)*(1 + y**2) + (x - 1/2)**2*y",
)


@dataclass(frozen=True)
class InterfacePressure(PortQuantity):
    """The interface pressure: Darcy's pressure trace, Stokes's normal
    stress."""

    def matching_sign(self) -> int:
        """-1: matched by continuity."""
        return -1


@dataclass(frozen=True)
class NormalVelocity(PortQuantity):
    """The normal velocity, tested against the skeleton basis."""

    def matching_sign(self) -> int:
        """+1: matched by mass balance."""
        return 1


class _VelocityForcing:
    """The momentum forcing in ``StokesPhysics``'s layout: values (npts,
    ndim)."""

    def __init__(self, supplier: Any) -> None:
        self._supplier = supplier

    def __call__(self, points: Np, time: float) -> Np:
        return np.asarray(self._supplier(np.atleast_2d(points), time)).T

    def is_time_dependent(self) -> bool:
        return bool(self._supplier.is_time_dependent())


def _stokes_terms(transient: bool = False) -> List[TermProtocol]:
    """Momentum rows div(-nu grad u_i + p e_i), with du_i/dt when
    ``transient``, and the continuity source."""
    terms: List[TermProtocol] = []
    if transient:
        terms += [TimeDerivative("u"), TimeDerivative("v")]
    for i, row in enumerate(("u", "v")):
        fluxes = [
            f"-({NU})*{row}_{c}" + (" + p" if i == j else "")
            for j, c in enumerate(("x", "y"))
        ]
        terms.append(Divergence(row, fluxes))
    terms.append(Source("p", "u_x + v_y"))
    return terms


def stokes_problem(
    solution: Solution = SMOOTH, transient: bool = False
) -> ManufacturedProblem:
    """The Stokes side's strong form and exact solution, with the
    velocity's time derivative when ``transient``; the interface's entry
    only completes the parts (its conditions are built apart)."""
    conditions: dict[str, ConditionProtocol] = {
        "left": Dirichlet(),
        "right": Neumann(),
        "bottom": Neumann(),
        "top": Neumann(),
    }
    return ManufacturedProblem(
        dict(zip(("u", "v", "p"), solution.stokes)),
        _stokes_terms(transient),
        Box([(0.0, 0.5), (0.0, 1.0)]),
        conditions,
    )


def _bjs(coords: Np) -> Np:
    """Per-component Robin coefficients (0, beta) on the interface: the
    normal component carries the port load, the tangential one BJS."""
    npts = coords.shape[1]
    return np.vstack([np.zeros(npts), np.full(npts, BETA)])


@dataclass(frozen=True)
class StokesSide:
    """The Stokes side's composed system before a response is built from
    it: the system, its port DOFs (node-major velocity components) and
    transfer, its problem and basis."""

    system: Any
    ports: list[int]
    transfer: PortTransferProtocol[Np, Np]
    problem: ManufacturedProblem
    basis: MixedBasis[Np]


def stokes_side(
    ncells: int,
    bjs: bool,
    sign: float,
    solution: Solution,
    skeleton: Skeleton,
    transient: bool = False,
) -> StokesSide:
    """Taylor-Hood Stokes on [0, 1/2] x [0, 1] (module docstring), its
    port every interface velocity DOF reading ``skeleton`` by the normal
    projection with sign ``sign``; with ``transient`` the velocity's time
    derivative joins the momentum rows (``solution`` may then use ``t``).
    BJS when ``bjs``."""
    bkd = _BKD
    problem = stokes_problem(solution, transient)
    mesh = StructuredMesh2D(ncells, ncells, [(0.0, 0.5), (0.0, 1.0)], bkd)
    basis: MixedBasis[Np] = TaylorHood().build(mesh)
    physics = StokesPhysics(
        basis,
        bkd,
        viscosity=NU,
        vel_forcing=_VelocityForcing(vector_forcing(problem, ("u", "v"))),
        pres_forcing=forcing(problem, "p"),
    )
    conditions: List[BoundaryConditionRole[Np]] = [
        bc
        for bc in vector_boundary_conditions(
            problem, basis, ("u", "v"), field_name="velocity"
        )
        if not (isinstance(bc, NeumannBC) and bc.boundary_name() == "right")
    ]
    velocity = basis.field("velocity")
    if not isinstance(velocity, ComponentDofsBasisProtocol):
        raise TypeError("the velocity field is not a vector field")
    if bjs:
        conditions.append(RobinBC(velocity, "right", TimeIndependent(_bjs), 0.0, bkd))
    system = compose_galerkin_system(physics, conditions)
    points: Np = np.asarray(basis.dof_coordinates())
    # Each component's interface DOFs along the interface; node-major.
    along = [
        sorted(
            (int(d) for d in np.asarray(velocity.get_dofs("right", (c,)))),
            key=lambda d: float(points[1, d]),
        )
        for c in range(2)
    ]
    if not np.allclose(points[:, along[0]], points[:, along[1]]):
        raise ValueError("the velocity components' interface DOFs differ")
    ports = [d for pair in zip(*along) for d in pair]
    nnodes = len(along[0])
    elements = [list(range(2 * e, 2 * e + 3)) for e in range(ncells)]
    trace = LagrangeSegmentTrace(
        points[:, along[0]], elements, (2 + skeleton.nskel + 1) // 2 + 1, bkd
    )
    stack: PortAssemblyProtocol[Np, CSRFormat, Np, Np] = ScipyAssemblyStack(np.float64)
    coupling = port_trace_coupling(
        skeleton.space,
        trace,
        skeleton.geometry,
        skeleton.faces,
        skeleton.dofs,
        AverageIncidentFaces(),
        stack,
    )
    inner = PortMatrixTransfer(
        BackendPortRows(skeleton.space.descriptor(), skeleton.dofs, bkd), coupling
    )
    normals = np.tile([[1.0, 0.0]], (nnodes, 1))
    transfer = NormalProjectionTransfer(inner, normals, bkd, sign)
    return StokesSide(system, ports, transfer, problem, basis)


class StokesDarcyStrip:
    """The coupled problem, its skeleton solve and each side's errors.

    Parameters
    ----------
    ncells
        Cells per direction on each side.
    nskel
        Skeleton nodes per face.
    bjs
        Impose Beavers-Joseph-Saffman; False leaves the tangential
        traction free (the BJS-dropped control).
    stokes_sign
        The Stokes transfer's sign; +1 flips the normal (the control).
    solution
        The manufactured solution.
    """

    def __init__(
        self,
        ncells: int,
        nskel: int,
        bjs: bool = True,
        stokes_sign: float = -1.0,
        solution: Solution = SMOOTH,
    ) -> None:
        bkd = _BKD
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
        self._stokes(ncells, bjs, stokes_sign, solution)
        faces, dofs = self._faces_and_dofs(1)
        darcy = skfem_adr_piece(
            Coefficients(K, None, None),
            solution.darcy,
            Skeleton(self._space, self._geometry, faces, dofs, nskel),
            ncells,
        )
        self._darcy = darcy
        self._darcy_response: LocalResponseProtocol[Np] = DeclaredKinds(
            darcy.response, InterfacePressure(), NormalVelocity()
        )

    def _faces_and_dofs(self, subdomain: int) -> tuple[list[int], list[int]]:
        faces = [
            f
            for f in self._topo.subdomain_entities(subdomain)
            if f in self._space.faces()
        ]
        return faces, list(subdomain_port_dofs(self._topo, self._space, subdomain))

    def _stokes(self, ncells: int, bjs: bool, sign: float, solution: Solution) -> None:
        faces, dofs = self._faces_and_dofs(0)
        side = stokes_side(
            ncells,
            bjs,
            sign,
            solution,
            Skeleton(self._space, self._geometry, faces, dofs, self._nskel),
        )
        self._stokes_transfer: PortTransferProtocol[Np, Np] = side.transfer
        self._stokes_response = skfem_loaded_response(
            side.system,
            side.ports,
            RecomputeAlways(ScipySparseLU()),
            newton(),
            receives=InterfacePressure(),
            returns=NormalVelocity(),
        )
        self._stokes_problem, self._stokes_basis = side.problem, side.basis

    def _pieces(
        self,
    ) -> list[tuple[LocalResponseProtocol[Np], PortTransferProtocol[Np, Np]]]:
        return [
            (self._stokes_response, self._stokes_transfer),
            (self._darcy_response, self._darcy.transfer),
        ]

    def skeleton(self, subdomain: int) -> Skeleton:
        """The skeleton as side ``subdomain`` reads it: 0 Stokes, 1
        Darcy."""
        faces, dofs = self._faces_and_dofs(subdomain)
        return Skeleton(self._space, self._geometry, faces, dofs, self._nskel)

    def space(self) -> ContinuousSkeletonSpace[Np]:
        """The skeleton space."""
        return self._space

    def states(self, lam: Np) -> tuple[Np, Np]:
        """The Stokes and Darcy states for the skeleton values ``lam``.
        Shapes: (nstates, 1)."""
        stokes, _ = self._state(self._stokes_response, self._stokes_transfer, lam)
        darcy, _ = self._state(self._darcy_response, self._darcy.transfer, lam)
        return stokes[:, None], darcy[:, None]

    def darcy_pinned(self) -> LocalResponseProtocol[Np]:
        """The Darcy side's pinned response, before its kinds are
        declared: a trace in, a flux out."""
        return self._darcy.response

    def layout(self) -> KindLayout:
        """The kind layout of the two pieces' declared kinds."""
        port_dofs: Sequence[Sequence[int]] = [
            self._faces_and_dofs(0)[1],
            self._faces_and_dofs(1)[1],
        ]
        return kind_layout(
            self._space.descriptor().size,
            port_dofs,
            [self._stokes_response, self._darcy_response],
        )

    def solve_skeleton(self) -> Np:
        """The interface pressure at the skeleton DOFs. Shape: (ndofs, 1)"""
        bkd, descriptor = _BKD, self._space.descriptor()
        space = BackendVectorSpace(descriptor, bkd)
        residual: SkeletonResidual[Np, Np, Np, DenseFormat[Np]] = SkeletonResidual(
            space,
            [CoupledSubdomain(r, t) for r, t in self._pieces()],
            DenseSchur(bkd),
        )
        lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(BackendDenseLU(bkd))
        outer: Newton[Np, Np] = Newton(atol=1e-10, rtol=0.0, maxiters=20)
        point = SteadySystem(residual, space, lu).solve(outer, space.zeros(1))
        if not point.converged():
            raise RuntimeError("the skeleton solve did not converge")
        return point.solution()

    def _state(
        self,
        response: LocalResponseProtocol[Np],
        transfer: PortTransferProtocol[Np, Np],
        lam: Np,
    ) -> tuple[Np, Np]:
        point = response.linearize(transfer.scatter(lam), _BKD.zeros((0, 1)), STEADY)
        if not isinstance(point, SteadyPinnedPoint):
            raise TypeError("expected a SteadyPinnedPoint")
        return point.state()[:, 0], point.output()

    def errors(self, lam: Np) -> dict[str, float]:
        """L2 errors of the Stokes velocity and pressure and the Darcy
        pressure for the skeleton values ``lam``."""
        stokes, _ = self._state(self._stokes_response, self._stokes_transfer, lam)
        darcy, _ = self._state(self._darcy_response, self._darcy.transfer, lam)
        velocity = GalerkinApproximation(
            self._stokes_problem,
            ["u", "v", "p"],
            self._stokes_basis,
            stokes,
            8,
            measured=["u", "v"],
        )
        pressure = GalerkinApproximation(
            self._stokes_problem,
            ["u", "v", "p"],
            self._stokes_basis,
            stokes,
            8,
            measured=["p"],
        )
        darcy_error = GalerkinApproximation(
            self._darcy.problem, ["u"], self._darcy.basis, darcy, 8
        )
        return {
            "velocity": velocity.l2_error(),
            "pressure": pressure.l2_error(),
            "darcy": darcy_error.l2_error(),
        }

    def gathered(self, lam: Np) -> tuple[Np, Np]:
        """Each side's output gathered onto the skeleton, Stokes first."""
        size = self._space.descriptor().size
        out = []
        for response, transfer in self._pieces():
            _, output = self._state(response, transfer, lam)
            out.append(transfer.gather_add(output, np.zeros((size, 1))))
        return out[0], out[1]

    def exact_pressure(self) -> Np:
        """The exact interface pressure at the skeleton DOFs. Shape:
        (ndofs, 1)"""
        values = solution_at(self._darcy.problem, self._coords, _BKD)[0]
        return np.asarray(values)[:, None]

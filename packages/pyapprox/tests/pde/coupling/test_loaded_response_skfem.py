r"""skfem subdomains as port-loaded (Neumann-to-Dirichlet) responses. NumPy
only: skfem is.

The unit square cut at x = 1/2, Q1 meshes on the skeleton's Lobatto
grids (``tests/_helpers/coupling_skfem.py``), :math:`-\Delta u + s(u) =
f` with u = 1 on the left edge, u = 2 on the right, natural on top and
bottom, s linear or cubic.

- The right piece as an NtD response (the tested outward flux in, the
  trace out) inverts its DtN response: given the flux the DtN response
  returns for the single-domain trace, it returns that trace (1e-10).
- The left piece DtN and the right one NtD, coupled through the kind
  layout (one trace and one flux unknown per interface DOF, the fixed
  coupling at every mixed DOF): the trace unknowns and both states equal
  the single-domain solve (1e-10), for both sinks.
- The NtD trace tangent passes central differences (cubic sink).
- ``vjp_trace`` passes a dot test against ``jvp_trace`` (rtol 1e-12,
  cubic sink) for the right piece as DtN and as NtD, through the sparse
  LU's transposed solve.
- The port load's transpose: <B T x, y> = <x, T^T B^T y> for a dense T.
- Refusals: a port DOF that is an essential row; an output outside its
  protocol; no port DOFs; a wrong shape or window at linearize.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Sequence

import numpy as np

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
from pyapprox.pde.coupling.kind_layout import kind_layout
from pyapprox.pde.coupling.loaded_response import PortLoad, SteadyLoadedResponse
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import SteadyPinnedPoint
from pyapprox.pde.coupling.ports import BackendPortRows, SplitRowsTransfer
from pyapprox.pde.coupling.responses import Flux, TimeWindow, Trace
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    DenseSchur,
    RowCoupling,
    SkeletonResidual,
)
from pyapprox.pde.coupling.stacks.galerkin import skfem_loaded_response
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendVectorSpace,
    DenseFormat,
    backend_dtype_name,
)
from pyapprox.util.linalg.operators import SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import CSRFormat, NumpyRowSet, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling_skfem import (
    STEADY,
    CoupledProblem,
    Np,
    TensorQuadMesh,
    Value,
    box_problem,
    newton,
    node_at,
)

_BKD = NumpyBkd()


def _forcing(x: Np) -> Np:
    return np.asarray(1.0 + x[0] + 2.0 * x[1])


def _sink(cubic: bool) -> tuple[Any, Any]:
    # The skfem physics adds R(u) as a source, so a sink is negative.
    if cubic:
        return (lambda x, u: -(u + u**3)), (lambda x, u: -(1.0 + 3.0 * u**2))
    return (lambda x, u: -u), (lambda x, u: -np.ones_like(u))


def _builder(cubic: bool) -> Any:
    def build(
        mesh: TensorQuadMesh, sides: Sequence[tuple[str, Value]], exempt: set[int]
    ) -> Any:
        bkd = mesh.bkd()
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=ConstantDiffusion(1.0),
            bkd=bkd,
            reaction=CallableReaction(*_sink(cubic)),
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


def _problem(cubic: bool) -> CoupledProblem:
    return box_problem((2, 1), _builder(cubic), [("left", 1.0), ("right", 2.0)])


def _ntd(problem: CoupledProblem, subdomain: int) -> tuple[Any, Np, tuple[int, ...]]:
    system, nodes, ports, dofs = problem.port_system(subdomain)
    response = skfem_loaded_response(
        system, ports, RecomputeAlways(ScipySparseLU()), newton()
    )
    return response, nodes, dofs


def _reference_at(problem: CoupledProblem, points: Np) -> Np:
    reference, ref_nodes = problem.single_domain()
    rows = problem.mesh_dofs(
        [node_at(ref_nodes, points[:, j]) for j in range(points.shape[1])]
    )
    return np.asarray(reference[rows])


@pytest.mark.parametrize("cubic", [False, True])
def test_ntd_inverts_the_dtn_response(cubic: bool) -> None:
    problem = _problem(cubic)
    dtn, nodes, dofs = problem.piece(1)
    ntd, _, _ = _ntd(problem, 1)
    skel = problem.skeleton_coords()
    trace = _reference_at(problem, skel[:, list(dofs)])
    params = _BKD.zeros((0, 1))
    flux = dtn.linearize(trace, params, STEADY).output()
    back = ntd.linearize(flux, params, STEADY).output()
    _BKD.assert_allclose(back, trace, atol=1e-10)


@pytest.mark.parametrize("cubic", [False, True])
def test_dtn_beside_ntd_equals_the_single_domain(cubic: bool) -> None:
    problem = _problem(cubic)
    dtn, dtn_nodes, dtn_dofs = problem.piece(0)
    ntd, ntd_nodes, ntd_dofs = _ntd(problem, 1)
    assert (dtn.receives(), ntd.receives()) == (Trace(), Flux())
    ndofs = problem.space.descriptor().size
    layout = kind_layout(ndofs, [dtn_dofs, ntd_dofs], [dtn, ntd])
    descriptor = SpaceDescriptor("mixed", layout.nunknowns(), backend_dtype_name(_BKD))
    subdomains = [
        CoupledSubdomain(
            piece,
            SplitRowsTransfer(
                BackendPortRows(descriptor, layout.inputs[i], _BKD),
                BackendPortRows(descriptor, layout.outputs[i], _BKD),
            ),
        )
        for i, piece in enumerate((dtn, ntd))
    ]
    entries = layout.coupling
    coupling = RowCoupling(
        BackendPortRows(descriptor, [e.unknown for e in entries], _BKD),
        BackendPortRows(descriptor, [e.row for e in entries], _BKD),
        _BKD.asarray([[float(e.sign)] for e in entries]),
    )
    space = BackendVectorSpace(descriptor, _BKD)
    residual: SkeletonResidual[Np, Np, Np, DenseFormat[Np]] = SkeletonResidual(
        space, subdomains, DenseSchur(_BKD), coupling=coupling
    )
    lu: RecomputeAlways[Np, DenseFormat[Np]] = RecomputeAlways(BackendDenseLU(_BKD))
    outer: Newton[Np, Np] = Newton(atol=1e-11, rtol=1e-11, maxiters=20)
    point = SteadySystem(residual, space, lu).solve(outer, space.zeros(1))
    assert point.converged()
    solution = point.solution()
    skel = problem.skeleton_coords()
    traces = [u for u, (_, kind) in enumerate(layout.unknowns) if kind == Trace()]
    trace_dofs = [layout.unknowns[u][0] for u in traces]
    _BKD.assert_allclose(
        solution[traces], _reference_at(problem, skel[:, trace_dofs]), atol=1e-10
    )
    for sub, nodes in zip(subdomains, (dtn_nodes, ntd_nodes)):
        solved = sub.response.linearize(
            sub.transfer.scatter(solution), _BKD.zeros((0, 1)), STEADY
        )
        assert isinstance(solved, SteadyPinnedPoint)
        state = solved.state()
        _BKD.assert_allclose(state, _reference_at(problem, nodes), atol=1e-10)


def test_ntd_trace_tangent_matches_finite_differences() -> None:
    problem = _problem(True)
    ntd, _, _ = _ntd(problem, 1)
    nport = ntd.nport()
    params = _BKD.zeros((0, 1))

    def fun(samples: Np) -> Np:
        cols = [
            ntd.linearize(samples[:, j : j + 1], params, STEADY).output()
            for j in range(samples.shape[1])
        ]
        return np.hstack(cols)

    def jvp(sample: Np, vec: Np) -> Np:
        tangent = ntd.linearize(sample, params, STEADY).derivatives().jvp_trace
        assert tangent is not None
        result: Np = tangent(vec)
        return result

    sample = np.linspace(-0.4, 0.3, nport)[:, None]
    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=nport, nvars=nport, fun=fun, jvp=jvp, bkd=_BKD)
    ).check_derivatives(sample, relative=True, central=True)[0]
    assert _BKD.to_float(_BKD.min(errors)) <= 1e-6


@pytest.mark.parametrize("loaded", [False, True])
def test_vjp_trace_is_the_transpose_of_jvp_trace(loaded: bool) -> None:
    """<J V, W> = <V, J^T W> on three-column blocks for the cubic
    subdomain, pinned (reaction flux) or loaded (port rows), through the
    sparse LU's transposed solve."""
    problem = _problem(True)
    response = _ntd(problem, 1)[0] if loaded else problem.piece(1)[0]
    nport = response.nport()
    sample = np.linspace(-0.4, 0.3, nport)[:, None]
    point = response.linearize(sample, _BKD.zeros((0, 1)), STEADY)
    bundle = point.derivatives()
    jvp, vjp = bundle.jvp_trace, bundle.vjp_trace
    assert jvp is not None and vjp is not None
    rng = np.random.default_rng(5)
    v, w = rng.normal(size=(nport, 3)), rng.normal(size=(nport, 3))
    lhs, rhs = float(np.sum(jvp(v) * w)), float(np.sum(v * vjp(w)))
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


def _port_data(nstates: int, ports: list[int]) -> PortPinnedData[Np, Np]:
    states = BackendVectorSpace(SpaceDescriptor("states", nstates), _BKD)
    params = BackendVectorSpace(SpaceDescriptor("params", len(ports)), _BKD)
    return PortPinnedData(
        NumpyRowSet(states.descriptor(), ports),
        BackendPortRows(states.descriptor(), ports, _BKD),
        BackendPortRows(params.descriptor(), list(range(len(ports))), _BKD),
        states,
        params,
    )


def test_port_load_transpose_is_its_adjoint() -> None:
    rng = np.random.default_rng(3)
    data = _port_data(6, [4, 1, 2])
    port = SpaceDescriptor("port", 3)
    matrix = rng.normal(size=(3, 3))
    load = PortLoad(data, BackendDenseOperator(matrix, port, _BKD, range=port))
    x, y = rng.normal(size=(3, 2)), rng.normal(size=(6, 2))
    lhs = float(np.sum(load.apply(x) * y))
    rhs = float(np.sum(x * load.apply_transpose(y)))
    assert abs(lhs - rhs) < 1e-12


def test_refusals() -> None:
    problem = _problem(False)
    system, _, ports, _ = problem.port_system(1)
    setups: RecomputeAlways[Np, CSRFormat] = RecomputeAlways(ScipySparseLU())
    with pytest.raises(ValueError):
        skfem_loaded_response(system, [], setups, newton())
    # A right-edge DOF carries the essential u = 2.
    disc = skfem_discretization(system)
    essential = sorted(set(range(disc.space().descriptor().size)) - set(ports))
    right = [
        d
        for d in essential
        if abs(float(problem.port_system(1)[1][0, d]) - 1.0) < 1e-12
    ]
    with pytest.raises(ValueError, match="essential"):
        skfem_loaded_response(system, [right[0]], setups, newton())
    residual = galerkin_steady_residual(disc)
    data = _port_data(disc.space().descriptor().size, ports)
    with pytest.raises(TypeError):
        SteadyLoadedResponse(
            residual,
            data,
            object(),  # type: ignore[arg-type]
            disc.space(),
            setups,
            newton(),
            _BKD,
        )
    ntd = skfem_loaded_response(system, ports, setups, newton())
    nport = ntd.nport()
    params = _BKD.zeros((0, 1))
    with pytest.raises(ValueError, match="port_data"):
        ntd.linearize(_BKD.zeros((nport + 1, 1)), params, STEADY)
    with pytest.raises(ValueError, match="params"):
        ntd.linearize(_BKD.zeros((nport, 1)), _BKD.zeros((1, 1)), STEADY)
    with pytest.raises(ValueError, match="steady"):
        ntd.linearize(_BKD.zeros((nport, 1)), params, TimeWindow(0.0, 1.0))

r"""A steady subdomain with pinned port rows as a Dirichlet-to-Neumann
response, on the mock P1 operator of ``tests._helpers.mock_elementwise``
(8 uniform cells of a unit interval, no advection), dense NumPy and
Torch.

- Against the closed form: :math:`-u'' = 1` with u(0) = 0.3 and the
  right end a port at :math:`\lambda = 0.8`; the response's flux equals
  ``KirchhoffResponse1D``'s outward flux to 1e-12 (P1 in 1D is nodally
  exact here, and the reaction recovers the end flux).
- Two subdomains, :math:`-u'' + u^2 = 1` on [0, 1] and [1, 2], u(0) = 1,
  u(2) = 2. The independent reference is the single-domain problem on
  [0, 1] with 16 cells, :math:`\kappa = 1/4`, which is the same
  discrete system halved (x = 2y): every element term scales by 1/2, so
  its nodes are the two-subdomain nodes. At its interface value the two
  fluxes balance to 1e-10, and each subdomain's state is the reference
  on its half to 1e-10.
- With the operator's modal coefficients as parameters, the output
  equals that of a response whose coefficients were set directly.
- The trace tangent passes central finite differences (smallest relative
  error below 1e-6) with one port (node 8, :math:`\lambda = 1.4`) and
  two (nodes 4 and 8, :math:`\lambda = (1.2, 0.9)`), with the reaction;
  a point's tangent is refused after the response is linearized again.
- The two pieces coupled through the skeleton and solved by Newton
  (atol 1e-11, matrix-free Schur, GMRES tol 1e-13) take more than one
  step and reach the halved single-domain interface value to 1e-10.
- Refusals: wrong shapes, an unsteady window, parameters without their
  placement.
"""

from typing import Generic, Optional, Sequence

import pytest

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJVPFromCallable,
)
from pyapprox.pde.boundary import DirichletConstraintSet
from pyapprox.pde.coupling.analytic import KirchhoffResponse1D
from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.pinned_ports import PortPinnedData, PortPlacement
from pyapprox.pde.coupling.pinned_response import SteadyPinnedResponse
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.responses import LocalResponseProtocol, TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    MatrixFreeSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.galerkin.system import GalerkinSteadySystem
from pyapprox.pde.galerkin.time_integration.dense_discretization import (
    DenseSteadyDiscretization,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.linalg.operators import MatrixFree
from pyapprox.util.linalg.operators.krylov import GMRES
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.mock_elementwise import (
    MockP1Operator,
    ModalCoefficientParameterization,
)
from tests._helpers.rom_problem import FixedValue

_NCELLS = 8
_STEADY = TimeWindow.steady()
_PARAM = [0.3, -0.2, 0.1, 0.4, -0.1, 0.2]


class _Piece(Generic[Array]):
    """The mock P1 problem with one essential node and one port node."""

    def __init__(
        self,
        bkd: Backend[Array],
        reaction: bool,
        essential: tuple[int, float],
        ports: Sequence[int],
        ncells: int = _NCELLS,
    ) -> None:
        self.bkd, self.n = bkd, ncells + 1
        self.operator: MockP1Operator[Array] = MockP1Operator(ncells, reaction, bkd)
        system = GalerkinSteadySystem(
            self.operator,
            DirichletConstraintSet(
                [FixedValue(essential[0], essential[1], bkd)], self.n, bkd
            ),
        )
        self.discretization: DenseSteadyDiscretization[Array] = (
            DenseSteadyDiscretization(system)
        )
        self.port_nodes = tuple(ports)

    def ports(self, nparams: int) -> PortPinnedData[Array, Array]:
        disc, bkd = self.discretization, self.bkd
        nport = len(self.port_nodes)
        states, theta = disc.space(), disc.parameter_space(nport + nparams)
        return PortPinnedData(
            BackendRowSet(states.descriptor(), self.port_nodes, bkd),
            BackendPortRows(states.descriptor(), self.port_nodes, bkd),
            BackendPortRows(theta.descriptor(), range(nport), bkd),
            states,
            theta,
        )

    def response(
        self, parameters: Optional[ParameterState[Array]] = None
    ) -> SteadyPinnedResponse[Array, Array, DenseFormat[Array]]:
        bkd, disc = self.bkd, self.discretization
        pinned = galerkin_steady_residual(disc, parameters)
        nparams = 0 if parameters is None else parameters.nparams()
        ports = self.ports(nparams)
        physical = None
        if parameters is not None:
            own = pinned.derivatives().param_space
            assert own is not None
            theta, nport = ports.param_space(), ports.nport()
            physical = PortPlacement(
                BackendPortRows(theta.descriptor(), range(nport, nport + nparams), bkd),
                BackendPortRows(own.descriptor(), range(nparams), bkd),
                theta,
                own,
            )
        newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        return SteadyPinnedResponse(
            pinned,
            ports,
            disc.space(),
            RecomputeAlways(BackendDenseLU(bkd)),
            newton,
            bkd,
            parameters,
            physical,
        )


def _no_params(bkd: Backend[Array]) -> Array:
    return bkd.zeros((0, 1))


def test_flux_matches_the_closed_form(bkd: Backend[Array]) -> None:
    response = _Piece(bkd, False, (0, 0.3), (_NCELLS,)).response()
    assert isinstance(response, LocalResponseProtocol)
    lam = bkd.asarray([[0.8]])
    flux = response.linearize(lam, _no_params(bkd), _STEADY).output()
    exact = KirchhoffResponse1D(1.0, 1.0, [1.0], 0.3, None, bkd)
    expected = exact.linearize(lam, _no_params(bkd), _STEADY).output()
    bkd.assert_allclose(flux, expected, atol=1e-12)


def _single_domain(bkd: Backend[Array]) -> Array:
    """The two-piece problem on [0, 2] halved onto [0, 1]: 16 cells,
    kappa = 1/4, the same forcing and reaction."""
    n = 2 * _NCELLS + 1
    operator: MockP1Operator[Array] = MockP1Operator(2 * _NCELLS, True, bkd)
    operator.set_coefficients(0.25 * bkd.ones((n - 1,)), bkd.ones((n,)))
    fixed = [FixedValue(0, 1.0, bkd), FixedValue(n - 1, 2.0, bkd)]
    disc: DenseSteadyDiscretization[Array] = DenseSteadyDiscretization(
        GalerkinSteadySystem(operator, DirichletConstraintSet(fixed, n, bkd))
    )
    residual: PinnedResidual[Array, DenseFormat[Array]] = galerkin_steady_residual(
        disc
    )
    system = SteadySystem(residual, disc.space(), RecomputeAlways(BackendDenseLU(bkd)))
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
    point = system.solve(newton, bkd.zeros((n, 1)))
    assert point.converged()
    return point.state()


def test_two_pieces_balance_at_the_single_domain_value(bkd: Backend[Array]) -> None:
    reference = _single_domain(bkd)
    lam = reference[_NCELLS : _NCELLS + 1]
    left = _Piece(bkd, True, (0, 1.0), (_NCELLS,)).response()
    right = _Piece(bkd, True, (_NCELLS, 2.0), (0,)).response()
    at_left = left.linearize(lam, _no_params(bkd), _STEADY)
    at_right = right.linearize(lam, _no_params(bkd), _STEADY)
    bkd.assert_allclose(
        at_left.output() + at_right.output(), bkd.zeros((1, 1)), atol=1e-10
    )
    bkd.assert_allclose(at_left.state(), reference[: _NCELLS + 1], atol=1e-10)
    bkd.assert_allclose(at_right.state(), reference[_NCELLS:], atol=1e-10)


def test_parameters_reach_the_subdomain(bkd: Backend[Array]) -> None:
    piece = _Piece(bkd, True, (0, 1.0), (_NCELLS,))
    parameters = ParameterState(ModalCoefficientParameterization(piece.operator, bkd))
    response = piece.response(parameters)
    assert response.nparams() == len(_PARAM)
    lam, param = bkd.asarray([[1.4]]), bkd.asarray(_PARAM)
    flux = response.linearize(lam, param[:, None], _STEADY).output()
    direct = _Piece(bkd, True, (0, 1.0), (_NCELLS,))
    ParameterState(ModalCoefficientParameterization(direct.operator, bkd)).set_param(
        param
    )
    expected = direct.response().linearize(lam, _no_params(bkd), _STEADY).output()
    bkd.assert_allclose(flux, expected, rtol=1e-12)


@pytest.mark.parametrize("ports,lam", [((_NCELLS,), [1.4]), ((4, _NCELLS), [1.2, 0.9])])
def test_trace_tangent_passes_finite_differences(
    bkd: Backend[Array], ports: tuple[int, ...], lam: list[float]
) -> None:
    response = _Piece(bkd, True, (0, 1.0), ports).response()
    nport = len(ports)

    def fun(sample: Array) -> Array:
        return response.linearize(sample, _no_params(bkd), _STEADY).output()

    def jvp(sample: Array, direction: Array) -> Array:
        point = response.linearize(sample, _no_params(bkd), _STEADY)
        return point.jvp_trace(direction)

    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=nport, nvars=nport, fun=fun, jvp=jvp, bkd=bkd)
    ).check_derivatives(bkd.asarray([[v] for v in lam]), relative=True, central=True)
    assert bkd.to_float(bkd.min(errors[0])) < 1e-6


def test_a_stale_point_is_refused(bkd: Backend[Array]) -> None:
    response = _Piece(bkd, True, (0, 1.0), (_NCELLS,)).response()
    first = response.linearize(bkd.asarray([[1.4]]), _no_params(bkd), _STEADY)
    response.linearize(bkd.asarray([[1.5]]), _no_params(bkd), _STEADY)
    with pytest.raises(ValueError, match="changed"):
        first.jvp_trace(bkd.ones((1, 1)))


def test_skeleton_newton_solves_the_two_pieces(bkd: Backend[Array]) -> None:
    r"""The two pieces of :math:`-u'' + u^2 = 1` coupled through the
    skeleton and solved by Newton with matrix-free GMRES: the interface
    value is the halved single-domain one to 1e-10."""
    topo = TensorBoxPartition((2,))
    basis = TensorLagrangeBasis(0, 1, bkd)
    numbering = VertexWeightNumbering(topo, basis)
    skeleton = ContinuousSkeletonSpace(topo, basis, numbering)
    space = skeleton.vector_space()
    pieces = [
        _Piece(bkd, True, (0, 1.0), (_NCELLS,)).response(),
        _Piece(bkd, True, (_NCELLS, 2.0), (0,)).response(),
    ]
    subdomains = [
        CoupledSubdomain(
            response,
            MatchingNodalTransfer(
                BackendPortRows(
                    skeleton.descriptor(), subdomain_port_dofs(topo, skeleton, s), bkd
                )
            ),
        )
        for s, response in enumerate(pieces)
    ]
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    residual = SkeletonResidual(space, subdomains, schur)
    setups: RecomputeAlways[Array, MatrixFree] = RecomputeAlways(
        GMRES(space, bkd, tol=1e-13)
    )
    newton: Newton[Array, Array] = Newton(atol=1e-11, rtol=1e-11, maxiters=20)
    point = SteadySystem(residual, space, setups).solve(newton, space.zeros(1))
    assert point.converged() and point.info().iterations > 1
    reference = _single_domain(bkd)
    bkd.assert_allclose(
        point.solution(), reference[_NCELLS : _NCELLS + 1], atol=1e-10
    )


@pytest.mark.parametrize(
    "lam_shape,params_shape,window",
    [((2, 1), (0, 1), _STEADY), ((1, 1), (1, 1), _STEADY), ((1, 1), (0, 1), None)],
)
def test_refusals(
    bkd: Backend[Array],
    lam_shape: Sequence[int],
    params_shape: Sequence[int],
    window: Optional[TimeWindow],
) -> None:
    response = _Piece(bkd, False, (0, 0.3), (_NCELLS,)).response()
    chosen = TimeWindow(0.0, 1.0) if window is None else window
    lam, params = bkd.zeros(tuple(lam_shape)), bkd.zeros(tuple(params_shape))
    with pytest.raises(ValueError):
        response.linearize(lam, params, chosen)


def test_parameters_need_their_placement(bkd: Backend[Array]) -> None:
    piece = _Piece(bkd, True, (0, 1.0), (_NCELLS,))
    parameters = ParameterState(ModalCoefficientParameterization(piece.operator, bkd))
    disc = piece.discretization
    with pytest.raises(ValueError, match="come together"):
        SteadyPinnedResponse(
            galerkin_steady_residual(disc, parameters),
            piece.ports(len(_PARAM)),
            disc.space(),
            RecomputeAlways(BackendDenseLU(bkd)),
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
            bkd,
            parameters,
        )

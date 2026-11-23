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
  ``vjp_trace`` passes a dot test against it there, with the modal
  coefficients as parameters beside the port data; a point's tangent
  and its transpose are refused after the response is linearized again.
- With the six modal coefficients as parameters (ports 4 and 8 at
  :math:`\lambda = (1.2, 0.9)`): ``jvp_param`` passes central finite
  differences (smallest relative error below 1e-6) and ``vjp_param`` a
  dot test against it (1e-12); a response without parameters carries
  neither field. Both include the reaction's explicit dependence
  :math:`-B^\top F_p`, which a dot test alone would not catch.
- The reaction declares that it reads the parameters exactly when its
  residual has them; ``FluxParamDerivatives`` refuses a tangent without
  its adjoint and a derivative of an output reading no parameters.
- The two pieces coupled through the skeleton and solved by Newton
  (atol 1e-11, matrix-free Schur, GMRES tol 1e-13) take more than one
  step and reach the halved single-domain interface value to 1e-10.
- The two pieces on the skeleton with the left piece's six modal
  coefficients as the coupled parameters (:class:`CoupledParameters`):
  :math:`S(\lambda; p)` at :math:`\lambda = 1.3` equals the pieces'
  outputs with the coefficients set directly (1e-12), and setting
  :math:`p` moves the version; the parameter Jacobian passes central
  differences (below 1e-6) and its transpose a dot test (1e-12).
- Refusals: wrong shapes, an unsteady window, parameters without their
  placement; on the skeleton, a response with parameters but no
  coupled parameters, rows of the wrong count, the wrong entry count,
  and setting parameters on a residual without them.
"""

from typing import Any, Generic, Optional, Sequence

import numpy as np
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
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.pinned_ports import PortPinnedData, PortPlacement
from pyapprox.pde.coupling.pinned_response import (
    FluxParamDerivatives,
    PinnedReactionFlux,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.responses import LocalResponseProtocol, TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledParameters,
    CoupledSubdomain,
    MatrixFreeSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
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
    BackendVectorSpace,
    DenseFormat,
)
from pyapprox.util.linalg.operators import MatrixFree, SpaceDescriptor
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
            PinnedReactionFlux(pinned, ports),
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
    residual: PinnedResidual[Array, DenseFormat[Array]] = galerkin_steady_residual(disc)
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


@pytest.mark.parametrize("ports,lam", [((_NCELLS,), [1.4]), ((4, _NCELLS), [1.2, 0.9])])
def test_vjp_trace_is_the_transpose_of_jvp_trace(
    bkd: Backend[Array], ports: tuple[int, ...], lam: list[float]
) -> None:
    """<J V, W> = <V, J^T W> on three-column blocks, with the reaction and
    parameters beside the port data."""
    piece = _Piece(bkd, True, (0, 1.0), ports)
    parameters = ParameterState(ModalCoefficientParameterization(piece.operator, bkd))
    response = piece.response(parameters)
    point = response.linearize(
        bkd.asarray([[v] for v in lam]), bkd.asarray(_PARAM)[:, None], _STEADY
    )
    bundle = point.derivatives()
    jvp, vjp = bundle.jvp_trace, bundle.vjp_trace
    assert jvp is not None and vjp is not None
    n = len(ports)
    v = bkd.asarray([[1.0, 0.5, -2.0], [0.3, -1.0, 0.7]][:n])
    w = bkd.asarray([[-0.6, 2.0, 0.1], [1.4, 0.2, -0.9]][:n])
    lhs = bkd.sum(jvp(v) * w)
    rhs = bkd.sum(v * vjp(w))
    bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-12)
    with pytest.raises(ValueError, match="weights"):
        vjp(bkd.zeros((n + 1, 1)))


def _parametric(bkd: Backend[Array]) -> SteadyPinnedResponse[Array, Array, Any]:
    piece = _Piece(bkd, True, (0, 1.0), (4, _NCELLS))
    return piece.response(
        ParameterState(ModalCoefficientParameterization(piece.operator, bkd))
    )


def test_parameter_tangent_passes_finite_differences(bkd: Backend[Array]) -> None:
    response = _parametric(bkd)
    lam = bkd.asarray([[1.2], [0.9]])

    def fun(sample: Array) -> Array:
        return response.linearize(lam, sample, _STEADY).output()

    def jvp(sample: Array, direction: Array) -> Array:
        point = response.linearize(lam, sample, _STEADY)
        jvp_param = point.derivatives().jvp_param
        assert jvp_param is not None
        return jvp_param(direction)

    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(
            nqoi=2, nvars=len(_PARAM), fun=fun, jvp=jvp, bkd=bkd
        )
    ).check_derivatives(bkd.asarray(_PARAM)[:, None], relative=True, central=True)
    assert bkd.to_float(bkd.min(errors[0])) < 1e-6


def test_vjp_param_is_the_transpose_of_jvp_param(bkd: Backend[Array]) -> None:
    """<J_p V, W> = <V, J_p^T W> on three-column blocks; a response
    without parameters carries no parameter fields."""
    point = _parametric(bkd).linearize(
        bkd.asarray([[1.2], [0.9]]), bkd.asarray(_PARAM)[:, None], _STEADY
    )
    bundle = point.derivatives()
    jvp, vjp = bundle.jvp_param, bundle.vjp_param
    assert jvp is not None and vjp is not None
    rng = np.random.default_rng(6)
    v = bkd.asarray(rng.normal(size=(len(_PARAM), 3)))
    w = bkd.asarray(rng.normal(size=(2, 3)))
    lhs = bkd.sum(jvp(v) * w)
    rhs = bkd.sum(v * vjp(w))
    bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-12)
    assert vjp(w).shape == (len(_PARAM), 3)
    with pytest.raises(ValueError, match="directions"):
        jvp(bkd.zeros((2, 1)))
    plain = _Piece(bkd, True, (0, 1.0), (_NCELLS,)).response()
    unparametric = plain.linearize(
        bkd.asarray([[1.4]]), _no_params(bkd), _STEADY
    ).derivatives()
    assert unparametric.jvp_param is None and unparametric.vjp_param is None


def test_flux_parameter_dependence_is_declared(bkd: Backend[Array]) -> None:
    """The reaction reads the parameters when its residual has them and
    not otherwise; the bundle's fields come together, and an output
    reading no parameters has none."""
    piece = _Piece(bkd, True, (0, 1.0), (_NCELLS,))
    disc = piece.discretization
    plain = galerkin_steady_residual(disc)
    assert (
        not PinnedReactionFlux(plain, piece.ports(0)).param_derivatives().reads_params
    )
    parametric = galerkin_steady_residual(
        disc, ParameterState(ModalCoefficientParameterization(piece.operator, bkd))
    )
    declared = PinnedReactionFlux(parametric, piece.ports(len(_PARAM)))
    bundle = declared.param_derivatives()
    assert bundle.reads_params and bundle.available()
    assert not FluxParamDerivatives.unavailable().available()
    assert FluxParamDerivatives.independent().available()
    with pytest.raises(ValueError, match="together"):
        FluxParamDerivatives(True, tangent=lambda s, v: v)
    with pytest.raises(ValueError, match="no derivative"):
        FluxParamDerivatives(False, lambda s, v: v, lambda s, w: w)


def test_a_stale_point_is_refused(bkd: Backend[Array]) -> None:
    response = _Piece(bkd, True, (0, 1.0), (_NCELLS,)).response()
    first = response.linearize(bkd.asarray([[1.4]]), _no_params(bkd), _STEADY)
    response.linearize(bkd.asarray([[1.5]]), _no_params(bkd), _STEADY)
    with pytest.raises(ValueError, match="changed"):
        first.jvp_trace(bkd.ones((1, 1)))
    with pytest.raises(ValueError, match="changed"):
        first.vjp_trace(bkd.ones((1, 1)))


def test_skeleton_newton_solves_the_two_pieces(bkd: Backend[Array]) -> None:
    r"""The two pieces of :math:`-u'' + u^2 = 1` coupled through the
    skeleton and solved by Newton with matrix-free GMRES: the interface
    value is the halved single-domain one to 1e-10."""
    topo = TensorBoxPartition((2,))
    bases = UniformBases(lobatto_lagrange_basis(0, 1, bkd))
    numbering = VertexWeightNumbering(topo, bases)
    skeleton = ContinuousSkeletonSpace(topo, bases, numbering)
    space = BackendVectorSpace(skeleton.descriptor(), bkd)
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
    bkd.assert_allclose(point.solution(), reference[_NCELLS : _NCELLS + 1], atol=1e-10)


def _parametric_chain(
    bkd: Backend[Array],
) -> tuple[
    SkeletonResidual[Array, Array, Array, MatrixFree], BackendVectorSpace[Array]
]:
    """The two pieces of the Newton test on one skeleton DOF, the left
    piece's six modal coefficients the coupled parameters."""
    topo = TensorBoxPartition((2,))
    bases = UniformBases(lobatto_lagrange_basis(0, 1, bkd))
    skeleton = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    space = BackendVectorSpace(skeleton.descriptor(), bkd)
    left = _Piece(bkd, True, (0, 1.0), (_NCELLS,))
    pieces = [
        left.response(
            ParameterState(ModalCoefficientParameterization(left.operator, bkd))
        ),
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
    params = SpaceDescriptor("params", len(_PARAM))
    parameters = CoupledParameters(
        BackendVectorSpace(params, bkd),
        [BackendPortRows(params, range(len(_PARAM)), bkd), None],
    )
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    residual = SkeletonResidual(space, subdomains, schur, parameters=parameters)
    return residual, space


def test_the_skeleton_reads_the_coupled_parameters(bkd: Backend[Array]) -> None:
    """S(lambda; p) equals the residual of pieces whose coefficients were
    set directly; setting p moves the version and changes S."""
    residual, space = _parametric_chain(bkd)
    lam = bkd.asarray([[1.3]])
    before = residual(lam)
    residual.set_param(bkd.asarray(_PARAM)[:, None])
    assert residual.parameters_version() == 1 and residual.nparams() == len(_PARAM)
    left = _Piece(bkd, True, (0, 1.0), (_NCELLS,))
    ParameterState(ModalCoefficientParameterization(left.operator, bkd)).set_param(
        bkd.asarray(_PARAM)
    )
    right = _Piece(bkd, True, (_NCELLS, 2.0), (0,)).response()
    left_flux = left.response().linearize(lam, _no_params(bkd), _STEADY).output()
    right_flux = right.linearize(lam, _no_params(bkd), _STEADY).output()
    bkd.assert_allclose(residual(lam), left_flux + right_flux, rtol=1e-12)
    assert bkd.to_float(bkd.norm(residual(lam) - before)) > 1e-3


def test_the_parameter_jacobian_passes_finite_differences(
    bkd: Backend[Array],
) -> None:
    """S_p passes central differences of p -> S(lambda; p), and its
    transpose a dot test (1e-12)."""
    residual, space = _parametric_chain(bkd)
    lam = bkd.asarray([[1.3]])

    def fun(sample: Array) -> Array:
        residual.set_param(sample)
        return residual(lam)

    def jvp(sample: Array, direction: Array) -> Array:
        residual.set_param(sample)
        param = residual.derivatives().param
        assert param is not None
        return param(lam).apply(direction)

    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(
            nqoi=1, nvars=len(_PARAM), fun=fun, jvp=jvp, bkd=bkd
        )
    ).check_derivatives(bkd.asarray(_PARAM)[:, None], relative=True, central=True)
    assert bkd.to_float(bkd.min(errors[0])) < 1e-6
    residual.set_param(bkd.asarray(_PARAM)[:, None])
    param = residual.derivatives().param
    assert param is not None
    op = param(lam)
    assert op.properties().transposable
    rng = np.random.default_rng(7)
    v = bkd.asarray(rng.normal(size=(len(_PARAM), 2)))
    w = bkd.asarray(rng.normal(size=(1, 2)))
    lhs, rhs = bkd.sum(op.apply(v) * w), bkd.sum(v * op.apply_transpose(w))
    bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-12)


def test_coupled_parameter_refusals(bkd: Backend[Array]) -> None:
    residual, space = _parametric_chain(bkd)
    subdomains = residual.subdomains()
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    with pytest.raises(ValueError, match="pass parameters"):
        SkeletonResidual(space, subdomains, schur)
    params = SpaceDescriptor("params", 2)
    with pytest.raises(ValueError, match="its rows 2"):
        SkeletonResidual(
            space,
            subdomains,
            schur,
            parameters=CoupledParameters(
                BackendVectorSpace(params, bkd),
                [BackendPortRows(params, range(2), bkd), None],
            ),
        )
    with pytest.raises(ValueError, match="entries"):
        SkeletonResidual(
            space,
            subdomains,
            schur,
            parameters=CoupledParameters(BackendVectorSpace(params, bkd), [None]),
        )
    plain = SkeletonResidual(space, [subdomains[1]], schur)
    with pytest.raises(ValueError, match="no parameters"):
        plain.set_param(bkd.zeros((1, 1)))
    assert plain.derivatives().param is None and plain.nparams() == 0


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
    pinned = galerkin_steady_residual(disc, parameters)
    ports = piece.ports(len(_PARAM))
    with pytest.raises(ValueError, match="come together"):
        SteadyPinnedResponse(
            pinned,
            ports,
            PinnedReactionFlux(pinned, ports),
            disc.space(),
            RecomputeAlways(BackendDenseLU(bkd)),
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
            bkd,
            parameters,
        )


def test_flux_must_satisfy_its_protocol(bkd: Backend[Array]) -> None:
    piece = _Piece(bkd, False, (0, 0.3), (_NCELLS,))
    disc = piece.discretization
    with pytest.raises(TypeError, match="PortFluxProtocol"):
        SteadyPinnedResponse(
            galerkin_steady_residual(disc),
            piece.ports(0),
            object(),  # type: ignore[arg-type]
            disc.space(),
            RecomputeAlways(BackendDenseLU(bkd)),
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
            bkd,
        )

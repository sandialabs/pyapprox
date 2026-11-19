"""A trace-receiving piece beside a flux-receiving one, coupled through
the kind layout, split-row transfers and the fixed coupling C.

[0, 1] is split at 1/2: piece A (u(0) = g0) receives the interface trace
lambda and returns its outward flux; piece B (u(1) = g1) receives its
outward flux mu and returns the trace. The skeleton unknowns are
(lambda, mu), the residual [q_A(lambda) + mu; t_B(mu) - lambda].

Values are checked against the single-domain solution in closed form:
K(u(x)) = K(g0) + (K(g1) - K(g0)) x + f x (1 - x) / 2, so lambda inverts
K(u(1/2)) and mu = -(K(g1) - K(g0)). The NtD response is checked against
the DtN response it inverts and its tangent against finite differences;
the coupled Jacobian against central differences of the residual.
"""

from typing import Any, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJVPFromCallable,
)
from pyapprox.pde.coupling.analytic import (
    KirchhoffNtDResponse1D,
    KirchhoffResponse1D,
)
from pyapprox.pde.coupling.kind_layout import KindLayout, kind_layout
from pyapprox.pde.coupling.ports import BackendPortRows, SplitRowsTransfer
from pyapprox.pde.coupling.responses import LocalResponseProtocol, TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    CouplingProtocol,
    DenseSchur,
    MatrixFreeSchur,
    RowCoupling,
    SkeletonResidual,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendVectorSpace,
    DenseFormat,
    backend_dtype_name,
)
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    MatrixFree,
    SpaceDescriptor,
)
from pyapprox.util.linalg.operators.krylov import GMRES
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.coupling import MockPortRows
from tests._helpers.linalg.mock_stack import MockVector
from tests._helpers.linalg.stacks import mock_case

_LABEL = LinearizationPoint(0, "test", provisional=True)
_G0, _G1, _F = 0.3, -0.2, 1.7
_CONSTANT = (2.0,)
_NONLINEAR = (1.5, 0.0, 1.5)


def _kirchhoff(coefs: Sequence[float], u: float) -> float:
    return sum(c * u ** (j + 1) / (j + 1) for j, c in enumerate(coefs))


def _exact(coefs: Sequence[float]) -> list[float]:
    """(lambda, mu) of the single-domain solution on [0, 1]."""
    k0, k1 = _kirchhoff(coefs, _G0), _kirchhoff(coefs, _G1)
    target = k0 + (k1 - k0) / 2 + _F / 8
    u = _G0
    for _ in range(50):
        conductivity = sum(c * u**j for j, c in enumerate(coefs))
        u -= (_kirchhoff(coefs, u) - target) / conductivity
    return [u, -(k1 - k0)]


def _pair(
    bkd: Backend[Array], coefs: Sequence[float]
) -> tuple[KirchhoffResponse1D[Array], KirchhoffNtDResponse1D[Array]]:
    dtn = KirchhoffResponse1D(0.5, _F, coefs, _G0, None, bkd)
    ntd = KirchhoffNtDResponse1D(0.5, _F, coefs, None, _G1, bkd)
    return dtn, ntd


def _coupling(
    layout: KindLayout, descriptor: SpaceDescriptor, bkd: Backend[Array]
) -> RowCoupling[Array, Array]:
    entries = layout.coupling
    return RowCoupling(
        BackendPortRows(descriptor, [e.unknown for e in entries], bkd),
        BackendPortRows(descriptor, [e.row for e in entries], bkd),
        bkd.asarray([[float(e.sign)] for e in entries]),
    )


def _mixed(
    bkd: Backend[Array], coefs: Sequence[float]
) -> tuple[
    list[CoupledSubdomain[Array, Array]],
    BackendVectorSpace[Array],
    CouplingProtocol[Array],
]:
    pieces: list[LocalResponseProtocol[Array]] = list(_pair(bkd, coefs))
    layout = kind_layout(1, [(0,), (0,)], pieces)
    descriptor = SpaceDescriptor("mixed", layout.nunknowns(), backend_dtype_name(bkd))
    subdomains = [
        CoupledSubdomain(
            piece,
            SplitRowsTransfer(
                BackendPortRows(descriptor, layout.inputs[i], bkd),
                BackendPortRows(descriptor, layout.outputs[i], bkd),
            ),
        )
        for i, piece in enumerate(pieces)
    ]
    return (
        subdomains,
        BackendVectorSpace(descriptor, bkd),
        _coupling(layout, descriptor, bkd),
    )


@pytest.mark.parametrize("coefs", [_CONSTANT, _NONLINEAR])
@pytest.mark.parametrize("port_left", [True, False])
def test_ntd_inverts_the_dtn_response(
    bkd: Backend[Array], coefs: tuple[float, ...], port_left: bool
) -> None:
    ends = (None, 0.4) if port_left else (0.4, None)
    ntd = KirchhoffNtDResponse1D(0.7, _F, coefs, ends[0], ends[1], bkd)
    dtn = KirchhoffResponse1D(0.7, _F, coefs, ends[0], ends[1], bkd)
    params, window = bkd.zeros((0, 1)), TimeWindow.steady()
    mu = bkd.asarray([[-0.35]])
    trace = ntd.linearize(mu, params, window).output()
    bkd.assert_allclose(dtn.linearize(trace, params, window).output(), mu, rtol=1e-12)


@pytest.mark.parametrize("port_left", [True, False])
def test_ntd_tangent_matches_finite_differences(
    bkd: Backend[Array], port_left: bool
) -> None:
    ends = (None, 0.4) if port_left else (0.4, None)
    ntd = KirchhoffNtDResponse1D(0.7, _F, _NONLINEAR, ends[0], ends[1], bkd)
    params, window = bkd.zeros((0, 1)), TimeWindow.steady()

    def fun(samples: Array) -> Array:
        cols = [
            ntd.linearize(samples[:, j : j + 1], params, window).output()
            for j in range(samples.shape[1])
        ]
        return bkd.hstack(cols)

    def jvp(sample: Array, vec: Array) -> Array:
        tangent = ntd.linearize(sample, params, window).derivatives().jvp_trace
        assert tangent is not None
        return tangent(vec)

    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=1, nvars=1, fun=fun, jvp=jvp, bkd=bkd)
    ).check_derivatives(bkd.asarray([[-0.35]]), relative=True, central=True)[0]
    assert bkd.to_float(bkd.min(errors)) <= 1e-6


def test_ntd_refuses_invalid_input(bkd: Backend[Array]) -> None:
    with pytest.raises(ValueError, match="floats"):
        KirchhoffNtDResponse1D(1.0, _F, _CONSTANT, None, None, bkd)
    with pytest.raises(ValueError, match="exactly one end"):
        KirchhoffNtDResponse1D(1.0, _F, _CONSTANT, 0.0, 0.0, bkd)
    ntd = KirchhoffNtDResponse1D(1.0, _F, _CONSTANT, None, 0.0, bkd)
    params, window = bkd.zeros((0, 1)), TimeWindow.steady()
    with pytest.raises(ValueError, match="port_data"):
        ntd.linearize(bkd.zeros((2, 1)), params, window)
    with pytest.raises(ValueError, match="params"):
        ntd.linearize(bkd.zeros((1, 1)), bkd.zeros((1, 1)), window)
    with pytest.raises(ValueError, match="steady"):
        ntd.linearize(bkd.zeros((1, 1)), params, TimeWindow(0.0, 1.0))
    point = ntd.linearize(bkd.zeros((1, 1)), params, window)
    with pytest.raises(ValueError, match="directions"):
        point.jvp_trace(bkd.zeros((2, 1)))
    stalled = KirchhoffNtDResponse1D(
        1.0,
        _F,
        _NONLINEAR,
        None,
        0.0,
        bkd,
        newton=Newton(atol=1e-30, rtol=0.0, maxiters=1),
    )
    with pytest.raises(RuntimeError, match="Kirchhoff"):
        stalled.linearize(bkd.asarray([[0.5]]), params, window)


def test_mixed_residual_is_flux_balance_and_trace_match(bkd: Backend[Array]) -> None:
    subdomains, space, coupling = _mixed(bkd, _NONLINEAR)
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    residual = SkeletonResidual(space, subdomains, schur, coupling=coupling)
    assert residual.coupling() is coupling
    dtn, ntd = _pair(bkd, _NONLINEAR)
    params, window = bkd.zeros((0, 1)), TimeWindow.steady()
    lam, mu = bkd.asarray([[0.1]]), bkd.asarray([[-0.4]])
    expected = bkd.vstack(
        [
            dtn.linearize(lam, params, window).output() + mu,
            ntd.linearize(mu, params, window).output() - lam,
        ]
    )
    bkd.assert_allclose(residual(bkd.vstack([lam, mu])), expected, rtol=1e-12)
    bkd.assert_allclose(
        residual(bkd.asarray([[v] for v in _exact(_NONLINEAR)])),
        bkd.zeros((2, 1)),
        atol=1e-12,
    )


def test_mixed_jacobian_matches_central_differences(bkd: Backend[Array]) -> None:
    subdomains, space, coupling = _mixed(bkd, _NONLINEAR)
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    residual = SkeletonResidual(space, subdomains, schur, coupling=coupling)
    jacobian = residual.derivatives().jacobian
    assert jacobian is not None

    def fun(samples: Array) -> Array:
        cols = [residual(samples[:, j : j + 1]) for j in range(samples.shape[1])]
        return bkd.hstack(cols)

    def jvp(sample: Array, vec: Array) -> Array:
        return jacobian(sample, _LABEL).apply(vec)

    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=2, nvars=2, fun=fun, jvp=jvp, bkd=bkd)
    ).check_derivatives(bkd.asarray([[0.1], [-0.4]]), relative=True, central=True)[0]
    assert bkd.to_float(bkd.min(errors)) <= 1e-6


def test_dense_schur_adds_the_coupling(bkd: Backend[Array]) -> None:
    subdomains, space, coupling = _mixed(bkd, _NONLINEAR)
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    free = SkeletonResidual(space, subdomains, schur, coupling=coupling)
    dense = SkeletonResidual(space, subdomains, DenseSchur(bkd), coupling=coupling)
    state = bkd.asarray([[0.1], [-0.4]])
    free_jacobian, dense_jacobian = (
        free.derivatives().jacobian,
        dense.derivatives().jacobian,
    )
    assert free_jacobian is not None and dense_jacobian is not None
    matrix = dense_jacobian(state, _LABEL).format().matrix()
    bkd.assert_allclose(
        matrix, free_jacobian(state, _LABEL).apply(bkd.eye(2)), rtol=1e-12
    )
    # The off-diagonal is C alone: the pieces read disjoint unknowns.
    bkd.assert_allclose(matrix[0, 1], bkd.asarray(1.0))
    bkd.assert_allclose(matrix[1, 0], bkd.asarray(-1.0))


@pytest.mark.parametrize("coefs", [_CONSTANT, _NONLINEAR])
def test_mixed_pair_matrix_free_equals_single_domain(
    bkd: Backend[Array], coefs: tuple[float, ...]
) -> None:
    subdomains, space, coupling = _mixed(bkd, coefs)
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    residual = SkeletonResidual(space, subdomains, schur, coupling=coupling)
    setups: RecomputeAlways[Array, MatrixFree] = RecomputeAlways(
        GMRES(space, bkd, tol=1e-13)
    )
    point = SteadySystem(residual, space, setups).solve(
        Newton(atol=1e-12, rtol=1e-12, maxiters=20), space.zeros(1)
    )
    assert point.converged()
    bkd.assert_allclose(
        point.solution(), bkd.asarray([[v] for v in _exact(coefs)]), rtol=1e-10
    )


@pytest.mark.parametrize("coefs", [_CONSTANT, _NONLINEAR])
def test_mixed_pair_dense_equals_single_domain(
    bkd: Backend[Array], coefs: tuple[float, ...]
) -> None:
    subdomains, space, coupling = _mixed(bkd, coefs)
    residual: SkeletonResidual[Array, Array, Array, DenseFormat[Array]] = (
        SkeletonResidual(space, subdomains, DenseSchur(bkd), coupling=coupling)
    )
    setups: RecomputeAlways[Array, DenseFormat[Array]] = RecomputeAlways(
        BackendDenseLU(bkd)
    )
    point = SteadySystem(residual, space, setups).solve(
        Newton(atol=1e-12, rtol=1e-12, maxiters=20), space.zeros(1)
    )
    assert point.converged()
    if coefs == _CONSTANT:
        assert point.info().iterations == 1
    bkd.assert_allclose(
        point.solution(), bkd.asarray([[v] for v in _exact(coefs)]), rtol=1e-10
    )


def test_mixed_pair_on_the_mock_stack() -> None:
    bkd, case = NumpyBkd(), mock_case()
    pieces: list[LocalResponseProtocol[NDArray[Any]]] = list(_pair(bkd, _NONLINEAR))
    layout = kind_layout(1, [(0,), (0,)], pieces)
    descriptor = SpaceDescriptor("mixed", layout.nunknowns())
    subdomains = [
        CoupledSubdomain(
            piece,
            SplitRowsTransfer(
                MockPortRows(descriptor, layout.inputs[i]),
                MockPortRows(descriptor, layout.outputs[i]),
            ),
        )
        for i, piece in enumerate(pieces)
    ]
    entries = layout.coupling
    coupling: RowCoupling[MockVector, NDArray[Any]] = RowCoupling(
        MockPortRows(descriptor, [e.unknown for e in entries]),
        MockPortRows(descriptor, [e.row for e in entries]),
        np.asarray([[float(e.sign)] for e in entries]),
    )
    space = case.space(descriptor)
    schur: MatrixFreeSchur[MockVector, NDArray[Any]] = MatrixFreeSchur()
    residual = SkeletonResidual(space, subdomains, schur, coupling=coupling)
    setups: RecomputeAlways[MockVector, MatrixFree] = RecomputeAlways(
        GMRES(space, case.bkd, tol=1e-13)
    )
    point = SteadySystem(residual, space, setups).solve(
        Newton(atol=1e-12, rtol=1e-12, maxiters=20), space.zeros(1)
    )
    assert point.converged()
    case.assert_vector(
        point.solution(),
        case.dense(np.asarray(_exact(_NONLINEAR))[:, None]),
        rtol=1e-10,
    )


def test_split_rows_transfer_reads_and_writes_apart(bkd: Backend[Array]) -> None:
    descriptor = SpaceDescriptor("s", 3, backend_dtype_name(bkd))
    transfer: SplitRowsTransfer[Array, Array] = SplitRowsTransfer(
        BackendPortRows(descriptor, [0, 2], bkd),
        BackendPortRows(descriptor, [1, 0], bkd),
        sign=-1.0,
    )
    assert transfer.nport() == 2
    lam = bkd.asarray([[1.0], [2.0], [3.0]])
    bkd.assert_allclose(transfer.scatter(lam), bkd.asarray([[-1.0], [-3.0]]))
    block = bkd.asarray([[5.0], [7.0]])
    bkd.assert_allclose(
        transfer.gather_add(block, lam), bkd.asarray([[-6.0], [-3.0], [3.0]])
    )
    # <E lam, r> = <lam, E^T r> when inputs and outputs coincide.
    same: SplitRowsTransfer[Array, Array] = SplitRowsTransfer(
        BackendPortRows(descriptor, [2, 0], bkd),
        BackendPortRows(descriptor, [2, 0], bkd),
        sign=-1.0,
    )
    lhs = bkd.sum(same.scatter(lam) * block)
    rhs = bkd.sum(lam * same.gather_add(block, bkd.zeros((3, 1))))
    bkd.assert_allclose(lhs, rhs)


def test_split_rows_and_coupling_refuse_invalid_input(bkd: Backend[Array]) -> None:
    descriptor = SpaceDescriptor("s", 3, backend_dtype_name(bkd))
    one = BackendPortRows(descriptor, [0], bkd)
    two = BackendPortRows(descriptor, [0, 1], bkd)
    with pytest.raises(ValueError, match="rows"):
        SplitRowsTransfer(one, two)
    with pytest.raises(ValueError, match="sign"):
        SplitRowsTransfer(one, one, sign=2.0)
    with pytest.raises(TypeError, match="inputs"):
        SplitRowsTransfer(object(), one)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="signs"):
        RowCoupling(one, two, bkd.ones((1, 1)))
    with pytest.raises(ValueError, match="signs"):
        RowCoupling(one, one, bkd.ones((2, 1)))
    with pytest.raises(TypeError, match="rows"):
        RowCoupling(one, object(), bkd.ones((1, 1)))  # type: ignore[arg-type]
    space = BackendVectorSpace(descriptor, bkd)
    schur: MatrixFreeSchur[Array, Array] = MatrixFreeSchur()
    with pytest.raises(TypeError, match="coupling"):
        SkeletonResidual(
            space,
            [],
            schur,
            coupling=object(),  # type: ignore[arg-type]
        )

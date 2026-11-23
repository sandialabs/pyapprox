"""Closed-form 1D responses: values against independent facts, their
trace directional derivatives against finite differences, and the
transposed products against the directional ones (dot tests).

Value checks are derivation-free: the outward fluxes of a subdomain sum
to minus the integrated source (divergence theorem); constant data with
no source give zero flux; and two halves of [0, 1] meeting at the exact
single-domain value u(1/2) = f/(8k) balance their fluxes.
"""

from typing import Optional

import pytest

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
from pyapprox.pde.coupling.responses import (
    Flux,
    LocalResponseProtocol,
    PortQuantity,
    ResponseDerivatives,
    ResponsePointProtocol,
    TimeWindow,
    Trace,
)
from pyapprox.util.backends.protocols import Array, Backend

_STEADY = TimeWindow.steady()
_LINEAR = (2.0,)
_NONLINEAR = (1.5, 0.0, 1.5)  # k(u) = 1.5 (1 + u^2)
_ENDS = [(None, None), (0.3, None), (None, -0.7)]


def _response(
    bkd: Backend[Array],
    coefs: tuple[float, ...],
    left: Optional[float],
    right: Optional[float],
) -> KirchhoffResponse1D[Array]:
    return KirchhoffResponse1D(0.8, 1.7, coefs, left, right, bkd)


def _flux(r: KirchhoffResponse1D[Array], data: Array) -> Array:
    return r.linearize(data, r.bkd().zeros((0, 1)), _STEADY).output()


@pytest.mark.parametrize("coefs", [_LINEAR, _NONLINEAR])
def test_fluxes_balance_the_source(
    bkd: Backend[Array], coefs: tuple[float, ...]
) -> None:
    r = _response(bkd, coefs, None, None)
    assert isinstance(r, LocalResponseProtocol)
    assert r.receives() == Trace() and r.returns() == Flux()
    kinds: list[PortQuantity] = [Trace(), Flux()]
    assert kinds[0] != kinds[1] and kinds[0] == Trace()
    point = r.linearize(bkd.asarray([[0.4], [-1.2]]), bkd.zeros((0, 1)), _STEADY)
    assert isinstance(point, ResponsePointProtocol)
    assert tuple(point.qoi().shape) == (0, 1)
    bkd.assert_allclose(
        bkd.sum(point.output(), axis=0), bkd.asarray([-1.7 * 0.8]), rtol=1e-12
    )


def test_constant_data_without_source_has_no_flux(bkd: Backend[Array]) -> None:
    r = KirchhoffResponse1D(0.8, 0.0, _NONLINEAR, None, None, bkd)
    bkd.assert_allclose(
        _flux(r, bkd.asarray([[0.6], [0.6]])), bkd.zeros((2, 1)), atol=1e-14
    )


def test_halves_balance_at_the_exact_interface_value(bkd: Backend[Array]) -> None:
    k, f = 2.0, 1.7
    left = KirchhoffResponse1D(0.5, f, (k,), 0.0, None, bkd)
    right = KirchhoffResponse1D(0.5, f, (k,), None, 0.0, bkd)
    lam = bkd.asarray([[f / (8.0 * k)]])
    total = _flux(left, lam) + _flux(right, lam)
    bkd.assert_allclose(total, bkd.zeros((1, 1)), atol=1e-14)
    # Away from it, the imbalance grows with the sum of the two
    # Steklov-Poincare slopes, +k/L each: positive, as for the Galerkin
    # reaction.
    shifted = _flux(left, lam + 0.1) + _flux(right, lam + 0.1)
    bkd.assert_allclose(shifted, bkd.asarray([[0.1 * 2.0 * k / 0.5]]), rtol=1e-12)


@pytest.mark.parametrize("coefs", [_LINEAR, _NONLINEAR])
@pytest.mark.parametrize("ends", _ENDS)
def test_jvp_trace_matches_finite_differences(
    bkd: Backend[Array],
    coefs: tuple[float, ...],
    ends: tuple[Optional[float], Optional[float]],
) -> None:
    r = _response(bkd, coefs, *ends)
    n = r.nport()

    def fun(samples: Array) -> Array:
        return bkd.hstack(
            [_flux(r, samples[:, j : j + 1]) for j in range(samples.shape[1])]
        )

    def jvp(sample: Array, vec: Array) -> Array:
        point = r.linearize(sample, bkd.zeros((0, 1)), _STEADY)
        tangent = point.derivatives().jvp_trace
        if tangent is None:
            raise AssertionError("the response declares jvp_trace")
        return tangent(vec)

    sample = bkd.asarray([[0.4], [-1.1]][:n])
    errors = DerivativeChecker(
        FunctionWithJVPFromCallable(nqoi=n, nvars=n, fun=fun, jvp=jvp, bkd=bkd)
    ).check_derivatives(sample, relative=True, central=True)[0]
    assert bkd.to_float(bkd.min(errors)) <= 1e-6


def test_jvp_trace_acts_on_blocks(bkd: Backend[Array]) -> None:
    r = _response(bkd, _NONLINEAR, None, None)
    point = r.linearize(bkd.asarray([[0.4], [-1.1]]), bkd.zeros((0, 1)), _STEADY)
    tangent = point.derivatives().jvp_trace
    assert tangent is not None
    block = bkd.asarray([[1.0, 0.5, -2.0], [0.3, -1.0, 0.7]])
    columns = bkd.hstack([tangent(block[:, j : j + 1]) for j in range(3)])
    bkd.assert_allclose(tangent(block), columns, rtol=1e-12)


@pytest.mark.parametrize("coefs", [_LINEAR, _NONLINEAR])
@pytest.mark.parametrize("ends", _ENDS)
def test_vjp_trace_is_the_transpose_of_jvp_trace(
    bkd: Backend[Array],
    coefs: tuple[float, ...],
    ends: tuple[Optional[float], Optional[float]],
) -> None:
    """<J V, W> = <V, J^T W> on blocks, one or two ports."""
    r = _response(bkd, coefs, *ends)
    n = r.nport()
    sample = bkd.asarray([[0.4], [-1.1]][:n])
    bundle = r.linearize(sample, bkd.zeros((0, 1)), _STEADY).derivatives()
    jvp, vjp = bundle.jvp_trace, bundle.vjp_trace
    assert jvp is not None and vjp is not None
    v = bkd.asarray([[1.0, 0.5, -2.0], [0.3, -1.0, 0.7]][:n])
    w = bkd.asarray([[-0.6, 2.0, 0.1], [1.4, 0.2, -0.9]][:n])
    lhs = bkd.sum(jvp(v) * w)
    rhs = bkd.sum(v * vjp(w))
    bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-13)
    with pytest.raises(ValueError):
        vjp(bkd.zeros((n + 1, 1)))


def test_ntd_vjp_trace_is_its_own_transpose(bkd: Backend[Array]) -> None:
    r = KirchhoffNtDResponse1D(0.8, 1.7, _NONLINEAR, 0.3, None, bkd)
    bundle = r.linearize(bkd.asarray([[0.5]]), bkd.zeros((0, 1)), _STEADY).derivatives()
    jvp, vjp = bundle.jvp_trace, bundle.vjp_trace
    assert jvp is not None and vjp is not None
    w = bkd.asarray([[0.7, -1.3]])
    bkd.assert_allclose(vjp(w), jvp(w), rtol=1e-14)


def test_invalid_input(bkd: Backend[Array]) -> None:
    with pytest.raises(ValueError, match="vjp_trace comes only with jvp_trace"):
        ResponseDerivatives(vjp_trace=lambda w: w)
    with pytest.raises(ValueError, match="jvp_param comes only with jvp_trace"):
        ResponseDerivatives(jvp_param=lambda v: v)
    with pytest.raises(ValueError, match="vjp_param comes only"):
        ResponseDerivatives(
            jvp_trace=lambda v: v, jvp_param=lambda v: v, vjp_param=lambda w: w
        )
    with pytest.raises(TypeError, match="jvp_param"):
        ResponseDerivatives(jvp_trace=lambda v: v, jvp_param=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        KirchhoffResponse1D(0.0, 1.0, _LINEAR, None, None, bkd)
    with pytest.raises(ValueError):
        KirchhoffResponse1D(1.0, 1.0, (), None, None, bkd)
    with pytest.raises(ValueError):
        KirchhoffResponse1D(1.0, 1.0, _LINEAR, 0.0, 0.0, bkd)
    r = _response(bkd, _LINEAR, 0.3, None)
    none = bkd.zeros((0, 1))
    with pytest.raises(ValueError):
        r.linearize(bkd.zeros((2, 1)), none, _STEADY)
    with pytest.raises(ValueError):
        r.linearize(bkd.zeros((1, 1)), bkd.zeros((1, 1)), _STEADY)
    with pytest.raises(ValueError):
        r.linearize(bkd.zeros((1, 1)), none, TimeWindow(0.0, 1.0))
    with pytest.raises(ValueError):
        TimeWindow(1.0, 0.0)
    tangent = r.linearize(bkd.zeros((1, 1)), none, _STEADY).derivatives().jvp_trace
    assert tangent is not None
    with pytest.raises(ValueError):
        tangent(bkd.zeros((2, 1)))
    assert ResponseDerivatives.none().jvp_trace is None
    with pytest.raises(TypeError):
        ResponseDerivatives(jvp_trace=1.0)  # type: ignore[arg-type]

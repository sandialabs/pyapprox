"""The flow-matching ODE residual declares no derivatives.

Its vector field supplies none, so its bundle is empty: the flow can be
stepped explicitly but not differentiated, and a consumer reading the
bundle refuses it rather than calling the raising ``jacobian``.
"""

from pyapprox.generative.flowmatching.ode_adapter import FlowODEResidual
from pyapprox.util.backends.protocols import Array, Backend


def _vector_field(samples: Array) -> Array:
    """v(t, x) = -x for a 2-state x: rows 1 and 2 of the input."""
    return -samples[1:3]


def test_bundle_is_empty(bkd: Backend[Array]) -> None:
    residual = FlowODEResidual(_vector_field, bkd, nstates=2)
    bundle = residual.derivatives()
    assert bundle.jacobian is None and bundle.param is None
    assert bundle.state_state is None and bundle.param_space is None
    bkd.assert_allclose(
        residual(bkd.asarray([1.0, -2.0]), 0.0), bkd.asarray([-1.0, 2.0])
    )

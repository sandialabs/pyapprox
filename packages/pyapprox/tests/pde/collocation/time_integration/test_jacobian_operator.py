"""Collocation's Jacobian as an operator, through the generic dense wrapper.

The collocation adapter's ``jacobian`` satisfies the wrapper's model
protocol, so collocation needs no operator code of its own. The operator
is compared with the adapter's matrix Jacobian for linear
advection-diffusion and nonlinear Burgers, and its action with a central
difference of the adapter's residual, on both backends.
"""

import math

import pytest

from pyapprox.ode.dense_jacobian_operator import (
    DenseJacobianOperator,
    JacobianMatrixProtocol,
)
from pyapprox.ode.protocols import JacobianOperatorProtocol
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics import (
    AdvectionDiffusionReaction,
    BurgersPhysics1D,
)
from pyapprox.pde.collocation.protocols import PhysicsProtocol
from pyapprox.pde.collocation.time_integration import (
    CollocationPhysicsToODEResidualAdapter,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import LinearizationPoint, SpaceDescriptor

_NPTS = 12


def _adapter(
    name: str, bkd: Backend[Array]
) -> tuple[CollocationPhysicsToODEResidualAdapter[Array], Array]:
    """The adapter, and a smooth nonzero state on the nodes."""
    basis = ChebyshevBasis1D(TransformedMesh1D(_NPTS, bkd), bkd)
    physics: PhysicsProtocol[Array]
    if name == "adr":
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=0.5)
    else:
        physics = BurgersPhysics1D(basis, bkd, viscosity=0.1)
    nodes = basis.nodes()
    state = bkd.sin(math.pi * nodes) + 0.5 * nodes
    return CollocationPhysicsToODEResidualAdapter(physics, bkd), state


@pytest.mark.parametrize("name", ["adr", "burgers"])
class TestCollocationJacobianOperator:
    def test_matches_the_adapter_jacobian(self, bkd: Backend[Array], name: str) -> None:
        adapter, state = _adapter(name, bkd)
        assert isinstance(adapter, JacobianMatrixProtocol)
        descriptor = SpaceDescriptor("state", _NPTS)
        producer: JacobianOperatorProtocol[Array, Array, DenseFormat[Array]] = (
            DenseJacobianOperator(adapter, descriptor, bkd)
        )
        point = LinearizationPoint(0, ("step", 1))
        op = producer.jacobian_operator(state, 0.0, point)
        reference = adapter.jacobian(state, 0.0)
        x = bkd.reshape(bkd.linspace(-1.0, 1.0, 2 * _NPTS), (_NPTS, 2))
        bkd.assert_allclose(op.apply(x), bkd.dot(reference, x), rtol=1e-13)
        bkd.assert_allclose(op.apply_transpose(x), bkd.dot(reference.T, x), rtol=1e-13)
        assert op.linearization() == point

    def test_action_is_the_residual_derivative(
        self, bkd: Backend[Array], name: str
    ) -> None:
        adapter, state = _adapter(name, bkd)
        op = DenseJacobianOperator(
            adapter, SpaceDescriptor("state", _NPTS), bkd
        ).jacobian_operator(state, 0.0, LinearizationPoint(0))
        direction = bkd.cos(bkd.linspace(0.0, 3.0, _NPTS))
        step = 1e-6
        difference = (
            adapter(state + step * direction, 0.0)
            - adapter(state - step * direction, 0.0)
        ) / (2 * step)
        bkd.assert_allclose(
            op.apply(direction[:, None])[:, 0], difference, rtol=1e-6, atol=1e-6
        )

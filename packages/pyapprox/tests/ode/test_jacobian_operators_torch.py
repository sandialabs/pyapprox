"""The dense and matrix-free Jacobian operators keep a torch graph intact.

References come from autodiff of the Lotka-Volterra residual alone, so
they share no code with either hand-written Jacobian: the action must
equal ``bkd.jvp`` of the residual, and differentiating the action with
respect to the state must equal ``bkd.jacobian`` of that jvp. Torch is
optional, so the module is skipped without it.
"""

import pytest

# Skip the module without torch; the imports below are then ordinary ones.
pytest.importorskip("torch")

import torch

from pyapprox.ode.dense_jacobian_operator import DenseJacobianOperator
from pyapprox.ode.matrix_free_jacobian_operator import MatrixFreeJacobianOperator
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.linalg.operators import LinearizationPoint, SpaceDescriptor
from tests._helpers.ode_models import LotkaVolterra

_Y = SpaceDescriptor("y", 2)
_POINT = LinearizationPoint(0)


def _producers(
    bkd: TorchBkd,
) -> list[
    DenseJacobianOperator[torch.Tensor] | MatrixFreeJacobianOperator[torch.Tensor]
]:
    model = LotkaVolterra(bkd)
    return [
        DenseJacobianOperator(model, _Y, bkd),
        MatrixFreeJacobianOperator(model, _Y),
    ]


class TestTorchAutograd:
    def test_action_is_the_autodiff_jvp(self, torch_bkd: TorchBkd) -> None:
        model = LotkaVolterra(torch_bkd)
        y = torch_bkd.asarray([10.0, 5.0])
        v = torch_bkd.asarray([[0.3], [-1.2]])
        expected = torch_bkd.jvp(lambda s: model(s, 0.0), y, v[:, 0])
        for producer in _producers(torch_bkd):
            op = producer.jacobian_operator(y, 0.0, _POINT)
            torch_bkd.assert_allclose(op.apply(v)[:, 0], expected, rtol=1e-14)

    def test_gradient_flows_from_the_action_to_the_state(
        self, torch_bkd: TorchBkd
    ) -> None:
        model = LotkaVolterra(torch_bkd)
        y = torch_bkd.asarray([10.0, 5.0])
        v = torch_bkd.asarray([[0.3], [-1.2]])
        expected = torch_bkd.jacobian(
            lambda s: torch_bkd.jvp(lambda t: model(t, 0.0), s, v[:, 0]), y
        )
        for producer in _producers(torch_bkd):
            actual = torch_bkd.jacobian(
                lambda s: producer.jacobian_operator(s, 0.0, _POINT).apply(v)[:, 0],
                y,
            )
            torch_bkd.assert_allclose(actual, expected, rtol=1e-14)

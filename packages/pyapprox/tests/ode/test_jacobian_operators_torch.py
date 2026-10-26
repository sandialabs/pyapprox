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
from pyapprox.ode.mass_matrix_system import (
    MassMatrixSystem,
    dense_system,
    matrix_free_system,
)
from pyapprox.ode.mass_solve import MassSolve
from pyapprox.ode.matrix_free_jacobian_operator import MatrixFreeJacobianOperator
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    MatrixFree,
    SpaceDescriptor,
)
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


class TestMassMatrixSystemAutograd:
    r"""The residual-form system against autodiff of its own residual
    :math:`F(y, \dot y) = M \dot y - f(y)`, with a nontrivial mass."""

    def _system(
        self, bkd: TorchBkd
    ) -> list[
        MassMatrixSystem[torch.Tensor, DenseFormat[torch.Tensor]]
        | MassMatrixSystem[torch.Tensor, MatrixFree]
    ]:
        model = LotkaVolterra(bkd, mass=bkd.asarray([[2.0, 0.5], [0.5, 1.0]]))
        return [dense_system(model), matrix_free_system(model, nstates=2)]

    def test_shifted_jacobian_is_the_autodiff_jacobian(
        self, torch_bkd: TorchBkd
    ) -> None:
        r""":math:`\sigma\,\partial F/\partial \dot y + \partial F/\partial y`
        from ``bkd.jacobian`` of the residual in each argument."""
        bkd, shift = torch_bkd, 3.0
        y, ydot = bkd.asarray([10.0, 5.0]), bkd.asarray([0.7, -1.3])
        for system in self._system(bkd):
            d_rate = bkd.jacobian(lambda r: system.residual(y, r, 0.0), ydot)
            d_state = bkd.jacobian(lambda s: system.residual(s, ydot, 0.0), y)
            op = system.shifted_jacobian(y, ydot, 0.0, shift, _POINT)
            identity = bkd.eye(2, dtype=bkd.default_dtype())
            bkd.assert_allclose(op.apply(identity), shift * d_rate + d_state)

    def test_gradient_flows_through_the_shifted_action(
        self, torch_bkd: TorchBkd
    ) -> None:
        bkd, shift = torch_bkd, 3.0
        y, ydot = bkd.asarray([10.0, 5.0]), bkd.asarray([0.7, -1.3])
        v = bkd.asarray([[0.3], [-1.2]])
        for system in self._system(bkd):
            expected = bkd.jacobian(
                lambda s: bkd.jvp(lambda t: system.residual(t, ydot, 0.0), s, v[:, 0]),
                y,
            )
            actual = bkd.jacobian(
                lambda s: system.shifted_jacobian(s, ydot, 0.0, shift, _POINT).apply(v)[
                    :, 0
                ],
                y,
            )
            bkd.assert_allclose(actual, expected, rtol=1e-12)


class TestMassSolveAutograd:
    def test_solve_differentiates_through_the_mass(self, torch_bkd: TorchBkd) -> None:
        r""":math:`\partial (M^{-1} b)/\partial M` through ``MassSolve``
        with the dense LU, against autodiff of a plain solve."""
        bkd = torch_bkd
        entries = bkd.asarray([2.0, 0.5, 0.3, 1.0])
        rhs = bkd.asarray([1.0, -2.0])

        def through_mass_solve(m: torch.Tensor) -> torch.Tensor:
            model = LotkaVolterra(bkd, mass=bkd.reshape(m, (2, 2)))
            return MassSolve(dense_system(model), bkd).solve(rhs)

        expected = bkd.jacobian(
            lambda m: bkd.solve(bkd.reshape(m, (2, 2)), rhs), entries
        )
        actual = bkd.jacobian(through_mass_solve, entries)
        bkd.assert_allclose(actual, expected, rtol=1e-12)

r"""The trajectory sweeps against torch autograd through the forward solve.

Autograd differentiates the discrete forward solve exactly, so the
sweeps' gradient and Hessian-vector product must equal it to rounding:
a far tighter check than finite differences. It also shows the forward
path (integrator, steppers, Newton, operators) keeps the graph intact.

Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` from (10, 5),
parameters (1, 0.1, 0.075, 1.5), four steps of 0.05 (a small graph),
Newton tolerance 1e-13, the Tikhonov-augmented MSE observing prey at
steps 1, 2, 4 and predator at 3, 4, as native misfit and Tikhonov terms
valued by the sink during the solve (so autograd also runs through the
terms' values). For backward Euler, Heun, implicit
midpoint, RK4, BDF2, Adams-Bashforth 2 and Adams-Moulton 3, plain and
with the curved time-dependent constraint and the parameter cross-term
initial state: the gradient equals autograd's and the HVP the
double-backward product to 1e-13, relative (measured: at most 4e-16 and
1.2e-15).

Torch only: the module is skipped without it.
"""

from typing import Callable, List

import pytest

# Skip the module without torch; the imports below are then ordinary ones.
pytest.importorskip("torch")

import torch

from pyapprox.ode.tableau import (
    backward_euler,
    classical_rk4,
    heun,
    implicit_midpoint,
)
from pyapprox.util.backends.torch import TorchBkd
from tests._helpers.ode.runge_kutta_cases import LV_PARAM
from tests._helpers.ode.trajectory_cases import (
    MULTISTEP,
    CurvedPin,
    Method,
    ScaledStart,
    TrajectoryProblem,
    random_direction,
)

_SHORT_TIMES = [0.05 * n for n in range(5)]
_METHODS: List[Callable[[], Method]] = [
    backward_euler,
    heun,
    implicit_midpoint,
    classical_rk4,
    *MULTISTEP,
]
# Which parameter-dependent pieces a case has: (constraint, initial state).
_PIECES = {"plain": (False, False), "pinned-start": (True, True)}


class TestAgainstAutograd:
    @pytest.mark.parametrize("pieces", list(_PIECES))
    @pytest.mark.parametrize("make", _METHODS, ids=[make().name for make in _METHODS])
    def test_gradient_and_hvp(
        self, torch_bkd: TorchBkd, make: Callable[[], Method], pieces: str
    ) -> None:
        bkd = torch_bkd
        pin, start = _PIECES[pieces]
        problem: TrajectoryProblem[torch.Tensor] = TrajectoryProblem(
            bkd,
            make(),
            CurvedPin(bkd) if pin else None,
            ScaledStart(bkd) if start else None,
            times=_SHORT_TIMES,
            prey=(1, 2, 4),
            predator=(3, 4),
        )
        param = torch.tensor(LV_PARAM, requires_grad=True)
        direction = random_direction(bkd)
        value = problem.value(param)[0, 0]
        (auto_gradient,) = torch.autograd.grad(value, param, create_graph=True)
        (auto_hvp,) = torch.autograd.grad((auto_gradient * direction).sum(), param)
        fixed = param.detach()
        bkd.assert_allclose(
            problem.gradient(fixed).T, auto_gradient.detach(), rtol=1e-13
        )
        bkd.assert_allclose(problem.hvp(fixed, direction), auto_hvp, rtol=1e-13)

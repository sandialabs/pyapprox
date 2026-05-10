"""Torch-native Adam optimizer satisfying BindableOptimizerProtocol.

Uses torch.optim.Adam (C++ implementation) with single forward+backward
per iteration. No numpy conversions, no redundant forward pass.
"""

from typing import Optional, Self, cast

import torch

from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.optimization.minimize.objective.protocols import (
    ObjectiveProtocol,
)
from pyapprox.optimization.minimize.result import OptimizerResult
from pyapprox.optimization.minimize.result_protocol import (
    OptimizerResultProtocol,
)
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.backends.torch import TorchBkd


class TorchAdamOptimizer:
    """Adam optimizer using torch.optim.Adam for maximum performance.

    Single forward + backward per iteration, all in torch space.

    Parameters
    ----------
    lr : float
        Learning rate. Default 1e-3.
    beta1 : float
        Exponential decay rate for first moment. Default 0.9.
    beta2 : float
        Exponential decay rate for second moment. Default 0.999.
    eps : float
        Numerical stability constant. Default 1e-8.
    maxiter : int
        Maximum iterations. Default 500.
    verbosity : int
        0 = silent, 1 = print every 100 iters. Default 0.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        maxiter: int = 500,
        verbosity: int = 0,
    ) -> None:
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._maxiter = maxiter
        self._verbosity = verbosity

        self._objective: Optional[ObjectiveProtocol[torch.Tensor]] = None
        self._bounds: Optional[torch.Tensor] = None
        self._bkd: Optional[TorchBkd] = None
        self._is_bound = False

    def bind(
        self,
        objective: ObjectiveProtocol[torch.Tensor],
        bounds: torch.Tensor,
        constraints: Optional[
            SequenceOfConstraintProtocols[torch.Tensor]
        ] = None,
    ) -> Self:
        if constraints is not None:
            raise NotImplementedError(
                "TorchAdamOptimizer does not support constraints."
            )
        if not isinstance(objective.bkd(), TorchBkd):
            raise TypeError(
                "TorchAdamOptimizer requires a TorchBkd backend, "
                f"got {type(objective.bkd()).__name__}"
            )
        self._objective = objective
        self._bounds = bounds
        self._bkd = objective.bkd()
        self._is_bound = True
        return self

    def is_bound(self) -> bool:
        return self._is_bound

    def copy(self) -> Self:
        return cast(
            Self,
            TorchAdamOptimizer(
                lr=self._lr,
                beta1=self._beta1,
                beta2=self._beta2,
                eps=self._eps,
                maxiter=self._maxiter,
                verbosity=self._verbosity,
            ),
        )

    def bkd(self) -> Backend[torch.Tensor]:
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._bkd is not None
        return self._bkd

    def minimize(
        self, init_guess: torch.Tensor
    ) -> OptimizerResultProtocol[torch.Tensor]:
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        assert self._bounds is not None

        objective = self._objective
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]

        param = torch.nn.Parameter(
            init_guess[:, 0].detach().clone()
        )
        n = param.shape[0]

        opt = torch.optim.Adam(
            [param],
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
        )

        best_f = float("inf")
        best_x = param.data.clone()
        hyp_list = objective.hyp_list()

        for t in range(1, self._maxiter + 1):
            # Detach internal hyp storage so in-place set_active_values
            # does not conflict with the previous iteration's graph.
            for hyp in hyp_list.hyperparameters():
                hyp._values = hyp._values.detach().clone()

            opt.zero_grad()
            loss = objective(param)[0, 0]
            loss.backward()
            opt.step()

            with torch.no_grad():
                param.clamp_(lb, ub)

            f_val = loss.item()
            if f_val < best_f:
                best_f = f_val
                best_x = param.data.clone()

            if self._verbosity >= 1 and t % 100 == 0:
                print(f"TorchAdam iter {t:5d}  f = {f_val:.6e}")

        return OptimizerResult(
            optima=best_x.detach().reshape(n, 1),
            fun=best_f,
            success=True,
            message="TorchAdam completed maximum iterations",
            nit=self._maxiter,
        )

"""Adam optimizer satisfying BindableOptimizerProtocol.

Implements the Adam algorithm (Kingma & Ba, 2015) for use as a warm-start
phase before a second-order local optimizer via ChainedOptimizer.
"""

from typing import Generic, Optional, Self, cast

from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.optimization.minimize.objective.protocols import (
    ObjectiveProtocol,
    ObjectiveWithJacobianProtocol,
)
from pyapprox.optimization.minimize.objective.validation import (
    validate_objective,
)
from pyapprox.optimization.minimize.result import OptimizerResult
from pyapprox.optimization.minimize.result_protocol import (
    OptimizerResultProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class AdamOptimizer(Generic[Array]):
    """Adam optimizer (Kingma & Ba, 2015).

    Implements the Adam algorithm for first-order gradient-based optimization.
    Works directly in backend space without numpy conversions.

    Parameters
    ----------
    lr : float
        Learning rate (step size). Default 1e-3.
    beta1 : float
        Exponential decay rate for first moment estimates. Default 0.9.
    beta2 : float
        Exponential decay rate for second moment estimates. Default 0.999.
    eps : float
        Small constant for numerical stability. Default 1e-8.
    maxiter : int
        Maximum number of iterations. Default 500.
    verbosity : int
        Verbosity level. 0 = silent, 1 = print every 100 iters. Default 0.
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

        self._objective: Optional[ObjectiveWithJacobianProtocol[Array]] = None
        self._bounds: Optional[Array] = None
        self._bkd: Optional[Backend[Array]] = None
        self._is_bound = False

    def bind(
        self,
        objective: ObjectiveProtocol[Array],
        bounds: Array,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
    ) -> Self:
        """Bind objective and bounds.

        Parameters
        ----------
        objective : ObjectiveProtocol[Array]
            Objective function (must have ``jacobian``).
        bounds : Array
            Parameter bounds, shape ``(nvars, 2)``.
        constraints : optional
            Not supported. Raises ``NotImplementedError`` if provided.

        Returns
        -------
        Self
        """
        if constraints is not None:
            raise NotImplementedError(
                "AdamOptimizer does not support constraints."
            )
        validate_objective(objective)
        if not isinstance(objective, ObjectiveWithJacobianProtocol):
            raise TypeError(
                "AdamOptimizer requires an objective with a jacobian method, "
                f"got {type(objective).__name__}"
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
            AdamOptimizer(
                lr=self._lr,
                beta1=self._beta1,
                beta2=self._beta2,
                eps=self._eps,
                maxiter=self._maxiter,
                verbosity=self._verbosity,
            ),
        )

    def bkd(self) -> Backend[Array]:
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._bkd is not None
        return self._bkd

    def minimize(
        self, init_guess: Array
    ) -> OptimizerResultProtocol[Array]:
        """Run Adam optimization.

        Parameters
        ----------
        init_guess : Array
            Initial parameter vector, shape ``(nvars, 1)``.

        Returns
        -------
        OptimizerResultProtocol[Array]
            Optimization result.
        """
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        assert self._bounds is not None
        assert self._bkd is not None

        bkd = self._bkd
        objective = self._objective

        x = init_guess[:, 0]
        n = len(x)
        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]

        m = bkd.zeros((n,))
        v = bkd.zeros((n,))
        beta1, beta2 = self._beta1, self._beta2
        lr, eps = self._lr, self._eps

        best_f = float("inf")
        best_x = x

        for t in range(1, self._maxiter + 1):
            x_col = bkd.reshape(x, (n, 1))
            f_val = objective(x_col)
            f_scalar = float(bkd.to_numpy(bkd.reshape(f_val, (-1,)))[0])

            grad = objective.jacobian(x_col)[0]

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad * grad

            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)

            x = x - lr * m_hat / (bkd.sqrt(v_hat) + eps)
            x = bkd.clip(x, lb, ub)

            if f_scalar < best_f:
                best_f = f_scalar
                best_x = x

            if self._verbosity >= 1 and t % 100 == 0:
                print(f"Adam iter {t:5d}  f = {f_scalar:.6e}")

        return OptimizerResult(
            optima=bkd.reshape(best_x, (n, 1)),
            fun=best_f,
            success=True,
            message="Adam completed maximum iterations",
            nit=self._maxiter,
        )

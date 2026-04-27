"""Adam optimizer satisfying BindableOptimizerProtocol.

Implements the Adam algorithm (Kingma & Ba, 2015) for use as a warm-start
phase before a second-order local optimizer via ChainedOptimizer.
"""

from typing import Generic, Optional, Self, Union, cast

import numpy as np
from scipy.optimize import OptimizeResult

from pyapprox.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)
from pyapprox.interface.functions.numpy.wrappers import (
    NumpyFunctionWithJacobianAndHVPWrapper,
    NumpyFunctionWithJacobianAndWHVPWrapper,
    NumpyFunctionWithJacobianWrapper,
    NumpyFunctionWrapper,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.optimization.minimize.objective.protocols import (
    ObjectiveProtocol,
)
from pyapprox.optimization.minimize.objective.validation import (
    validate_objective,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.util.backends.protocols import Array, Backend

_WrappedObjective = Union[
    NumpyFunctionWrapper[Array],
    NumpyFunctionWithJacobianWrapper[Array],
    NumpyFunctionWithJacobianAndHVPWrapper[Array],
    NumpyFunctionWithJacobianAndWHVPWrapper[Array],
]


class AdamOptimizer(Generic[Array]):
    """Adam optimizer (Kingma & Ba, 2015).

    Implements the Adam algorithm for first-order gradient-based optimization.
    Satisfies ``BindableOptimizerProtocol`` so it can be used standalone or
    as the global phase of a ``ChainedOptimizer``.

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

        self._objective: Optional[_WrappedObjective[Array]] = None
        self._lb: Optional[np.ndarray] = None
        self._ub: Optional[np.ndarray] = None
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
        self._objective = numpy_function_wrapper_factory(objective)
        if not hasattr(self._objective, "jacobian"):
            raise TypeError(
                "AdamOptimizer requires an objective with a jacobian method."
            )
        np_bounds = self._objective.bkd().to_numpy(bounds)
        self._lb = np_bounds[:, 0]
        self._ub = np_bounds[:, 1]
        self._is_bound = True
        return self

    def is_bound(self) -> bool:
        """Return True if bound to an objective."""
        return self._is_bound

    def copy(self) -> Self:
        """Return an unbound copy with same options."""
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
        """Return the backend from the bound objective."""
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        return self._objective.bkd()

    def minimize(
        self, init_guess: Array
    ) -> ScipyOptimizerResultWrapper[Array]:
        """Run Adam optimization.

        Parameters
        ----------
        init_guess : Array
            Initial parameter vector, shape ``(nvars, 1)``.

        Returns
        -------
        ScipyOptimizerResultWrapper[Array]
            Optimization result.
        """
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        assert self._lb is not None
        assert self._ub is not None

        bkd = self._objective.bkd()
        x = bkd.to_numpy(init_guess[:, 0]).copy()
        n = len(x)

        m = np.zeros(n)
        v = np.zeros(n)
        beta1, beta2 = self._beta1, self._beta2
        lr, eps = self._lr, self._eps
        lb, ub = self._lb, self._ub

        best_f = np.inf
        best_x = x.copy()

        for t in range(1, self._maxiter + 1):
            obj_out = self._objective(x[:, None])
            f_val = float(obj_out[0, 0])
            grad = self._objective.jacobian(x[:, None])[0]  # type: ignore[union-attr]

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad**2

            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)

            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

            # Project onto bounds
            x = np.clip(x, lb, ub)

            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

            if self._verbosity >= 1 and t % 100 == 0:
                print(f"Adam iter {t:5d}  f = {f_val:.6e}")

        scipy_result = OptimizeResult(
            x=best_x,
            fun=best_f,
            success=True,
            message="Adam completed maximum iterations",
            nit=self._maxiter,
        )
        return ScipyOptimizerResultWrapper(scipy_result, bkd)

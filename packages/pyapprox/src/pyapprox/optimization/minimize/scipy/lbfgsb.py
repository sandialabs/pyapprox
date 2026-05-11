"""L-BFGS-B optimizer satisfying BindableOptimizerProtocol."""

from typing import Any, Callable, Generic, Optional, Self, Union, cast

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize as scipy_minimize

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


class LBFGSBOptimizer(Generic[Array]):
    """Optimizer using SciPy's L-BFGS-B method.

    Parameters
    ----------
    objective : Optional[ObjectiveProtocol[Array]]
        Objective function. If None, must call bind() before minimize().
    bounds : Optional[Array]
        Bounds for optimization variables, shape (nvars, 2).
    verbosity : int
        Verbosity level. 0 = silent.
    maxiter : Optional[int]
        Maximum number of iterations.
    ftol : Optional[float]
        Function tolerance for termination.
    gtol : Optional[float]
        Gradient tolerance for termination.
    callback : Optional[Callable]
        Callback function called after each iteration.
    """

    def __init__(
        self,
        objective: Optional[ObjectiveProtocol[Array]] = None,
        bounds: Optional[Array] = None,
        verbosity: int = 0,
        maxiter: Optional[int] = None,
        ftol: Optional[float] = None,
        gtol: Optional[float] = None,
        callback: Optional[Callable[..., Any]] = None,
    ):
        self._verbosity = verbosity
        self._maxiter = maxiter
        self._ftol = ftol
        self._gtol = gtol
        self._user_callback = callback

        self._opts: dict[str, Any] = {}
        if maxiter is not None:
            self._opts["maxiter"] = maxiter
        if ftol is not None:
            self._opts["ftol"] = ftol
        if gtol is not None:
            self._opts["gtol"] = gtol

        self._objective: Optional[_WrappedObjective[Array]] = None
        self._bounds: Optional[Bounds] = None
        self._is_bound = False

        if objective is not None:
            if bounds is None:
                raise ValueError(
                    "bounds must be provided when objective is provided"
                )
            self.bind(objective, bounds)

    def bind(
        self,
        objective: ObjectiveProtocol[Array],
        bounds: Array,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
    ) -> Self:
        """Bind objective and bounds. Returns self for chaining."""
        validate_objective(objective)
        self._objective = numpy_function_wrapper_factory(objective)
        self._bounds = self._convert_bounds(
            bounds, self._objective.nvars(), self._objective.bkd()
        )
        self._is_bound = True
        return self

    def is_bound(self) -> bool:
        return self._is_bound

    def copy(self) -> Self:
        return cast(
            Self,
            LBFGSBOptimizer(
                objective=None,
                bounds=None,
                verbosity=self._verbosity,
                maxiter=self._maxiter,
                ftol=self._ftol,
                gtol=self._gtol,
                callback=self._user_callback,
            ),
        )

    def bkd(self) -> Backend[Array]:
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        return self._objective.bkd()

    def _convert_bounds(
        self, bounds: Array, nvars: int, bkd: Backend[Array]
    ) -> Bounds:
        if bounds is None:
            return Bounds(
                np.full((nvars,), -np.inf),
                np.full((nvars,), np.inf),
            )
        np_bounds = bkd.to_numpy(bounds)
        return Bounds(np_bounds[:, 0], np_bounds[:, 1])

    def _objective_gradient_from_jacobian(self, sample: Array) -> Array:
        assert self._objective is not None
        return self._objective.jacobian(sample[:, None])[0]  # type: ignore[union-attr,no-any-return]

    def minimize(self, init_guess: Array) -> ScipyOptimizerResultWrapper[Array]:
        """Perform L-BFGS-B optimization.

        Parameters
        ----------
        init_guess : Array
            Initial guess, shape (nvars, 1).

        Returns
        -------
        ScipyOptimizerResultWrapper[Array]
        """
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        assert self._bounds is not None

        jac = (
            self._objective_gradient_from_jacobian
            if hasattr(self._objective, "jacobian")
            else None
        )

        callback = self._user_callback
        if callback is None and self._verbosity > 0:
            _iter = [0]

            def callback(intermediate_result: Any) -> None:
                _iter[0] += 1
                fun = intermediate_result.fun
                print(f"L-BFGS-B iter {_iter[0]}: fun={fun:.6e}")

        scipy_result = scipy_minimize(
            lambda x: self._objective(x[:, None])[:, 0],
            self.bkd().to_numpy(init_guess[:, 0]),
            method="L-BFGS-B",
            jac=jac,
            bounds=self._bounds,
            options=self._opts,
            callback=callback,
        )
        return ScipyOptimizerResultWrapper(scipy_result, self.bkd())

"""L-BFGS-B optimizer satisfying BindableOptimizerProtocol."""

from typing import Any, Callable, Generic, Optional, Self

import numpy as np
from scipy.optimize import Bounds, OptimizeResult
from scipy.optimize import minimize as scipy_minimize

from pyapprox.interface.functions.numpy.adapter import (
    NumpyArray,
    NumpyDerivativesAdapter,
    NumpyFn,
)
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.optimization.minimize.objective.validation import (
    validate_objective,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.util.backends.protocols import Array, Backend


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

        self._objective: Optional[NumpyDerivativesAdapter[Array]] = None
        # derivative capability captured once at bind(); value varies,
        # attribute shape never does
        self._np_jac: Optional[NumpyFn] = None
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
        adapter = NumpyDerivativesAdapter(
            objective, objective.derivatives()
        )
        self._objective = adapter
        self._np_jac = adapter.jacobian()
        self._bounds = self._convert_bounds(
            bounds, adapter.nvars(), adapter.bkd()
        )
        self._is_bound = True
        return self

    def is_bound(self) -> bool:
        return self._is_bound

    def copy(self) -> Self:
        return type(self)(
            objective=None,
            bounds=None,
            verbosity=self._verbosity,
            maxiter=self._maxiter,
            ftol=self._ftol,
            gtol=self._gtol,
            callback=self._user_callback,
        )

    def bkd(self) -> Backend[Array]:
        objective = self._objective
        if objective is None:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        return objective.bkd()

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
        objective = self._objective
        if objective is None or self._bounds is None:
            raise RuntimeError("Optimizer not bound. Call bind() first.")

        np_jac = self._np_jac
        jac: Optional[NumpyFn] = None
        if np_jac is not None:

            def jac(x: NumpyArray) -> NumpyArray:
                # np.asarray: numpy stubs type __getitem__ as Any
                return np.asarray(np_jac(x[:, None])[0])

        callback = self._user_callback
        if callback is None and self._verbosity > 0:
            _iter = [0]

            def callback(intermediate_result: OptimizeResult) -> None:
                _iter[0] += 1
                fun = intermediate_result.fun
                print(f"L-BFGS-B iter {_iter[0]}: fun={fun:.6e}")

        scipy_result = scipy_minimize(
            lambda x: objective(x[:, None])[:, 0],
            self.bkd().to_numpy(init_guess[:, 0]),
            method="L-BFGS-B",
            jac=jac,
            bounds=self._bounds,
            options=self._opts,
            callback=callback,
        )
        return ScipyOptimizerResultWrapper(scipy_result, self.bkd())

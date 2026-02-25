from typing import Generic, Optional, Self, Union, cast

import numpy as np
from scipy.optimize import direct, Bounds

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)
from pyapprox.optimization.minimize.objective.validation import (
    validate_objective,
)
from pyapprox.interface.functions.numpy.wrappers import (
    NumpyFunctionWrapper,
    NumpyFunctionWithJacobianWrapper,
    NumpyFunctionWithJacobianAndHVPWrapper,
    NumpyFunctionWithJacobianAndWHVPWrapper,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)

_WrappedObjective = Union[
    NumpyFunctionWrapper[Array],
    NumpyFunctionWithJacobianWrapper[Array],
    NumpyFunctionWithJacobianAndHVPWrapper[Array],
    NumpyFunctionWithJacobianAndWHVPWrapper[Array],
]


class ScipyDirectOptimizer(Generic[Array]):
    """Optimizer using SciPy's DIRECT (DIviding RECTangles) algorithm.

    DIRECT is a deterministic global optimizer that does not require
    gradient information. It systematically divides the search space
    and evaluates the objective at the center of each rectangle.

    Supports two usage patterns:

    1. Direct binding:
    ```python
    optimizer = ScipyDirectOptimizer(objective=obj, bounds=bounds)
    result = optimizer.minimize(init_guess)
    ```

    2. Deferred binding:
    ```python
    optimizer = ScipyDirectOptimizer(maxiter=1000)
    optimizer.bind(objective, bounds)
    result = optimizer.minimize(init_guess)
    ```
    """

    def __init__(
        self,
        objective: Optional[FunctionProtocol[Array]] = None,
        bounds: Optional[Array] = None,
        maxfun: Optional[int] = None,
        maxiter: int = 1000,
        vol_tol: float = 1e-16,
        len_tol: float = 1e-6,
        f_min: float = -np.inf,
        f_min_rtol: float = 1e-4,
        locally_biased: bool = True,
        raise_on_failure: bool = True,
    ):
        self._maxfun = maxfun
        self._maxiter = maxiter
        self._vol_tol = vol_tol
        self._len_tol = len_tol
        self._f_min = f_min
        self._f_min_rtol = f_min_rtol
        self._locally_biased = locally_biased
        self._raise_on_failure = raise_on_failure

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
        objective: FunctionProtocol[Array],
        bounds: Array,
        constraints: Optional[object] = None,
    ) -> Self:
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
            ScipyDirectOptimizer(
                objective=None,
                bounds=None,
                maxfun=self._maxfun,
                maxiter=self._maxiter,
                vol_tol=self._vol_tol,
                len_tol=self._len_tol,
                f_min=self._f_min,
                f_min_rtol=self._f_min_rtol,
                locally_biased=self._locally_biased,
                raise_on_failure=self._raise_on_failure,
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

    def minimize(
        self, init_guess: Array
    ) -> ScipyOptimizerResultWrapper[Array]:
        if not self._is_bound:
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        assert self._objective is not None
        assert self._bounds is not None

        kwargs = {
            "func": lambda x: float(
                self._objective(x[:, None])[:, 0]
            ),
            "bounds": self._bounds,
            "maxiter": self._maxiter,
            "vol_tol": self._vol_tol,
            "len_tol": self._len_tol,
            "f_min": self._f_min,
            "f_min_rtol": self._f_min_rtol,
            "locally_biased": self._locally_biased,
        }
        if self._maxfun is not None:
            kwargs["maxfun"] = self._maxfun

        scipy_result = direct(**kwargs)

        if self._raise_on_failure and not scipy_result.success:
            raise RuntimeError(
                f"DIRECT optimization failed: {scipy_result.message}"
            )

        return ScipyOptimizerResultWrapper(scipy_result, self.bkd())

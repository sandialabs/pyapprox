from typing import Generic, Union, Optional, cast
import numpy as np
from scipy.optimize import Bounds, minimize as scipy_minimize
from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.optimization.linear_constraint import (
    PyApproxLinearConstraint,
)
from pyapprox.typing.optimization.constraint_protocols import (
    UnionOfNonlinearConstraintProtocols,
    SequenceOfUnionOfConstraintProtocols,
    validate_constraints,
)
from pyapprox.typing.interface.functions.hessian_protocols import (
    FunctionWithJacobianAndHVPProtocol,
)
from pyapprox.typing.interface.functions.numpy.numpy_function_factory import (
    numpy_function_wrapper_factory,
)
from pyapprox.typing.optimization.scipy.scipy_constraint_factory import (
    convert_constraints,
)
from pyapprox.typing.optimization.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)

UnionOfObjectiveProtocols = UnionOfNonlinearConstraintProtocols[Array]


def validate_objective(objective: UnionOfObjectiveProtocols[Array]) -> None:
    """
    Validate that the given objective satisfies the required protocol.

    Parameters
    ----------
    objective : UnionOfObjectiveProtocols
        The objective function to validate.

    Raises
    ------
    TypeError
        If the objective does not satisfy the required protocol.
    ValueError
        If the objective does not have exactly one quantity of interest (nqoi != 1).
    """
    if not isinstance(objective, FunctionWithJacobianAndHVPProtocol):
        raise TypeError(
            "The provided objective must satisfy the "
            "'FunctionWithJacobianAndHVPProtocol'. "
            f"Got an object of type {type(objective).__name__}."
        )
    if objective.nqoi() != 1:
        raise ValueError(
            f"The provided objective must have exactly one quantity of interest (nqoi=1). "
            f"Got nqoi={objective.nqoi()}."
        )


class ScipyTrustConstrOptimizer(Generic[Array]):
    """
    Optimizer using SciPy's trust-constr method.

    This class wraps SciPy's trust-constr optimizer and integrates with PyApprox's
    function and constraint wrappers.
    """

    def __init__(
        self,
        objective: UnionOfObjectiveProtocols[Array],
        bounds: Array,
        constraints: Optional[
            SequenceOfUnionOfConstraintProtocols[Array]
        ] = None,
        verbosity: int = 0,
        maxiter: Optional[int] = None,
        gtol: Optional[float] = None,
        xtol: Optional[float] = None,
        barrier_tol: Optional[float] = None,
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        objective : UnionOfObjectiveProtocols
            Objective function for the optimization problem.
        bounds : Array
            Bounds for the optimization variables.
        constraints : Optional[SequenceOfUnionOfConstraintProtocols], optional
            Constraints for the optimization problem. Defaults to None.
        verbosity : int, optional
            Verbosity level for the optimizer. Defaults to 0.
        maxiter : Optional[int], optional
            Maximum number of iterations. Defaults to SciPy's default.
        gtol : Optional[float], optional
            Gradient tolerance for termination. Defaults to SciPy's default.
        xtol : Optional[float], optional
            Step tolerance for termination. Defaults to SciPy's default.
        barrier_tol : Optional[float], optional
            Barrier tolerance for termination. Defaults to SciPy's default.
        """
        self._verbosity = verbosity

        # Explicit options for trust-constr with defaults
        self._opts = {
            "maxiter": maxiter,
            "gtol": gtol,
            "xtol": xtol,
            "barrier_tol": barrier_tol,
            "verbose": verbosity,
        }
        # Remove None values to let SciPy use its defaults
        self._opts = {
            key: value
            for key, value in self._opts.items()
            if value is not None
        }

        # Validate and wrap the objective using the factory
        validate_objective(objective)
        self._objective = numpy_function_wrapper_factory(
            objective, sample_ndim=1
        )

        # Wrap the bounds
        self._bounds = self._convert_bounds(bounds, self._objective.nvars())

        # Validate and wrap the constraints using the factory if provided
        if constraints:
            validate_constraints(constraints)
        self._constraints = convert_constraints(constraints)

    def bkd(self) -> Backend[Array]:
        """
        Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations.
        """
        return self._objective.bkd()

    def _convert_bounds(self, bounds: Array, nvars: int) -> Bounds:
        """
        Convert bounds to a SciPy-compatible Bounds object.

        Parameters
        ----------
        bounds : Array
            Bounds for the optimization variables.
        nvars : int
            Number of variables in the optimization problem.

        Returns
        -------
        Bounds
            SciPy-compatible bounds object.
        """
        if bounds is None:
            return Bounds(
                np.full((nvars,), -np.inf),
                np.full((nvars,), np.inf),
                keep_feasible=True,
            )
        np_bounds = self.bkd().to_numpy(bounds)
        return Bounds(np_bounds[:, 0], np_bounds[:, 1], keep_feasible=True)

    def _objective_gradient_from_jacobian(self, sample: Array) -> Array:
        return self._objective.jacobian(sample)[0]

    def _objective_hessp_from_hvp(self, sample: Array, vec: Array) -> Array:
        return self._objective.hvp(sample, vec)[:, 0]

    def minimize(
        self, init_guess: Array
    ) -> ScipyOptimizerResultWrapper[Array]:
        """
        Perform the optimization.

        Parameters
        ----------
        init_guess : Array
            Initial guess for the optimization variables.

        Returns
        -------
        ScipyOptimizerResultWrapper
            Wrapped optimization result.
        """
        jac = (
            self._objective_gradient_from_jacobian
            if hasattr(self._objective, "jacobian")
            else None
        )
        hessp = (
            self._objective_hessp_from_hvp
            if hasattr(self._objective, "hvp")
            else None
        )

        scipy_result = scipy_minimize(  # type: ignore
            self._objective,
            self.bkd().to_numpy(init_guess[:, 0]),
            method="trust-constr",
            jac=jac,
            hessp=hessp,
            bounds=self._bounds,
            constraints=self._constraints,
            options=self._opts,
        )
        # Wrap the SciPy result
        return ScipyOptimizerResultWrapper(scipy_result, self.bkd())

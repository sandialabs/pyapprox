from typing import Generic

from scipy.optimize import OptimizeResult

from pyapprox.util.backends.protocols import Array, Backend


class ScipyOptimizerResultWrapper(Generic[Array]):
    """
    Wrapper for SciPy's OptimizeResult to satisfy the OptimizerResultProtocol.

    This class provides a clean interface for accessing optimization results
    while adhering to the OptimizerResultProtocol. It also provides access to
    the raw SciPy result.
    """

    def __init__(self, scipy_result: OptimizeResult, bkd: Backend[Array]):
        """
        Initialize the wrapper.

        Parameters
        ----------
        scipy_result : OptimizeResult
            The raw SciPy optimization result.

        Raises
        ------
        TypeError
            If the input is not a valid SciPy OptimizeResult.
        """
        self._validate_scipy_result(scipy_result)
        self._scipy_result = scipy_result
        self._bkd = bkd

    def _validate_scipy_result(self, scipy_result: OptimizeResult) -> None:
        """
        Validate that the input is a valid SciPy OptimizeResult.

        Parameters
        ----------
        scipy_result : OptimizeResult
            The raw SciPy optimization result.

        Raises
        ------
        TypeError
            If the input is not a valid SciPy OptimizeResult.
        """
        if not isinstance(scipy_result, OptimizeResult):
            raise TypeError(
                f"The provided result must be a valid SciPy OptimizeResult. "
                f"Got an object of type {type(scipy_result).__name__}."
            )

    def fun(self) -> float:
        """
        Get the minimum value of the objective function.

        Returns
        -------
        float
            Minimum value of the objective function.
        """
        return self._scipy_result.fun

    def success(self) -> bool:
        """
        Get the success status of the optimization.

        Returns
        -------
        bool
            True if the optimization was successful, False otherwise.
        """
        return self._scipy_result.success

    def optima(self) -> Array:
        """
        Get the solution vector (optimal values of decision variables).

        Returns
        -------
        Array
            Solution vector if available, otherwise None.
        """
        return self._bkd.asarray(self._scipy_result.x[:, None])

    def get_raw_result(self) -> OptimizeResult:
        """
        Get the raw SciPy optimization result.

        Returns
        -------
        OptimizeResult
            The raw SciPy optimization result.
        """
        return self._scipy_result

    def message(self) -> str:
        """
        Get the termination message from the optimizer.

        Returns
        -------
        str
            Message describing why the optimizer terminated.
        """
        return getattr(self._scipy_result, "message", "No message available")

    def __repr__(self) -> str:
        """
        Return a string representation of the optimization result.

        Returns
        -------
        str
            String representation of the optimization result.
        """
        return (
            f"{self.__class__.__name__}("
            f"fun={self.fun()}, "
            f"success={self.success()}, "
            f"optima={self.optima()},"
        )

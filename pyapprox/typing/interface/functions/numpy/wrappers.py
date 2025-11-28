from typing import Generic, Any

from numpy.typing import NDArray

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.interface.functions.function import FunctionProtocol
from pyapprox.typing.interface.functions.jacobian_protocols import (
    FunctionWithJacobianProtocol,
)
from pyapprox.typing.interface.functions.hessian_protocols import (
    FunctionWithJacobianAndHVPProtocol,
)


class NumpyFunctionWrapper(Generic[Array]):
    """
    Wrapper for functions when a numpy interface is required, e.g.
    when using scipy.minimize.

    This class ensures compatibility between functions that use different
    backends and optimization frameworks. It provides methods for evaluating
    the function.
    """

    def __init__(self, function: FunctionProtocol[Array]):
        self._bkd = function.bkd()
        self._function = function

    def bkd(self) -> Backend[Array]:
        return self._function.bkd()

    def nvars(self) -> int:
        return self._function.nvars()

    def nqoi(self) -> int:
        return self._function.nqoi()

    def __call__(self, samples: NDArray[Any]) -> NDArray[Any]:
        return self._bkd.to_numpy(self._function(self._bkd.asarray(samples)))


class NumpyFunctionWithJacobianWrapper(Generic[Array]):
    """
    Wrapper for functions with Jacobian when a numpy interface is required, e.g.,
    when using scipy.optimize.

    This class ensures compatibility between functions that use different
    backends and optimization frameworks. It provides methods for evaluating
    the function and its Jacobian.
    """

    def __init__(self, function: FunctionWithJacobianProtocol[Array]):
        self._bkd = function.bkd()
        self._function = function

    def bkd(self) -> Backend[Array]:
        return self._function.bkd()

    def nvars(self) -> int:
        return self._function.nvars()

    def nqoi(self) -> int:
        return self._function.nqoi()

    def __call__(self, samples: NDArray[Any]) -> NDArray[Any]:
        """
        Evaluate the function at the given samples.

        Parameters
        ----------
        samples : NDArray[Any]
            Input samples as a NumPy array.

        Returns
        -------
        NDArray[Any]
            Function evaluations as a NumPy array.
        """
        return self._bkd.to_numpy(self._function(self._bkd.asarray(samples)))

    def jacobian(self, sample: NDArray[Any]) -> NDArray[Any]:
        """
        Compute the Jacobian of the function at the given sample.

        Parameters
        ----------
        sample : NDArray[Any]
            Input sample as a NumPy array.

        Returns
        -------
        NDArray[Any]
            Jacobian matrix as a NumPy array.
        """
        return self._bkd.to_numpy(
            self._function.jacobian(self._bkd.asarray(sample))
        )


class NumpyFunctionWithJacobianAndHVPWrapper(Generic[Array]):
    """
    Wrapper for functions with Jacobian and Hessian-vector product when a numpy
    interface is required, e.g., when using scipy.optimize.

    This class ensures compatibility between functions that use different
    backends and optimization frameworks. It provides methods for evaluating
    the function, its Jacobian, and Hessian-vector products.
    """

    def __init__(self, function: FunctionWithJacobianAndHVPProtocol[Array]):
        self._bkd = function.bkd()
        self._function = function

    def bkd(self) -> Backend[Array]:
        return self._function.bkd()

    def nvars(self) -> int:
        return self._function.nvars()

    def nqoi(self) -> int:
        return self._function.nqoi()

    def __call__(self, samples: NDArray[Any]) -> NDArray[Any]:
        """
        Evaluate the function at the given samples.

        Parameters
        ----------
        samples : NDArray[Any]
            Input samples as a NumPy array.

        Returns
        -------
        NDArray[Any]
            Function evaluations as a NumPy array.
        """
        return self._bkd.to_numpy(self._function(self._bkd.asarray(samples)))

    def jacobian(self, sample: NDArray[Any]) -> NDArray[Any]:
        """
        Compute the Jacobian of the function at the given sample.

        Parameters
        ----------
        sample : NDArray[Any]
            Input sample as a NumPy array.

        Returns
        -------
        NDArray[Any]
            Jacobian matrix as a NumPy array.
        """
        return self._bkd.to_numpy(
            self._function.jacobian(self._bkd.asarray(sample))
        )

    def hvp(self, sample: NDArray[Any], vec: NDArray[Any]) -> NDArray[Any]:
        """
        Compute the Hessian-vector product of the function at the given sample.

        Parameters
        ----------
        sample : NDArray[Any]
            Input sample as a NumPy array.
        vec : NDArray[Any]
            Vector for the Hessian-vector product.

        Returns
        -------
        NDArray[Any]
            Hessian-vector product as a NumPy array.
        """
        return self._bkd.to_numpy(
            self._function.hvp(
                self._bkd.asarray(sample), self._bkd.asarray(vec)
            )
        )

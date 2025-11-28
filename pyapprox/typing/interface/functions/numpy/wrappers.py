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

    def __init__(
        self, function: FunctionProtocol[Array], sample_ndim: int = 2
    ):
        self._bkd = function.bkd()
        self._function = function
        # PyApprox assumes samples are always 2D but numpy functions, e.g. from
        # scipy, may only use 1D arrays
        # sample_ndim is the size of the array the numpy function passes to
        # this model
        self._sample_ndim = sample_ndim

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._function.nvars()

    def nqoi(self) -> int:
        return self._function.nqoi()

    def _convert_samples_from_numpy(self, samples: NDArray[Any]) -> Array:
        if self._sample_ndim == 2:
            return self._bkd.asarray(samples)
        return self._bkd.asarray(samples[:, None])

    def __call__(self, samples: NDArray[Any]) -> NDArray[Any]:
        return self._bkd.to_numpy(
            self._function(self._convert_samples_from_numpy(samples))
        )


class NumpyFunctionWithJacobianWrapper(Generic[Array]):
    """
    Wrapper for functions with Jacobian when a numpy interface is required, e.g.,
    when using scipy.optimize.

    This class ensures compatibility between functions that use different
    backends and optimization frameworks. It provides methods for evaluating
    the function and its Jacobian.
    """

    def __init__(
        self,
        function: FunctionWithJacobianProtocol[Array],
        sample_ndim: int = 2,
    ):
        self._bkd = function.bkd()
        self._function = function
        self._sample_ndim = sample_ndim

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._function.nvars()

    def nqoi(self) -> int:
        return self._function.nqoi()

    def _convert_samples_from_numpy(self, samples: NDArray[Any]) -> Array:
        if self._sample_ndim == 2:
            return self._bkd.asarray(samples)
        return self._bkd.asarray(samples[:, None])

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
        return self._bkd.to_numpy(
            self._function(self._convert_samples_from_numpy(samples))
        )

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
            self._function.jacobian(self._convert_samples_from_numpy(sample))
        )


class NumpyFunctionWithJacobianAndHVPWrapper(Generic[Array]):
    """
    Wrapper for functions with Jacobian and Hessian-vector product when a numpy
    interface is required, e.g., when using scipy.optimize.

    This class ensures compatibility between functions that use different
    backends and optimization frameworks. It provides methods for evaluating
    the function, its Jacobian, and Hessian-vector products.
    """

    def __init__(
        self,
        function: FunctionWithJacobianAndHVPProtocol[Array],
        sample_ndim=2,
    ):
        self._bkd = function.bkd()
        self._function = function
        self._sample_ndim = sample_ndim

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._function.nvars()

    def nqoi(self) -> int:
        return self._function.nqoi()

    def _convert_samples_from_numpy(self, samples: NDArray[Any]) -> Array:
        if self._sample_ndim == 2:
            return self._bkd.asarray(samples)
        return self._bkd.asarray(samples[:, None])

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
        return self._bkd.to_numpy(
            self._function(self._convert_samples_from_numpy(samples))
        )

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
            self._function.jacobian(self._convert_samples_from_numpy(sample))
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
        bkd_vec = self._bkd.asarray(vec)
        if self._sample_ndim == 1:
            bkd_vec = bkd_vec[:, None]
        hvp = self._bkd.to_numpy(
            self._function.hvp(
                self._convert_samples_from_numpy(sample),
                bkd_vec,
            )
        )
        return hvp

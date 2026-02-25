from pyapprox.util.backends.protocols import Array
from pyapprox.optimization.implicitfunction.functionals.mean_squared_error import (
    MSEFunctional,
)


class TikhonovMSEFunctional(MSEFunctional):
    """
    Tikhonov regularized Mean Squared Error (MSE) functional.
    """

    def __call__(self, state: Array, param: Array) -> Array:
        """
        Compute the value of the functional.

        Returns
        -------
        Array
            Value of the functional.
        """
        return (
            super().__call__(state, param)
            + self._bkd.sum(param**2, keepdims=True) / 2.0
        )

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """
        Compute the Jacobian of the functional with respect to the parameters.

        Returns
        -------
        Array
            Jacobian matrix with respect to the parameters.
        """
        return param.T

    def param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        """
        Compute the Hessian-vector product with respect to the parameters.

        Returns
        -------
        Array
            Hessian-vector product result.
        """
        return vvec

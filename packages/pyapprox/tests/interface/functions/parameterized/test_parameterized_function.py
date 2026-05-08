from typing import Generic

from pyapprox.interface.functions.parameterized.factory import (
    _convert_to_function_of_parameters as convert_to_function_of_parameters,
    # convert_to_function_of_parameters,
)
from pyapprox.util.backends.protocols import Array, Backend


class ExampleParameterizedFunctionWithHVP(Generic[Array]):
    """
    Example implementation of a parameterized function with full-rank Hessian.

    The function is defined as:
        f(x, p) = sum_j (p[0]**2 * x[j]**2 + p[1]**3 * x[j] + p[2]**4 + p[0] * p[1]),
    where the summation is over the second dimension of x.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 1

    def nparams(self) -> int:
        return 3

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate f(x, p) = sum_j (p[0]**2 * x[j]**2 + p[1]**3 * x[j] + p[2]**4 + p[0] *
        p[1]).

        functions using protocol must have both the same type and the
        same argument names. We derive from Function here so must
        use samples
        """
        if not hasattr(self, "_param"):
            raise ValueError("Parameter p has not been set.")
        return self._bkd.sum(
            self._param[0] ** 2 * samples**2
            + self._param[1] ** 3 * samples
            + self._param[2] ** 4
            + self._param[0] * self._param[1],
            axis=1,  # Sum over the second dimension of x
        )

    def set_parameter(self, p: Array) -> None:
        """
        Set the parameter p.
        """
        self._param = p

    def jacobian_wrt_parameters(self, x: Array) -> Array:
        """
        Compute the Jacobian of f(x, p) with respect to p.

        Jacobian:
        - df/dp[0] = sum_j (2 * p[0] * x[j]**2 + p[1])
        - df/dp[1] = sum_j (3 * p[1]**2 * x[j] + p[0])
        - df/dp[2] = sum_j (4 * p[2]**3)
        """
        return self._bkd.stack(
            [
                self._bkd.sum(
                    2 * self._param[0] * x**2 + self._param[1], axis=1
                ),  # df/dp[0]
                self._bkd.sum(
                    3 * self._param[1] ** 2 * x + self._param[0], axis=1
                ),  # df/dp[1]
                self._bkd.sum(4 * self._param[2] ** 3 + x * 0.0, axis=1),  # df/dp[2]
            ],
            axis=1,
        )

    def hvp_wrt_parameters(self, x: Array, vec: Array) -> Array:
        """
        Compute the Hessian-vector product of f(x, p) with respect to p.

        Hessian:
        - d^2f/dp[0]^2 = sum_j (2 * x[j]**2)
        - d^2f/dp[1]^2 = sum_j (6 * p[1] * x[j])
        - d^2f/dp[2]^2 = sum_j (12 * p[2]**2)
        - d^2f/dp[0]dp[1] = sum_j (1)
        """
        hessian = self._bkd.zeros((self.nparams(), self.nparams()))
        hessian[0, 0] = self._bkd.sum(2 * x[0] ** 2, axis=0)  # d^2f/dp[0]^2
        hessian[1, 1] = self._bkd.sum(6 * self._param[1] * x[0], axis=0)  # d^2f/dp[1]^2
        hessian[2, 2] = self._bkd.sum(
            12 * self._param[2] ** 2 + x[0] * 0.0, axis=0
        )  # d^2f/dp[2]^2
        hessian[0, 1] = self._bkd.sum(1.0 + x[0] * 0.0, axis=0)  # d^2f/dp[0]dp[1]
        hessian[1, 0] = hessian[0, 1]  # Symmetry of Hessian

        # Compute Hessian-vector product
        return hessian @ vec


class TestFunctionOfParameters:
    def _setup(self, bkd):
        self.fixed_x = bkd.reshape(
            bkd.linspace(0, 1, 5), (1, -1)
        )  # Shape (1, npts)
        self.param_func_with_full_rank_hvp = ExampleParameterizedFunctionWithHVP(
            bkd
        )

    def test_function_of_parameters(self, bkd) -> None:
        """
        Test the functionality of FunctionOfParametersWithJacobian.
        """
        self._setup(bkd)
        param = bkd.array([1.0, 2.0, 3.0])[:, None]  # Single parameter vector
        self.param_func_with_full_rank_hvp.set_parameter(param)
        func_of_params = convert_to_function_of_parameters(
            self.param_func_with_full_rank_hvp, self.fixed_x
        )

        # Evaluate the function for a set of parameters
        result = func_of_params(param)

        # Expected result computation:
        expected_result = bkd.asarray(
            [
                [
                    bkd.sum(
                        param[0] ** 2 * self.fixed_x**2
                        + param[1] ** 3 * self.fixed_x
                        + param[2] ** 4
                        + param[0] * param[1]
                    )
                ]
            ]
        )
        bkd.assert_allclose(result, expected_result)

        jacobian = func_of_params.jacobian(param)

        # Expected Jacobian computation:
        expected_jacobian = bkd.stack(
            [
                bkd.sum(
                    2.0 * param[0] * self.fixed_x[0] ** 2 + param[1], axis=0
                ),  # df/dp[0]
                bkd.sum(
                    3.0 * param[1] ** 2 * self.fixed_x[0] + param[0], axis=0
                ),  # df/dp[1]
                bkd.sum(
                    4.0 * param[2] ** 3 + self.fixed_x[0] * 0.0, axis=0
                ),  # df/dp[2]
            ],
            axis=0,
        )[None, :]
        bkd.assert_allclose(jacobian, expected_jacobian)

        vec = bkd.array([1.0, 1.0, 1.0])[:, None]  # Vector for HVP
        hvp = func_of_params.hvp(param, vec)

        # Expected HVP computation:
        expected_hvp = bkd.stack(
            [
                bkd.sum(2.0 * self.fixed_x[0] ** 2, axis=0) * vec[0]
                + self.fixed_x.shape[1]
                * vec[1],  # d^2f/dp[0]^2 * vec[0] + d^2f/dp[0]dp[1] * vec[1]
                bkd.sum(6.0 * param[1] * self.fixed_x[0], axis=0) * vec[1]
                + self.fixed_x.shape[1]
                * vec[0],  # d^2f/dp[1]^2 * vec[1] + d^2f/dp[0]dp[1] * vec[0]
                self.fixed_x.shape[1] * 12.0 * param[2] ** 2 * vec[2],
                # d^2f/dp[2]^2 * vec[2]
            ],
            axis=0,
        )
        bkd.assert_allclose(hvp, expected_hvp)

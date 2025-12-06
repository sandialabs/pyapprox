from typing import Optional, Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.interface.functions.protocols.jacobian import (
    FunctionWithJVPProtocol,
)
from pyapprox.typing.interface.functions.protocols.validation import (
    validate_sample,
)


class JVPChecker(Generic[Array]):
    """
    Base class for checking the correctness of jacobian vector products.

    Parameters
    ----------
    function_object : DerivativeCheckerProtocol
        The function object satisfying the required protocol.
    backend : Backend
        The backend to use for computations.
    """

    def __init__(
        self,
        function: FunctionWithJVPProtocol[Array],
        symb: str = "J",
        fd_eps: Optional[Array] = None,
        direction: Optional[Array] = None,
        relative: bool = True,
        verbosity: int = 0,
    ):
        self._validate_function(function)
        self._fun = function
        self._bkd = function.bkd()

        if fd_eps is None:
            fd_eps = self._bkd.flip(self._bkd.logspace(-13, 0, 14))
        if direction is None:
            nvars = self._fun.nvars()
            direction = self._bkd.asarray(np.random.normal(0, 1, (nvars, 1)))
            direction /= self._bkd.norm(direction)

        self._symb = symb
        self._fd_eps = fd_eps
        self._direction = direction
        self._relative = relative
        self._verbosity = verbosity

    def _validate_function(
        self,
        function: FunctionWithJVPProtocol[Array],
    ) -> None:
        if not isinstance(function, FunctionWithJVPProtocol):
            raise ValueError(
                "The provided function must satisfy "
                "'FunctionWithJVPProtocol'. "
                f"Got an object of type {type(function).__name__}."
            )

    def check(self, sample: Array) -> Array:
        """
        Compare the result of an jvp with directional
        finite difference approximations.
        """
        validate_sample(self._fun.nvars(), sample)
        errors = []
        val = self._fun(sample)
        directional_grad = self._fun.jvp(sample, self._direction)
        relative = self._relative
        if self._bkd.norm(directional_grad) < 1e-16:
            if self._verbosity > 0:
                print("Gradient is zero so setting relative=False")
            relative = False

        row_format = "{:<12} {:<25} {:<25} {:<25}"
        headers = [
            "Eps",
            "norm({0}v)".format(self._symb),
            "norm({0}v_fd)".format(self._symb),
            "Rel. Errors" if relative else "Abs. Errors",
        ]
        if self._verbosity > 0:
            print(row_format.format(*headers))
        row_format = "{:<12.2e} {:<25} {:<25} {:<25}"

        for ii in range(self._fd_eps.shape[0]):
            sample_perturbed = (
                self._bkd.copy(sample) + self._fd_eps[ii] * self._direction
            )
            perturbed_val = self._fun(sample_perturbed)
            fd_directional_grad = (perturbed_val - val) / self._fd_eps[ii]
            errors.append(
                self._bkd.norm(
                    fd_directional_grad.reshape(directional_grad.shape)
                    - directional_grad
                )
            )
            if relative:
                if self._bkd.norm(directional_grad) < 1e-16:
                    raise RuntimeError(
                        "directional grad is zero thus grad is likely zero. "
                        "Set relative=False"
                    )
                errors[-1] /= self._bkd.norm(directional_grad)
            if self._verbosity > 0:
                print(
                    row_format.format(
                        self._fd_eps[ii],
                        self._bkd.norm(directional_grad),
                        self._bkd.norm(fd_directional_grad),
                        errors[ii],
                    )
                )
        return self._bkd.asarray(errors)

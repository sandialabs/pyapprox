"""
Hastings ecology ODE residual.

Implements a three-species ecology model (food chain dynamics):
    dY_1/dT = Y_1*(1 - Y_1) - a_1*Y_1*Y_2/(1 + b_1*Y_1)
    dY_2/dT = a_1*Y_1*Y_2/(1 + b_1*Y_1) - a_2*Y_2*Y_3/(1 + b_2*Y_2) - d_1*Y_2
    dY_3/dT = a_2*Y_2*Y_3/(1 + b_2*Y_2) - d_2*Y_3

References:
    Hastings, Alan, and Thomas Powell. "Chaos in a Three-Species Food Chain."
    Ecology 72, no. 3 (1991): 896-903.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.validation import validate_backend


class HastingsEcologyResidual(Generic[Array]):
    """
    Hastings ecology ODE residual.

    Three-species food chain model with saturating functional response.

    States: [y_1, y_2, y_3] (population densities)
    Parameters: [a_1, b_1, a_2, b_2, d_1, d_2, y_1_0, y_2_0, y_3_0]
        - a_1, a_2: predation rates
        - b_1, b_2: half-saturation constants
        - d_1, d_2: death rates
        - y_1_0, y_2_0, y_3_0: initial conditions

    Parameters
    ----------
    bkd : Backend
        Backend for array operations.
    """

    def __init__(self, bkd: Backend[Array]):
        validate_backend(bkd)
        self._bkd = bkd
        self._nstates = 3
        self._nparams = 9  # 6 physical params + 3 initial conditions
        self._time = 0.0
        self._param = None

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._nparams

    def set_time(self, time: float) -> None:
        """Set the current time."""
        self._time = time

    def set_param(self, param: Array) -> None:
        """Set the parameters."""
        if param.ndim == 2:
            param = param.flatten()
        if param.shape[0] != self._nparams:
            raise ValueError(
                f"Expected {self._nparams} parameters, got {param.shape[0]}"
            )
        self._param = param

    def get_initial_condition(self) -> Array:
        """Return initial condition from parameters."""
        return self._param[6:]

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the ODE right-hand side.

        Parameters
        ----------
        state : Array
            State [y_1, y_2, y_3]. Shape: (3,)

        Returns
        -------
        Array
            f(state). Shape: (3,)
        """
        y1, y2, y3 = state
        a1, b1, a2, b2, d1, d2 = self._param[:6]
        return self._bkd.stack([
            y1 * (1 - y1) - a1 * y1 * y2 / (1 + b1 * y1),
            a1 * y1 * y2 / (1 + b1 * y1)
            - a2 * y2 * y3 / (1 + b2 * y2)
            - d1 * y2,
            a2 * y2 * y3 / (1 + b2 * y2) - d2 * y3,
        ], axis=0)

    def jacobian(self, state: Array) -> Array:
        """
        Compute the state Jacobian df/dy.

        Parameters
        ----------
        state : Array
            State [y_1, y_2, y_3]. Shape: (3,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (3, 3)
        """
        y1, y2, y3 = state
        a1, b1, a2, b2, d1, d2 = self._param[:6]
        zero = y1 * 0.0

        # df_1/dy_1 = 1 - 2*y_1 - a_1*y_2/(1+b_1*y_1)^2
        df1_dy1 = (
            1
            - 2 * y1
            - a1 * y2 / (1 + b1 * y1)
            + a1 * b1 * y1 * y2 / (1 + b1 * y1) ** 2
        )
        df1_dy2 = -a1 * y1 / (1 + b1 * y1)

        df2_dy1 = a1 * y2 / (1 + b1 * y1) ** 2
        df2_dy2 = (
            -d1
            + a1 * y1 / (1 + b1 * y1)
            - a2 * y3 / (1 + b2 * y2) ** 2
        )
        df2_dy3 = -a2 * y2 / (1 + b2 * y2)

        df3_dy2 = a2 * y3 / (1 + b2 * y2) ** 2
        df3_dy3 = -d2 + a2 * y2 / (1 + b2 * y2)

        return self._bkd.stack([
            self._bkd.hstack([df1_dy1, df1_dy2, zero]),
            self._bkd.hstack([df2_dy1, df2_dy2, df2_dy3]),
            self._bkd.hstack([zero, df3_dy2, df3_dy3]),
        ], axis=0)

    def mass_matrix(self, nstates: int) -> Array:
        """Return the identity mass matrix."""
        return self._bkd.eye(nstates)

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix to a vector (identity, returns vec)."""
        return vec

    def param_jacobian(self, state: Array) -> Array:
        """
        Compute the parameter Jacobian df/dp.

        Parameters
        ----------
        state : Array
            State [y_1, y_2, y_3]. Shape: (3,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (3, 9)
        """
        y1, y2, y3 = state
        a1, b1, a2, b2, d1, d2 = self._param[:6]
        zero = y1 * 0.0

        row0 = self._bkd.hstack([
            -y1 * y2 / (1 + b1 * y1),
            a1 * y1**2 * y2 / (1 + b1 * y1) ** 2,
            zero, zero, zero, zero,
            zero, zero, zero,
        ])
        row1 = self._bkd.hstack([
            y1 * y2 / (1 + b1 * y1),
            -a1 * y1**2 * y2 / (1 + b1 * y1) ** 2,
            -y2 * y3 / (1 + b2 * y2),
            a2 * y2**2 * y3 / (1 + b2 * y2) ** 2,
            -y2,
            zero,
            zero, zero, zero,
        ])
        row2 = self._bkd.hstack([
            zero, zero,
            y2 * y3 / (1 + b2 * y2),
            -a2 * y2**2 * y3 / (1 + b2 * y2) ** 2,
            zero,
            -y3,
            zero, zero, zero,
        ])
        return self._bkd.stack([row0, row1, row2], axis=0)

    def initial_param_jacobian(self) -> Array:
        """
        Compute initial condition parameter Jacobian.

        The last 3 parameters are initial conditions, so this is -I
        for those columns.
        """
        return self._bkd.hstack([
            self._bkd.zeros((self._nstates, 6)),
            -self._bkd.eye(3),
        ])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nparams={self._nparams})"

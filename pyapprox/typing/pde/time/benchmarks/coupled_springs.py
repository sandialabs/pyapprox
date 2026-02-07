"""
Coupled springs ODE residual.

Implements a two-mass spring system with friction:
    x'_1 = y_1
    y'_1 = (-b_1*y_1 - k_1*(x_1 - L_1) + k_2*(x_2 - x_1 - L_2)) / m_1
    x'_2 = y_2
    y'_2 = (-b_2*y_2 - k_2*(x_2 - x_1 - L_2)) / m_2
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.validation import validate_backend


class CoupledSpringsResidual(Generic[Array]):
    """
    Coupled springs ODE residual.

    Two masses connected by springs with friction. The left end of the
    left spring is fixed.

    States: [x_1, y_1, x_2, y_2] (positions and velocities)
    Parameters: [m_1, m_2, k_1, k_2, L_1, L_2, b_1, b_2, x_1_0, y_1_0, x_2_0, y_2_0]
        - m_1, m_2: masses
        - k_1, k_2: spring constants
        - L_1, L_2: natural spring lengths
        - b_1, b_2: friction coefficients
        - x_1_0, y_1_0, x_2_0, y_2_0: initial conditions

    Parameters
    ----------
    bkd : Backend
        Backend for array operations.
    """

    def __init__(self, bkd: Backend[Array]):
        validate_backend(bkd)
        self._bkd = bkd
        self._nstates = 4
        self._nparams = 12  # 8 physical params + 4 initial conditions
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
        return self._param[8:]

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the ODE right-hand side.

        Parameters
        ----------
        state : Array
            State [x_1, y_1, x_2, y_2]. Shape: (4,)

        Returns
        -------
        Array
            f(state). Shape: (4,)
        """
        x1, y1, x2, y2 = state
        m1, m2, k1, k2, L1, L2, b1, b2 = self._param[:8]
        return self._bkd.hstack([
            y1,
            (-b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1,
            y2,
            (-b2 * y2 - k2 * (x2 - x1 - L2)) / m2,
        ])

    def jacobian(self, state: Array) -> Array:
        """
        Compute the state Jacobian df/dy.

        Parameters
        ----------
        state : Array
            State [x_1, y_1, x_2, y_2]. Shape: (4,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (4, 4)
        """
        x1, y1, x2, y2 = state
        m1, m2, k1, k2, L1, L2, b1, b2 = self._param[:8]
        zero = x1 * 0.0
        one = x1 * 0.0 + 1.0
        return self._bkd.stack([
            self._bkd.hstack([zero, one, zero, zero]),
            self._bkd.hstack([(-k1 - k2) / m1, -b1 / m1, k2 / m1, zero]),
            self._bkd.hstack([zero, zero, zero, one]),
            self._bkd.hstack([k2 / m2, zero, -k2 / m2, -b2 / m2]),
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
            State [x_1, y_1, x_2, y_2]. Shape: (4,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (4, 12)
        """
        x1, y1, x2, y2 = state
        m1, m2, k1, k2, L1, L2, b1, b2 = self._param[:8]
        zero = x1 * 0.0
        numer1 = -b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)
        numer2 = -b2 * y2 - k2 * (x2 - x1 - L2)

        row0 = self._bkd.zeros((self._nparams,))
        row1 = self._bkd.hstack([
            -numer1 / m1**2,
            zero,
            -(x1 - L1) / m1,
            (x2 - x1 - L2) / m1,
            k1 / m1,
            -k2 / m1,
            -y1 / m1,
            zero,
            zero, zero, zero, zero,
        ])
        row2 = self._bkd.zeros((self._nparams,))
        row3 = self._bkd.hstack([
            zero,
            -numer2 / m2**2,
            zero,
            -(x2 - x1 - L2) / m2,
            zero,
            k2 / m2,
            zero,
            -y2 / m2,
            zero, zero, zero, zero,
        ])
        return self._bkd.stack([row0, row1, row2, row3], axis=0)

    def initial_param_jacobian(self) -> Array:
        """
        Compute initial condition parameter Jacobian.

        The last 4 parameters are initial conditions, so this is -I
        for those columns.
        """
        return self._bkd.hstack([
            self._bkd.zeros((self._nstates, 8)),
            -self._bkd.eye(4),
        ])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nparams={self._nparams})"

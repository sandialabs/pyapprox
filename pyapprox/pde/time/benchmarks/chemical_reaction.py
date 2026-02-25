"""
Chemical reaction ODE residual.

Implements a surface adsorption model with three species:
    du/dt = a*z - c*u - 4*d*u*v
    dv/dt = 2*b*z^2 - 4*d*u*v
    dw/dt = e*z - f*w

where z = 1 - u - v - w (fraction of unoccupied surface).

Species:
    u: monomer
    v: dimer
    w: inert

References:
    Vigil et al., Phys. Rev. E., 1996.
    Makeev et al., J. Chem. Phys., 2002.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class ChemicalReactionResidual(Generic[Array]):
    """
    Chemical reaction ODE residual.

    Surface adsorption model describing species adsorbing onto a surface.

    States: [u, v, w] (monomer, dimer, inert)
    Parameters: [a, b, c, d, e, f]
        - a: adsorption rate for monomer
        - b: dimer formation rate
        - c: desorption rate for monomer
        - d: reaction rate
        - e: adsorption rate for inert
        - f: desorption rate for inert

    Initial condition is fixed at [0, 0, 0] (empty surface).

    Parameters
    ----------
    bkd : Backend
        Backend for array operations.
    """

    def __init__(self, bkd: Backend[Array]):
        validate_backend(bkd)
        self._bkd = bkd
        self._nstates = 3
        self._nparams = 6
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
        """Return initial condition (empty surface)."""
        return self._bkd.zeros(self._nstates)

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the ODE right-hand side.

        Parameters
        ----------
        state : Array
            State [u, v, w]. Shape: (3,)

        Returns
        -------
        Array
            f(state). Shape: (3,)
        """
        u, v, w = state
        a, b, c, d, e, f = self._param
        z = 1.0 - u - v - w
        return self._bkd.hstack([
            a * z - c * u - 4 * d * u * v,
            2 * b * z**2 - 4 * d * u * v,
            e * z - f * w,
        ])

    def jacobian(self, state: Array) -> Array:
        """
        Compute the state Jacobian df/dy.

        Parameters
        ----------
        state : Array
            State [u, v, w]. Shape: (3,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (3, 3)
        """
        u, v, w = state
        a, b, c, d, e, f = self._param
        z = 1.0 - u - v - w
        return self._bkd.stack([
            self._bkd.hstack([
                -a - c - 4 * d * v,
                -a - 4 * d * u,
                -a
            ]),
            self._bkd.hstack([
                -4 * b * z - 4 * d * v,
                -4 * b * z - 4 * d * u,
                -4 * b * z
            ]),
            self._bkd.hstack([-e, -e, -e - f]),
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
            State [u, v, w]. Shape: (3,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (3, 6)
        """
        u, v, w = state
        a, b, c, d, e, f = self._param
        z = 1.0 - u - v - w
        zero = u * 0.0

        row0 = self._bkd.hstack([z, zero, -u, -4 * u * v, zero, zero])
        row1 = self._bkd.hstack([zero, 2 * z**2, zero, -4 * u * v, zero, zero])
        row2 = self._bkd.hstack([zero, zero, zero, zero, z, -w])
        return self._bkd.stack([row0, row1, row2], axis=0)

    def initial_param_jacobian(self) -> Array:
        """
        Compute initial condition parameter Jacobian.

        Initial condition is fixed at zeros, so this is zero.
        """
        return self._bkd.zeros((self._nstates, self._nparams))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nparams={self._nparams})"

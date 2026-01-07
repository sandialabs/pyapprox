"""
Minimax objective function.

The minimax objective simply returns the slack variable t, which represents
the upper bound on all component objectives.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class MinimaxObjective(Generic[Array]):
    """
    Objective function for minimax optimization.

    Returns the slack variable t as the objective value. The constraints
    ensure t >= f_i(x) for all components i.

    Parameters
    ----------
    nmodel_vars : int
        Number of variables in the original problem (excluding slack).
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    The optimization variables are [t, x_1, ..., x_n] where:
    - t is the slack variable (minimax bound)
    - x_1, ..., x_n are the original problem variables

    The objective is simply J([t, x]) = t, with Jacobian [1, 0, ..., 0].

    This class satisfies SlackBasedObjectiveProtocol.
    """

    def __init__(self, nmodel_vars: int, bkd: Backend[Array]) -> None:
        self._nmodel_vars = nmodel_vars
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get computational backend."""
        return self._bkd

    def nslack(self) -> int:
        """Number of slack variables (always 1 for minimax)."""
        return 1

    def nvars(self) -> int:
        """Total number of variables (slack + original)."""
        return self._nmodel_vars + self.nslack()

    def nqoi(self) -> int:
        """Number of quantities of interest (always 1)."""
        return 1

    def __call__(self, sample: Array) -> Array:
        """
        Evaluate the minimax objective.

        Parameters
        ----------
        sample : Array
            Optimization variables [t, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            Objective value (just t). Shape: (1, 1)
        """
        return self._bkd.reshape(sample[0, 0], (1, 1))

    def jacobian(self, sample: Array) -> Array:
        """
        Jacobian of minimax objective.

        Parameters
        ----------
        sample : Array
            Optimization variables. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian [1, 0, ..., 0]. Shape: (1, nvars)
        """
        jac = self._bkd.zeros((1, self.nvars()))
        jac[0, 0] = 1.0
        return jac

    def hvp(self, sample: Array, vec: Array) -> Array:
        """
        Hessian-vector product (always zero for linear objective).

        Parameters
        ----------
        sample : Array
            Optimization variables. Shape: (nvars, 1)
        vec : Array
            Direction vector. Shape: (nvars, 1)

        Returns
        -------
        Array
            HVP (zero). Shape: (nvars, 1)
        """
        return self._bkd.zeros((self.nvars(), 1))

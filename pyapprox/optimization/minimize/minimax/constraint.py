"""
Minimax constraint.

Converts a multi-QoI objective into constraints t >= f_i(x) for minimax.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend

from .protocols import MultiQoIObjectiveProtocol


class MinimaxConstraint(Generic[Array]):
    """
    Constraint for minimax optimization: t - f_i(x) >= 0 for all i.

    This constraint converts a multi-QoI objective into inequality constraints
    that enforce the slack variable t to be an upper bound on all objectives.

    Parameters
    ----------
    model : MultiQoIObjectiveProtocol[Array]
        Multi-output objective function to minimize the maximum of.

    Notes
    -----
    The optimization variables are [t, x_1, ..., x_n] where t is the slack.

    The constraint values are:
        g_i([t, x]) = t - f_i(x) >= 0

    The Jacobian is:
        dg_i/d[t, x] = [1, -df_i/dx]

    This class is compatible with nonlinear constraint protocols.
    """

    def __init__(self, model: MultiQoIObjectiveProtocol[Array]) -> None:
        self._model = model
        self._bkd = model.bkd()

    def bkd(self) -> Backend[Array]:
        """Get computational backend."""
        return self._bkd

    def nslack(self) -> int:
        """Number of slack variables."""
        return 1

    def nvars(self) -> int:
        """Total number of variables (slack + original)."""
        return self._model.nvars() + self.nslack()

    def nqoi(self) -> int:
        """Number of constraints (same as model nqoi)."""
        return self._model.nqoi()

    def lb(self) -> Array:
        """
        Get lower bounds for constraints.

        Returns
        -------
        Array
            Lower bounds (zeros). Shape: (nqoi,)
        """
        return self._bkd.zeros((self._model.nqoi(),))

    def ub(self) -> Array:
        """
        Get upper bounds for constraints.

        Returns
        -------
        Array
            Upper bounds (inf). Shape: (nqoi,)
        """
        return self._bkd.full((self._model.nqoi(),), float("inf"))

    def __call__(self, sample: Array) -> Array:
        """
        Evaluate constraint: g_i = t - f_i(x).

        Parameters
        ----------
        sample : Array
            Optimization variables [t, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            Constraint values. Shape: (nqoi, 1)
        """
        t = sample[0, 0]
        x = sample[1:]
        f_vals = self._model(x)  # Shape: (nqoi, 1)
        return t - f_vals

    def jacobian(self, sample: Array) -> Array:
        """
        Jacobian of constraint: [1, -df/dx].

        Parameters
        ----------
        sample : Array
            Optimization variables. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nqoi, nvars)
        """
        x = sample[1:]
        model_jac = self._model.jacobian(x)  # Shape: (nqoi, nmodel_vars)
        ones_col = self._bkd.ones((model_jac.shape[0], 1))
        return self._bkd.hstack([ones_col, -model_jac])

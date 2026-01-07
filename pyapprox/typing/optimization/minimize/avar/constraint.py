"""
AVaR constraint.

Converts a multi-QoI objective into constraints for AVaR optimization:
s_i + t - f_i(x) >= 0 for all i.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.optimization.minimize.minimax.protocols import (
    MultiQoIObjectiveProtocol,
)


class AVaRConstraint(Generic[Array]):
    """
    Constraint for AVaR optimization: s_i + t - f_i(x) >= 0 for all i.

    This constraint ensures that the excess slack s_i captures any amount
    by which f_i(x) exceeds the VaR estimate t.

    Parameters
    ----------
    model : MultiQoIObjectiveProtocol[Array]
        Multi-output objective function to minimize the AVaR of.

    Notes
    -----
    The optimization variables are [t, s_1, ..., s_n, x_1, ..., x_m] where:
    - t is the VaR estimate
    - s_i are excess slack variables (one per scenario)
    - x are the original problem variables

    The constraint values are:
        g_i([t, s, x]) = t + s_i - f_i(x) >= 0

    The Jacobian is:
        dg_i/d[t, s, x] = [1, 0, ..., 1_i, ..., 0, -df_i/dx]
                               ^
                               i-th position (1-indexed after t)

    The constraint s_i >= 0 is handled by bounds, not this constraint.
    """

    def __init__(self, model: MultiQoIObjectiveProtocol[Array]) -> None:
        self._model = model
        self._bkd = model.bkd()
        self._nscenarios = model.nqoi()

    def bkd(self) -> Backend[Array]:
        """Get computational backend."""
        return self._bkd

    def nslack(self) -> int:
        """Number of slack variables (1 + nscenarios)."""
        return 1 + self._nscenarios

    def nvars(self) -> int:
        """Total number of variables (slack + original)."""
        return self._model.nvars() + self.nslack()

    def nqoi(self) -> int:
        """Number of constraints (same as model nqoi = nscenarios)."""
        return self._nscenarios

    def lb(self) -> Array:
        """
        Get lower bounds for constraints.

        Returns
        -------
        Array
            Lower bounds (zeros). Shape: (nqoi,)
        """
        return self._bkd.zeros((self._nscenarios,))

    def ub(self) -> Array:
        """
        Get upper bounds for constraints.

        Returns
        -------
        Array
            Upper bounds (inf). Shape: (nqoi,)
        """
        return self._bkd.full((self._nscenarios,), float("inf"))

    def __call__(self, sample: Array) -> Array:
        """
        Evaluate constraint: g_i = t + s_i - f_i(x).

        Parameters
        ----------
        sample : Array
            Optimization variables [t, s, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            Constraint values. Shape: (nqoi, 1)
        """
        t = sample[0, 0]
        s = sample[1:1 + self._nscenarios]  # Shape: (nscenarios, 1)
        x = sample[1 + self._nscenarios:]  # Shape: (nmodel_vars, 1)
        f_vals = self._model(x)  # Shape: (nqoi, 1)
        return t + s - f_vals

    def jacobian(self, sample: Array) -> Array:
        """
        Jacobian of constraint.

        The Jacobian for constraint i is:
        [1, 0, ..., 1, ..., 0, -df_i/dx]
            ^       ^
            s_1     s_i (i-th position is 1)

        Parameters
        ----------
        sample : Array
            Optimization variables. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nqoi, nvars)
        """
        x = sample[1 + self._nscenarios:]  # Shape: (nmodel_vars, 1)
        model_jac = self._model.jacobian(x)  # Shape: (nqoi, nmodel_vars)

        nqoi = self._nscenarios
        nvars = self.nvars()
        nmodel_vars = self._model.nvars()

        jac = self._bkd.zeros((nqoi, nvars))

        # Derivative w.r.t. t (first column is 1 for all constraints)
        jac[:, 0] = 1.0

        # Derivative w.r.t. s_i: constraint i has 1 in position (i, 1+i)
        for i in range(nqoi):
            jac[i, 1 + i] = 1.0

        # Derivative w.r.t. x: -df_i/dx
        jac[:, 1 + nqoi:] = -model_jac

        return jac

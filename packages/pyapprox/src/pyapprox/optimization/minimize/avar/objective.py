"""
AVaR (Average Value at Risk) objective function.

The AVaR objective returns t + (1/(n*(1-alpha))) * sum(s_i), where t is the
VaR estimate and s_i are the excess slack variables.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class AVaRObjective(Generic[Array]):
    """
    Objective function for AVaR optimization.

    Evaluates: J = t + (1/(n*(1-alpha))) * sum(s_i)

    Parameters
    ----------
    nmodel_vars : int
        Number of variables in the original problem (excluding slack).
    nscenarios : int
        Number of scenarios (objectives) being averaged.
    alpha : float
        Risk level in [0, 1). AVaR_alpha averages the worst (1-alpha) outcomes.
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    The optimization variables are [t, s_1, ..., s_n, x_1, ..., x_m] where:
    - t is the VaR estimate (1 variable)
    - s_1, ..., s_n are excess slack variables (n variables)
    - x_1, ..., x_m are the original problem variables (m variables)

    The objective is J([t, s, x]) = t + (1/(n*(1-alpha))) * sum(s),
    with Jacobian [1, c, c, ..., c, 0, ..., 0] where c = 1/(n*(1-alpha)).
    """

    def __init__(
        self,
        nmodel_vars: int,
        nscenarios: int,
        alpha: float,
        bkd: Backend[Array],
    ) -> None:
        if alpha < 0 or alpha >= 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self._nmodel_vars = nmodel_vars
        self._nscenarios = nscenarios
        self._alpha = alpha
        self._bkd = bkd
        # Coefficient for excess slack: 1/(n*(1-alpha))
        self._excess_coeff = 1.0 / (nscenarios * (1.0 - alpha))

    def bkd(self) -> Backend[Array]:
        """Get computational backend."""
        return self._bkd

    def alpha(self) -> float:
        """Risk level in [0, 1)."""
        return self._alpha

    def nslack(self) -> int:
        """Number of slack variables (1 + nscenarios)."""
        return 1 + self._nscenarios

    def nvars(self) -> int:
        """Total number of variables (slack + original)."""
        return self._nmodel_vars + self.nslack()

    def nqoi(self) -> int:
        """Number of quantities of interest (always 1)."""
        return 1

    def __call__(self, sample: Array) -> Array:
        """
        Evaluate the AVaR objective.

        Parameters
        ----------
        sample : Array
            Optimization variables [t, s, x]. Shape: (nvars, 1)

        Returns
        -------
        Array
            Objective value. Shape: (1, 1)
        """
        t = sample[0, 0]
        s = sample[1 : 1 + self._nscenarios, 0]  # Shape: (nscenarios,)
        obj = t + self._excess_coeff * self._bkd.sum(s)
        return self._bkd.reshape(obj, (1, 1))

    def jacobian(self, sample: Array) -> Array:
        """
        Jacobian of AVaR objective.

        Parameters
        ----------
        sample : Array
            Optimization variables. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian [1, c, ..., c, 0, ..., 0]. Shape: (1, nvars)
        """
        jac = self._bkd.zeros((1, self.nvars()))
        jac[0, 0] = 1.0  # Derivative w.r.t. t
        jac[0, 1 : 1 + self._nscenarios] = self._excess_coeff  # Derivatives w.r.t. s_i
        # Derivatives w.r.t. x are zero
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

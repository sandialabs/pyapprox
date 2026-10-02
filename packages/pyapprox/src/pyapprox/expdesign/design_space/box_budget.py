r"""Box-bounded design weights with a fixed total.

The feasible set is

.. math::

    \{w \in \mathbb{R}^d : l \le w_i \le u, \ \textstyle\sum_i w_i = n_b\},

with lower bound :math:`l`, upper bound :math:`u` and budget :math:`n_b`.
With :math:`l = 0`, :math:`u = 1` and :math:`n_b = 1` this is the
probability simplex; with :math:`n_b = k` its vertices are the designs
selecting :math:`k` of the :math:`d` candidates.
"""

from typing import Generic

from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.util.backends.protocols import Array, Backend


class BoxBudgetDesignSpace(Generic[Array]):
    """Design weights in ``[lower, upper]`` summing to ``budget``.

    Parameters
    ----------
    nvars : int
        Number of design weights.
    budget : float
        Total weight ``sum(w)``. Must lie in
        ``[nvars * lower, nvars * upper]``.
    bkd : Backend[Array]
        Computational backend.
    lower : float
        Lower bound of every weight. Default 0.
    upper : float
        Upper bound of every weight. Default 1.
    """

    def __init__(
        self,
        nvars: int,
        budget: float,
        bkd: Backend[Array],
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> None:
        if nvars < 1:
            raise ValueError(f"nvars must be positive, got {nvars}")
        if not lower < upper:
            raise ValueError(f"lower must be less than upper, got {lower} and {upper}")
        if not nvars * lower <= budget <= nvars * upper:
            raise ValueError(
                f"budget {budget} is infeasible: it must lie in "
                f"[{nvars * lower}, {nvars * upper}]"
            )
        self._nvars = nvars
        self._budget = float(budget)
        self._bkd = bkd
        self._lower = float(lower)
        self._upper = float(upper)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of design weights."""
        return self._nvars

    def lower(self) -> float:
        """Lower bound of every weight."""
        return self._lower

    def upper(self) -> float:
        """Upper bound of every weight."""
        return self._upper

    def budget(self) -> float:
        """Total weight ``sum(w)``."""
        return self._budget

    def bounds(self) -> Array:
        """Bounds of each weight. Shape: (nvars, 2)"""
        bounds = self._bkd.zeros((self._nvars, 2))
        bounds[:, 0] = self._lower
        bounds[:, 1] = self._upper
        return bounds

    def constraints(self) -> SequenceOfConstraintProtocols[Array]:
        """The single equality constraint ``sum(w) = budget``."""
        ones = self._bkd.ones((1, self._nvars))
        total = self._bkd.asarray([self._budget])
        return [PyApproxLinearConstraint(ones, total, total, self._bkd)]

    def initial(self) -> Array:
        """Uniform weights ``budget / nvars``. Shape: (nvars, 1)

        Feasible because the budget lies in ``[nvars * lower,
        nvars * upper]``.
        """
        return self._bkd.full((self._nvars, 1), self._budget / self._nvars)

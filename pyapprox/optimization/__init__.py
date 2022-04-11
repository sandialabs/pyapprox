"""The :mod:`pyapprox.optimization` module implements a number of popular tools for risk-averse regression and design.
"""

from pyapprox.optimization.first_order_stochastic_dominance import (
    solve_FSD_constrained_least_squares_smooth)

from pyapprox.optimization.second_order_stochastic_dominance import (
    solve_SSD_constrained_least_squares_smooth)
from pyapprox.optimization.cvar_regression import cvar_regression
from pyapprox.optimization.quantile_regression import (
    solve_quantile_regression, solve_least_squares_regression
)


__all__ = ["solve_FSD_constrained_least_squares_smooth",
           "solve_SSD_constrained_least_squares_smooth",
           "cvar_regression", "solve_quantile_regression",
           "solve_least_squares_regression"]

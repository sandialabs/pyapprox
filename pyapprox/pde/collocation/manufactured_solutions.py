from abc import ABC, abstractmethod
from functools import partial
import textwrap

import sympy as sp
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.autopde.sympy_utils import (
    _evaluate_sp_lambda,
    _evaluate_list_of_sp_lambda,
)


class ManufactureSolution(ABC):
    def __init__(
        self,
        nvars: int,
        sol_string: str,
        transient: bool = False,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._sol_string = sol_string
        self._transient = transient
        self._oned = oned
        self._solution_expression()
        self.sympy_expressions()
        self._expressions_to_functions()

    @abstractmethod
    def sympy_expressions(self):
        raise NotImplementedError

    @abstractmethod
    def _solution_expression(self):
        raise NotImplementedError

    def cartesian_symbols(self):
        return sp.symbols(["x", "y", "z"])[: self.nvars()]

    def time_symbol(self):
        return sp.symbols(["t"])

    def all_symbols(self):
        if not self._transient:
            return self.cartesian_symbols()
        return self.cartesian_symbols() + (self.time_symbol(),)

    def _expression_to_function(self, expr):
        all_symbs = self.all_symbols()
        forc_lambda = sp.lambdify(all_symbs, expr, "numpy")
        return partial(
            _evaluate_sp_lambda, forc_lambda, bkd=self._bkd, oned=self._oned
        )

    def _expression_list_to_function(self, exprs):
        all_symbs = self.all_symbols()
        forc_lambda = [sp.lambdify(all_symbs, expr, "numpy") for expr in exprs]
        return partial(
            _evaluate_list_of_sp_lambda,
            forc_lambda,
            bkd=self._bkd,
            oned=False,  # when passing a list must always return 2d array
        )

    def _expressions_to_functions(self):
        self.functions = dict()
        for name, expr in self._expressions.items():
            if isinstance(expr, list):
                self.functions[name] = self._expression_list_to_function(expr)
            else:
                self.functions[name] = self._expression_to_function(expr)

    def nvars(self):
        return self._nvars

    def __repr__(self):
        fields = f"{self._expressions}".split(",")
        expr_string = ",\n".join(fields)[1:-1].replace("'", "")
        return "{0}(\n {1}\n)".format(self.__class__.__name__, expr_string)


class ScalarSolutionMixin:
    def _solution_expression(self):
        sol_expr = sp.sympify(self._sol_string)
        self._expressions = {
            "solution": sol_expr,
            "forcing": 0,
        }

    def solution_symbols(self):
        return sp.symbols(["u"])


class DiffusionMixin:
    def sympy_diffusion_expressions(self):
        cartesian_symbs = self.cartesian_symbols()
        diff_expr = sp.sympify(self._diff_string)
        sol_expr = self._expressions["solution"]
        laplace_expr = sum(
            [
                (diff_expr * sol_expr.diff(symb, 1)).diff(symb, 1)
                for symb in cartesian_symbs
            ]
        )
        forc_expr = -laplace_expr
        flux_exprs = [
            diff_expr*sol_expr.diff(symb, 1) for symb in cartesian_symbs
        ]
        self._expressions["diffusion"] = diff_expr
        self._expressions["forcing"] += forc_expr
        self._expressions["flux"] = flux_exprs


class ReactionMixin:
    def sympy_reaction_expressions(self):
        react_str = self._react_str.replace(
            "u", "({0})".format(self._sol_string)
        )
        self._expressions["reaction"] = sp.sympify(react_str)
        self._expressions["forcing"] -= self._expressions["reaction"]


class AdvectionMixin:
    def sympy_advection_expressions(self):
        cartesian_symbs = self.cartesian_symbols()
        vel_exprs = [
            sp.sympify(vel_string) for vel_string in self._vel_strs
        ]
        advection_expr = sum(
            [vel_expr*self._expressions["solution"].diff(symb, 1)
             for vel_expr, symb in zip(vel_exprs, cartesian_symbs)])
        self._expressions["velocity"] = vel_exprs
        self._expressions["advection"] = advection_expr
        self._expressions["forcing"] += self._expressions["advection"]


class Diffusion(ScalarSolutionMixin, DiffusionMixin, ManufactureSolution):
    def __init__(
        self,
        nvars: int,
        sol_string: str,
        diff_string: str,
        transient: bool = False,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._diff_string = diff_string
        super().__init__(nvars, sol_string, transient, bkd, oned)

    def sympy_expressions(self):
        self.sympy_diffusion_expressions()


class ReactionDiffusion(ReactionMixin, Diffusion):
    def __init__(
        self,
        nvars: int,
        sol_string: str,
        diff_string: str,
        react_str: str,
        transient: bool = False,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._react_str = react_str
        super().__init__(nvars, sol_string, diff_string, transient, bkd, oned)

    def sympy_expressions(self):
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()


class AdvectionDiffusionReaction(AdvectionMixin, ReactionDiffusion):
    def __init__(
        self,
        nvars: int,
        sol_string: str,
        diff_string: str,
        react_str: str,
        vel_strs: list[str],
        transient: bool = False,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._vel_strs = vel_strs
        super().__init__(
            nvars, sol_string, diff_string, react_str, transient, bkd, oned
        )

    def sympy_expressions(self):
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()

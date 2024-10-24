from abc import ABC, abstractmethod
from functools import partial
import textwrap

import sympy as sp
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.autopde.sympy_utils import (
    _evaluate_sp_lambda,
    _evaluate_list_of_sp_lambda,
    _evaluate_transient_sp_lambda,
    _evaluate_list_of_transient_sp_lambda,
)


class ManufacturedSolution(ABC):
    def __init__(
        self,
        nvars: int,
        sol_string: str,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._sol_string = sol_string
        self._oned = oned
        self._expressions = dict()
        self.transient = dict()
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
        return sp.symbols(["T"])

    def all_symbols(self):
        return self.cartesian_symbols() + self.time_symbol()

    def _steady_expression_to_function(self, expr):
        all_symbs = self.cartesian_symbols()
        expr_lambda = sp.lambdify(all_symbs, expr, "numpy")
        return partial(
            _evaluate_sp_lambda, expr_lambda, bkd=self._bkd, oned=self._oned
        )

    def _steady_expression_list_to_function(self, exprs):
        all_symbs = self.cartesian_symbols()
        expr_lambda = [sp.lambdify(all_symbs, expr, "numpy") for expr in exprs]
        return partial(
            _evaluate_list_of_sp_lambda,
            expr_lambda,
            bkd=self._bkd,
            oned=False,  # when passing a list must always return 2d array
        )

    def _transient_expression_to_function(self, expr):
        all_symbs = self.all_symbols()
        expr_lambda = sp.lambdify(all_symbs, expr, "numpy")
        print(all_symbs)
        return partial(
            _evaluate_transient_sp_lambda, expr_lambda, bkd=self._bkd,
            oned=self._oned
        )

    def _transient_expression_list_to_function(self, exprs):
        all_symbs = self.all_symbols()
        expr_lambda = [sp.lambdify(all_symbs, expr, "numpy") for expr in exprs]
        return partial(
            _evaluate_list_of_transient_sp_lambda,
            expr_lambda,
            bkd=self._bkd,
            oned=False,  # when passing a list must always return 2d array
        )

    def is_transient(self):
        return self._bkd.any(list(self.transient.values()))

    def _expressions_to_functions(self):
        self.transient["forcing"] = self.is_transient()
        if (
                self._bkd.any(list(self.transient.values()))
                and not self.transient["solution"]
        ):
            raise ValueError(
                "solution must be transient because another function is"
            )
        self.functions = dict()
        for name, expr in self._expressions.items():
            if isinstance(expr, list):
                if not self.transient[name]:
                    self.functions[name] = (
                        self._steady_expression_list_to_function(expr)
                    )
                else:
                    self.functions[name] = (
                        self._transient_expression_list_to_function(expr)
                    )
            else:
                if not self.transient[name]:
                    self.functions[name] = self._steady_expression_to_function(
                        expr
                    )
                else:
                    self.functions[name] = (
                        self._transient_expression_to_function(expr)
                    )

    def nvars(self):
        return self._nvars

    def _set_expression_from_bool(self, name: str, expr, transient: bool):
        self._expressions[name] = expr
        self.transient[name] = transient

    def _set_expression(self, name: str, expr, expr_str: str):
        print(expr_str)
        if "T" in expr_str:
            transient = True
        else:
            transient = False
        self._set_expression_from_bool(name, expr, transient)

    def __repr__(self):
        fields = f"{self._expressions}".split(",")
        expr_string = ",\n".join(fields)[1:-1].replace("'", "")
        return "{0}(\n {1}\n)".format(self.__class__.__name__, expr_string)


class ScalarSolutionMixin:
    def _solution_expression(self):
        sol_expr = sp.sympify(self._sol_string)
        self._set_expression("solution", sol_expr, self._sol_string)
        # do not use set expression for forcing as we will
        # only know if it is transient once all functions have been
        # parsed
        self._expressions["forcing"] = 0

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
        self._set_expression("diffusion", diff_expr, self._diff_string)
        self._set_expression("flux", flux_exprs, self._sol_string)
        self._expressions["forcing"] += forc_expr


class ReactionMixin:
    def sympy_reaction_expressions(self):
        react_str = self._react_str.replace(
            "u", "({0})".format(self._sol_string)
        )
        self._set_expression("reaction", sp.sympify(react_str), react_str)
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
        self._set_expression(
            "velocity",  vel_exprs, ", ".join(self._vel_strs)
        )
        self._expressions["forcing"] += advection_expr


class AdvectionDiffusionReaction(
        ScalarSolutionMixin,
        DiffusionMixin,
        ReactionMixin,
        AdvectionMixin,
        ManufacturedSolution,
):
    def __init__(
        self,
        nvars: int,
        sol_string: str,
        diff_string: str,
        react_str: str,
        vel_strs: list[str],
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._diff_string = diff_string
        self._react_str = react_str
        self._vel_strs = vel_strs
        super().__init__(nvars, sol_string, bkd, oned)

    def sympy_expressions(self):
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()

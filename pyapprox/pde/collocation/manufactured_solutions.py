from abc import ABC, abstractmethod
from functools import partial
import textwrap
from typing import List

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
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._oned = oned
        self._expressions = dict()
        self.transient = dict()
        self._solution_expression()
        self.sympy_expressions()
        self.sympy_temporal_derivative_expression()
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
        return partial(
            _evaluate_transient_sp_lambda,
            expr_lambda,
            bkd=self._bkd,
            oned=self._oned,
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
        return self._bkd.any(self._bkd.array(list(self.transient.values())))

    def _expressions_to_functions(self):
        self.transient["forcing"] = self.is_transient()
        if (
            self._bkd.any(self._bkd.array(list(self.transient.values())))
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
        if name in self._expressions:
            if not isinstance(expr, list):
                self._expressions[name] += expr
            else:
                for ii in range(len(self._expressions[name])):
                    self._expressions[name][ii] += expr[ii]
            self.transient[name] = transient or self.transient[name]
        else:
            self._expressions[name] = expr
            self.transient[name] = transient

    def _set_expression(self, name: str, expr, expr_str: str):
        if "T" in expr_str:
            transient = True
        else:
            transient = False
        self._set_expression_from_bool(name, expr, transient)

    def __repr__(self):
        fields = f"{self._expressions}".split(",")
        expr_str = ",\n".join(fields)[1:-1].replace("'", "")
        return "{0}(\n {1}\n)".format(self.__class__.__name__, expr_str)


class ScalarSolutionMixin:
    def __init__(self, sol_str, *args, **kwargs):
        self._sol_str = sol_str
        super().__init__(*args, **kwargs)

    def _solution_expression(self):
        sol_expr = sp.sympify(self._sol_str)
        self._set_expression("solution", sol_expr, self._sol_str)
        # do not use set expression for forcing as we will
        # only know if it is transient once all functions have been
        # parsed
        self._expressions["forcing"] = 0

    def solution_symbols(self):
        return sp.symbols(["u"])

    def sympy_temporal_derivative_expression(self):
        if self.is_transient():
            self._expressions["forcing"] += self._expressions["solution"].diff(
                self.time_symbol()[0]
            )


class DiffusionMixin:
    def sympy_diffusion_expressions(self):
        cartesian_symbs = self.cartesian_symbols()
        diff_expr = sp.sympify(self._diff_str)
        sol_expr = self._expressions["solution"]
        laplace_expr = sum(
            [
                (diff_expr * sol_expr.diff(symb, 1)).diff(symb, 1)
                for symb in cartesian_symbs
            ]
        )
        forc_expr = -laplace_expr
        flux_exprs = [
            diff_expr * sol_expr.diff(symb, 1) for symb in cartesian_symbs
        ]
        self._set_expression("diffusion", diff_expr, self._diff_str)
        self._set_expression("flux", flux_exprs, self._sol_str)
        self._expressions["forcing"] += forc_expr


class ReactionMixin:
    def sympy_reaction_expressions(self):
        react_str = self._react_str.replace("u", "({0})".format(self._sol_str))
        self._set_expression("reaction", sp.sympify(react_str), react_str)
        self._expressions["forcing"] -= self._expressions["reaction"]


class AdvectionMixin:
    def sympy_advection_expressions(self):
        cartesian_symbs = self.cartesian_symbols()
        vel_exprs = [sp.sympify(vel_str) for vel_str in self._vel_strs]
        advection_expr = sum(
            [
                vel_expr * self._expressions["solution"].diff(symb, 1)
                for vel_expr, symb in zip(vel_exprs, cartesian_symbs)
            ]
        )
        self._set_expression("velocity", vel_exprs, ", ".join(self._vel_strs))
        self._expressions["forcing"] += advection_expr
        flux_exprs = [
            -vel_expr * self._expressions["solution"] for vel_expr in vel_exprs
        ]
        self._set_expression("flux", flux_exprs, self._sol_str)


class AdvectionDiffusionReaction(
    ScalarSolutionMixin,
    DiffusionMixin,
    ReactionMixin,
    AdvectionMixin,
    ManufacturedSolution,
):
    def __init__(
        self,
        sol_str: str,
        nvars: int,
        diff_str: str,
        react_str: str,
        vel_strs: list[str],
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._diff_str = diff_str
        self._react_str = react_str
        self._vel_strs = vel_strs
        super().__init__(sol_str, nvars, bkd, oned)

    def sympy_expressions(self):
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()
        self.sympy_advection_expressions()


class Helmholtz(
    ScalarSolutionMixin, DiffusionMixin, ReactionMixin, ManufacturedSolution
):
    def __init__(
        self,
        sol_str: str,
        nvars: int,
        sqwavenum_str: str,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._diff_str = "1"
        self._sqwavenum_str = sqwavenum_str
        self._react_str = f"u*{sqwavenum_str}"
        self._vel_strs = ["0"] * nvars
        super().__init__(sol_str, nvars, bkd, oned)

    def sympy_expressions(self):
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()
        self._set_expression(
            "sqwavenum",
            sp.sympify(self._sqwavenum_str),
            self._sqwavenum_str,
        )


class ShallowIce(ScalarSolutionMixin, ManufacturedSolution):
    def __init__(
        self,
        sol_str: str,
        nvars: int,
        bed_str: str,
        friction_str: str,
        A: float,
        rho: float,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._bed_str = bed_str
        self._friction_str = friction_str
        self._A = A
        self._rho = rho
        self._n = 3
        self._g = 9.81
        self._gamma = (
            2 * self._A * (self._rho * self._g) ** self._n / (self._n + 2)
        )
        super().__init__(sol_str, nvars, bkd, oned)

    def sympy_expressions(self):
        cartesian_symbs = self.cartesian_symbols()
        bed_expr = sp.sympify(self._bed_str)
        friction_expr = sp.sympify(self._friction_str)
        sol_expr = self._expressions["solution"]
        surface_expr = bed_expr + sol_expr
        surface_grad_exprs = [surface_expr.diff(s, 1) for s in cartesian_symbs]
        diffusion = (
            self._gamma
            * sol_expr ** (self._n + 2)
            * sum([gs**2 for gs in surface_grad_exprs]) ** ((self._n - 1) / 2)
            + self._rho * self._g / friction_expr * sol_expr**2
        )

        flux_exprs = [diffusion * gs for gs in surface_grad_exprs]
        forc_expr = -sum(
            [
                flux_expr.diff(symb, 1)
                for symb, flux_expr in zip(cartesian_symbs, flux_exprs)
            ]
        )
        self._set_expression("bed", bed_expr, self._bed_str)
        self._set_expression("friction", friction_expr, self._friction_str)
        self._set_expression("diffusion", diffusion, "")
        self._set_expression("flux", flux_exprs, self._sol_str)
        self._expressions["forcing"] += forc_expr


class VectorSolutionMixin:
    def __init__(self, sol_strs, *args, **kwargs):
        self._sol_strs = sol_strs
        self._nelems = len(sol_strs)
        super().__init__(*args, **kwargs)

    def _solution_expression(self):
        sol_exprs = [sp.sympify(sol_str) for sol_str in self._sol_strs]
        # next line will not work if one component is time dependent
        # while others are not
        self._set_expression("solution", sol_exprs, self._sol_strs[0])
        # do not use set expression for forcing as we will
        # only know if it is transient once all functions have been
        # parsed
        self._expressions["forcing"] = [0 for ii in range(self._nelems)]

    def solution_symbols(self):
        return sp.symbols(["u{ii+1}" for ii in range(self._nelems)])

    def sympy_temporal_derivative_expression(self):
        if self.is_transient():
            for ii in range(self._nvars+1):
                self._expressions["forcing"][ii] += (
                    self._expressions["solution"][ii].diff(
                        self.time_symbol()[0]
                    )
                )


class ShallowWave(VectorSolutionMixin, ManufacturedSolution):
    def __init__(
        self,
        nvars: int,
        depth_str: str,
        vel_strs: List[str],
        bed_str: str,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._depth_str = depth_str
        self._vel_strs = vel_strs
        self._bed_str = bed_str
        self._g = 9.81
        mom_strs = [f"{depth_str}*{vel_str}" for vel_str in vel_strs]
        super().__init__([depth_str] + mom_strs, nvars, bkd, oned)

    def sympy_expressions(self):
        cartesian_symbs = self.cartesian_symbols()
        bed_expr = sp.sympify(self._bed_str)
        depth_expr = self._expressions["solution"][0]
        mom_exprs = self._expressions["solution"][1:]

        flux_exprs = [None for ii in range(self._nvars+1)]
        flux_exprs[0] = [-mom_expr for mom_expr in mom_exprs]
        flux_exprs[1] = [None for ii in range(self._nvars)]
        flux_exprs[1][0] = -(
            mom_exprs[0] ** 2 * depth_expr + 0.5 * self._g * depth_expr**2
        )
        if self._nvars > 1:
            flux_exprs[1][1] = -mom_exprs[0] * mom_exprs[1] / depth_expr
            flux_exprs[2] = [
                -mom_exprs[0] * mom_exprs[1] / depth_expr,
                -(
                    mom_exprs[1] ** 2 / depth_expr
                    + 0.5 * self._g * depth_expr**2
                ),
            ]

        forc_expr = [
            -sum(
                [
                    flux.diff(s, 1)
                    for flux, s in zip(flux_exprs[ii], cartesian_symbs)
                ]
            )
            for ii in range(self._nvars+1)
        ]
        self._set_expression("bed", bed_expr, self._bed_str)
        self._set_expression("flux", flux_exprs, self._depth_str)
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_expr)
        ]

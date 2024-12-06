from abc import ABC, abstractmethod
from functools import partial
from typing import List
import copy

import sympy as sp
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.autopde.sympy_utils import (
    _evaluate_sp_lambda,
    _evaluate_list_of_sp_lambda,
    _evaluate_list_of_list_of_sp_lambda,
    _evaluate_transient_sp_lambda,
    _evaluate_list_of_transient_sp_lambda,
    _evaluate_list_of_list_of_transient_sp_lambda,
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
    def ncomponents(self) -> int:
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
            oned=False,
        )

    def _steady_expression_list_of_lists_to_function(self, exprs):
        all_symbs = self.cartesian_symbols()
        expr_lambda = [
            [sp.lambdify(all_symbs, expr, "numpy") for expr in row]
            for row in exprs
        ]
        return partial(
            _evaluate_list_of_list_of_sp_lambda,
            expr_lambda,
            bkd=self._bkd,
            oned=False,
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
            oned=False,
        )

    def _transient_expression_list_of_lists_to_function(self, exprs):
        all_symbs = self.all_symbols()
        expr_lambda = [
            [sp.lambdify(all_symbs, expr, "numpy") for expr in row]
            for row in exprs
        ]
        return partial(
            _evaluate_list_of_list_of_transient_sp_lambda,
            expr_lambda,
            bkd=self._bkd,
            oned=False,
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
            if isinstance(expr, list) and not isinstance(expr[0], list):
                if not self.transient[name]:
                    self.functions[name] = (
                        self._steady_expression_list_to_function(expr)
                    )
                else:
                    self.functions[name] = (
                        self._transient_expression_list_to_function(expr)
                    )
            elif isinstance(expr, list) and isinstance(expr[0], list):
                if not self.transient[name]:
                    self.functions[name] = (
                        self._steady_expression_list_of_lists_to_function(expr)
                    )
                else:
                    self.functions[name] = (
                        self._transient_expression_list_of_lists_to_function(
                            expr
                        )
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
            self._set_expression(
                "forcing_without_time_deriv",
                self._expressions["forcing"],
                self._sol_str,
            )
            self._expressions["forcing"] += self._expressions["solution"].diff(
                self.time_symbol()[0]
            )

    def ncomponents(self) -> int:
        return 1


class DiffusionMixin:
    def _sympy_diffusion_expressions(self, diff_str, sol_expr):
        cartesian_symbs = self.cartesian_symbols()
        diff_expr = sp.sympify(diff_str)
        # laplace_expr = sum(
        #     [
        #         (diff_expr * sol_expr.diff(symb, 1)).diff(symb, 1)
        #         for symb in cartesian_symbs
        #     ]
        # )
        # forc_expr = -laplace_expr
        flux_exprs = [
            -diff_expr * sol_expr.diff(symb, 1) for symb in cartesian_symbs
        ]
        forc_expr = sum(
            flux.diff(symb, 1)
            for flux, symb in zip(flux_exprs, cartesian_symbs)
        )
        return diff_expr, flux_exprs, forc_expr

    def sympy_diffusion_expressions(self):
        diff_expr, flux_exprs, forc_expr = self._sympy_diffusion_expressions(
            self._diff_str, self._expressions["solution"]
        )
        self._set_expression("diffusion", diff_expr, self._diff_str)
        self._set_expression("flux", flux_exprs, self._sol_str)
        self._expressions["forcing"] += forc_expr


class ReactionMixin:
    def _sympy_reaction_expressions(self, react_str, sol_str):
        # react_expr_wrt_sol = sp.sympify(react_str)
        # sol_symb = self.solution_symbols()[0]
        # react_prime_expr = react_expr_wrt_sol.diff(sol_symb)
        # react_prime_expr = react_prime_expr.subs(
        #     sol_symb, self._expressions["solution"]
        # )
        react_str = react_str.replace("u", "({0})".format(sol_str))
        return react_str, sp.sympify(react_str)  # , react_prime_expr

    def sympy_reaction_expressions(self):
        react_str, react_expr = self._sympy_reaction_expressions(
            self._react_str, self._sol_str
        )
        self._set_expression("reaction", react_expr, react_str)
        # self._set_expression("reaction_prime", react_prime_expr, react_str)
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
        if self._conservative:
            flux_exprs = [
                vel_expr * self._expressions["solution"]
                for vel_expr in vel_exprs
            ]
            self._set_expression("flux", flux_exprs, self._sol_str)


class ManufacturedAdvectionDiffusionReaction(
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
        conservative: bool = True,
    ):
        self._diff_str = diff_str
        self._react_str = react_str
        self._vel_strs = vel_strs
        self._conservative = conservative
        super().__init__(sol_str, nvars, bkd, oned)

    def sympy_expressions(self):
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()
        self.sympy_advection_expressions()


class ManufacturedNonLinearAdvectionDiffusionReaction(
    ManufacturedAdvectionDiffusionReaction
):
    def __init__(
        self,
        sol_str: str,
        nvars: int,
        linear_diff_str: str,
        nonlinear_diff_op_str: str,
        react_str: str,
        vel_strs: list[str],
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
        conservative: bool = True,
    ):
        nonlinear_diff_str = f"({linear_diff_str}) * ({nonlinear_diff_op_str})"
        self._linear_diff_str = linear_diff_str
        diff_str = nonlinear_diff_str.replace("u", "({0})".format(sol_str))
        super().__init__(
            sol_str,
            nvars,
            diff_str,
            react_str,
            vel_strs,
            bkd,
            oned,
            conservative,
        )

    def sympy_expressions(self):
        super().sympy_expressions()
        linear_diff = sp.sympify(self._linear_diff_str)
        self._set_expression("linear_diffusion", linear_diff, self._sol_str)


class ManufacturedHelmholtz(
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


class ManufacturedShallowIce(ScalarSolutionMixin, ManufacturedSolution):
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
        self._ncomponents = len(sol_strs)
        super().__init__(*args, **kwargs)

    def ncomponents(self) -> int:
        return self._ncomponents

    def _solution_expression(self):
        sol_exprs = [sp.sympify(sol_str) for sol_str in self._sol_strs]
        # next line will not work if one component is time dependent
        # while others are not
        self._set_expression("solution", sol_exprs, self._sol_strs[0])
        # do not use set expression for forcing as we will
        # only know if it is transient once all functions have been
        # parsed
        self._expressions["forcing"] = [0 for ii in range(self._ncomponents)]

    def solution_symbols(self):
        return sp.symbols(["u{ii+1}" for ii in range(self._ncomponents)])

    def sympy_temporal_derivative_expression(self):
        if self.is_transient():
            self._set_expression(
                "forcing_without_time_deriv",
                copy.deepcopy(self._expressions["forcing"]),
                self._sol_strs[0],
            )

            for ii in range(self.ncomponents()):
                self._expressions["forcing"][ii] += self._expressions[
                    "solution"
                ][ii].diff(self.time_symbol()[0])


class ManufacturedShallowWave(VectorSolutionMixin, ManufacturedSolution):
    def __init__(
        self,
        nvars: int,
        depth_str: str,
        mom_strs: List[str],
        bed_str: str,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._depth_str = depth_str
        self._mom_strs = mom_strs
        self._bed_str = bed_str
        self._g = 9.81
        super().__init__([depth_str] + mom_strs, nvars, bkd, oned)

    def sympy_expressions(self):
        cartesian_symbs = self.cartesian_symbols()
        bed_expr = sp.sympify(self._bed_str)

        if self._nvars == 1:
            h, uh = self._expressions["solution"]
            flux_exprs = [
                [uh],
                [(uh**2) / h + 0.5 * self._g * h**2],
            ]
        else:
            h, uh, vh = self._expressions["solution"]
            uvh = uh * vh / h
            flux_exprs = [
                [uh, vh],
                [(uh**2) / h + 0.5 * self._g * h**2, uvh],
                [uvh, (vh**2) / h + 0.5 * self._g * h**2],
            ]
        forc_expr = [
            sum(
                [
                    flux.diff(s, 1)
                    for flux, s in zip(flux_exprs[ii], cartesian_symbs)
                ]
            ).simplify()
            for ii in range(self._nvars + 1)
        ]
        # assume code always applies bed gradient forcing
        for ii in range(self._nvars):
            forc_expr[ii + 1] += (
                self._g * h * bed_expr.diff(cartesian_symbs[ii], 1)
            )
        self._set_expression("bed", bed_expr, self._bed_str)
        self._set_expression("flux", flux_exprs, self._depth_str)
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_expr)
        ]


class ManufacturedTwoSpeciesReactionDiffusion(
    VectorSolutionMixin,
    DiffusionMixin,
    ReactionMixin,
    ManufacturedSolution,
):
    r"""
    Reaction Vector: R(u0, u1) = [c0*u0**p0-u1, c1*u1**p1+u0]
    for coefficients c0, c1 and powers p0 and p1
    """

    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        diff_strs: List[str],
        react_strs: List[str],
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._diff_strs = diff_strs
        self._react_strs = react_strs
        super().__init__(sol_strs, nvars, bkd, oned)

    def sympy_diffusion_expressions(self):
        diff_expr0, flux_exprs0, forc_expr0 = (
            self._sympy_diffusion_expressions(
                self._diff_strs[0], self._expressions["solution"][0]
            )
        )
        diff_expr1, flux_exprs1, forc_expr1 = (
            self._sympy_diffusion_expressions(
                self._diff_strs[1], self._expressions["solution"][1]
            )
        )
        forc_exprs = [forc_expr0, forc_expr1]
        flux_exprs = [flux_exprs0, flux_exprs1]
        diff_exprs = [diff_expr0, diff_expr1]

        self._set_expression("diffusion", diff_exprs, self._diff_strs[0])
        self._set_expression("flux", flux_exprs, self._sol_strs[0])
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]

    def sympy_reaction_expressions(self):
        react_str0, react_expr0 = self._sympy_reaction_expressions(
            self._react_strs[0], self._sol_strs[0]
        )
        react_str1, react_expr1 = self._sympy_reaction_expressions(
            self._react_strs[1], self._sol_strs[1]
        )
        react_exprs = [react_expr0, react_expr1]
        react_exprs = [
            react_expr0 - self._expressions["solution"][1],
            react_expr1 + self._expressions["solution"][0],
        ]
        self._set_expression("reaction", react_exprs, self._react_strs[0])
        self._expressions["forcing"] = [
            f - g for f, g in zip(self._expressions["forcing"], react_exprs)
        ]

    def sympy_expressions(self):
        self.sympy_diffusion_expressions()
        self.sympy_reaction_expressions()


class ManufacturedShallowShelfVelocityEquations(
    VectorSolutionMixin,
    ManufacturedSolution,
):
    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        bed_str: str,
        depth_str: str,
        friction_str: str,
        A: float,
        rho: float,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):

        self._bed_str = bed_str
        self._depth_str = depth_str
        self._friction_str = friction_str
        self._A = A
        self._rho = rho
        self._g = 9.81
        self._n = 3
        super().__init__(sol_strs, nvars, bkd, oned)

    def velocity_expressions(self):
        return self._expressions["solution"]

    def sympy_expressions(self):
        cartesian_symbs = self.cartesian_symbols()
        bed_expr = sp.sympify(self._bed_str)
        depth_expr = sp.sympify(self._depth_str)
        friction_expr = sp.sympify(self._friction_str)
        surface_expr = bed_expr + depth_expr
        surface_grad_exprs = [surface_expr.diff(s, 1) for s in cartesian_symbs]

        u, v = self.velocity_expressions()
        ux = u.diff(cartesian_symbs[0])
        uy = u.diff(cartesian_symbs[1])
        vx = v.diff(cartesian_symbs[0])
        vy = v.diff(cartesian_symbs[1])

        effective_strain_rate = (
            ux**2 + vy**2 + ux * vy + 0.25 * (uy + vx) ** 2
        ) ** (1 / 2)
        mu_expr = (
            0.5
            * self._A ** (-1 / self._n)
            * effective_strain_rate ** (1 / self._n - 1)
        )
        offdiag_strain = 0.5 * (uy + vx)
        strain_tensor = [
            [2. * ux + vy, offdiag_strain],
            [offdiag_strain, ux + 2. * vy],
        ]
        flux = [[2 * mu_expr * s for s in row] for row in strain_tensor]
        forc_expr0 = (
            -(depth_expr * flux[0][0]).diff(cartesian_symbs[0])
            - (depth_expr * flux[0][1]).diff(cartesian_symbs[1])
            + friction_expr * u
            + self._rho * self._g * depth_expr * surface_grad_exprs[0]
        )
        forc_expr1 = (
            -(depth_expr * flux[1][0]).diff(cartesian_symbs[0])
            - (depth_expr * flux[1][1]).diff(cartesian_symbs[1])
            + friction_expr * v
            + self._rho * self._g * depth_expr * surface_grad_exprs[1]
        )
        forc_exprs = [forc_expr0, forc_expr1]
        # for now assume always steady
        self._set_expression("bed", bed_expr, self._bed_str)
        self._set_expression("depth", depth_expr, self._depth_str)
        self._set_expression("surface", surface_expr, self._depth_str)
        self._set_expression("friction", friction_expr, self._friction_str)
        self._set_expression(
            "effective_strain_rate", effective_strain_rate, self._sol_strs[0]
        )
        self._set_expression("flux", flux, self._sol_strs[0])
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]


class ManufacturedShallowShelfVelocityAndDepthEquations(
    ManufacturedShallowShelfVelocityEquations
):
    def __init__(
        self,
        vel_strs: List[str],
        nvars: int,
        bed_str: str,
        depth_str: str,
        friction_str: str,
        A: float,
        rho: float,
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        sol_strs = [depth_str] + vel_strs
        super().__init__(
            sol_strs, nvars, bed_str, depth_str, friction_str, A, rho, bkd, oned
        )

    def velocity_expressions(self):
        return self._expressions["solution"][1:]

    def sympy_expressions(self):
        super().sympy_expressions()
        cartesian_symbs = self.cartesian_symbols()
        depth = self._expressions["depth"]
        vel_exprs = self.velocity_expressions()
        # note sign in front of divergence is the negative of that used in
        # diffusion equation
        depth_forc = sum(
            [
                (depth * vel_expr).diff(symb, 1)
                for vel_expr, symb in zip(vel_exprs, cartesian_symbs)
            ]
        )
        self._expressions["velocity_forcing"] = self._expressions["forcing"]
        self._expressions["forcing"] = [depth_forc] + self._expressions["forcing"]
        self._expressions["depth_forcing"] = depth_forc
        self.transient["depth_forcing"] = self.is_transient()
        self.transient["velocity_forcing"] = self.is_transient()
        velocity_flux = self._expressions["flux"]
        depth_flux = [[depth*vel for vel in vel_exprs]]
        flux = depth_flux + velocity_flux
        del self._expressions["flux"]
        self._set_expression("flux", flux, self._sol_strs[0])

    def sympy_temporal_derivative_expression(self):
        if not self.is_transient():
            raise ValueError("Equations must be transient")
        # only depth equations depends on temporal derivative
        self._set_expression(
                "depth_forcing_without_time_deriv",
                copy.deepcopy(self._expressions["depth_forcing"]),
                self._sol_strs[0],
        )
        self._expressions["depth_forcing"] += self._expressions[
                "solution"
            ][0].diff(self.time_symbol()[0], 1)


class ManufacturedLinearElasticityEquations(
    VectorSolutionMixin,
    ManufacturedSolution,
):
    def __init__(
        self,
        sol_strs: List[str],
        nvars: int,
        lambda_str: int,
        mu_str: int,
        # body_forc_strs: List[str],
        bkd=NumpyLinAlgMixin,
        oned: bool = False,
    ):
        self._lambda_str = lambda_str
        self._mu_str = mu_str
        # self._body_forc_strs = body_forc_strs
        super().__init__(sol_strs, nvars, bkd, oned)

    def sympy_expressions(self):
        cartesian_symbs = self.cartesian_symbols()
        lambda_expr = sp.sympify(self._lambda_str)
        mu_expr = sp.sympify(self._mu_str)
        # body_forc_exprs = [
        #     sp.sympify(bforce_str) for bforce_str in self._bforce_strs
        # ]
        self._set_expression("lambda", lambda_expr, self._lambda_str)
        self._set_expression("mu", mu_expr, self._mu_str)
        # self._set_expression(
        #     "body_force", body_forc_exprs, self._body_forc_strs[0]
        # )

        disp_expr = self._expressions["solution"]
        exx = disp_expr[0].diff(cartesian_symbs[0], 1)
        exy = 0.5 * (
            disp_expr[0].diff(cartesian_symbs[1], 1)
            + disp_expr[1].diff(cartesian_symbs[0], 1)
        )
        eyy = disp_expr[1].diff(cartesian_symbs[1], 1)

        trace_e = exx + eyy
        tauxx = lambda_expr * trace_e + 2. * mu_expr * exx
        tauxy = 2.0 * mu_expr * exy
        tauyy = lambda_expr * trace_e + 2. * mu_expr * eyy
        tau = [
            [tauxx, tauxy],
            [tauxy, tauyy]
        ]
        # compute divergence of tau
        forc_exprs = [
            -sum(
                [
                    tau[ii][jj].diff(cartesian_symbs[jj], 1)
                    #- body_forc_exprs[ii]
                    for jj in range(2)
                ]
            )
            for ii in range(2)
        ]
        # tau is the flux
        self._set_expression("flux", tau, self._sol_strs[0])
        self._expressions["forcing"] = [
            f + g for f, g in zip(self._expressions["forcing"], forc_exprs)
        ]

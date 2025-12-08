"""Mixins for building manufactured solutions.

These mixins add specific PDE terms (diffusion, reaction, advection)
to manufactured solution classes.
"""

from typing import List, Tuple, Any, Dict, Callable

import sympy as sp


class DiffusionMixin:
    """Mixin for diffusion terms: -div(D * grad(u)).

    Adds diffusion-related sympy expressions to a manufactured solution.
    The forcing term contribution is: div(D * grad(u)).
    """

    _diff_str: str
    _sol_str: str
    _expressions: Dict[str, Any]
    cartesian_symbols: Callable
    _set_expression: Callable

    def _sympy_diffusion_expressions(
        self, diff_str: str, sol_expr: sp.Expr
    ) -> Tuple[sp.Expr, List[sp.Expr], sp.Expr]:
        """Compute sympy expressions for diffusion terms.

        Parameters
        ----------
        diff_str : str
            String representation of diffusion coefficient.
        sol_expr : sp.Expr
            Sympy expression for the solution.

        Returns
        -------
        diff_expr : sp.Expr
            Diffusion coefficient expression.
        flux_exprs : List[sp.Expr]
            Flux components: -D * du/dx_i for each dimension.
        forc_expr : sp.Expr
            Forcing contribution: div(flux) = div(-D * grad(u)).
        """
        cartesian_symbs = self.cartesian_symbols()
        diff_expr = sp.sympify(diff_str)
        # Flux = -D * grad(u)
        flux_exprs = [
            -diff_expr * sol_expr.diff(symb, 1) for symb in cartesian_symbs
        ]
        # Forcing = div(flux) = sum(d/dx_i(-D * du/dx_i))
        forc_expr = sum(
            flux.diff(symb, 1)
            for flux, symb in zip(flux_exprs, cartesian_symbs)
        )
        return diff_expr, flux_exprs, forc_expr

    def sympy_diffusion_expressions(self) -> None:
        """Build and store diffusion sympy expressions."""
        diff_expr, flux_exprs, forc_expr = self._sympy_diffusion_expressions(
            self._diff_str, self._expressions["solution"]
        )
        self._set_expression("diffusion", diff_expr, self._diff_str)
        self._set_expression("flux", flux_exprs, self._sol_str)
        self._expressions["forcing"] += forc_expr


class ReactionMixin:
    """Mixin for reaction terms: -R(u).

    Adds reaction-related sympy expressions to a manufactured solution.
    The forcing term contribution is: -R(u).
    """

    _react_str: str
    _sol_str: str
    _expressions: Dict[str, Any]
    _set_expression: Callable

    def _sympy_reaction_expressions(
        self, react_str: str, sol_str: str
    ) -> Tuple[str, sp.Expr]:
        """Compute sympy expressions for reaction terms.

        Parameters
        ----------
        react_str : str
            String representation of reaction term (may contain 'u').
        sol_str : str
            String representation of the solution.

        Returns
        -------
        react_str_subst : str
            Reaction string with 'u' substituted with solution.
        react_expr : sp.Expr
            Reaction expression evaluated at solution.
        """
        # Substitute 'u' with the solution expression
        react_str_subst = react_str.replace("u", "({0})".format(sol_str))
        return react_str_subst, sp.sympify(react_str_subst)

    def sympy_reaction_expressions(self) -> None:
        """Build and store reaction sympy expressions."""
        react_str, react_expr = self._sympy_reaction_expressions(
            self._react_str, self._sol_str
        )
        self._set_expression("reaction", react_expr, react_str)
        # Forcing contribution: -reaction (since PDE is du/dt = ... - reaction)
        self._expressions["forcing"] -= self._expressions["reaction"]


class AdvectionMixin:
    """Mixin for advection terms: v . grad(u).

    Adds advection-related sympy expressions to a manufactured solution.
    The forcing term contribution is: v . grad(u).
    """

    _vel_strs: List[str]
    _conservative: bool
    _sol_str: str
    _expressions: Dict[str, Any]
    cartesian_symbols: Callable
    _set_expression: Callable

    def sympy_advection_expressions(self) -> None:
        """Build and store advection sympy expressions."""
        cartesian_symbs = self.cartesian_symbols()
        vel_exprs = [sp.sympify(vel_str) for vel_str in self._vel_strs]
        # Advection = v . grad(u)
        advection_expr = sum(
            vel_expr * self._expressions["solution"].diff(symb, 1)
            for vel_expr, symb in zip(vel_exprs, cartesian_symbs)
        )
        self._set_expression("velocity", vel_exprs, ", ".join(self._vel_strs))
        self._expressions["forcing"] += advection_expr
        if self._conservative:
            # Conservative form: flux = v * u
            flux_exprs = [
                vel_expr * self._expressions["solution"]
                for vel_expr in vel_exprs
            ]
            self._set_expression("flux", flux_exprs, self._sol_str)

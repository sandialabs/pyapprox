"""Base classes for manufactured solutions.

Provides the base ManufacturedSolution class and solution mixins
for scalar and vector PDEs.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Generic, List, Dict, Callable, Union, Any
import copy

import sympy as sp

from pyapprox.typing.util.backends.protocols import Array, Backend


def _evaluate_sp_lambda(
    sp_lambda: Callable,
    xx: Array,
    bkd: Backend[Array],
    oned: bool = False,
) -> Array:
    """Evaluate a sympy lambda function at points.

    Parameters
    ----------
    sp_lambda : Callable
        Sympy lambdified function.
    xx : Array
        Points to evaluate at. Shape: (nvars, npts) or (npts,) for 1D.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D array instead of column vector.

    Returns
    -------
    Array
        Function values. Shape: (npts,) if oned else (npts, 1).
    """
    if len(xx.shape) == 1:
        sp_args = (xx,)
    else:
        sp_args = tuple(xx[ii, :] for ii in range(xx.shape[0]))
    vals = sp_lambda(*sp_args)
    # Check if vals is array-like (has shape attribute)
    if hasattr(vals, 'shape'):
        if oned:
            return bkd.asarray(vals)
        vals_arr = bkd.asarray(vals)
        return vals_arr[:, None] if vals_arr.ndim == 1 else vals_arr
    # vals is a scalar
    if oned:
        npts = xx.shape[0] if len(xx.shape) == 1 else xx.shape[1]
        return bkd.full((npts,), float(vals))
    npts = xx.shape[0] if len(xx.shape) == 1 else xx.shape[1]
    return bkd.full((npts, 1), float(vals))


def _evaluate_transient_sp_lambda(
    sp_lambda: Callable,
    xx: Array,
    time: float,
    bkd: Backend[Array],
    oned: bool = False,
) -> Array:
    """Evaluate a transient sympy lambda function at points and time.

    Parameters
    ----------
    sp_lambda : Callable
        Sympy lambdified function (takes space coords + time).
    xx : Array
        Points to evaluate at. Shape: (nvars, npts).
    time : float
        Time value.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D array instead of column vector.

    Returns
    -------
    Array
        Function values. Shape: (npts,) if oned else (npts, 1).
    """
    if len(xx.shape) == 1:
        sp_args = (xx, time)
    else:
        sp_args = tuple(xx[ii, :] for ii in range(xx.shape[0])) + (time,)
    vals = sp_lambda(*sp_args)
    # Check if vals is array-like (has shape attribute)
    if hasattr(vals, 'shape'):
        if oned:
            return bkd.asarray(vals)
        vals_arr = bkd.asarray(vals)
        return vals_arr[:, None] if vals_arr.ndim == 1 else vals_arr
    # vals is a scalar
    if oned:
        npts = xx.shape[0] if len(xx.shape) == 1 else xx.shape[1]
        return bkd.full((npts,), float(vals))
    npts = xx.shape[0] if len(xx.shape) == 1 else xx.shape[1]
    return bkd.full((npts, 1), float(vals))


def _evaluate_list_of_sp_lambda(
    sp_lambdas: List[Callable],
    xx: Array,
    bkd: Backend[Array],
    oned: bool = False,
) -> Array:
    """Evaluate a list of sympy lambda functions.

    Parameters
    ----------
    sp_lambdas : List[Callable]
        List of sympy lambdified functions.
    xx : Array
        Points to evaluate at.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays.

    Returns
    -------
    Array
        Stacked values. Shape: (npts, nfuncs).
    """
    vals = [
        _evaluate_sp_lambda(sp_lambda, xx, bkd, oned)
        for sp_lambda in sp_lambdas
    ]
    return bkd.hstack(vals)


def _evaluate_list_of_transient_sp_lambda(
    sp_lambdas: List[Callable],
    xx: Array,
    time: float,
    bkd: Backend[Array],
    oned: bool = False,
) -> Array:
    """Evaluate a list of transient sympy lambda functions."""
    vals = [
        _evaluate_transient_sp_lambda(sp_lambda, xx, time, bkd, oned)
        for sp_lambda in sp_lambdas
    ]
    return bkd.hstack(vals)


def _evaluate_list_of_list_of_sp_lambda(
    sp_lambdas: List[List[Callable]],
    xx: Array,
    bkd: Backend[Array],
    oned: bool = False,
) -> Array:
    """Evaluate a list of lists of sympy lambda functions."""
    vals = [
        _evaluate_list_of_sp_lambda(row, xx, bkd, oned)
        for row in sp_lambdas
    ]
    return bkd.stack(vals, axis=0)


def _evaluate_list_of_list_of_transient_sp_lambda(
    sp_lambdas: List[List[Callable]],
    xx: Array,
    time: float,
    bkd: Backend[Array],
    oned: bool = False,
) -> Array:
    """Evaluate a list of lists of transient sympy lambda functions."""
    vals = [
        _evaluate_list_of_transient_sp_lambda(row, xx, time, bkd, oned)
        for row in sp_lambdas
    ]
    return bkd.stack(vals, axis=0)


class ManufacturedSolution(ABC, Generic[Array]):
    """Base class for manufactured solutions.

    Uses sympy to symbolically compute forcing terms and derivatives
    for the Method of Manufactured Solutions (MMS).

    Parameters
    ----------
    nvars : int
        Number of spatial dimensions.
    bkd : Backend
        Computational backend.
    oned : bool
        If True, return 1D arrays from functions.
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        oned: bool = False,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._oned = oned
        self._expressions: Dict[str, Any] = {}
        self.transient: Dict[str, bool] = {}
        self._solution_expression()
        self.sympy_expressions()
        self.sympy_temporal_derivative_expression()
        self._expressions_to_functions()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    @abstractmethod
    def sympy_expressions(self) -> None:
        """Build sympy expressions for all PDE terms."""
        raise NotImplementedError

    @abstractmethod
    def ncomponents(self) -> int:
        """Return number of solution components."""
        raise NotImplementedError

    @abstractmethod
    def _solution_expression(self) -> None:
        """Set up the solution expression."""
        raise NotImplementedError

    def cartesian_symbols(self) -> List[sp.Symbol]:
        """Return sympy symbols for spatial coordinates."""
        return list(sp.symbols(["x", "y", "z"])[: self.nvars()])

    def time_symbol(self) -> List[sp.Symbol]:
        """Return sympy symbol for time."""
        return list(sp.symbols(["T"]))

    def all_symbols(self) -> List[sp.Symbol]:
        """Return all sympy symbols (space + time)."""
        return self.cartesian_symbols() + self.time_symbol()

    def _steady_expression_to_function(self, expr) -> Callable:
        """Convert a steady expression to a callable function."""
        all_symbs = self.cartesian_symbols()
        expr_lambda = sp.lambdify(all_symbs, expr, "numpy")
        return partial(
            _evaluate_sp_lambda, expr_lambda, bkd=self._bkd, oned=self._oned
        )

    def _steady_expression_list_to_function(self, exprs: List) -> Callable:
        """Convert a list of steady expressions to a callable function."""
        all_symbs = self.cartesian_symbols()
        expr_lambda = [sp.lambdify(all_symbs, expr, "numpy") for expr in exprs]
        return partial(
            _evaluate_list_of_sp_lambda,
            expr_lambda,
            bkd=self._bkd,
            oned=False,
        )

    def _steady_expression_list_of_lists_to_function(
        self, exprs: List[List]
    ) -> Callable:
        """Convert a list of lists of steady expressions to a callable."""
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

    def _transient_expression_to_function(self, expr) -> Callable:
        """Convert a transient expression to a callable function."""
        all_symbs = self.all_symbols()
        expr_lambda = sp.lambdify(all_symbs, expr, "numpy")
        return partial(
            _evaluate_transient_sp_lambda,
            expr_lambda,
            bkd=self._bkd,
            oned=self._oned,
        )

    def _transient_expression_list_to_function(self, exprs: List) -> Callable:
        """Convert a list of transient expressions to a callable function."""
        all_symbs = self.all_symbols()
        expr_lambda = [sp.lambdify(all_symbs, expr, "numpy") for expr in exprs]
        return partial(
            _evaluate_list_of_transient_sp_lambda,
            expr_lambda,
            bkd=self._bkd,
            oned=False,
        )

    def _transient_expression_list_of_lists_to_function(
        self, exprs: List[List]
    ) -> Callable:
        """Convert a list of lists of transient expressions to a callable."""
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

    def is_transient(self) -> bool:
        """Check if any expression is time-dependent."""
        if not self.transient:
            return False
        return any(self.transient.values())

    def _expressions_to_functions(self) -> None:
        """Convert all sympy expressions to callable functions."""
        self.transient["forcing"] = self.is_transient()
        if (
            any(self.transient.values())
            and not self.transient.get("solution", False)
        ):
            raise ValueError(
                "solution must be transient because another function is"
            )
        self.functions: Dict[str, Callable] = {}
        for name, expr in self._expressions.items():
            if isinstance(expr, list) and not isinstance(expr[0], list):
                if not self.transient.get(name, False):
                    self.functions[name] = (
                        self._steady_expression_list_to_function(expr)
                    )
                else:
                    self.functions[name] = (
                        self._transient_expression_list_to_function(expr)
                    )
            elif isinstance(expr, list) and isinstance(expr[0], list):
                if not self.transient.get(name, False):
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
                if not self.transient.get(name, False):
                    self.functions[name] = self._steady_expression_to_function(
                        expr
                    )
                else:
                    self.functions[name] = (
                        self._transient_expression_to_function(expr)
                    )

    def nvars(self) -> int:
        """Return number of spatial dimensions."""
        return self._nvars

    def _set_expression_from_bool(
        self, name: str, expr: Any, transient: bool
    ) -> None:
        """Set an expression with explicit transient flag."""
        if name in self._expressions:
            if not isinstance(expr, list):
                self._expressions[name] += expr
            else:
                for ii in range(len(self._expressions[name])):
                    self._expressions[name][ii] += expr[ii]
            self.transient[name] = transient or self.transient.get(name, False)
        else:
            self._expressions[name] = expr
            self.transient[name] = transient

    def _set_expression(self, name: str, expr: Any, expr_str: str) -> None:
        """Set an expression, inferring transient from 'T' in string."""
        transient = "T" in expr_str
        self._set_expression_from_bool(name, expr, transient)

    def __repr__(self) -> str:
        fields = f"{self._expressions}".split(",")
        expr_str = ",\n".join(fields)[1:-1].replace("'", "")
        return "{0}(\n {1}\n)".format(self.__class__.__name__, expr_str)


class ScalarSolutionMixin:
    """Mixin for scalar (single component) solutions.

    Provides _solution_expression and sympy_temporal_derivative_expression
    for scalar PDEs.
    """

    _sol_str: str
    _expressions: Dict[str, Any]
    transient: Dict[str, bool]
    _set_expression: Callable
    time_symbol: Callable
    is_transient: Callable

    def __init__(self, sol_str: str, *args, **kwargs):
        self._sol_str = sol_str
        super().__init__(*args, **kwargs)

    def _solution_expression(self) -> None:
        """Set up scalar solution expression."""
        sol_expr = sp.sympify(self._sol_str)
        self._set_expression("solution", sol_expr, self._sol_str)
        # Initialize forcing to 0, will be built by mixins
        self._expressions["forcing"] = sp.Integer(0)

    def solution_symbols(self) -> List[sp.Symbol]:
        """Return sympy symbol for solution."""
        return list(sp.symbols(["u"]))

    def sympy_temporal_derivative_expression(self) -> None:
        """Add temporal derivative to forcing for transient problems."""
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
        """Return 1 for scalar solutions."""
        return 1


class VectorSolutionMixin:
    """Mixin for vector (multi-component) solutions.

    Provides _solution_expression and sympy_temporal_derivative_expression
    for vector PDEs.
    """

    _sol_strs: List[str]
    _ncomponents: int
    _expressions: Dict[str, Any]
    transient: Dict[str, bool]
    _set_expression: Callable
    time_symbol: Callable
    is_transient: Callable

    def __init__(self, sol_strs: List[str], *args, **kwargs):
        self._sol_strs = sol_strs
        self._ncomponents = len(sol_strs)
        super().__init__(*args, **kwargs)

    def ncomponents(self) -> int:
        """Return number of solution components."""
        return self._ncomponents

    def _solution_expression(self) -> None:
        """Set up vector solution expressions."""
        sol_exprs = [sp.sympify(sol_str) for sol_str in self._sol_strs]
        self._set_expression("solution", sol_exprs, self._sol_strs[0])
        # Initialize forcing to zeros for each component
        self._expressions["forcing"] = [
            sp.Integer(0) for _ in range(self._ncomponents)
        ]

    def solution_symbols(self) -> List[sp.Symbol]:
        """Return sympy symbols for solution components."""
        return list(sp.symbols([f"u{ii+1}" for ii in range(self._ncomponents)]))

    def sympy_temporal_derivative_expression(self) -> None:
        """Add temporal derivatives to forcing for transient problems."""
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

import os
from typing import List

import sympy as sp

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin


def convert_sympy_equations_to_latex(
    equations: List[str], tex_filename: str, compile_pdf: bool = True
):
    out = r"""
    \documentclass[]{article}
    \usepackage{amsmath,amssymb}
    \begin{document}
    """
    print(len(equations))
    # lines = [r"\begin{align*}%s\end{align*}"%sp.latex(eq) for eq in equations]
    lines = [sp.latex(eq, mode="equation*") for eq in equations]
    out += os.linesep.join(lines) + r"\end{document}"
    print(out)

    makefile_trunk = os.path.dirname(tex_filename)
    if not os.path.exists(makefile_trunk):
        os.makedirs(makefile_trunk)

    with open(tex_filename, "w") as f:
        f.write(out)

    if compile_pdf:
        cur_dir = os.path.abspath(os.path.curdir)
        os.chdir(makefile_trunk)
        import subprocess

        tex_out = subprocess.check_output(
            [
                "pdflatex",
                "--interaction=nonstopmode",
                os.path.split(tex_filename)[-1],
            ],
            stderr=subprocess.STDOUT,
        )
        os.chdir(cur_dir)


def _evaluate_sp_lambda(
    sp_lambda: callable,
    xx: Array,
    bkd: BackendMixin = NumpyMixin,
    oned: bool = False,
):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args)
    if isinstance(vals, bkd.array_type()):
        if oned:
            return vals
        return vals[:, None]
    if oned:
        return bkd.full((xx.shape[1],), vals)
    return bkd.full((xx.shape[1], 1), vals)


def _evaluate_transient_sp_lambda(
    sp_lambda: callable,
    xx: Array,
    time: float,
    bkd: BackendMixin = NumpyMixin,
    oned: bool = False,
):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args, time)
    if isinstance(vals, bkd.array_type()):
        if oned:
            return vals
        return vals[:, None]
    if oned:
        return bkd.full((xx.shape[1],), vals)
    return bkd.full((xx.shape[1], 1), vals)


def _evaluate_list_of_sp_lambda(
    sp_lambdas: callable,
    xx: Array,
    as_list: bool = False,
    bkd: BackendMixin = NumpyMixin,
    oned: bool = False,
):
    # sp_lambda returns list of values from multiple functions
    vals = [
        _evaluate_sp_lambda(sp_lambda, xx, bkd, oned)
        for sp_lambda in sp_lambdas
    ]
    if as_list:
        return vals
    return bkd.hstack(vals)


def _evaluate_list_of_transient_sp_lambda(
    sp_lambdas: callable,
    xx: Array,
    time: float,
    as_list: bool = False,
    bkd: BackendMixin = NumpyMixin,
    oned: bool = False,
):
    # sp_lambda returns list of values from multiple functions
    vals = [
        _evaluate_transient_sp_lambda(sp_lambda, xx, time, bkd, oned)
        for sp_lambda in sp_lambdas
    ]
    if as_list:
        return vals
    return bkd.hstack(vals)


def _evaluate_list_of_list_of_sp_lambda(
    sp_lambdas: List[callable],
    xx: Array,
    as_list: bool = False,
    bkd: BackendMixin = NumpyMixin,
    oned: bool = False,
):
    # sp_lambda returns list of values from multiple functions
    vals = [
        _evaluate_list_of_sp_lambda(row, xx, as_list, bkd, oned)
        for row in sp_lambdas
    ]
    if as_list:
        return vals
    return bkd.stack(vals, axis=0)


def _evaluate_list_of_list_of_transient_sp_lambda(
    sp_lambdas: List[callable],
    xx: Array,
    time: float,
    as_list=False,
    bkd: BackendMixin = NumpyMixin,
    oned: bool = False,
):
    # sp_lambda returns list of values from multiple functions
    vals = [
        _evaluate_list_of_transient_sp_lambda(
            row, xx, time, as_list, bkd, oned
        )
        for row in sp_lambdas
    ]
    if as_list:
        return vals
    return bkd.stack(vals, axis=0)


# if __name__== "__main__":
#     import subprocess
#     s0 = sp.Symbol(r'\sigma_0^2')
#     s1 = sp.Symbol(r'\sigma_1^2')
#     a01 = sp.Symbol('a_{01}')
#     a02 = sp.Symbol('a_{02}')
#     a12 = sp.Symbol('a_{12}')

#     y = sp.Symbol('y')
#     x = sp.Matrix([[sp.Symbol('x_0')],[sp.Symbol('x_1')]])

#     Sxx = sp.Matrix([[s0,a01*s0],[a01*s0,s1]])
#     a = sp.Matrix([[a02],[a12]])
#     print(Sxx)
#     print((a.T*Sxx*a)[0,0])
#     from pyapprox.sympy import convert_sympy_equations_to_latex
#     convert_sympy_equations_to_latex([
#         sp.Eq(sp.MatrixSymbol(r'\Sigma_{xx}',2,2),Sxx),
#         sp.Eq(y,(a.T*x)[0]),
#         sp.simplify((a.T*Sxx*a)[0,0]),],'temp/temp.tex')

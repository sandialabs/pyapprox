#!/usr/bin/env python
import os, sympy as sp
import subprocess
def convert_sympy_equations_to_latex(equations,tex_filename,compile_pdf=True):
    out = r"""
    \documentclass[]{article}
    \usepackage{amsmath,amssymb}
    \begin{document}
    """
    print(len(equations))
    #lines = [r"\begin{align*}%s\end{align*}"%sp.latex(eq) for eq in equations]
    lines = [sp.latex(eq,mode='equation*')
             for eq in equations]
    out += os.linesep.join(lines) + r"\end{document}"
    print (out)

    makefile_trunk = os.path.dirname(tex_filename)
    if not os.path.exists(makefile_trunk):
        os.makedirs(makefile_trunk)
    
    with open(tex_filename, 'w') as f:
        f.write(out)

    if compile_pdf:
        cur_dir = os.path.abspath(os.path.curdir)
        os.chdir(makefile_trunk)
        import subprocess
        tex_out = subprocess.check_output(
            ['pdflatex', '--interaction=nonstopmode',
             os.path.split(tex_filename)[-1]],stderr=subprocess.STDOUT)
        os.chdir(cur_dir)

if __name__== "__main__":    
    import sympy as sp
    s0 = sp.Symbol(r'\sigma_0^2')
    s1 = sp.Symbol(r'\sigma_1^2')
    a01 = sp.Symbol('a_{01}')
    a02 = sp.Symbol('a_{02}')
    a12 = sp.Symbol('a_{12}')

    y = sp.Symbol('y')
    x = sp.Matrix([[sp.Symbol('x_0')],[sp.Symbol('x_1')]])

    Sxx = sp.Matrix([[s0,a01*s0],[a01*s0,s1]])
    a = sp.Matrix([[a02],[a12]])
    print(Sxx)
    print((a.T*Sxx*a)[0,0])
    from pyapprox.sympy_utilities import convert_sympy_equations_to_latex
    convert_sympy_equations_to_latex([
        sp.Eq(sp.MatrixSymbol(r'\Sigma_{xx}',2,2),Sxx),
        sp.Eq(y,(a.T*x)[0]),
        sp.simplify((a.T*Sxx*a)[0,0]),],'temp/temp.tex')

"""Mathematical expressions from text, evaluated with a backend.

- :mod:`~pyapprox.util.expressions.names`: what an expression may refer
  to (variables, named constants, extra functions).
- :mod:`~pyapprox.util.expressions.parsing`: text to sympy, without
  running the text.
- :mod:`~pyapprox.util.expressions.evaluation`: sympy to values on
  backend arrays.

Parsing needs no backend and evaluation needs no text, so symbolic code
imports only the first two.
"""

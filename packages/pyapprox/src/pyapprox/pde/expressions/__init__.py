"""PDE expressions: the naming conventions over
:mod:`pyapprox.util.expressions`.

- :mod:`~pyapprox.pde.expressions.symbols`: coordinates, time, normals,
  field jets, and the names a PDE expression may use (symbolic only).
- :mod:`~pyapprox.pde.expressions.evaluation`: PDE expressions at points
  of ``(ndim, npts)`` on a backend.

Nothing here assumes an exact solution: deployed models use it for
forcing and boundary data given as text, and the verification harness for
manufactured data.
"""

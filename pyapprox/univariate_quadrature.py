from warnings import warn
msg = f"The module pyappprox.{__name__} has now moved to "
msg += "pyappprox.univariate_polynomials.quadrature"
warn(msg, DeprecationWarning)
from pyapprox.univariate_polynomials.quadrature import *
